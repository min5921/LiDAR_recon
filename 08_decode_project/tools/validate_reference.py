import argparse
import csv
import math
from pathlib import Path

import numpy as np


H = W = 468
S = H * W


def load(path, channels):
    return np.fromfile(path, np.float32).reshape(channels, H, W)


def corners(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    half_x, half_y = dx / 2, dy / 2
    return [
        (x + px * cosine - py * sine, y + px * sine + py * cosine)
        for px, py in [(-half_x, -half_y), (half_x, -half_y),
                       (half_x, half_y), (-half_x, half_y)]
    ]


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def intersect(a, b, p, q):
    first, second = cross(p, q, a), cross(p, q, b)
    scale = first / (first - second) if abs(first - second) > 1e-12 else 1.0
    return (a[0] + (b[0] - a[0]) * scale,
            a[1] + (b[1] - a[1]) * scale)


def iou(first, second):
    bound = (first[3] + first[4] + second[3] + second[4]) / 2
    if abs(first[0] - second[0]) > bound or abs(first[1] - second[1]) > bound:
        return 0.0
    polygon, clip = corners(first), corners(second)
    for edge in range(4):
        p, q = clip[edge], clip[(edge + 1) % 4]
        old, polygon = polygon, []
        if not old:
            break
        for index, current in enumerate(old):
            previous = old[index - 1]
            current_inside = cross(p, q, current) >= -1e-9
            previous_inside = cross(p, q, previous) >= -1e-9
            if current_inside != previous_inside:
                polygon.append(intersect(previous, current, p, q))
            if current_inside:
                polygon.append(current)
    intersection = abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon)))) / 2 if polygon else 0
    union = first[3] * first[4] + second[3] * second[4] - intersection
    return intersection / union if intersection else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--decode-dir", required=True, type=Path)
    args = parser.parse_args()

    reg = load(args.head_dir / "reg.bin", 2)
    height = load(args.head_dir / "height.bin", 1)
    dim = load(args.head_dir / "dim.bin", 3)
    rot = load(args.head_dir / "rot.bin", 2)
    heatmap = load(args.head_dir / "hm.bin", 3)
    candidates = []
    for source_index in range(S):
        y, x = divmod(source_index, W)
        logits = heatmap[:, y, x]
        label = int(np.argmax(logits))
        score = float(1 / (1 + np.exp(-np.float64(logits[label]))))
        if score <= 0.1:
            continue
        box = [
            (x + float(reg[0, y, x])) * 0.32 - 74.88,
            (y + float(reg[1, y, x])) * 0.32 - 74.88,
            float(height[0, y, x]),
            *np.exp(dim[:, y, x].astype(np.float64)).tolist(),
            float(np.arctan2(rot[0, y, x], rot[1, y, x])),
            score, label, source_index,
        ]
        finite = all(math.isfinite(value) for value in box[:8])
        in_range = (-80 <= box[0] <= 80 and -80 <= box[1] <= 80
                    and -10 <= box[2] <= 10)
        if finite and in_range:
            candidates.append(box)

    candidates.sort(key=lambda box: (-box[7], box[9]))
    candidates = candidates[:4096]
    kept = []
    for box in candidates:
        if all(iou(previous, box) <= 0.7 for previous in kept):
            kept.append(box)
            if len(kept) == 500:
                break

    with (args.decode_dir / "detections.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == len(kept), (len(actual), len(kept))
    names = ["x", "y", "z", "dx", "dy", "dz", "yaw", "score"]
    maximum_difference = 0.0
    for row, expected in zip(actual, kept):
        assert int(row["label"]) == expected[8]
        assert int(row["source_index"]) == expected[9]
        maximum_difference = max(
            maximum_difference,
            max(abs(float(row[name]) - expected[index])
                for index, name in enumerate(names)),
        )
    print(f"candidates: {len(candidates)}")
    print(f"kept: {len(kept)}")
    print(f"max_abs_diff: {maximum_difference:.9g}")
    if maximum_difference > 1e-4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
