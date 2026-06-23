import argparse
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def is_probably_zero_filled(path: Path, byte_count: int = 512) -> bool:
    with path.open("rb") as f:
        data = f.read(byte_count)
    return bool(data) and all(byte == 0 for byte in data)


def inspect_tar(path: Path, max_members: int = 5):
    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()[:max_members]
            return {
                "is_tar": True,
                "members": [member.name for member in members],
                "error": None,
            }
    except Exception as exc:
        return {
            "is_tar": False,
            "members": [],
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--inspect-tar", action="store_true")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"root does not exist: {args.root}")

    files = [path for path in args.root.rglob("*") if path.is_file()]
    by_suffix = Counter(path.suffix.lower() for path in files)
    by_parent = defaultdict(int)
    total_size = 0

    suspicious = []
    tar_reports = []
    for path in files:
        total_size += path.stat().st_size
        by_parent[str(path.parent)] += 1

        suffixes = "".join(path.suffixes).lower()
        if ".gstmp" in suffixes:
            suspicious.append({
                "path": str(path),
                "reason": "download temp suffix .gstmp",
                "size": path.stat().st_size,
            })
        elif path.suffix.lower() == ".tar" and is_probably_zero_filled(path):
            suspicious.append({
                "path": str(path),
                "reason": "first bytes are zero-filled",
                "size": path.stat().st_size,
            })

        if args.inspect_tar and path.suffix.lower() == ".tar":
            report = inspect_tar(path)
            report["path"] = str(path)
            report["size"] = path.stat().st_size
            tar_reports.append(report)

    report = {
        "root": str(args.root),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "extensions": dict(sorted(by_suffix.items())),
        "parent_file_counts": dict(sorted(by_parent.items())),
        "suspicious": suspicious,
        "tar_reports": tar_reports,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

