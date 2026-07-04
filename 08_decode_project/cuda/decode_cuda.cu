#include "centerpoint/decode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace centerpoint {
namespace {
void ok(cudaError_t s, const char *op) {
  if (s != cudaSuccess)
    throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
struct Buf {
  float *p = nullptr;
  std::size_t n = 0;
  Buf() = default;
  explicit Buf(std::size_t x) : n(x) {
    ok(cudaMalloc(&p, n * 4), "cudaMalloc");
  }
  ~Buf() {
    if (p)
      cudaFree(p);
  }
  Buf(const Buf &) = delete;
  Buf &operator=(const Buf &) = delete;
  Buf(Buf &&o) noexcept : p(o.p), n(o.n) {
    o.p = nullptr;
    o.n = 0;
  }
  Buf &operator=(Buf &&o) noexcept {
    if (this != &o) {
      if (p)
        cudaFree(p);
      p = o.p;
      n = o.n;
      o.p = nullptr;
      o.n = 0;
    }
    return *this;
  }
};
Buf upload(const std::vector<float> &v) {
  Buf b(v.size());
  ok(cudaMemcpy(b.p, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload");
  return b;
}
struct RawDetection {
  float x, y, z, dx, dy, dz, yaw, score;
  int label, index;
};
__global__ void decode_kernel(const float *reg, const float *hei,
                              const float *dim, const float *rot,
                              const float *hm, RawDetection *out, int *count,
                              int h, int w, float threshold, float pcx,
                              float pcy, float vx, float vy, float minx,
                              float miny, float minz, float maxx, float maxy,
                              float maxz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int spatial = h * w;
  if (i >= spatial)
    return;
  int label = 0;
  float logit = hm[i];
  for (int c = 1; c < 3; ++c) {
    float v = hm[c * spatial + i];
    if (v > logit) {
      logit = v;
      label = c;
    }
  }
  float score = 1.0F / (1.0F + expf(-logit));
  if (!(score > threshold))
    return;
  int yy = i / w, xx = i % w;
  float x = (xx + reg[i]) * vx + pcx;
  float y = (yy + reg[spatial + i]) * vy + pcy;
  float z = hei[i];
  float dx = expf(dim[i]), dy = expf(dim[spatial + i]),
        dz = expf(dim[2 * spatial + i]);
  float yaw = atan2f(rot[i], rot[spatial + i]);
  if (!(isfinite(x) && isfinite(y) && isfinite(z) && isfinite(dx) &&
        isfinite(dy) && isfinite(dz) && isfinite(yaw)))
    return;
  if (x < minx || x > maxx || y < miny || y > maxy || z < minz || z > maxz)
    return;
  int dst = atomicAdd(count, 1);
  out[dst] = {x, y, z, dx, dy, dz, yaw, score, label, i};
}
struct Point {
  double x, y;
};
double cross(Point a, Point b, Point c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
Point intersection(Point a, Point b, Point p, Point q) {
  double a1 = cross(p, q, a), a2 = cross(p, q, b), den = a1 - a2;
  if (std::abs(den) < 1e-12)
    return b;
  double t = a1 / den;
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}
std::array<Point, 4> corners(const Detection &b) {
  double c = std::cos(b.yaw), s = std::sin(b.yaw), hx = b.dx * 0.5,
         hy = b.dy * 0.5;
  std::array<Point, 4> local = {Point{-hx, -hy}, Point{hx, -hy}, Point{hx, hy},
                                Point{-hx, hy}};
  for (auto &p : local) {
    double x = p.x, y = p.y;
    p = {b.x + x * c - y * s, b.y + x * s + y * c};
  }
  return local;
}
double polygon_area(const std::vector<Point> &p) {
  double a = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto &q = p[(i + 1) % p.size()];
    a += p[i].x * q.y - q.x * p[i].y;
  }
  return std::abs(a) * 0.5;
}
double rotated_iou(const Detection &a, const Detection &b) {
  auto ca = corners(a), cb = corners(b);
  std::vector<Point> poly(ca.begin(), ca.end());
  for (int e = 0; e < 4 && !poly.empty(); ++e) {
    Point p = cb[e], q = cb[(e + 1) % 4];
    std::vector<Point> next;
    for (std::size_t i = 0; i < poly.size(); ++i) {
      Point cur = poly[i], prev = poly[(i + poly.size() - 1) % poly.size()];
      bool cin = cross(p, q, cur) >= -1e-9, pin = cross(p, q, prev) >= -1e-9;
      if (cin != pin)
        next.push_back(intersection(prev, cur, p, q));
      if (cin)
        next.push_back(cur);
    }
    poly = std::move(next);
  }
  double inter = poly.empty() ? 0 : polygon_area(poly);
  double uni = static_cast<double>(a.dx) * a.dy +
               static_cast<double>(b.dx) * b.dy - inter;
  return uni > 0 ? inter / uni : 0;
}
std::vector<Detection> nms(const std::vector<Detection> &input,
                           const DecodeConfig &cfg) {
  std::vector<Detection> sorted = input;
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const auto &a, const auto &b) {
                     if (a.score != b.score)
                       return a.score > b.score;
                     return a.source_index < b.source_index;
                   });
  if (sorted.size() > static_cast<std::size_t>(cfg.pre_max_size))
    sorted.resize(cfg.pre_max_size);
  std::vector<Detection> kept;
  std::vector<unsigned char> suppressed(sorted.size());
  for (std::size_t i = 0;
       i < sorted.size() &&
       kept.size() < static_cast<std::size_t>(cfg.post_max_size);
       ++i) {
    if (suppressed[i])
      continue;
    kept.push_back(sorted[i]);
    for (std::size_t j = i + 1; j < sorted.size(); ++j)
      if (!suppressed[j] &&
          rotated_iou(sorted[i], sorted[j]) > cfg.nms_iou_threshold)
        suppressed[j] = 1;
  }
  return kept;
}
} // namespace
DecodeResult decode_and_nms(const HeadMaps &m, const DecodeConfig &c) {
  const int spatial = m.height_size * m.width_size;
  Buf reg = upload(m.reg), hei = upload(m.height), dim = upload(m.dim),
      rot = upload(m.rot), hm = upload(m.hm);
  RawDetection *device = nullptr;
  int *count = nullptr;
  ok(cudaMalloc(&device,
                static_cast<std::size_t>(spatial) * sizeof(RawDetection)),
     "candidate malloc");
  ok(cudaMalloc(&count, sizeof(int)), "count malloc");
  ok(cudaMemset(count, 0, sizeof(int)), "count zero");
  cudaEvent_t start, stop;
  ok(cudaEventCreate(&start), "event");
  ok(cudaEventCreate(&stop), "event");
  try {
    ok(cudaEventRecord(start), "record");
    decode_kernel<<<(spatial + 255) / 256, 256>>>(
        reg.p, hei.p, dim.p, rot.p, hm.p, device, count, m.height_size,
        m.width_size, c.score_threshold, c.pc_x, c.pc_y, c.voxel_x, c.voxel_y,
        c.post_range[0], c.post_range[1], c.post_range[2], c.post_range[3],
        c.post_range[4], c.post_range[5]);
    ok(cudaGetLastError(), "decode kernel");
    ok(cudaEventRecord(stop), "record");
    ok(cudaEventSynchronize(stop), "sync");
    DecodeResult r;
    ok(cudaEventElapsedTime(&r.cuda_ms, start, stop), "elapsed");
    int n = 0;
    ok(cudaMemcpy(&n, count, sizeof(int), cudaMemcpyDeviceToHost),
       "copy count");
    std::vector<RawDetection> raw(n);
    ok(cudaMemcpy(raw.data(), device, raw.size() * sizeof(RawDetection),
                  cudaMemcpyDeviceToHost),
       "copy candidates");
    r.before_nms.reserve(raw.size());
    for (const auto &x : raw)
      r.before_nms.push_back(
          {x.x, x.y, x.z, x.dx, x.dy, x.dz, x.yaw, x.score, x.label, x.index});
    auto t = std::chrono::steady_clock::now();
    r.detections = nms(r.before_nms, c);
    r.nms_ms = std::chrono::duration<float, std::milli>(
                   std::chrono::steady_clock::now() - t)
                   .count();
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(count);
    cudaFree(device);
    return r;
  } catch (...) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(count);
    cudaFree(device);
    throw;
  }
}
} // namespace centerpoint
