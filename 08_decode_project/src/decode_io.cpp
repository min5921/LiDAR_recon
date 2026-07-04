#include "centerpoint/decode.hpp"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace centerpoint {
namespace {
std::vector<float> read_floats(const std::filesystem::path &p,
                               std::size_t expected) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + p.string());
  auto bytes = f.tellg();
  if (bytes < 0 || static_cast<std::uintmax_t>(bytes) != expected * 4)
    throw std::runtime_error("unexpected tensor size: " + p.string());
  std::vector<float> v(expected);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!f)
    throw std::runtime_error("cannot read " + p.string());
  return v;
}
} // namespace
HeadMaps read_head_maps(const std::filesystem::path &d) {
  constexpr std::size_t s = 468ULL * 468ULL;
  HeadMaps m;
  m.reg = read_floats(d / "reg.bin", 2 * s);
  m.height = read_floats(d / "height.bin", s);
  m.dim = read_floats(d / "dim.bin", 3 * s);
  m.rot = read_floats(d / "rot.bin", 2 * s);
  m.hm = read_floats(d / "hm.bin", 3 * s);
  return m;
}
void write_detections(const std::filesystem::path &d, const DecodeResult &r) {
  std::filesystem::create_directories(d);
  std::ofstream csv(d / "detections.csv");
  csv << std::setprecision(9);
  csv << "x,y,z,dx,dy,dz,yaw,score,label,source_index\n";
  for (const auto &x : r.detections)
    csv << x.x << ',' << x.y << ',' << x.z << ',' << x.dx << ',' << x.dy << ','
        << x.dz << ',' << x.yaw << ',' << x.score << ',' << x.label << ','
        << x.source_index << '\n';
  std::ofstream bin(d / "detections.bin", std::ios::binary);
  for (const auto &x : r.detections) {
    const float values[9] = {x.x,   x.y,     x.z,
                             x.dx,  x.dy,    x.dz,
                             x.yaw, x.score, static_cast<float>(x.label)};
    bin.write(reinterpret_cast<const char *>(values), sizeof(values));
  }
  std::ofstream meta(d / "detections_metadata.json");
  meta << "{\n  \"box_layout\": \"x,y,z,dx,dy,dz,yaw\",\n  \"class_names\": "
          "[\"VEHICLE\", \"PEDESTRIAN\", \"CYCLIST\"],\n  "
          "\"candidates_before_nms\": "
       << r.before_nms.size()
       << ",\n  \"detections_after_nms\": " << r.detections.size()
       << ",\n  \"cuda_decode_ms\": " << r.cuda_ms
       << ",\n  \"rotated_nms_ms\": " << r.nms_ms << "\n}\n";
}
} // namespace centerpoint
