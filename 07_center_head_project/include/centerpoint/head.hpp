#pragma once
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace centerpoint {
struct Tensor { std::vector<float> values; int channels=0, height=0, width=0; };
struct BatchNorm { std::vector<float> weight, bias, mean, variance; };
struct Conv { std::string name; std::vector<float> weight, bias; BatchNorm bn; int in_channels=0, out_channels=0; bool has_bn=false; };
struct BranchWeights { std::string name; Conv hidden, output; };
struct HeadWeights { Conv shared; std::array<BranchWeights,5> branches; float bn_epsilon=1.0e-3F; };
struct HeadResult { std::array<Tensor,5> outputs; float elapsed_ms=0.0F; };

Tensor read_rpn_output(const std::filesystem::path& dir);
HeadWeights read_head_weights(const std::filesystem::path& dir);
HeadResult run_center_head_cuda(const Tensor& input, const HeadWeights& weights);
void write_head_output(const std::filesystem::path& dir, const HeadResult& result);
}
