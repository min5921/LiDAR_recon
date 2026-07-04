#pragma once

#include "centerpoint/rpn_full_types.hpp"

namespace centerpoint {

FullRpnResult run_full_rpn_cuda(const HostTensor& input,
                               const FullRpnWeights& weights);

}  // namespace centerpoint
