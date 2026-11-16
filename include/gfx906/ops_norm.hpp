// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm family (extracted from csrc/rocm/gfx906_layernorm_kernels).
#pragma once

#include <cstdint>
#include <vector>

#include "detail/aten.hpp"

namespace gfx906 {

using gx::Tensor;

Tensor rms_norm_gated_gfx906(Tensor input, Tensor weight, Tensor gate,
                             double epsilon, bool norm_before_gate);

Tensor gemma_rms_norm_gfx906(Tensor input, Tensor weight, double epsilon);

std::vector<Tensor> gemma_fused_add_rms_norm_gfx906(Tensor input,
                                                    Tensor residual,
                                                    Tensor weight,
                                                    double epsilon);

}  // namespace gfx906
