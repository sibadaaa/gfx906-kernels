// SPDX-License-Identifier: Apache-2.0
//
// Minimal stand-in for <torch/headeronly/util/Exception.h>.
//
// dtype_fp8.cuh only needs STD_TORCH_CHECK; route it to the module's own check
// so errors surface as gx::Error through the C ABI.
#pragma once

#include "gfx906/detail/aten.hpp"

#define STD_TORCH_CHECK(cond, ...) \
  TORCH_CHECK(cond, __VA_ARGS__)
