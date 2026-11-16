// SPDX-License-Identifier: Apache-2.0
//
// Native C++ API for the gfx906 compute kernels.
//
// This is the "C++ engine" entry point: no torch, no Python.  It is the same
// symbol set the C ABI wraps (include/gfx906/gfx906.h); the C ABI is the
// stable surface, this header is the zero-cost one.
//
// Tensors are `gx::Tensor` non-owning views (see detail/aten.hpp).  A caller
// that already owns device memory builds a view with gx::Tensor::view(); a
// caller that wants the library to allocate passes TensorOptions and releases
// the returned tensor by letting it go out of scope.
#pragma once

#include "detail/aten.hpp"
#include "ops_attnd.hpp"
#include "ops_attention.hpp"
#include "ops_linear.hpp"
#include "ops_mamba.hpp"
#include "ops_norm.hpp"

namespace gfx906 {

using gx::Tensor;

}  // namespace gfx906
