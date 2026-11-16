// SPDX-License-Identifier: Apache-2.0
#pragma once

// HIPified selection: vLLM's build rewrites the CUDA dtype headers to the
// `*_hip.cuh` siblings during hipify; this module vendors the HIP variants
// directly so the templates are identical to what the production .so compiles.
#include "dtype_generic.cuh"
#include "dtype_float16_hip.cuh"
#include "dtype_float32.cuh"
#include "dtype_bfloat16_hip.cuh"
#include "dtype_fp8.cuh"
