// SPDX-License-Identifier: Apache-2.0
//
// Materialise a strided tensor into a contiguous buffer.
//
// Only used by `gx::Tensor::contiguous()` when the fast row-copy path does not
// apply (i.e. the innermost dimension is not unit-stride).  The RMSNorm host
// wrappers call `.contiguous()` on their input, so this has to be correct for
// arbitrary 1-D/2-D layouts rather than silently assuming row-major.
#include <hip/hip_runtime.h>

#include <cstring>

#include "gfx906/detail/aten.hpp"

namespace gx {

namespace {

struct StridedDesc {
  int64_t sizes[8];
  int64_t strides[8];
  int32_t ndim;
};

__global__ void copy_strided_kernel(char* __restrict__ dst,
                                    const char* __restrict__ src, int64_t n,
                                    StridedDesc desc, int64_t esize) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t remaining = i;
  int64_t offset = 0;
  for (int32_t d = desc.ndim - 1; d >= 0; --d) {
    const int64_t size = desc.sizes[d];
    const int64_t idx = size > 0 ? remaining % size : 0;
    remaining = size > 0 ? remaining / size : 0;
    offset += idx * desc.strides[d];
  }
  const char* s = src + offset * esize;
  char* d = dst + i * esize;
  if (esize == 1) {
    *d = *s;
  } else if (esize == 2) {
    uint16_t v;
    std::memcpy(&v, s, 2);
    std::memcpy(d, &v, 2);
  } else if (esize == 4) {
    uint32_t v;
    std::memcpy(&v, s, 4);
    std::memcpy(d, &v, 4);
  } else {
    uint64_t v;
    std::memcpy(&v, s, 8);
    std::memcpy(d, &v, 8);
  }
}

}  // namespace

void copy_strided(const Tensor& src, Tensor& dst) {
  StridedDesc desc{};
  desc.ndim = static_cast<int32_t>(src.dim());
  for (int32_t i = 0; i < desc.ndim; ++i) {
    desc.sizes[i] = src.size(i);
    desc.strides[i] = src.stride(i);
  }
  const int64_t n = src.numel();
  if (n == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  copy_strided_kernel<<<blocks, threads, 0, stream::current()>>>(
      static_cast<char*>(dst.data_ptr()),
      static_cast<const char*>(src.data_ptr()), n, desc, src.element_size());
}

}  // namespace gx
