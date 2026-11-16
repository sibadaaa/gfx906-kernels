// SPDX-License-Identifier: Apache-2.0
//
// Runtime hooks for the torch-free tensor shim: allocator, stream, device.
//
// The defaults are plain HIP.  A host that already owns an allocator (torch
// caching allocator, an inference engine's arena, ...) installs its own hooks
// with gx_set_allocator() and the kernels then return buffers from that arena
// -- see torch/gfx906_torch.cpp for the torch adapter.
#include <hip/hip_runtime.h>

#include <mutex>
#include <new>

#include "gfx906/detail/aten.hpp"

namespace gx {

namespace {

struct AllocatorState {
  AllocFn alloc = nullptr;
  FreeFn free_fn = nullptr;
  void* ctx = nullptr;
};

AllocatorState& allocator_state() {
  static AllocatorState state;
  return state;
}

std::mutex& allocator_mutex() {
  static std::mutex m;
  return m;
}

thread_local hipStream_t tls_stream = nullptr;
thread_local int tls_device = -1;

void* default_alloc(size_t bytes, void* /*ctx*/) {
  void* ptr = nullptr;
  if (hipMalloc(&ptr, bytes) != hipSuccess) {
    throw std::bad_alloc();
  }
  return ptr;
}

void default_free(void* ptr, void* /*ctx*/) {
  if (ptr != nullptr) (void)hipFree(ptr);
}

}  // namespace

void set_allocator(AllocFn alloc, FreeFn free_fn, void* ctx) {
  std::lock_guard<std::mutex> lock(allocator_mutex());
  allocator_state().alloc = alloc;
  allocator_state().free_fn = free_fn;
  allocator_state().ctx = ctx;
}

void* alloc_bytes(size_t bytes, int device) {
  AllocFn alloc;
  void* ctx;
  {
    std::lock_guard<std::mutex> lock(allocator_mutex());
    alloc = allocator_state().alloc;
    ctx = allocator_state().ctx;
  }
  const int previous = current_device();
  const bool switch_device = device >= 0 && device != previous;
  if (switch_device) (void)hipSetDevice(device);
  void* ptr = alloc != nullptr ? alloc(bytes, ctx) : default_alloc(bytes, nullptr);
  if (switch_device) (void)hipSetDevice(previous);
  return ptr;
}

namespace detail {

std::shared_ptr<Storage> make_storage(void* ptr, size_t bytes) {
  FreeFn free_fn;
  void* ctx;
  {
    std::lock_guard<std::mutex> lock(allocator_mutex());
    free_fn = allocator_state().free_fn;
    ctx = allocator_state().ctx;
  }
  auto* storage = new Storage{ptr, bytes};
  // The deleter captures the allocator that produced the block: a host may
  // swap allocators between calls, and a block must go back to its owner.
  return std::shared_ptr<Storage>(storage, [ptr, free_fn, ctx](Storage* raw) {
    if (free_fn != nullptr) {
      free_fn(ptr, ctx);
    } else {
      default_free(ptr, nullptr);
    }
    delete raw;
  });
}

}  // namespace detail

namespace stream {

hipStream_t current() { return tls_stream; }

void set(hipStream_t s) { tls_stream = s; }

}  // namespace stream

int current_device() {
  if (tls_device >= 0) return tls_device;
  int dev = 0;
  (void)hipGetDevice(&dev);
  return dev;
}

void set_current_device(int device) { tls_device = device; }

}  // namespace gx
