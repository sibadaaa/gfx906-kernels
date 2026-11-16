// SPDX-License-Identifier: Apache-2.0
//
// A deliberately small, torch-free subset of the ATen tensor API.
//
// Why this exists
// ---------------
// The gfx906 kernels in this module were written inside vLLM, where every host
// entry point is spelled in `torch::Tensor` / `at::cuda::` terms.  Rather than
// rewriting ~2k lines of *host* code (launch geometry, dispatch tables, shape
// gates) and risking a silent numerical drift, the device code and the host
// code are carried over verbatim and compiled against this shim.
//
// The contract is:
//   * device code is byte-identical to the vLLM source it was extracted from;
//   * `at::`/`c10::`/`torch::` below only provide what those sources use;
//   * nothing here links against libtorch, so an engine that has no torch can
//     consume the library (C ABI in include/gfx906/gfx906.h).
//
// The aliases are opt-out (`GX_NO_ATEN_ALIASES`) so a torch adapter can be
// compiled in the same build: it keeps torch's real `at::Tensor` and converts
// to `gx::Tensor` views at the boundary.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fp8_e4m3.h"

namespace gx {

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

// Numeric values intentionally match torch ScalarType so the C ABI can hand
// descriptors to/from torch without a translation table.
enum class ScalarType : int32_t {
  Byte = 0,
  Char = 1,
  Short = 2,
  Int = 3,
  Long = 4,
  Half = 5,
  Float = 6,
  Double = 7,
  Bool = 11,
  BFloat16 = 15,
  Float8_e5m2 = 23,
  Float8_e4m3fn = 24,
  Float8_e5m2fnuz = 25,
  Float8_e4m3fnuz = 26,
  UInt16 = 27,
  UInt32 = 28,
  UInt64 = 29,
};

inline constexpr ScalarType kByte = ScalarType::Byte;
inline constexpr ScalarType kChar = ScalarType::Char;
inline constexpr ScalarType kShort = ScalarType::Short;
inline constexpr ScalarType kInt = ScalarType::Int;
inline constexpr ScalarType kLong = ScalarType::Long;
inline constexpr ScalarType kHalf = ScalarType::Half;
inline constexpr ScalarType kFloat = ScalarType::Float;
inline constexpr ScalarType kDouble = ScalarType::Double;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
inline constexpr ScalarType kFloat8_e5m2 = ScalarType::Float8_e5m2;
inline constexpr ScalarType kFloat8_e4m3fn = ScalarType::Float8_e4m3fn;
inline constexpr ScalarType kFloat8_e5m2fnuz = ScalarType::Float8_e5m2fnuz;
inline constexpr ScalarType kFloat8_e4m3fnuz = ScalarType::Float8_e4m3fnuz;

using Half = __half;
using BFloat16 = __hip_bfloat16;

inline int64_t element_size(ScalarType t) {
  switch (t) {
    case ScalarType::Byte:
    case ScalarType::Char:
    case ScalarType::Bool:
    case ScalarType::Float8_e5m2:
    case ScalarType::Float8_e4m3fn:
    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fnuz:
      return 1;
    case ScalarType::Short:
    case ScalarType::Half:
    case ScalarType::BFloat16:
    case ScalarType::UInt16:
      return 2;
    case ScalarType::Int:
    case ScalarType::Float:
    case ScalarType::UInt32:
      return 4;
    case ScalarType::Long:
    case ScalarType::Double:
    case ScalarType::UInt64:
      return 8;
  }
  return 0;
}

inline const char* dtype_name(ScalarType t) {
  switch (t) {
    case ScalarType::Byte: return "uint8";
    case ScalarType::Char: return "int8";
    case ScalarType::Short: return "int16";
    case ScalarType::Int: return "int32";
    case ScalarType::Long: return "int64";
    case ScalarType::Half: return "float16";
    case ScalarType::Float: return "float32";
    case ScalarType::Double: return "float64";
    case ScalarType::BFloat16: return "bfloat16";
    case ScalarType::Float8_e5m2: return "float8_e5m2";
    case ScalarType::Float8_e4m3fn: return "float8_e4m3fn";
    case ScalarType::Float8_e5m2fnuz: return "float8_e5m2fnuz";
    case ScalarType::Float8_e4m3fnuz: return "float8_e4m3fnuz";
    default: return "unknown";
  }
}

enum class MemoryFormat : int32_t { Contiguous = 0, Preserve = 1 };

// ---------------------------------------------------------------------------
// device / options
// ---------------------------------------------------------------------------

struct Device {
  int32_t index_ = -1;

  Device() = default;
  explicit Device(int32_t index) : index_(index) {}

  int64_t index() const { return index_; }
  bool has_index() const { return index_ >= 0; }
  bool is_cuda() const { return true; }
  bool is_cpu() const { return false; }
};

inline bool operator==(const Device& a, const Device& b) {
  return a.index_ == b.index_;
}
inline bool operator!=(const Device& a, const Device& b) { return !(a == b); }

inline std::ostream& operator<<(std::ostream& os, const Device& d) {
  os << "cuda:" << d.index_;
  return os;
}
inline std::ostream& operator<<(std::ostream& os, ScalarType t) {
  os << dtype_name(t);
  return os;
}

struct TensorOptions {
  ScalarType dtype_ = ScalarType::Float;
  Device device_{};

  TensorOptions() = default;

  TensorOptions dtype(ScalarType t) const {
    TensorOptions c = *this;
    c.dtype_ = t;
    return c;
  }
  TensorOptions device(Device d) const {
    TensorOptions c = *this;
    c.device_ = d;
    return c;
  }
  TensorOptions device(std::optional<Device> d) const {
    return d.has_value() ? device(*d) : *this;
  }
  TensorOptions memory_format(MemoryFormat) const { return *this; }

  ScalarType dtype() const { return dtype_; }
  Device device() const { return device_; }
  bool has_device() const { return device_.has_index(); }
};

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

// Stream inserter for size/stride vectors: TORCH_CHECK messages in the
// extracted host code format `tensor.sizes()`, and the formatter below streams
// its arguments.  Declared in namespace gx so the dependent `os << v` inside
// fmt() finds it by ordinary lookup.
inline std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& v) {
  os << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) os << ", ";
    os << v[i];
  }
  return os << ']';
}

inline std::string fmt() { return std::string(); }

template <typename T, typename... Rest>
std::string fmt(const T& v, const Rest&... rest) {
  std::ostringstream os;
  os << v;
  return os.str() + fmt(rest...);
}

[[noreturn]] inline void throw_check(const char* file, int line,
                                     const char* expr, const std::string& msg) {
  std::ostringstream os;
  os << "gfx906: check failed: " << msg << " (" << file << ":" << line;
  if (expr != nullptr) os << ", `" << expr << "`";
  os << ")";
  throw Error(os.str());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// allocator / stream / device
// ---------------------------------------------------------------------------

using AllocFn = void* (*)(size_t bytes, void* ctx);
using FreeFn = void (*)(void* ptr, void* ctx);

void set_allocator(AllocFn alloc, FreeFn free_fn, void* ctx);
void* alloc_bytes(size_t bytes, int device);

namespace stream {
hipStream_t current();
void set(hipStream_t s);
}  // namespace stream

int current_device();
void set_current_device(int device);

class Tensor;

namespace detail {

struct Storage {
  void* ptr = nullptr;
  size_t bytes = 0;
};

inline std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

std::shared_ptr<Storage> make_storage(void* ptr, size_t bytes);

}  // namespace detail

Tensor empty(std::vector<int64_t> sizes, TensorOptions opts);
Tensor empty(std::initializer_list<int64_t> sizes, TensorOptions opts);

// Materialise `src` (arbitrary strides) into `dst` (contiguous, same sizes).
// Defined in src/kernels/copy_strided.hip.
void copy_strided(const Tensor& src, Tensor& dst);
Tensor empty_like(const Tensor& t, TensorOptions opts);
Tensor zeros(std::vector<int64_t> sizes, TensorOptions opts);

// ---------------------------------------------------------------------------
// tensor
// ---------------------------------------------------------------------------

class Tensor {
 public:
  Tensor() = default;

  static Tensor view(void* data, std::vector<int64_t> sizes,
                     std::vector<int64_t> strides, ScalarType dtype,
                     int device) {
    Tensor t;
    t.data_ = data;
    t.sizes_ = std::move(sizes);
    t.strides_ = std::move(strides);
    t.dtype_ = dtype;
    t.device_ = Device(device);
    return t;
  }

  bool defined() const { return data_ != nullptr || storage_ != nullptr; }

  void* data_ptr() const { return data_; }

  template <typename T>
  T* data_ptr() const {
    return reinterpret_cast<T*>(data_);
  }

  int64_t size(int64_t dim) const { return sizes_[normalize(dim)]; }
  const std::vector<int64_t>& sizes() const { return sizes_; }
  int64_t stride(int64_t dim) const { return strides_[normalize(dim)]; }
  int64_t dim() const { return static_cast<int64_t>(sizes_.size()); }
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t s : sizes_) n *= s;
    return n;
  }
  int64_t element_size() const { return gx::element_size(dtype_); }

  ScalarType scalar_type() const { return dtype_; }
  ScalarType dtype() const { return dtype_; }
  Device device() const { return device_; }

  bool is_cuda() const { return true; }
  bool is_cpu() const { return false; }

  bool is_contiguous() const {
    int64_t expected = 1;
    for (int64_t i = dim() - 1; i >= 0; --i) {
      if (sizes_[i] == 1) continue;
      if (strides_[i] != expected) return false;
      expected *= sizes_[i];
    }
    return true;
  }

  TensorOptions options() const {
    return TensorOptions().dtype(dtype_).device(device_);
  }

  Tensor new_empty(std::vector<int64_t> sizes) const {
    return gx::empty(std::move(sizes), options());
  }
  Tensor new_empty(std::vector<int64_t> sizes, TensorOptions opts) const {
    return gx::empty(std::move(sizes), opts);
  }

  Tensor contiguous() const;
  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
  Tensor t() const;

  void set_storage(std::shared_ptr<detail::Storage> storage) {
    storage_ = std::move(storage);
  }

 private:
  int64_t normalize(int64_t d) const {
    return d < 0 ? d + static_cast<int64_t>(sizes_.size()) : d;
  }

  std::shared_ptr<detail::Storage> storage_;
  void* data_ = nullptr;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  ScalarType dtype_ = ScalarType::Float;
  Device device_{};
};

inline Tensor empty(std::vector<int64_t> sizes, TensorOptions opts) {
  const int64_t esize = element_size(opts.dtype());
  int64_t n = 1;
  for (int64_t s : sizes) n *= s;
  const size_t bytes =
      static_cast<size_t>(n) * static_cast<size_t>(esize);
  const int dev = opts.has_device() ? static_cast<int>(opts.device().index())
                                    : current_device();
  if (bytes == 0) {
    const auto strides = detail::contiguous_strides(sizes);
    return Tensor::view(nullptr, std::move(sizes), strides, opts.dtype(), dev);
  }
  void* ptr = alloc_bytes(bytes, dev);
  Tensor t = Tensor::view(ptr, sizes, detail::contiguous_strides(sizes),
                          opts.dtype(), dev);
  t.set_storage(detail::make_storage(ptr, bytes));
  return t;
}

inline Tensor empty(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return empty(std::vector<int64_t>(sizes), opts);
}

inline Tensor empty_like(const Tensor& t, TensorOptions opts) {
  if (!opts.has_device()) opts = opts.device(t.device());
  return empty(t.sizes(), opts);
}

inline Tensor zeros(std::vector<int64_t> sizes, TensorOptions opts) {
  Tensor t = empty(std::move(sizes), opts);
  if (t.defined() && t.numel() > 0) {
    (void)hipMemsetAsync(t.data_ptr(), 0,
                   static_cast<size_t>(t.numel()) *
                       static_cast<size_t>(t.element_size()),
                   stream::current());
  }
  return t;
}

inline Tensor zeros(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return zeros(std::vector<int64_t>(sizes), opts);
}

inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
  const int64_t d = normalize(dim);
  Tensor out = *this;
  out.sizes_[d] = length;
  out.data_ = static_cast<char*>(data_) +
              static_cast<int64_t>(start) * strides_[d] * element_size();
  return out;
}

inline Tensor Tensor::t() const {
  Tensor out = *this;
  if (out.sizes_.size() == 2) {
    std::swap(out.sizes_[0], out.sizes_[1]);
    std::swap(out.strides_[0], out.strides_[1]);
  }
  return out;
}

inline Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;

  // copy_strided() carries an 8-dimension descriptor; anything wider is
  // rejected rather than silently truncating the layout.
  if (sizes_.size() > 8) {
    throw Error("gfx906: contiguous() supports at most 8 dimensions");
  }

  Tensor out = empty(sizes_, options());
  const int64_t esize = element_size();

  // Fast path: innermost dimension is unit-stride, so rows can be blitted.
  if (sizes_.back() == 1 || strides_.back() == 1) {
    const size_t row_bytes =
        static_cast<size_t>(sizes_.back()) * static_cast<size_t>(esize);
    if (sizes_.size() == 1) {
      (void)hipMemcpyAsync(out.data_, data_, row_bytes,
                           hipMemcpyDeviceToDevice, stream::current());
      return out;
    }
    (void)hipMemcpy2DAsync(out.data_, row_bytes, data_,
                           static_cast<size_t>(strides_[0]) * esize, row_bytes,
                           static_cast<size_t>(sizes_[0]),
                           hipMemcpyDeviceToDevice, stream::current());
    return out;
  }

  // General path (e.g. a transposed operand reaching `.contiguous()`).
  copy_strided(*this, out);
  return out;
}

// ---------------------------------------------------------------------------
// device management
// ---------------------------------------------------------------------------

inline std::optional<Device> device_of(const Tensor& t) {
  return t.defined() ? std::optional<Device>(t.device()) : std::nullopt;
}

inline std::optional<Device> device_of(const std::optional<Tensor>& t) {
  return t.has_value() ? device_of(*t) : std::nullopt;
}

class DeviceGuard {
 public:
  explicit DeviceGuard(Device d) : previous_(current_device()) {
    target_ = static_cast<int>(d.index());
    active_ = target_ >= 0 && target_ != previous_;
    if (active_) (void)hipSetDevice(target_);
  }
  explicit DeviceGuard(std::optional<Device> d)
      : previous_(current_device()), target_(-1), active_(false) {
    if (d.has_value()) {
      target_ = static_cast<int>(d->index());
      active_ = target_ >= 0 && target_ != previous_;
      if (active_) (void)hipSetDevice(target_);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  ~DeviceGuard() {
    if (active_) (void)hipSetDevice(previous_);
  }

 private:
  int previous_ = 0;
  int target_ = -1;
  bool active_ = false;
};

using OptionalDeviceGuard = DeviceGuard;

// ---------------------------------------------------------------------------
// dispatch helpers (subset of AT_DISPATCH_*)
// ---------------------------------------------------------------------------

template <ScalarType T>
struct ScalarTypeToCPP;
template <>
struct ScalarTypeToCPP<ScalarType::Float> {
  using type = float;
};
template <>
struct ScalarTypeToCPP<ScalarType::Half> {
  using type = Half;
};
template <>
struct ScalarTypeToCPP<ScalarType::BFloat16> {
  using type = BFloat16;
};
template <>
struct ScalarTypeToCPP<ScalarType::Double> {
  using type = double;
};
template <>
struct ScalarTypeToCPP<ScalarType::Char> {
  using type = int8_t;
};
template <>
struct ScalarTypeToCPP<ScalarType::Byte> {
  using type = uint8_t;
};
template <>
struct ScalarTypeToCPP<ScalarType::Int> {
  using type = int32_t;
};
template <>
struct ScalarTypeToCPP<ScalarType::Float8_e4m3fn> {
  using type = Float8_e4m3fn;
};

}  // namespace gx

// ---------------------------------------------------------------------------
// ATen-compatible aliases
// ---------------------------------------------------------------------------

#ifndef GX_NO_ATEN_ALIASES

namespace at {

using gx::BFloat16;
using gx::Device;
using gx::Half;
using gx::MemoryFormat;
using gx::ScalarType;
using gx::Tensor;
using gx::TensorOptions;

using gx::kBFloat16;
using gx::kByte;
using gx::kChar;
using gx::kDouble;
using gx::kFloat;
using gx::kFloat8_e4m3fn;
using gx::kFloat8_e4m3fnuz;
using gx::kFloat8_e5m2;
using gx::kFloat8_e5m2fnuz;
using gx::kHalf;
using gx::kInt;
using gx::kLong;

inline Tensor empty(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return gx::empty(sizes, opts);
}
inline Tensor empty(std::vector<int64_t> sizes, TensorOptions opts) {
  return gx::empty(std::move(sizes), opts);
}
inline Tensor zeros(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return gx::zeros(sizes, opts);
}
inline Tensor zeros(std::vector<int64_t> sizes, TensorOptions opts) {
  return gx::zeros(std::move(sizes), opts);
}
inline Tensor empty_like(const Tensor& t, TensorOptions opts) {
  return gx::empty_like(t, opts);
}

inline std::optional<Device> device_of(const Tensor& t) {
  return gx::device_of(t);
}
inline std::optional<Device> device_of(const std::optional<Tensor>& t) {
  return gx::device_of(t);
}

namespace cuda {
inline hipStream_t getCurrentCUDAStream() { return gx::stream::current(); }
inline hipStream_t getCurrentCUDAStream(int) { return gx::stream::current(); }
using CUDAGuard = gx::DeviceGuard;
using OptionalCUDAGuard = gx::DeviceGuard;
}  // namespace cuda

}  // namespace at

namespace torch {

using gx::Tensor;
using gx::TensorOptions;

using gx::kBFloat16;
using gx::kByte;
using gx::kChar;
using gx::kDouble;
using gx::kFloat;
using gx::kFloat8_e4m3fn;
using gx::kHalf;
using gx::kInt;
using gx::kLong;

inline constexpr gx::ScalarType kFloat32 = gx::ScalarType::Float;

inline Tensor empty(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return gx::empty(sizes, opts);
}
inline Tensor empty(std::vector<int64_t> sizes, TensorOptions opts) {
  return gx::empty(std::move(sizes), opts);
}
inline Tensor zeros(std::vector<int64_t> sizes, TensorOptions opts) {
  return gx::zeros(std::move(sizes), opts);
}
inline Tensor zeros(std::initializer_list<int64_t> sizes, TensorOptions opts) {
  return gx::zeros(sizes, opts);
}
inline Tensor empty_like(const Tensor& t, TensorOptions opts) {
  return gx::empty_like(t, opts);
}

}  // namespace torch

namespace c10 {

using ScalarType = gx::ScalarType;
using Float8_e4m3fn = gx::Float8_e4m3fn;
using Float8_e4m3fnuz = gx::Float8_e4m3fnuz;
using MemoryFormat = gx::MemoryFormat;

namespace cuda {
inline int current_device() { return gx::current_device(); }
using CUDAGuard = gx::DeviceGuard;
using OptionalCUDAGuard = gx::DeviceGuard;
}  // namespace cuda

}  // namespace c10

// --- TORCH_CHECK ------------------------------------------------------------

#define TORCH_CHECK(cond, ...)                                    \
  do {                                                            \
    if (!(cond)) {                                                \
      ::gx::detail::throw_check(__FILE__, __LINE__, #cond,         \
                                ::gx::detail::fmt(__VA_ARGS__));  \
    }                                                             \
  } while (0)

#define TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)

// --- AT_DISPATCH ------------------------------------------------------------

#define GX_DISPATCH_CASE(ENUM, ...)                      \
  case ENUM: {                                           \
    using scalar_t = ::gx::ScalarTypeToCPP<ENUM>::type;  \
    return __VA_ARGS__();                                \
  }

#define GX_DISPATCH_SWITCH(TYPE, NAME, ...)                      \
  [&] {                                                          \
    (void)(NAME);                                                \
    switch (TYPE) {                                              \
      __VA_ARGS__                                                \
      default:                                                   \
        ::gx::detail::throw_check(__FILE__, __LINE__, nullptr,   \
                                  "unsupported dtype in " NAME); \
    }                                                            \
  }()

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  GX_DISPATCH_CASE(::gx::ScalarType::Float, __VA_ARGS__) \
  GX_DISPATCH_CASE(::gx::ScalarType::Half, __VA_ARGS__)  \
  GX_DISPATCH_CASE(::gx::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  GX_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_HALF_TYPES(...)            \
  GX_DISPATCH_CASE(::gx::ScalarType::Half, __VA_ARGS__) \
  GX_DISPATCH_CASE(::gx::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  GX_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))

#endif  // GX_NO_ATEN_ALIASES
