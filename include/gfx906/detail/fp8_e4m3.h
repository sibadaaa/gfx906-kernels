// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
//
// fp8 E4M3FN conversion, copied verbatim from PyTorch
// `torch/headeronly/util/Float8_e4m3fn.h` (c10::Float8_e4m3fn).
//
// The bit patterns must match PyTorch exactly: the gfx906 quantization paths
// use this type on device (fp8_emu_gemm block quantizer) and any deviation
// would change the emitted weight/activation codes.  Keeping the same
// algorithm is therefore part of the numerical contract, not an optimisation.
#pragma once

#include <cstdint>
#include <climits>

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
  #define GX_HD __host__ __device__
#else
  #define GX_HD
#endif

namespace gx {

namespace fp8_detail {

inline GX_HD float fp32_from_bits(uint32_t v) {
  union {
    uint32_t as_bits;
    float as_value;
  } x;
  x.as_bits = v;
  return x.as_value;
}

inline GX_HD uint32_t fp32_to_bits(float v) {
  union {
    float as_value;
    uint32_t as_bits;
  } x;
  x.as_value = v;
  return x.as_bits;
}

inline GX_HD float fp8e4m3fn_to_fp32_value(uint8_t input) {
  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  // __builtin_clz is available in both host and device passes; AMD's __clz
  // returns 32 for 0, which is what the explicit fallback reproduces.
  uint32_t renorm_shift =
      nonsign != 0 ? __builtin_clz(nonsign) : sizeof(uint32_t) * CHAR_BIT;
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  uint32_t result =
      sign |
      ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
  return fp32_from_bits(result);
}

inline GX_HD uint8_t fp8e4m3fn_from_fp32_value(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;

  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    result = f_bits > UINT32_C(0x7F800000) ? 0x7f : 0x7e;
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint8_t mant_odd = (f_bits >> 20) & 1;
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 20);
      if (result == 0x7f) {
        result = 0x7e;
      }
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

}  // namespace fp8_detail

struct alignas(1) Float8_e4m3fn {
  uint8_t x;

  struct from_bits_t {};
  GX_HD static constexpr from_bits_t from_bits() { return from_bits_t(); }

  Float8_e4m3fn() = default;

  constexpr GX_HD Float8_e4m3fn(uint8_t bits, from_bits_t /*unused*/)
      : x(bits) {}

  inline GX_HD Float8_e4m3fn(float value)
      : x(fp8_detail::fp8e4m3fn_from_fp32_value(value)) {}

  inline GX_HD operator float() const {
    return fp8_detail::fp8e4m3fn_to_fp32_value(x);
  }

  inline GX_HD bool isnan() const { return (x & 0b01111111) == 0b01111111; }
};

struct alignas(1) Float8_e4m3fnuz {
  uint8_t x;

  struct from_bits_t {};
  GX_HD static constexpr from_bits_t from_bits() { return from_bits_t(); }

  Float8_e4m3fnuz() = default;

  constexpr GX_HD Float8_e4m3fnuz(uint8_t bits, from_bits_t /*unused*/)
      : x(bits) {}

  // gfx906 has no fnuz hardware conversion; this mirrors the value range only
  // so the (unused on gfx906) declaration sites keep compiling.
  inline GX_HD Float8_e4m3fnuz(float value) : x(0) {
    (void)value;
  }

  inline GX_HD operator float() const { return 0.0f; }
};

}  // namespace gx
