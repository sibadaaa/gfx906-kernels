// SPDX-License-Identifier: Apache-2.0
//
// Host-only dump used by tests/test_fp8_conversion.py.
//
// It prints, for every 8-bit e4m3fn code, the fp32 value our conversion
// produces, and for a deterministic set of float32 inputs the code our
// conversion produces.  The Python side compares both against PyTorch's own
// conversion on CPU -- the fp8 LUT and the in-kernel block quantizer depend on
// these being bit-identical.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gfx906/detail/fp8_e4m3.h"

namespace {

float bits_to_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint32_t float_to_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

// Deterministic, wide-coverage float32 stream: exponents from the subnormal
// range up to the e4m3fn saturation point, both signs, plus inf/NaN codes.
uint32_t next_bits(uint32_t i) {
  const uint32_t sign = (i & 1u) ? 0x80000000u : 0u;
  const uint32_t exp = (i * 2654435761u) % 256u;
  const uint32_t man = (i * 40503u) & 0x7FFFFFu;
  if (i % 97 == 0) return sign | 0x7F800000u;              // inf
  if (i % 89 == 0) return sign | 0x7FC00000u | man;        // NaN
  return sign | (exp << 23) | man;
}

}  // namespace

int main() {
  for (int i = 0; i < 256; ++i) {
    gx::Float8_e4m3fn v(static_cast<uint8_t>(i), gx::Float8_e4m3fn::from_bits());
    std::printf("D %d %u\n", i, float_to_bits(static_cast<float>(v)));
  }
  for (uint32_t i = 0; i < 4096; ++i) {
    const uint32_t bits = next_bits(i);
    const float f = bits_to_float(bits);
    gx::Float8_e4m3fn q(f);
    std::printf("Q %u %u\n", bits, static_cast<uint32_t>(q.x));
  }
  return 0;
}
