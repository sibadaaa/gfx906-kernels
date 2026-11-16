// SPDX-License-Identifier: Apache-2.0
//
// Minimal stand-in for <torch/headeronly/core/ScalarType.h>.
//
// The extracted device headers only use this for compile-time type tags
// (`SRC_DTYPE == ScalarType::Half`) in conversion dispatch macros, so a value
// table with torch's numeric encoding is enough.  Keeping the numeric values
// identical means a tag that crosses the ABI still means the same dtype.
#pragma once

namespace torch {
namespace headeronly {

enum class ScalarType : int8_t {
  Byte = 0,
  Char = 1,
  Short = 2,
  Int = 3,
  Long = 4,
  Half = 5,
  Float = 6,
  Double = 7,
  ComplexHalf = 8,
  ComplexFloat = 9,
  ComplexDouble = 10,
  Bool = 11,
  QInt8 = 12,
  QUInt8 = 13,
  QInt32 = 14,
  BFloat16 = 15,
  Float8_e5m2 = 23,
  Float8_e4m3fn = 24,
  Float8_e5m2fnuz = 25,
  Float8_e4m3fnuz = 26,
  UInt16 = 27,
  UInt32 = 28,
  UInt64 = 29,
};

inline const char* toString(ScalarType t) {
  switch (t) {
    case ScalarType::Byte: return "uint8";
    case ScalarType::Char: return "int8";
    case ScalarType::Half: return "float16";
    case ScalarType::Float: return "float32";
    case ScalarType::BFloat16: return "bfloat16";
    case ScalarType::Float8_e4m3fn: return "float8_e4m3fn";
    default: return "unknown";
  }
}

}  // namespace headeronly
}  // namespace torch
