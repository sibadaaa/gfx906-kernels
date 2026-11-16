#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness of the fp8 e4m3fn conversion against PyTorch (CPU only).

The gfx906 FP8 path does two things with this conversion: it builds the
256-entry fp8->fp16 LUT used by every fp8 GEMV, and it quantizes activations
inside the block-quantizer kernel.  Both must match PyTorch bit for bit, and
both are pure arithmetic, so they can be checked without a GPU.

    python tests/test_fp8_conversion.py
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "build", "tests", "fp8_dump")


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def run_dump() -> tuple[list[int], list[tuple[int, int]]]:
    if not os.path.exists(BIN):
        raise SystemExit(
            f"{BIN} missing; run tools/build_tests.sh first"
        )
    out = subprocess.run([BIN], check=True, capture_output=True, text=True).stdout
    decode: list[int] = [0] * 256
    quantize: list[tuple[int, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "D":
            decode[int(parts[1])] = int(parts[2])
        elif parts[0] == "Q":
            quantize.append((int(parts[1]), int(parts[2])))
    return decode, quantize


def main() -> int:
    import torch

    decode, quantize = run_dump()
    failures = 0

    # 1) fp8 code -> float32 (this is what the LDS LUT is built from).
    ref = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    ref = ref.view(torch.float8_e4m3fn).to(torch.float32)
    for i in range(256):
        a = bits_to_float(decode[i])
        b = float(ref[i])
        same = struct.pack("<f", a) == struct.pack("<f", b)
        if not same:
            failures += 1
            if failures <= 8:
                print(f"  decode mismatch code={i:#04x}: ours={a!r} ref={b!r}")

    # 2) float32 -> fp8 code (block quantizer + weight staging).
    vals = torch.tensor(
        [bits_to_float(b) for b, _ in quantize], dtype=torch.float32
    )
    ref_codes = vals.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    for (bits, ours), ref_code in zip(quantize, ref_codes.tolist()):
        if ours != ref_code:
            failures += 1
            if failures <= 16:
                print(
                    f"  quantize mismatch input=0x{bits:08x} "
                    f"({bits_to_float(bits)!r}): ours={ours:#04x} "
                    f"ref={ref_code:#04x}"
                )

    total = 256 + len(quantize)
    if failures:
        print(f"FAIL: {failures}/{total} fp8 conversions differ from PyTorch")
        return 1
    print(f"ok: {total} fp8 conversions bit-exact vs PyTorch (CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
