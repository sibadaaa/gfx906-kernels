#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit: are the migrated device functions identical to their vLLM source?

The numerical contract of this module is "device code untouched".  This script
proves it directly instead of relying on review: it extracts every function
whose signature mentions ``__global__`` or ``__device__`` from the vLLM source
and from the migrated file, normalises whitespace/comments, and compares the
bodies function by function.

    python tests/audit_device_code.py            # summary + exit code
    python tests/audit_device_code.py --verbose  # per-file function counts

Exit code 0 means every source device function exists in the target with a
byte-identical body (after whitespace normalisation).
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VLLM = os.path.dirname(ROOT)

PAIRS = [
    ("csrc/rocm/bf16_emu_gemv.hip", "src/kernels/bf16_emu_gemv.hip"),
    ("csrc/rocm/w8a8_sdot4.hip", "src/kernels/w8a8_sdot4.hip"),
    ("csrc/rocm/sdot4_pc.hip", "src/kernels/sdot4_pc.hip"),
    ("csrc/rocm/fp8_emu_gemm.hip", "src/kernels/fp8_emu_gemm.hip"),
    ("csrc/rocm/attnd_inner.hip", "src/kernels/attnd_inner.hip"),
    ("csrc/rocm/attnd_inner_kvint8.hip", "src/kernels/attnd_inner_kvint8.hip"),
    ("csrc/rocm/attention.hip", "src/kernels/attention.hip"),
    ("csrc/rocm/gfx906_layernorm_kernels.hip",
     "src/kernels/gfx906_layernorm_kernels.hip"),
    ("csrc/mamba/gfx906_decode_kernels.hip",
     "src/kernels/gfx906_decode_kernels.hip"),
]


def strip_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\":
                    i += 1
                    if i < n:
                        out.append(text[i])
                        i += 1
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_device_functions(text: str):
    """Yield (signature_key, body_norm) for __global__/__device__ functions."""
    clean = strip_comments(text)
    results = []
    for m in re.finditer(r"__global__|__device__", clean):
        # Walk back to the start of the declaration: allow a template/attribute
        # prefix, stop at ';' '}' '{' or a preprocessor line.
        start = m.start()
        while start > 0 and clean[start - 1] not in ";}{":
            start -= 1
        # Skip a leading template<...> so it is part of the key.
        brace = clean.find("{", m.end())
        if brace < 0:
            continue
        header = clean[start:brace]
        # A declaration (no body) also has no brace before the next ';'.
        semi = clean.find(";", m.end())
        if semi >= 0 and semi < brace:
            continue
        depth = 0
        i = brace
        while i < len(clean):
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        body = clean[brace : i + 1]
        key = norm(header)
        # Keep only the callable name + template args, which is what lets us
        # match a source function to its migrated twin.
        probe = re.sub(r"__launch_bounds__\s*\([^)]*\)", " ", header)
        probe = re.sub(r"template\s*<[^>]*>", " ", probe)
        probe = re.sub(r"__global__|__device__|__host__|inline|static", " ", probe)
        probe = re.sub(r"\b(?:const|unsigned|signed|struct|class|typename)\b", " ", probe)
        name = re.search(r"([A-Za-z_]\w*)\s*\(", probe)
        if name is None:
            continue
        results.append((key, body, norm(name.group(1))))
    return results


def extract_launches(text: str):
    """Return the normalised text of every hipLaunchKernelGGL(...) call."""
    clean = strip_comments(text)
    out = []
    for m in re.finditer(r"hipLaunchKernelGGL", clean):
        i = clean.find("(", m.end())
        if i < 0:
            continue
        depth = 0
        j = i
        while j < len(clean):
            if clean[j] == "(":
                depth += 1
            elif clean[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(norm(clean[i : j + 1]))
    return out


def main() -> int:
    verbose = "--verbose" in sys.argv
    failures = 0
    total = 0

    for src_rel, dst_rel in PAIRS:
        src_path = os.path.join(VLLM, src_rel)
        dst_path = os.path.join(ROOT, dst_rel)
        if not os.path.exists(src_path) or not os.path.exists(dst_path):
            print(f"  MISSING {src_rel} / {dst_rel}")
            failures += 1
            continue

        src_fns = [(name, norm(body))
                   for _, body, name in extract_device_functions(open(src_path).read())]
        dst_fns = [(name, norm(body))
                   for _, body, name in extract_device_functions(open(dst_path).read())]

        # Multiset comparison: the same file can define several variants of a
        # kernel under #ifdef (e.g. the gfx9 MFMA path in attention.hip), so
        # matching by name alone is ambiguous.
        from collections import Counter

        src_counter = Counter(src_fns)
        dst_counter = Counter(dst_fns)
        missing_items = list((src_counter - dst_counter).elements())
        matched = len(src_fns) - len(missing_items)

        for name, _ in missing_items[:3]:
            print(f"  BODY DIFF {dst_rel}: {name}")
        if missing_items:
            failures += len(missing_items)
        if verbose:
            print(
                f"  {os.path.basename(dst_rel):40s} device_fns={len(src_fns):4d} "
                f"identical={matched:4d} diff={len(missing_items)}"
            )
        total += len(src_fns)

        # Launch geometry: every hipLaunchKernelGGL call site must match too.
        src_launch = Counter(extract_launches(open(src_path).read()))
        dst_launch = Counter(extract_launches(open(dst_path).read()))
        launch_missing = list((src_launch - dst_launch).elements())
        if launch_missing:
            failures += len(launch_missing)
            print(f"  LAUNCH DIFF {dst_rel}: {len(launch_missing)} call site(s)")
        if verbose:
            print(
                f"  {'':40s} launches={sum(src_launch.values()):4d} "
                f"identical={sum(src_launch.values()) - len(launch_missing):4d}"
            )

    print(
        f"device functions checked: {total}; "
        f"{'all bodies identical' if failures == 0 else str(failures) + ' problem(s)'}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
