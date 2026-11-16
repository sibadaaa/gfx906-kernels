#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every C ABI function declared in the header is really exported.

Catches the classic split-brain bug where the header grows a declaration but
the library never defines it (link succeeds, call fails at runtime).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HEADER = os.path.join(ROOT, "include", "gfx906", "gfx906.h")
LIB = os.path.join(ROOT, "build", "libgfx906_kernels.so")


def main() -> int:
    if not os.path.exists(LIB):
        print(f"missing {LIB}; run tools/build.sh", file=sys.stderr)
        return 2
    declared = set(
        re.findall(r"\b(gx906_[a-z0-9_]+)\s*\(", open(HEADER).read())
    )
    nm = subprocess.run(
        ["nm", "-D", "--defined-only", LIB], capture_output=True, text=True
    ).stdout
    defined = set(re.findall(r" T (gx906_[a-z0-9_]+)", nm))
    missing = sorted(declared - defined)
    extra = sorted(defined - declared)
    if missing:
        print(f"declared but not defined: {missing}")
    if extra:
        print(f"defined but not declared: {extra}")
    if missing or extra:
        return 1
    print(f"ok: {len(declared)} C ABI functions declared == defined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
