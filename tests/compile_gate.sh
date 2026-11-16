#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile gate: every extracted kernel must compile standalone for gfx906.
# Needs no GPU (compile only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ ! -f "$DTK_ROOT/env.sh" ]]; then
  echo "[gfx906] DTK not found at '$DTK_ROOT'; set DTK_ROOT to your DTK tree" >&2
  exit 2
fi
export DTK_ROOT
# shellcheck disable=SC1090
set +u
source "$DTK_ROOT/env.sh"
set -u

# DTK's clang looks for a gcc matching its own default; point it at the host's
# C++ headers explicitly (same fix as vLLM's tools/dcu/env.sh).
for _cxx in /usr/include/c++/*; do
  [[ -d "$_cxx" ]] || continue
  export CPLUS_INCLUDE_PATH="$_cxx:/usr/include/x86_64-linux-gnu/c++/$(basename "$_cxx")${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
done
unset _cxx

ARCH="${GX906_ARCH:-gfx906}"
OUT="${TMPDIR:-/tmp}/gfx906-gate"
mkdir -p "$OUT"

# -mno-fma / -fno-gpu-rdc mirror the vLLM production compile line; FMA
# contraction changes 1-ulp results (GDN state update) and breaks bit-exactness.
FLAGS=(-c --offload-arch="$ARCH" -O3 -std=c++17 -fPIC -DUSE_ROCM
       -mno-fma -fno-gpu-rdc -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DNDEBUG
       -Iinclude -Isrc -Isrc/vendor -Wall -Wno-unused-result)

fail=0
for src in src/runtime.cpp src/capi/gfx906_capi.cpp src/kernels/*.hip; do
  obj="$OUT/$(basename "${src%.*}").o"
  printf '  %-52s' "$src"
  if "${DTK_ROOT}/bin/hipcc" "${FLAGS[@]}" "$src" -o "$obj" 2>"$OUT/err.log"; then
    echo "ok"
  else
    echo "FAIL"
    grep -E "error:" "$OUT/err.log" | head -5
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[gfx906] compile gate FAILED"
  exit 1
fi
echo "[gfx906] compile gate passed ($(ls src/kernels/*.hip | wc -l) kernel files)"
