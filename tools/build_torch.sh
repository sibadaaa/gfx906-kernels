#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the optional torch adapter (libgfx906_torch.so) with hipcc.
#
#   DTK_ROOT=... PYTHON=.venv/bin/python tools/build_torch.sh
#
# The adapter only talks to the C ABI; it is a plain shared object that
# `torch.ops.load_library()` can load.  Building it with hipcc (instead of the
# host compiler) matches how the ROCm torch headers are meant to be consumed.
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

PYTHON="${PYTHON:-$HERE/../.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

# Resolve the torch package directory without importing torch: the DTK
# runtime libs may not be on this shell's loader path yet, and an import would
# fail for reasons unrelated to the build.
TORCH_PREFIX="$("$PYTHON" - <<'PY'
import os, sysconfig
print(os.path.join(sysconfig.get_paths()["purelib"], "torch"))
PY
)"
if [[ ! -d "$TORCH_PREFIX/include" ]]; then
  echo "[gfx906] torch headers not found under $TORCH_PREFIX" >&2
  echo "[gfx906] set PYTHON to the interpreter whose torch you build against" >&2
  exit 1
fi
TORCH_INC="$TORCH_PREFIX/include"
TORCH_LIB="$TORCH_PREFIX/lib"

BUILD="${GX906_BUILD_DIR:-$HERE/build}"
NATIVE="$BUILD/libgfx906_kernels.so"
if [[ ! -f "$NATIVE" ]]; then
  echo "[gfx906] $NATIVE missing; run tools/build.sh first" >&2
  exit 1
fi

LIBDIRS=()
for _dir in /usr/lib/gcc/x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu; do
  [[ -d "$_dir" ]] && LIBDIRS+=(-L"$_dir")
done
unset _dir

"${DTK_ROOT}/bin/hipcc" \
  -shared -fPIC --offload-arch="${GX906_ARCH:-gfx906}" \
  -std=c++17 -O3 -DUSE_ROCM -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$HERE/include" -I"$TORCH_INC" -I"$TORCH_INC/torch/csrc/api/include" \
  torch/gfx906_torch.cpp \
  -o "$BUILD/libgfx906_torch.so" \
  "${LIBDIRS[@]}" \
  -L"$BUILD" -lgfx906_kernels -Wl,-rpath,"$BUILD" \
  -L"$TORCH_LIB" -ltorch -ltorch_cpu -lc10 -ltorch_hip \
  -L"$DTK_ROOT/lib" -lamdhip64

echo "[gfx906] ok: $BUILD/libgfx906_torch.so"
