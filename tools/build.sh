#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build libgfx906_kernels.so from the extracted HIP sources.
#
#   DTK_ROOT=/opt/dtk tools/build.sh [release|debug]
#
# Everything is compiled with hipcc for gfx906 only.  There is no torch in the
# link line: the library depends on libamdhip64 and the C++ runtime only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODE="${1:-release}"
DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ ! -f "$DTK_ROOT/env.sh" ]]; then
  echo "[gfx906] DTK not found at '$DTK_ROOT'; set DTK_ROOT to your DTK tree" >&2
  exit 2
fi
export DTK_ROOT
export DTK_ROOT
# shellcheck disable=SC1090
set +u
source "${DTK_ROOT}/env.sh"
set -u

# DTK's clang looks for a gcc matching its own default; point it at the host's
# C++ headers explicitly (same fix as vLLM's tools/dcu/env.sh).
for _cxx in /usr/include/c++/*; do
  [[ -d "$_cxx" ]] || continue
  export CPLUS_INCLUDE_PATH="$_cxx:/usr/include/x86_64-linux-gnu/c++/$(basename "$_cxx")${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
done
unset _cxx

ARCH="${GX906_ARCH:-gfx906}"
OPT=(-O3 -g0)
[[ "$MODE" == "debug" ]] && OPT=(-O0 -g -DGX906_DEBUG=1)

BUILD="${GX906_BUILD_DIR:-$HERE/build}"
mkdir -p "$BUILD"

CXX_SOURCES=(src/runtime.cpp src/capi/gfx906_capi.cpp)
HIP_SOURCES=(src/kernels/*.hip)

echo "[gfx906] arch=$ARCH mode=$MODE out=$BUILD/libgfx906_kernels.so"
echo "[gfx906] hip=$(${DTK_ROOT}/bin/hipcc --version 2>/dev/null | head -1 || echo unknown)"

LIBDIRS=()
for _dir in /usr/lib/gcc/x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu; do
  [[ -d "$_dir" ]] && LIBDIRS+=(-L"$_dir")
done
unset _dir

"${DTK_ROOT}/bin/hipcc" \
  -shared -fPIC --offload-arch="$ARCH" \
  -std=c++17 "${OPT[@]}" -DUSE_ROCM -DGX906_KERNELS_BUILD=1 \
  -mno-fma -fno-gpu-rdc -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DNDEBUG \
  -Iinclude -Isrc -Isrc/vendor \
  "${CXX_SOURCES[@]}" "${HIP_SOURCES[@]}" \
  -o "$BUILD/libgfx906_kernels.so" \
  "${LIBDIRS[@]}" \
  -L"${DTK_ROOT}/lib" -lamdhip64

echo "[gfx906] ok: $BUILD/libgfx906_kernels.so"
