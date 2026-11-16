#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the GPU-free test artifacts:
#   build/tests/fp8_dump     host-only fp8 conversion oracle
#   build/tests/minimal_gemv C ABI example, compiled with the *host* cc
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

# DTK's clang needs the host's libstdc++ headers explicitly.
for _cxx in /usr/include/c++/*; do
  [[ -d "$_cxx" ]] || continue
  export CPLUS_INCLUDE_PATH="$_cxx:/usr/include/x86_64-linux-gnu/c++/$(basename "$_cxx")${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
done
unset _cxx

mkdir -p build/tests
LIBDIRS=()
for _dir in /usr/lib/gcc/x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu; do
  [[ -d "$_dir" ]] && LIBDIRS+=(-L"$_dir")
done
unset _dir

"${DTK_ROOT}/bin/hipcc" "${LIBDIRS[@]}" -O2 -std=c++17 --offload-arch="${GX906_ARCH:-gfx906}" -Iinclude \
  tests/fp8_dump.cpp -o build/tests/fp8_dump

# Deliberately the host C compiler: the public header must not need HIP/torch.
cc -O2 -std=c99 -Wall -Wextra -Iinclude \
  examples/minimal_gemv.c -Lbuild -lgfx906_kernels \
  -Wl,-rpath,"$PWD/build" -o build/tests/minimal_gemv

echo "[gfx906] ok: build/tests/fp8_dump build/tests/minimal_gemv"
