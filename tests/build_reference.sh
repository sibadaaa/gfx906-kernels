#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the A/B reference extensions from the *vLLM sources in this repo*,
# using the production compile flags captured in
# .s56/rocm_c_rebuild/stmt.txt.
#
# This is only needed to run tests/compare_with_vllm.py on a checkout whose
# prebuilt vllm/*.so does not match the current environment.  It produces:
#
#   build/ref/_rocm_C.abi3.so   linear + attention ops  (torch.ops._rocm_C)
#   build/ref/_C_ref.abi3.so    RMSNorm + GDN ops       (torch.ops._C)
#
# Environment:
#   REPO        repo root (default: the parent of this module)
#   DTK_ROOT    DTK 26.04 host tree (default: /opt/dtk)
#   VLLM_REPO   local vLLM checkout the kernels came from (default: parent dir)
#   ROCM_BUILD_TREE  vLLM CMake build tree (default: <VLLM_REPO>/build-sync/temp)
#   TORCH_DIR   torch package to compile against; must match the interpreter
#               used for the A/B run (default: <VLLM_REPO>/.venv/...)
#   OUT         output dir (default: build/ref)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-$(dirname "$HERE")}"
DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
OUT="${OUT:-$HERE/build/ref}"

S56="$REPO/.s56/rocm_c_rebuild"
B="${ROCM_BUILD_TREE:-$REPO/build-sync/temp}"
TORCH_DIR="${TORCH_DIR:-$REPO/.venv/lib/python3.10/site-packages/torch}"
if [[ ! -f "$S56/stmt.txt" ]]; then
  echo "[ref] missing $S56/stmt.txt (the captured production compile line)." >&2
  echo "[ref] This harness is for A/B against a local vLLM checkout; point" >&2
  echo "[ref] VLLM_REPO/STMTS at that checkout and its captured build line." >&2
  exit 2
fi
if [[ ! -d "$TORCH_DIR/include" || ! -d "$TORCH_DIR/lib" ]]; then
  echo "[ref] torch headers/libs not found in $TORCH_DIR" >&2
  echo "[ref] set TORCH_DIR to the torch package you will run the A/B with" >&2
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "$DTK_ROOT/env.sh" >/dev/null 2>&1
set -u
for _cxx in /usr/include/c++/*; do
  [[ -d "$_cxx" ]] || continue
  export CPLUS_INCLUDE_PATH="$_cxx:/usr/include/x86_64-linux-gnu/c++/$(basename "$_cxx")${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
done
unset _cxx

mkdir -p "$OUT" "$OUT/shim/ATen/cuda" "$OUT/shim/c10/cuda" "$OUT/shim/core"

# The production build hipifies these includes; reproduce the mapping here so
# the untouched vLLM sources compile against the HIP torch headers.
cat > "$OUT/shim/ATen/dtk_macros.h" <<'EOF'
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
EOF
cat > "$OUT/shim/ATen/cuda/CUDAContext.h" <<'EOF'
#pragma once
#include <ATen/hip/HIPContext.h>
EOF
cat > "$OUT/shim/c10/cuda/CUDAGuard.h" <<'EOF'
#pragma once
#include <c10/hip/HIPGuard.h>
EOF
cat > "$OUT/shim/core/cuda_compat.h" <<'EOF'
#pragma once
#include "core/hip_compat.h"
EOF
cat > "$OUT/shim/torch_utils.h" <<'EOF'
#pragma once
#include <hip/hip_runtime.h>
EOF

get() { grep -m1 "^  $1 = " "$S56/stmt.txt" | sed "s/^  $1 = //"; }
rebase() {
  printf '%s' "$1" | sed 's/\\//g; s/"//g' \
    | sed "s|/usr/local/lib/python3.10/dist-packages/torch|$TORCH_DIR|g" \
    | sed "s|/opt/dtk-26.04-DCC2602-0317|$DTK_ROOT|g" \
    | sed "s|/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26105.2065-3f2faa34|$DTK_ROOT/hip/lib/libgalaxyhip.so.5|g" \
    | sed "s|/opt/dtk/|$DTK_ROOT/|g" \
    | sed "s|/src/csrc/dtk_rocprim_compat.h|$REPO/csrc/dtk_rocprim_compat.h|g"
}
DEFINES=$(get DEFINES | sed 's/GOOGLE_GLOG_DLL_DECL=""/GOOGLE_GLOG_DLL_DECL=/g')
FLAGS=$(rebase "$(get FLAGS)")
INCLUDES="-I$OUT/shim $(rebase "$(get INCLUDES)" | sed "s|-I$B/csrc|-I$REPO/csrc -I$B/csrc|")"
INCLUDES="$INCLUDES -isystem $S56/sysroot/usr/include"
LINK_FLAGS=$(rebase "$(get LINK_FLAGS)")
LINK_LIBRARIES=$(rebase "$(get LINK_LIBRARIES)" \
  | sed 's|/usr/lib/x86_64-linux-gnu/libglog.so.0.4.0||' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2||' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libunwind.so$|/usr/lib/x86_64-linux-gnu/libunwind.so.8|' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libunwind.so |/usr/lib/x86_64-linux-gnu/libunwind.so.8 |' \
  | sed 's|/usr/lib/x86_64-linux-gnu/librt.so$|/usr/lib/x86_64-linux-gnu/librt.so.1|' \
  | sed 's|/usr/lib/x86_64-linux-gnu/librt.so |/usr/lib/x86_64-linux-gnu/librt.so.1 |')
LINK_PATH=$(rebase "$(get LINK_PATH)")

cd "$B"
compile() {  # src lang out
  echo "[ref] $1"
  "$DTK_ROOT/llvm/bin/clang++" $DEFINES $INCLUDES $FLAGS -x "$2" -c "$1" -o "$3"
}
link() {  # out objs...
  local out=$1; shift
  echo "[ref] linking $out"
  "$DTK_ROOT/hip/bin/hipcc_cmake_linker_helper" "$DTK_ROOT/llvm/bin" \
    $LINK_FLAGS "$@" -o "$out" $LINK_PATH $LINK_LIBRARIES -shared
}

# --- linear + attention ------------------------------------------------
OBJS=("$OUT/_rocm_C.binding.o" "$OUT/_rocm_C.w8a8.o" "$OUT/_rocm_C.fp8.o" \
      "$OUT/_rocm_C.bf16.o" "$OUT/_rocm_C.attnd.o" "$OUT/_rocm_C.kvint8.o" \
      "$OUT/_rocm_C.sdot4_pc.o" "$OUT/_rocm_C.attention.o")
compile "$REPO/csrc/rocm/torch_bindings.cpp" c++ "${OBJS[0]}"
i=1
for f in w8a8_sdot4 fp8_emu_gemm bf16_emu_gemv attnd_inner attnd_inner_kvint8 sdot4_pc attention; do
  compile "$REPO/csrc/rocm/$f.hip" hip "${OBJS[$i]}"
  i=$((i + 1))
done
link "$OUT/_rocm_C.abi3.so" "${OBJS[@]}"

# --- norm + mamba ------------------------------------------------------
COBJS=("$OUT/_C.gfx906_bindings.o" "$OUT/_C.layernorm.o" "$OUT/_C.mamba.o")
compile "$REPO/csrc/rocm/gfx906_torch_bindings.cpp" c++ "${COBJS[0]}"
compile "$REPO/csrc/rocm/gfx906_layernorm_kernels.hip" hip "${COBJS[1]}"
compile "$REPO/csrc/mamba/gfx906_decode_kernels.hip" hip "${COBJS[2]}"
link "$OUT/_C_ref.abi3.so" "${COBJS[@]}"

ls -la "$OUT"/*.so
echo "[ref] done"
