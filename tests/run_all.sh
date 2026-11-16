#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run every check that does NOT need a GPU:
#   build -> compile gate -> device-code audit -> fp8 vs PyTorch -> ABI symbols
#   -> C99 example (rejection path)
#
#   tests/run_all.sh              # full (rebuilds the library)
#   tests/run_all.sh --no-build   # reuse build/
#
# The GPU-only gate (bit-exact vs vLLM operators) is tests/compare_with_vllm.py
# and must be run on the Z100L host.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ ! -f "$DTK_ROOT/env.sh" ]]; then
  echo "[gfx906] DTK not found at '$DTK_ROOT'; set DTK_ROOT to your DTK tree" >&2
  exit 2
fi
export DTK_ROOT

NO_BUILD=0
[[ "${1:-}" == "--no-build" ]] && NO_BUILD=1

# DTK runtime for the library + example.
set +u
# shellcheck disable=SC1090
source "$DTK_ROOT/env.sh" >/dev/null 2>&1
set -u

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for cand in "$HERE/../.venv/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
      PYTHON="$cand"
      break
    fi
  done
fi

echo "== build =="
if [[ $NO_BUILD -eq 0 ]]; then
  tools/build.sh >/dev/null
fi
tools/build_tests.sh >/dev/null

echo "== compile gate =="
tests/compile_gate.sh | tail -1

echo "== device-code audit =="
"$PYTHON" tests/audit_device_code.py

echo "== fp8 conversion vs PyTorch (CPU) =="
# torch wheels often need their own lib dir (and libomp from the DTK LLVM tree)
# on the loader path; add both without touching the caller's environment.
TORCH_LIB="$("$PYTHON" - <<'PY' 2>/dev/null || true
import os, sysconfig
print(os.path.join(sysconfig.get_paths()["purelib"], "torch", "lib"))
PY
)"
if [[ -n "$TORCH_LIB" && -d "$TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="$TORCH_LIB:$DTK_ROOT/llvm/lib:${LD_LIBRARY_PATH:-}"
fi
if "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  "$PYTHON" tests/test_fp8_conversion.py
else
  echo "  skipped: this interpreter cannot import torch (run it manually with"
  echo "           the engine's own LD_LIBRARY_PATH to include this check)"
fi

echo "== C ABI symbols =="
"$PYTHON" tests/check_abi_symbols.py

echo "== C99 example =="
LD_LIBRARY_PATH="$HERE/build:${LD_LIBRARY_PATH:-}" ./build/tests/minimal_gemv

echo
echo "all GPU-free checks passed."
echo "GPU gate (run on the Z100L host): $PYTHON tests/compare_with_vllm.py"
