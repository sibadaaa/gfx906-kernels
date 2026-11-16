# SPDX-License-Identifier: Apache-2.0
"""ctypes binding for libgfx906_kernels.so (no torch required).

An engine that already has HIP buffers can describe them with :func:`view` and
call the ops directly.  The library allocates outputs; ``Tensor.release()``
drops that reference.
"""

from __future__ import annotations

import ctypes as C
import os
from contextlib import contextmanager
from typing import Optional, Sequence

MAX_DIMS = 8

OK = 0
ERR_INVALID_ARGUMENT = 1
ERR_INVALID_SHAPE = 2
ERR_UNSUPPORTED = 3
ERR_HIP = 4
ERR_OUT_OF_MEMORY = 5
ERR_INTERNAL = 6

# torch ScalarType encoding
U8, I8, I16, I32, I64 = 0, 1, 2, 3, 4
F16, F32, F64, BF16 = 5, 6, 7, 15
F8E5M2, F8E4M3, F8E5M2FNUZ, F8E4M3FNUZ = 23, 24, 25, 26


class GxTensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("sizes", C.c_int64 * MAX_DIMS),
        ("strides", C.c_int64 * MAX_DIMS),
        ("ndim", C.c_int32),
        ("dtype", C.c_int32),
        ("device", C.c_int32),
        ("reserved0", C.c_int32),
        ("owner", C.c_void_p),
        ("reserved1", C.c_void_p),
    ]


class Error(RuntimeError):
    pass


def _find_library() -> str:
    candidates = []
    env = os.environ.get("GX906_LIB")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    candidates += [
        os.path.join(root, "build", "libgfx906_kernels.so"),
        os.path.join(root, "build", "libgfx906_kernels.dylib"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise OSError(
        "libgfx906_kernels.so not found; run tools/build.sh or set GX906_LIB"
    )


_lib = C.CDLL(_find_library())

_lib.gx906_abi_version.restype = C.c_int32
_lib.gx906_last_error.restype = C.c_char_p
_lib.gx906_set_stream.argtypes = [C.c_void_p]
_lib.gx906_get_stream.restype = C.c_void_p
_lib.gx906_set_device.argtypes = [C.c_int32]
_lib.gx906_get_device.restype = C.c_int32
_lib.gx906_build_id.restype = C.c_char_p

ALLOC_FN = C.CFUNCTYPE(C.c_void_p, C.c_size_t, C.c_void_p)
FREE_FN = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
_lib.gx906_set_allocator.argtypes = [ALLOC_FN, FREE_FN, C.c_void_p]

_lib.gx906_tensor_view.argtypes = [
    C.POINTER(GxTensor),
    C.c_void_p,
    C.POINTER(C.c_int64),
    C.POINTER(C.c_int64),
    C.c_int32,
    C.c_int32,
    C.c_int32,
]
_lib.gx906_tensor_release.argtypes = [C.POINTER(GxTensor)]

_P = C.POINTER(GxTensor)


def _sig(name: str, args: Sequence, restype=C.c_int32) -> None:
    fn = getattr(_lib, name)
    fn.argtypes = list(args)
    fn.restype = restype


_T = C.c_float
_I = C.c_int64
_B = C.c_int32

_sig("gx906_bf16_emu_gemv",
     [_P, _P, _P, _I, _I, _B, _P])
_sig("gx906_fp16_gemv_stream_m1", [_P, _P, _P, _I, _P])
_sig("gx906_w8a8_sdot4_gemv", [_P, _P, _P, _P, _P, _B, _I, _P])
_sig("gx906_w8a8_sdot4_gemm", [_P, _P, _P, _P, _P, _B, _I, _P])
_sig("gx906_w8a8_sdot4_gemm_pc", [_P, _P, _P, _P, _P, _B, _I, _P])
_sig("gx906_fp8_emu_gemv", [_P, _P, _P, _P, _P, _P, _B, _I, _P])
_sig("gx906_fp8_emu_gemm", [_P, _P, _P, _P, _P, _P, _B, _I, _P])
_sig("gx906_fp8_emu_gemv_block",
     [_P, _P, _P, _P, _P, _P, _B, _I, _I, _I, _P])
_sig("gx906_fp8_emu_gemm_block",
     [_P, _P, _P, _P, _P, _P, _B, _I, _I, _I, _P])
_sig("gx906_fp8_emu_gemm_block_w8a16",
     [_P, _P, _P, _P, _P, _B, _I, _I, _I, _I, _P])
_sig("gx906_fp8_emu_gemm_block_w8a16_smul",
     [_P, _P, _P, _P, _P, _B, _I, _I, _I, _P])
_sig("gx906_fp8_dequant_to_fp16_out", [_P, _P, _P, _P, _I, _I])
_sig("gx906_fp8_lut_fill", [_P])
_sig("gx906_rms_norm_gated", [_P, _P, _P, _T, _B, _P])
_sig("gx906_gemma_rms_norm", [_P, _P, _T, _P])
_sig("gx906_gemma_fused_add_rms_norm", [_P, _P, _P, _T, _P, _P])


def _last_error() -> str:
    return _lib.gx906_last_error().decode("utf-8", "replace")


def check(status: int) -> None:
    if status != OK:
        raise Error(f"gfx906 status={status}: {_last_error()}")


def set_stream(hip_stream: int) -> None:
    _lib.gx906_set_stream(C.c_void_p(hip_stream))


def get_stream() -> Optional[int]:
    return _lib.gx906_get_stream()


def set_device(device: int) -> None:
    _lib.gx906_set_device(device)


def get_device() -> int:
    return _lib.gx906_get_device()


def build_id() -> str:
    return _lib.gx906_build_id().decode()


@contextmanager
def current_hip_stream(stream: int):
    """Scope the library's stream (e.g. torch.cuda.current_stream().cuda_stream)."""
    previous = get_stream()
    set_stream(stream)
    try:
        yield
    finally:
        set_stream(previous or 0)


class Tensor:
    """A gx_tensor.  Non-owning for views; release() for library outputs."""

    __slots__ = ("_t",)

    def __init__(self, inner: Optional[GxTensor] = None):
        self._t = inner if inner is not None else GxTensor()

    @classmethod
    def view(
        cls,
        data: int,
        sizes: Sequence[int],
        dtype: int,
        device: int = 0,
        strides: Optional[Sequence[int]] = None,
    ) -> "Tensor":
        ndim = len(sizes)
        c_sizes = (C.c_int64 * ndim)(*sizes)
        c_strides = (C.c_int64 * ndim)(*strides) if strides else None
        out = cls()
        _lib.gx906_tensor_view(
            C.byref(out._t), C.c_void_p(data), c_sizes, c_strides, ndim, dtype,
            device,
        )
        return out

    @property
    def ptr(self) -> int:
        return self._t.data or 0

    @property
    def sizes(self) -> tuple:
        return tuple(self._t.sizes[i] for i in range(self._t.ndim))

    @property
    def dtype(self) -> int:
        return self._t.dtype

    @property
    def device(self) -> int:
        return self._t.device

    def release(self) -> None:
        # Views have no owner: releasing them would just zero the descriptor.
        if self._t.owner:
            _lib.gx906_tensor_release(C.byref(self._t))

    def _p(self):
        return C.byref(self._t)

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.release()
        except Exception:
            pass


def _opt(t: Optional[Tensor]):
    return t._p() if t is not None else None


# --- convenience wrappers -------------------------------------------------

def bf16_emu_gemv(x, w, bias=None, nthreads=256, route=0, range_safe=False):
    out = Tensor()
    check(_lib.gx906_bf16_emu_gemv(x._p(), w._p(), _opt(bias), nthreads, route,
                                   int(range_safe), out._p()))
    return out


def fp16_gemv_stream_m1(x, w, bias=None, nthreads=256):
    out = Tensor()
    check(_lib.gx906_fp16_gemv_stream_m1(x._p(), w._p(), _opt(bias), nthreads,
                                         out._p()))
    return out


def w8a8_sdot4_gemv(xq, w, x_s, w_s, bias=None, out_dtype=F16, nthreads=256):
    out = Tensor()
    check(_lib.gx906_w8a8_sdot4_gemv(xq._p(), w._p(), x_s._p(), w_s._p(),
                                     _opt(bias), out_dtype, nthreads, out._p()))
    return out


def w8a8_sdot4_gemm(xq, w, x_s, w_s, bias=None, out_dtype=F16, cfg=0):
    out = Tensor()
    check(_lib.gx906_w8a8_sdot4_gemm(xq._p(), w._p(), x_s._p(), w_s._p(),
                                     _opt(bias), out_dtype, cfg, out._p()))
    return out


def fp8_emu_gemv(a, w, a_s, w_s, lut, bias=None, out_dtype=F16, nthreads=256):
    out = Tensor()
    check(_lib.gx906_fp8_emu_gemv(a._p(), w._p(), a_s._p(), w_s._p(), lut._p(),
                                  _opt(bias), out_dtype, nthreads, out._p()))
    return out


def fp8_emu_gemm(a, w, a_s, w_s, lut, bias=None, out_dtype=F16, cfg=0):
    out = Tensor()
    check(_lib.gx906_fp8_emu_gemm(a._p(), w._p(), a_s._p(), w_s._p(), lut._p(),
                                  _opt(bias), out_dtype, cfg, out._p()))
    return out


def fp8_emu_gemm_block_w8a16(a, w, w_s, lut, bias=None, out_dtype=F16,
                             nthreads=256, cfg=0, block_n=128, block_k=128):
    out = Tensor()
    check(_lib.gx906_fp8_emu_gemm_block_w8a16(
        a._p(), w._p(), w_s._p(), lut._p(), _opt(bias), out_dtype, nthreads, cfg,
        block_n, block_k, out._p()))
    return out


def fp8_lut_fill(lut) -> None:
    check(_lib.gx906_fp8_lut_fill(lut._p()))


def rms_norm_gated(x, weight, gate, epsilon, norm_before_gate):
    out = Tensor()
    check(_lib.gx906_rms_norm_gated(x._p(), weight._p(), gate._p(),
                                    float(epsilon), int(norm_before_gate),
                                    out._p()))
    return out


def gemma_rms_norm(x, weight, epsilon):
    out = Tensor()
    check(_lib.gx906_gemma_rms_norm(x._p(), weight._p(), float(epsilon),
                                    out._p()))
    return out


def gemma_fused_add_rms_norm(x, residual, weight, epsilon):
    out, residual_out = Tensor(), Tensor()
    check(_lib.gx906_gemma_fused_add_rms_norm(
        x._p(), residual._p(), weight._p(), float(epsilon), out._p(),
        residual_out._p()))
    return out, residual_out
