# SPDX-License-Identifier: Apache-2.0
"""gfx906 compute kernels for Hygon DCU Z100 (gfx906).

Two entry points:

* :func:`load_torch_ops` -- register ``torch.ops.gfx906_kernels.*`` (torch
  engines: vLLM, SGLang, custom torch stacks);
* :mod:`gfx906_kernels._ffi` -- ctypes bindings for engines that already own
  HIP buffers and do not want torch on the critical path.
"""

from __future__ import annotations

import os

from . import _ffi
from ._ffi import (  # noqa: F401
    BF16,
    F16,
    F32,
    F8E4M3,
    I8,
    Error,
    Tensor,
    build_id,
    get_device,
    get_stream,
    set_device,
    set_stream,
)

__all__ = [
    "Error",
    "Tensor",
    "BF16",
    "F16",
    "F32",
    "F8E4M3",
    "I8",
    "build_id",
    "get_device",
    "get_stream",
    "set_device",
    "set_stream",
    "load_torch_ops",
    "torch_ops",
    "is_available",
]

_TORCH_OPS = None


def is_available() -> bool:
    """True when the native library loads (it does not need a GPU to load)."""
    return _ffi._lib is not None


def _build_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.environ.get("GX906_BUILD_DIR", os.path.join(root, "build"))


def load_torch_ops():
    """Build (once) and load the torch adapter, returning the loaded module.

    The adapter is a plain shared object (``build/libgfx906_torch.so``) that
    registers ``torch.ops.gfx906_kernels.*``; it links the native library and
    nothing else.  Build it with ``tools/build_torch.sh``.
    """
    global _TORCH_OPS
    if _TORCH_OPS is not None:
        return _TORCH_OPS

    import torch

    build = _build_dir()
    adapter = os.environ.get("GX906_TORCH_LIB",
                             os.path.join(build, "libgfx906_torch.so"))
    if not os.path.exists(adapter):
        raise OSError(
            f"{adapter} not found; run tools/build_torch.sh "
            "(or set GX906_TORCH_LIB)"
        )
    torch.ops.load_library(adapter)
    _TORCH_OPS = torch.ops.gfx906_kernels
    return _TORCH_OPS


def torch_ops():
    """The registered ``torch.ops.gfx906_kernels`` namespace."""
    return load_torch_ops()
