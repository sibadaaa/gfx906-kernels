# SPDX-License-Identifier: Apache-2.0
"""Bit-exact gate: gfx906_kernels vs the in-tree vLLM `_rocm_C` operators.

Run on the 4x Z100L host from the repository root:

    .venv/bin/python gfx906-kernels/tests/compare_with_vllm.py

Every case calls the production operator and the extracted one on identical
inputs and compares the raw output bits (NaN-safe, no tolerance).  A mismatch
prints the first differing index, both values and both bit patterns.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

# Reference: either the in-tree vLLM build (``import vllm``) or an explicitly
# given extension, e.g. a relink of the same sources with production flags:
#   GX906_REF_LIB=/path/_rocm_C.abi3.so python tests/compare_with_vllm.py
_REF_LIB = os.environ.get("GX906_REF_LIB")
_REF_C_LIB = os.environ.get("GX906_REF_C_LIB")
if _REF_LIB:
    torch.ops.load_library(_REF_LIB)
    if _REF_C_LIB:
        # Norm / GDN registers into `_C`, linear+attention into `_rocm_C`.
        torch.ops.load_library(_REF_C_LIB)
else:
    import vllm  # noqa: E402,F401  (registers torch.ops._rocm_C and _C)

import gfx906_kernels  # noqa: E402

gfx906_kernels.load_torch_ops()

try:
    REF = torch.ops._rocm_C
except AttributeError:  # pragma: no cover
    raise SystemExit(
        "torch.ops._rocm_C is not registered; run this from the gfx906 vLLM "
        "environment (import vllm must load the rocm extension)"
    )
NEW = torch.ops.gfx906_kernels
try:
    REF_C = torch.ops._C
except AttributeError:  # pragma: no cover
    REF_C = None

DEV = "cuda:0"
FAILURES: list[str] = []
KNOWN: list[str] = []
SKIPPED: list[str] = []


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float16, torch.bfloat16, torch.int16, torch.uint16):
        return t.view(torch.int16)
    if t.dtype in (torch.float32, torch.int32):
        return t.view(torch.int32)
    if t.dtype in (torch.float64, torch.int64):
        return t.view(torch.int64)
    return t


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape or a.dtype != b.dtype:
        FAILURES.append(
            f"{name}: metadata mismatch ref={tuple(a.shape)}/{a.dtype} "
            f"new={tuple(b.shape)}/{b.dtype}"
        )
        return
    ba, bb = _bits(a), _bits(b)
    if torch.equal(ba, bb):
        print(f"  ok    {name}  {tuple(a.shape)} {a.dtype}")
        return
    diff = (ba != bb).nonzero()
    idx = tuple(diff[0].tolist())
    FAILURES.append(
        f"{name}: {diff.shape[0]}/{a.numel()} differ; first {idx} "
        f"ref={a[idx].item()!r} (0x{int(ba[idx]):x}) "
        f"new={b[idx].item()!r} (0x{int(bb[idx]):x})"
    )


def case(fn):
    if os.environ.get("GX906_TRACE"):
        print(f"case: {fn.__name__}", flush=True)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{fn.__name__}: raised {exc!r}\n{traceback.format_exc()}")
    return fn


def _rand_half(shape, dtype, scale=1.0):
    return (torch.randn(shape, device=DEV, dtype=torch.float32) * scale).to(dtype)


FK = [512, 2048, 5120]


@case
def gemv_dense():
    for dtype in (torch.float16, torch.bfloat16):
        for M in (1, 2, 4, 8):
            for N, K in ((512, 512), (2048, 5120), (17408, 5120)):
                x = _rand_half((M, K), dtype)
                w = _rand_half((N, K), dtype)
                for bias in (None, _rand_half((N,), dtype)):
                    a = REF.bf16_emu_gemv(x, w, bias, 256, 0, False)
                    b = NEW.bf16_emu_gemv(x, w, bias, 256, 0, False)
                    compare(
                        f"bf16_emu_gemv {dtype} M={M} N={N} K={K} "
                        f"bias={bias is not None}", a, b)


@case
def gemv_stream_m1():
    for N, K in ((2048, 5120), (5120, 5120), (17408, 5120)):
        x = _rand_half((1, K), torch.float16)
        w = _rand_half((N, K), torch.float16)
        a = REF.fp16_gemv_stream_m1(x, w, None, 256)
        b = NEW.fp16_gemv_stream_m1(x, w, None, 256)
        compare(f"fp16_gemv_stream_m1 N={N} K={K}", a, b)


def _int8_inputs(M, N, K):
    xq = torch.randint(-127, 127, (M, K), device=DEV, dtype=torch.int8)
    w = torch.randint(-127, 127, (N, K), device=DEV, dtype=torch.int8)
    x_s = torch.rand((M,), device=DEV, dtype=torch.float32) * 0.01 + 1e-4
    w_s = torch.rand((N,), device=DEV, dtype=torch.float32) * 0.01 + 1e-4
    return xq, w, x_s, w_s


@case
def sdot4_gemv():
    for M in (1, 2, 4, 8):
        for N, K in ((512, 512), (2048, 5120), (17408, 5120)):
            xq, w, x_s, w_s = _int8_inputs(M, N, K)
            bias = _rand_half((N,), torch.float16)
            a = REF.w8a8_sdot4_gemv(xq, w, x_s, w_s, bias, torch.float16, 256)
            b = NEW.w8a8_sdot4_gemv(xq, w, x_s, w_s, bias, torch.float16, 256)
            compare(f"w8a8_sdot4_gemv M={M} N={N} K={K}", a, b)


@case
def sdot4_gemm():
    for M in (16, 128, 512):
        for N, K in ((512, 512), (2048, 5120)):
            xq, w, x_s, w_s = _int8_inputs(M, N, K)
            for cfg in (0,):
                a = REF.w8a8_sdot4_gemm(xq, w, x_s, w_s, None, torch.float16, cfg)
                b = NEW.w8a8_sdot4_gemm(xq, w, x_s, w_s, None, torch.float16, cfg)
                compare(f"w8a8_sdot4_gemm M={M} N={N} K={K} cfg={cfg}", a, b)


def _fp8_inputs(M, N, K, block=None, dtype=torch.float16):
    a = _rand_half((M, K), dtype)
    w = _rand_half((N, K), torch.float32).to(torch.float8_e4m3fn)
    if block is None:
        a_s = torch.rand((M,), device=DEV, dtype=torch.float32) * 0.01 + 1e-3
        w_s = torch.rand((N,), device=DEV, dtype=torch.float32) * 0.01 + 1e-3
        return a, w, a_s, w_s
    block_n, block_k = block
    nkb = (K + block_k - 1) // block_k
    a_s = torch.rand((M, nkb), device=DEV, dtype=torch.float32) * 0.5
    w_s = torch.rand(((N + block_n - 1) // block_n, nkb), device=DEV,
                     dtype=torch.float32) * 0.5
    return a, w, a_s, w_s


def _lut():
    # Bit-exact copy of the analytic table vLLM's gfx906_fp8.py builds
    # (fp8_to_fp16_lut): e4m3fn, bias 7, no inf codes, 0x7F/0xFF are NaN.
    bits = torch.arange(256, dtype=torch.int32)
    sign = torch.where((bits >> 7) & 1 == 1, -1.0, 1.0)
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    mant_scale = 1.0 / 8.0
    bias = 7
    nan_codes = (exp == 0xF) & (man == 0x7)
    man_f = man.to(torch.float32)
    exp_f = exp.to(torch.float32)
    val = torch.where(
        exp == 0,
        man_f * mant_scale * 2.0 ** (1 - bias),
        (1.0 + man_f * mant_scale) * torch.pow(2.0, exp_f - bias),
    )
    out = sign * val
    out = torch.where(nan_codes, torch.tensor(float("nan")), out)
    return out.to(torch.half).to(DEV).contiguous()


@case
def fp8_emu():
    lut = _lut()
    for dtype in (torch.float16, torch.bfloat16):
        for M in (1, 4):
            N, K = 2048, 5120
            a, w, a_s, w_s = _fp8_inputs(M, N, K, dtype=dtype)
            a = (a.to(torch.float32) * a_s.view(-1, 1)).to(torch.float8_e4m3fn)
            ra = REF.fp8_emu_gemv(a, w, a_s, w_s, lut, None, dtype, 256)
            rb = NEW.fp8_emu_gemv(a, w, a_s, w_s, lut, None, dtype, 256)
            compare(f"fp8_emu_gemv {dtype} M={M}", ra, rb)


@case
def fp8_block_w8a16():
    lut = _lut()
    # fp8 block checkpoints write scale_inv as [cdiv(N,bn), cdiv(K,bk)].
    for M in (1, 4):
        for block in ((128, 128),):
            N, K = 2048, 5120
            a, w, _, w_s = _fp8_inputs(M, N, K, block=block)
            ra = REF.fp8_emu_gemm_block_w8a16(a, w, w_s, lut, None,
                                              torch.float16, 256, 0, *block)
            rb = NEW.fp8_emu_gemm_block_w8a16(a, w, w_s, lut, None,
                                              torch.float16, 256, 0, *block)
            compare(f"fp8_emu_gemm_block_w8a16 M={M} block={block}", ra, rb)


@case
def fp8_block_and_gemm():
    lut = _lut()
    N, K, block = 2048, 5120, (128, 128)
    for M in (1, 4, 16, 128):
        a, w, a_s, w_s = _fp8_inputs(M, N, K, block=block)
        # block-scaled GEMM/GEMV consume an already-quantised activation
        a = a.to(torch.float8_e4m3fn)
        ra = REF.fp8_emu_gemm_block(a, w, a_s, w_s, lut, None, torch.float16,
                                    0, *block)
        rb = NEW.fp8_emu_gemm_block(a, w, a_s, w_s, lut, None, torch.float16,
                                    0, *block)
        compare(f"fp8_emu_gemm_block M={M}", ra, rb)

        a_q, w_q, a_sq, w_sq = _fp8_inputs(M, N, K)
        a_q = (a_q.to(torch.float32) * a_sq.view(-1, 1)).to(torch.float8_e4m3fn)
        ra = REF.fp8_emu_gemm(a_q, w_q, a_sq, w_sq, lut, None, torch.float16, 0)
        rb = NEW.fp8_emu_gemm(a_q, w_q, a_sq, w_sq, lut, None, torch.float16, 0)
        compare(f"fp8_emu_gemm M={M}", ra, rb)


@case
def w8a16_smul():
    # The whole-A staging variant is env-gated at library load time, so the
    # switch has to be set before the process starts:
    #   VLLM_DCU_FP8_GEMV_WA=1 python tests/compare_with_vllm.py
    if os.environ.get("VLLM_DCU_FP8_GEMV_WA") != "1":
        SKIPPED.append("w8a16 smul: rerun with VLLM_DCU_FP8_GEMV_WA=1")
        return
    lut = _lut()
    N, K = 2048, 5120
    a = _rand_half((1, 2 * K), torch.float16, 0.5)
    w = _rand_half((N, K), torch.float32).to(torch.float8_e4m3fn)
    w_s = torch.rand(((N + 127) // 128, K // 128), device=DEV,
                     dtype=torch.float32) * 0.5
    ra = REF.fp8_emu_gemm_block_w8a16_smul(a, w, w_s, lut, None,
                                           torch.float16, 256, 128, 128)
    rb = NEW.fp8_emu_gemm_block_w8a16_smul(a, w, w_s, lut, None,
                                           torch.float16, 256, 128, 128)
    compare("fp8_emu_gemm_block_w8a16_smul", ra, rb)


@case
def prefill_int8_pc():
    lut = _lut()
    N, K = 2048, 5120
    w = _rand_half((N, K), torch.float32).to(torch.float8_e4m3fn)
    w_s = torch.rand(((N + 127) // 128, K // 128), device=DEV,
                     dtype=torch.float32) * 0.5

    qa = REF.fp8_stage_int8_pc(w, w_s, lut, 128, 128)
    qb = NEW.fp8_stage_int8_pc(w, w_s, lut, 128, 128)
    for name, xa, xb in zip(("quant_w", "scale_w"), qa, qb):
        compare(f"fp8_stage_int8_pc {name}", xa, xb)

    for M in (128, 512):
        xq = torch.randint(-127, 127, (M, K), device=DEV, dtype=torch.int8)
        x_s = torch.rand((M,), device=DEV, dtype=torch.float32) * 0.01 + 1e-4
        bias = _rand_half((N,), torch.float16)
        for cfg in (0, 4):
            ra = REF.w8a8_sdot4_gemm_pc(xq, qa[0], x_s, qa[1], bias,
                                        torch.float16, cfg)
            rb = NEW.w8a8_sdot4_gemm_pc(xq, qb[0], x_s, qb[1], bias,
                                        torch.float16, cfg)
            compare(f"w8a8_sdot4_gemm_pc M={M} cfg={cfg}", ra, rb)

        ra = REF.fp8_emu_mm_int8_pc(xq, w, w_s, x_s, lut, bias,
                                    torch.float16, 128, 128)
        rb = NEW.fp8_emu_mm_int8_pc(xq, w, w_s, x_s, lut, bias,
                                    torch.float16, 128, 128)
        compare(f"fp8_emu_mm_int8_pc M={M}", ra, rb)


@case
def fp8_block_quant():
    lut = _lut()
    N, K, block = 2048, 5120, (128, 128)
    for M in (1, 16):
        a, w, _, w_s = _fp8_inputs(M, N, K, block=block)
        for eps in (1e-10, 1e-3):
            ra = REF.fp8_emu_gemm_block_quant(a, w, w_s, lut, None,
                                              torch.float16, 256, 0, *block, eps)
            rb = NEW.fp8_emu_gemm_block_quant(a, w, w_s, lut, None,
                                              torch.float16, 256, 0, *block, eps)
            compare(f"fp8_emu_gemm_block_quant M={M} eps={eps}", ra, rb)

            ta = REF.fp8_emu_gemm_block_quant_x(a, w, w_s, lut, None,
                                                torch.float16, 256, 0, *block,
                                                eps)
            tb = NEW.fp8_emu_gemm_block_quant_x(a, w, w_s, lut, None,
                                                torch.float16, 256, 0, *block,
                                                eps)
            for name, xa, xb in zip(("out", "x_q", "x_s"), ta, tb):
                compare(
                    f"fp8_emu_gemm_block_quant_x M={M} eps={eps} {name}",
                    xa, xb)


@case
def gemma_fused_add_norm():
    if REF_C is None or not hasattr(REF_C, "gemma_fused_add_rms_norm_gfx906"):
        SKIPPED.append("gemma fused add: reference extension does not register it")
        return
    for dtype in (torch.float16, torch.bfloat16):
        for T, H in ((1, 5120), (4, 5120), (128, 5120)):
            x = _rand_half((T, H), dtype)
            w = _rand_half((H,), dtype)
            for res_dtype in (dtype, torch.float32):
                residual = _rand_half((T, H), res_dtype)
                ra = REF_C.gemma_fused_add_rms_norm_gfx906(
                    x, residual.clone(), w, 1e-6)
                rb = NEW.gemma_fused_add_rms_norm_gfx906(
                    x, residual.clone(), w, 1e-6)
                compare(f"gemma_fused_add {dtype}/{res_dtype} {T}x{H} out",
                        ra[0], rb[0])
                compare(f"gemma_fused_add {dtype}/{res_dtype} {T}x{H} residual",
                        ra[1], rb[1])


@case
def norm_family():
    if REF_C is None or not hasattr(REF_C, "rms_norm_gated_gfx906"):
        SKIPPED.append("norm family: reference extension does not register it")
        return
    for dtype in (torch.float16, torch.bfloat16):
        for T, H in ((1, 5120), (4, 5120), (512, 5120)):
            x = _rand_half((T, H), dtype)
            gate = _rand_half((T, H), dtype)
            w = _rand_half((H,), dtype)
            for before in (True, False):
                ra = REF_C.rms_norm_gated_gfx906(x, w, gate, 1e-6, before)
                rb = NEW.rms_norm_gated_gfx906(x, w, gate, 1e-6, before)
                compare(f"rms_norm_gated {dtype} {T}x{H} before={before}", ra, rb)
            ra = REF_C.gemma_rms_norm_gfx906(x, w, 1e-6)
            rb = NEW.gemma_rms_norm_gfx906(x, w, 1e-6)
            compare(f"gemma_rms_norm {dtype} {T}x{H}", ra, rb)


@case
def gdn_decode_case():
    if REF_C is None or not hasattr(REF_C, "fused_sigmoid_gating_delta_rule_gfx906_decode"):
        SKIPPED.append("GDN decode: reference extension does not register it")
        return
    torch.manual_seed(13)
    tokens, heads, key_dim, value_dim, slots = 4, 4, 128, 128, 8
    for dtype in (torch.float16, torch.bfloat16):
        for state_dtype in (dtype, torch.float32):
            q = _rand_half((1, tokens, heads, key_dim), dtype, 0.5)
            k = _rand_half((1, tokens, heads, key_dim), dtype, 0.5)
            v = _rand_half((1, tokens, heads, value_dim), dtype, 0.5)
            A_log = _rand_half((heads,), dtype, 0.2)
            dt_bias = _rand_half((heads,), dtype, 0.2)
            a = _rand_half((tokens, heads), dtype, 0.5)
            b = _rand_half((tokens, heads), dtype, 0.5)
            state = _rand_half((slots, heads, value_dim, key_dim), state_dtype,
                               0.2)
            for l2norm in (False, True):
                def run(op, st):
                    return op(A_log, a, b, dt_bias, q, k, v, st, 1.0, 20.0,
                              key_dim ** -0.5, l2norm), st

                ref_out, ref_state = run(
                    REF_C.fused_sigmoid_gating_delta_rule_gfx906_decode,
                    state.clone())
                new_out, new_state = run(
                    NEW.fused_sigmoid_gating_delta_rule_gfx906_decode,
                    state.clone())
                compare(
                    f"gdn_decode {dtype}/{state_dtype} l2norm={l2norm} out",
                    ref_out, new_out)
                compare(
                    f"gdn_decode {dtype}/{state_dtype} l2norm={l2norm} state",
                    ref_state, new_state)


@case
def attnd_decode_inner_case():
    if not hasattr(REF, "gfx906_attnd_decode_inner"):
        SKIPPED.append("attnd decode inner: reference extension does not register it")
        return
    torch.manual_seed(17)
    q_len, nq, nkv, head, block_size, num_blocks = 32, 6, 1, 256, 16, 8
    num_segments = 2
    rows = q_len
    scale = head ** -0.5

    query = _rand_half((q_len, nq, head), torch.float16, 0.5)
    # production hybrid page: [blocks, block_size, nkv, head], K|V packed per
    # token in the real cache (stride 512); separate K/V tensors reproduce the
    # same strides.
    key = _rand_half((num_blocks, block_size, nkv, head), torch.float16, 0.5)
    value = _rand_half((num_blocks, block_size, nkv, head), torch.float16, 0.5)
    block_table = torch.tensor([[0, 1]], device=DEV, dtype=torch.int32)
    seq_lens = torch.tensor([q_len], device=DEV, dtype=torch.int32)
    cu_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)

    def run(op, segm_dtype, delayed, descale):
        segm_output = torch.zeros((rows, nq, num_segments, head), device=DEV,
                                  dtype=segm_dtype)
        segm_max = torch.full((rows, nq, num_segments), float("-inf"),
                              device=DEV, dtype=torch.float32)
        segm_expsum = torch.zeros((rows, nq, num_segments), device=DEV,
                                  dtype=torch.float32)
        kd = vd = None
        if descale:
            kd = torch.full((1, nkv), 0.5, device=DEV, dtype=torch.float32)
            vd = torch.full((1, nkv), 2.0, device=DEV, dtype=torch.float32)
        op(segm_output, segm_max, segm_expsum, query, key, value, block_table,
           seq_lens, cu_q, scale, num_segments, delayed, 1, kd, vd)
        return segm_output, segm_max, segm_expsum

    for segm_dtype in (torch.float16, torch.float32):
        for delayed in (False, True):
            for descale in (False, True):
                a = run(REF.gfx906_attnd_decode_inner, segm_dtype, delayed,
                        descale)
                b = run(NEW.gfx906_attnd_decode_inner, segm_dtype, delayed,
                        descale)
                for name, xa, xb in zip(("segm_output", "segm_max", "segm_expsum"),
                                        a, b):
                    compare(
                        f"attnd_inner {segm_dtype} delayed={delayed} "
                        f"descale={descale} {name}", xa, xb)


@case
def attnd_decode_inner_kvint8_case():
    if not hasattr(REF, "gfx906_attnd_decode_inner_kvint8"):
        SKIPPED.append("attnd kvint8: reference extension does not register it")
        return
    torch.manual_seed(19)
    q_len, nkv, block_size, head, num_blocks = 32, 2, 16, 256, 8
    nq = nkv * 6
    num_segments = 2
    row_bytes = head + 4  # 256 B of int8 codes + inline fp32 scale
    scale = head ** -0.5

    query = _rand_half((q_len, nq, head), torch.float16, 0.5)
    key = torch.randint(-127, 127, (num_blocks, nkv, block_size, row_bytes),
                        device=DEV, dtype=torch.int8)
    value = torch.randint(-127, 127, (num_blocks, nkv, block_size, row_bytes),
                          device=DEV, dtype=torch.int8)
    k_scale = torch.rand((num_blocks, block_size, nkv), device=DEV,
                         dtype=torch.float32) * 0.01 + 1e-4
    v_scale = torch.rand((num_blocks, block_size, nkv), device=DEV,
                         dtype=torch.float32) * 0.01 + 1e-4
    block_table = torch.tensor([[0, 1]], device=DEV, dtype=torch.int32)
    seq_lens = torch.tensor([q_len], device=DEV, dtype=torch.int32)
    cu_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)

    def run(op, segm_dtype, delayed):
        segm_output = torch.zeros((q_len, nq, num_segments, head), device=DEV,
                                  dtype=segm_dtype)
        segm_max = torch.full((q_len, nq, num_segments), float("-inf"),
                              device=DEV, dtype=torch.float32)
        segm_expsum = torch.zeros((q_len, nq, num_segments), device=DEV,
                                  dtype=torch.float32)
        op(segm_output, segm_max, segm_expsum, query, key, value, k_scale,
           v_scale, block_table, seq_lens, cu_q, scale, num_segments, delayed,
           1, 0)
        return segm_output, segm_max, segm_expsum

    for segm_dtype in (torch.float16, torch.float32):
        for delayed in (False, True):
            a = run(REF.gfx906_attnd_decode_inner_kvint8, segm_dtype, delayed)
            b = run(NEW.gfx906_attnd_decode_inner_kvint8, segm_dtype, delayed)
            for name, xa, xb in zip(("segm_output", "segm_max", "segm_expsum"),
                                    a, b):
                compare(
                    f"attnd_kvint8 {segm_dtype} delayed={delayed} {name}", xa, xb)


@case
def gdn_packed_decode_case():
    if REF_C is None or not hasattr(
            REF_C, "fused_recurrent_gated_delta_rule_gfx906_packed_decode"):
        SKIPPED.append("GDN packed decode: reference does not register it")
        return
    torch.manual_seed(23)
    tokens, kv_heads, key_dim, value_dim, slots = 4, 2, 128, 128, 8
    qk_dim = 2 * kv_heads * key_dim
    qkv_dim = qk_dim + kv_heads * value_dim
    state_indices = torch.tensor([0, 1, 2, 3], device=DEV, dtype=torch.int32)

    for dtype in (torch.float16, torch.bfloat16):
        for state_dtype in (dtype, torch.float32):
            mixed = _rand_half((tokens, qkv_dim), dtype, 0.5)
            a = _rand_half((tokens, kv_heads), dtype, 0.5)
            b = _rand_half((tokens, kv_heads), dtype, 0.5)
            A_log = _rand_half((kv_heads,), dtype, 0.2)
            dt_bias = _rand_half((kv_heads,), dtype, 0.2)
            state = _rand_half((slots, kv_heads, value_dim, key_dim),
                               state_dtype, 0.2)
            for l2norm in (False, True):
                for tiled in (False, True):
                    for transposed in (False,):
                        def run(op, st, o):
                            return op(mixed, a, b, A_log, dt_bias, st, o,
                                      state_indices, key_dim ** -0.5, l2norm,
                                      tiled, transposed), st, o

                        ref_out, ref_state, _ = run(
                            REF_C.fused_recurrent_gated_delta_rule_gfx906_packed_decode,
                            state.clone(),
                            torch.empty((tokens, 1, kv_heads, value_dim),
                                        device=DEV, dtype=dtype))
                        new_out, new_state, _ = run(
                            NEW.fused_recurrent_gated_delta_rule_gfx906_packed_decode,
                            state.clone(),
                            torch.empty((tokens, 1, kv_heads, value_dim),
                                        device=DEV, dtype=dtype))
                        compare(
                            f"gdn_packed {dtype}/{state_dtype} l2norm={l2norm} "
                            f"tiled={tiled} out", ref_out, new_out)
                        compare(
                            f"gdn_packed {dtype}/{state_dtype} "
                            f"l2norm={l2norm} tiled={tiled} state",
                            ref_state, new_state)


@case
def paged_attention_case():
    # csrc/rocm/attention.hip compiles the selected kernel body to
    # UNREACHABLE_CODE on gfx906 (it is the gfx9 MFMA variant), so running it
    # trips a device-side assert in BOTH implementations.  The gfx906 decode
    # path is `gfx906_attnd_decode_inner*`, not this legacy op.  Opt in with
    # GX906_TEST_PAGED=1 only when you know the dispatch selects a usable body.
    if not os.environ.get("GX906_TEST_PAGED"):
        SKIPPED.append(
            "paged attention: upstream kernel asserts on gfx906 (MFMA-only body); "
            "set GX906_TEST_PAGED=1 to run anyway"
        )
        return
    if not hasattr(REF, "paged_attention"):
        SKIPPED.append("paged attention: reference extension does not register it")
        return
    torch.manual_seed(7)
    for head_size, block_size in ((128, 16), (64, 32)):
        for dtype in (torch.float16, torch.bfloat16):
            num_seqs, num_heads, num_kv_heads = 3, 8, 8
            max_seq_len = 96
            seq_lens_host = [96, 33, 1]
            blocks_per_seq = (max_seq_len + block_size - 1) // block_size
            num_blocks = num_seqs * blocks_per_seq + 3
            x = 16 // dtype.itemsize

            query = _rand_half((num_seqs, num_heads, head_size), dtype, 0.1)
            key_cache = _rand_half(
                (num_blocks, num_kv_heads, head_size // x, block_size, x),
                dtype, 0.1)
            value_cache = _rand_half(
                (num_blocks, num_kv_heads, head_size, block_size), dtype, 0.1)
            table = torch.randperm(num_blocks, device=DEV)[
                : num_seqs * blocks_per_seq
            ].reshape(num_seqs, blocks_per_seq).to(torch.int32).contiguous()
            seq_lens = torch.tensor(seq_lens_host, device=DEV, dtype=torch.int32)

            max_partitions = (max_seq_len + 511) // 512
            scale = head_size ** -0.5

            def run(op):
                out = torch.empty((num_seqs, num_heads, head_size), device=DEV,
                                  dtype=dtype)
                exp_sums = torch.empty((num_seqs, num_heads, max_partitions),
                                       device=DEV, dtype=torch.float32)
                max_logits = torch.empty_like(exp_sums)
                tmp_out = torch.empty(
                    (num_seqs, num_heads, max_partitions, head_size), device=DEV,
                    dtype=dtype)
                op(out, exp_sums, max_logits, tmp_out, query, key_cache,
                   value_cache, num_kv_heads, scale, table, seq_lens, None,
                   block_size, max_seq_len, None, "auto",
                   torch.ones(1, device=DEV), torch.ones(1, device=DEV), None,
                   "f16", False)
                return out, exp_sums, max_logits, tmp_out

            a = run(REF.paged_attention)
            b = run(NEW.paged_attention)
            for name, xa, xb in zip(("out", "exp_sums", "max_logits", "tmp_out"),
                                    a, b):
                compare(
                    f"paged_attention {dtype} hs={head_size} bs={block_size} {name}",
                    xa, xb)


@case
def conv1d_decode_case():
    if REF_C is None or not hasattr(REF_C, "causal_conv1d_gfx906_decode_update"):
        SKIPPED.append("conv1d decode: reference extension does not register it")
        return
    torch.manual_seed(11)
    for dtype in (torch.float16, torch.bfloat16):
        for state_dtype in (dtype, torch.float32):
            batch, dim, width, state_len, cache = 4, 256, 4, 3, 8
            x = _rand_half((batch, dim), dtype, 0.5)
            weight = _rand_half((dim, width), dtype, 0.5)
            bias = _rand_half((dim,), dtype, 0.1)
            state = _rand_half((cache, dim, state_len), state_dtype, 0.5)
            indices = torch.tensor([2, 0, 7, 1], device=DEV, dtype=torch.int32)
            for use_bias in (False, True):
                for use_idx in (False, True):
                    for silu in (False, True):
                        def run(op, state_copy):
                            return op(x, state_copy, weight,
                                      bias if use_bias else None,
                                      indices if use_idx else None, -1, silu)

                        a = run(REF_C.causal_conv1d_gfx906_decode_update,
                                state.clone())
                        b = run(NEW.causal_conv1d_gfx906_decode_update,
                                state.clone())
                        compare(
                            f"conv1d_decode {dtype}/{state_dtype} "
                            f"bias={use_bias} idx={use_idx} silu={silu}", a, b)


def main() -> int:
    if not torch.cuda.is_available():
        print("no GPU visible; this gate must run on the Z100L host", file=sys.stderr)
        return 2
    print(f"gfx906_kernels build: {gfx906_kernels.build_id()}")
    print(f"reference: torch.ops._rocm_C")
    print()
    for note in SKIPPED:
        print(f"skipped: {note}")
    for note in KNOWN:
        print(f"known deviation: {note}")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print("\nall cases bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
