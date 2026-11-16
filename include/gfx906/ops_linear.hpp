// SPDX-License-Identifier: Apache-2.0
//
// Linear-layer kernels: dense GEMV, W8A8 int8 (v_dot4), W8A8/W8A16 FP8 emu.
#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "detail/aten.hpp"

namespace gfx906 {

using gx::Tensor;

Tensor bf16_emu_gemv(Tensor x, Tensor w, std::optional<Tensor> bias,
                     int64_t nthreads, int64_t route, bool range_safe);

Tensor bf16_emu_gemv_meta(Tensor x, Tensor w, std::optional<Tensor> bias,
                          int64_t nthreads, int64_t route, bool range_safe);

Tensor fp16_gemv_stream_m1(Tensor x, Tensor w, std::optional<Tensor> bias,
                           int64_t nthreads);

Tensor fp16_gemv_stream_m1_meta(Tensor x, Tensor w, std::optional<Tensor> bias,
                                int64_t nthreads);

Tensor w8a8_sdot4_gemv(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                       std::optional<Tensor> bias, gx::ScalarType out_dtype,
                       int64_t nthreads);

Tensor w8a8_sdot4_gemm(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                       std::optional<Tensor> bias, gx::ScalarType out_dtype,
                       int64_t cfg);

Tensor w8a8_sdot4_gemv_meta(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                            std::optional<Tensor> bias,
                            gx::ScalarType out_dtype, int64_t nthreads);

Tensor w8a8_sdot4_gemm_meta(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                            std::optional<Tensor> bias,
                            gx::ScalarType out_dtype, int64_t cfg);

Tensor w8a8_sdot4_gemm_pc(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                          std::optional<Tensor> bias, gx::ScalarType out_dtype,
                          int64_t cfg);

Tensor w8a8_sdot4_gemm_pc_meta(Tensor xq, Tensor w, Tensor x_s, Tensor w_s,
                               std::optional<Tensor> bias,
                               gx::ScalarType out_dtype, int64_t cfg);

std::tuple<Tensor, Tensor> fp8_stage_int8_pc(Tensor w, Tensor w_s, Tensor lut,
                                             int64_t block_n, int64_t block_k);

std::tuple<Tensor, Tensor> fp8_stage_int8_pc_meta(Tensor w, Tensor w_s,
                                                  Tensor lut, int64_t block_n,
                                                  int64_t block_k);

Tensor fp8_emu_mm_int8_pc(Tensor xq, Tensor w, Tensor w_s, Tensor x_s,
                          Tensor lut, std::optional<Tensor> bias,
                          gx::ScalarType out_dtype, int64_t block_n,
                          int64_t block_k);

Tensor fp8_emu_mm_int8_pc_meta(Tensor xq, Tensor w, Tensor w_s, Tensor x_s,
                               Tensor lut, std::optional<Tensor> bias,
                               gx::ScalarType out_dtype, int64_t block_n,
                               int64_t block_k);

// ---------------------------------------------------------------------------
// FP8 emulation on the fp16 datapath (src/kernels/fp8_emu_gemm.hip).
//
// fp8 weights/scales stay as laid out by the caller; results are
// `(acc * a_s) * w_s`, then cast to `out_dtype`, then bias in `out_dtype`.
// `fp16_gemv_stream_m1` / `fp16_gemv_stream_m1_meta` are declared above (same
// translation unit).
// ---------------------------------------------------------------------------

Tensor fp8_emu_gemv(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut,
                    std::optional<Tensor> bias, gx::ScalarType out_dtype,
                    int64_t nthreads);

Tensor fp8_emu_gemm(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut,
                    std::optional<Tensor> bias, gx::ScalarType out_dtype,
                    int64_t cfg);

Tensor fp8_emu_gemv_meta(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                         Tensor lut, std::optional<Tensor> bias,
                         gx::ScalarType out_dtype, int64_t nthreads);

Tensor fp8_emu_gemm_meta(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                         Tensor lut, std::optional<Tensor> bias,
                         gx::ScalarType out_dtype, int64_t cfg);

Tensor fp8_emu_gemv_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                          Tensor lut, std::optional<Tensor> bias,
                          gx::ScalarType out_dtype, int64_t nthreads,
                          int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemm_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                          Tensor lut, std::optional<Tensor> bias,
                          gx::ScalarType out_dtype, int64_t cfg,
                          int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemv_block_meta(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                               Tensor lut, std::optional<Tensor> bias,
                               gx::ScalarType out_dtype, int64_t nthreads,
                               int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemm_block_meta(Tensor a, Tensor w, Tensor a_s, Tensor w_s,
                               Tensor lut, std::optional<Tensor> bias,
                               gx::ScalarType out_dtype, int64_t cfg,
                               int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemm_block_quant(Tensor a, Tensor w, Tensor w_s, Tensor lut,
                                std::optional<Tensor> bias,
                                gx::ScalarType out_dtype, int64_t nthreads,
                                int64_t cfg, int64_t block_n, int64_t block_k,
                                double eps);

std::tuple<Tensor, Tensor, Tensor> fp8_emu_gemm_block_quant_x(
    Tensor a, Tensor w, Tensor w_s, Tensor lut, std::optional<Tensor> bias,
    gx::ScalarType out_dtype, int64_t nthreads, int64_t cfg, int64_t block_n,
    int64_t block_k, double eps);

Tensor fp8_emu_gemm_block_quant_meta(Tensor a, Tensor w, Tensor w_s,
                                     Tensor lut, std::optional<Tensor> bias,
                                     gx::ScalarType out_dtype, int64_t nthreads,
                                     int64_t cfg, int64_t block_n,
                                     int64_t block_k, double eps);

std::tuple<Tensor, Tensor, Tensor> fp8_emu_gemm_block_quant_x_meta(
    Tensor a, Tensor w, Tensor w_s, Tensor lut, std::optional<Tensor> bias,
    gx::ScalarType out_dtype, int64_t nthreads, int64_t cfg, int64_t block_n,
    int64_t block_k, double eps);

Tensor fp8_emu_gemm_block_w8a16(Tensor a, Tensor w, Tensor w_s, Tensor lut,
                                std::optional<Tensor> bias,
                                gx::ScalarType out_dtype, int64_t nthreads,
                                int64_t cfg, int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemm_block_w8a16_meta(Tensor a, Tensor w, Tensor w_s,
                                     Tensor lut, std::optional<Tensor> bias,
                                     gx::ScalarType out_dtype, int64_t nthreads,
                                     int64_t cfg, int64_t block_n,
                                     int64_t block_k);

Tensor fp8_emu_gemm_block_w8a16_smul(Tensor a, Tensor w, Tensor w_s,
                                     Tensor lut, std::optional<Tensor> bias,
                                     gx::ScalarType out_dtype, int64_t nthreads,
                                     int64_t block_n, int64_t block_k);

Tensor fp8_emu_gemm_block_w8a16_smul_meta(Tensor a, Tensor w, Tensor w_s,
                                          Tensor lut,
                                          std::optional<Tensor> bias,
                                          gx::ScalarType out_dtype,
                                          int64_t nthreads, int64_t block_n,
                                          int64_t block_k);

void fp8_dequant_to_fp16_out(Tensor out, Tensor wq, Tensor ws, Tensor lut,
                             int64_t block_n, int64_t block_k);

}  // namespace gfx906
