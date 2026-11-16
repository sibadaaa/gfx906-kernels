// SPDX-License-Identifier: Apache-2.0
//
// gfx906 compute kernels -- stable C ABI.
//
// Design goals
// ------------
//   * no torch, no libstdc++ types in the ABI: an inference engine in C, C++,
//     Rust (bindgen) or Python (ctypes/cffi) can call this directly;
//   * every call takes an explicit HIP stream and device, so the library never
//     guesses about the caller's scheduling;
//   * tensors are plain descriptors (pointer + sizes + strides + dtype +
//     device).  The library never copies a weight, so callers keep their
//     checkpoint layout;
//   * outputs are allocated by the library through a caller-installable
//     allocator (default: hipMalloc).  A torch host installs the caching
//     allocator and gets torch-owned buffers back.
//
// Ownership: a `gx_tensor` filled by an op owns a reference until
// gx_tensor_release() is called.  `gx_tensor` values the caller builds by hand
// (see gx_tensor_view) are non-owning and must not be released.
#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define GX906_ABI_VERSION 1

typedef enum {
  GX_OK = 0,
  GX_ERR_INVALID_ARGUMENT = 1,
  GX_ERR_INVALID_SHAPE = 2,
  GX_ERR_UNSUPPORTED = 3,
  GX_ERR_HIP = 4,
  GX_ERR_OUT_OF_MEMORY = 5,
  GX_ERR_INTERNAL = 6,
} gx_status;

// Numeric values match torch's ScalarType encoding on purpose: a torch host
// can pass static_cast<int32_t>(tensor.scalar_type()) without a mapping.
typedef enum {
  GX_DTYPE_U8 = 0,
  GX_DTYPE_I8 = 1,
  GX_DTYPE_I16 = 2,
  GX_DTYPE_I32 = 3,
  GX_DTYPE_I64 = 4,
  GX_DTYPE_F16 = 5,
  GX_DTYPE_F32 = 6,
  GX_DTYPE_F64 = 7,
  GX_DTYPE_BF16 = 15,
  GX_DTYPE_F8E5M2 = 23,
  GX_DTYPE_F8E4M3 = 24,
  GX_DTYPE_F8E5M2FNUZ = 25,
  GX_DTYPE_F8E4M3FNUZ = 26,
} gx_dtype;

#define GX906_MAX_DIMS 8

typedef struct gx_tensor {
  void* data;
  int64_t sizes[GX906_MAX_DIMS];
  int64_t strides[GX906_MAX_DIMS];  // in elements, like torch
  int32_t ndim;
  int32_t dtype;
  int32_t device;
  int32_t reserved0;
  void* owner;      // library-private for outputs; NULL for views
  void* reserved1;
} gx_tensor;

typedef void* (*gx_alloc_fn)(size_t bytes, void* ctx);
typedef void (*gx_free_fn)(void* ptr, void* ctx);

// ---------------------------------------------------------------------------
// library / runtime
// ---------------------------------------------------------------------------

// Returns GX906_ABI_VERSION.
int32_t gx906_abi_version(void);

// Last error message for the calling thread; never NULL, never to be freed.
const char* gx906_last_error(void);

// HIP stream (hipStream_t).  NULL selects the default stream.
void gx906_set_stream(void* hip_stream);
void* gx906_get_stream(void);

// The kernels always run on the device recorded in the tensor descriptors.
// gx906_set_device pins the device used for allocations and validation.
void gx906_set_device(int32_t device);
int32_t gx906_get_device(void);

// Install a host allocator for op outputs.  Call before the first op.
void gx906_set_allocator(gx_alloc_fn alloc, gx_free_fn free_fn, void* ctx);

// Human-readable build identity (git describe / version).
const char* gx906_build_id(void);

// ---------------------------------------------------------------------------
// tensor helpers
// ---------------------------------------------------------------------------

// Build a non-owning view.  `strides` may be NULL for a contiguous layout.
void gx906_tensor_view(gx_tensor* t, void* data, const int64_t* sizes,
                       const int64_t* strides, int32_t ndim, int32_t dtype,
                       int32_t device);

// Drop the library reference held by an op output.  Safe on views (no-op).
void gx906_tensor_release(gx_tensor* t);

// ---------------------------------------------------------------------------
// dense fp16 / bf16 decode GEMV (M <= 8)
//   x [M,K] f16/bf16, w [N,K] same dtype, out [M,N]
// route: 0 auto, 1 fp32 accumulate, 2 v_dot2_f32_f16
// ---------------------------------------------------------------------------
gx_status gx906_bf16_emu_gemv(const gx_tensor* x, const gx_tensor* w,
                              const gx_tensor* bias, int64_t nthreads,
                              int64_t route, int32_t range_safe,
                              gx_tensor* out);

// lm_head-shaped streaming fp16 GEMV, M == 1.
gx_status gx906_fp16_gemv_stream_m1(const gx_tensor* x, const gx_tensor* w,
                                    const gx_tensor* bias, int64_t nthreads,
                                    gx_tensor* out);

// ---------------------------------------------------------------------------
// W8A8 int8 GEMM/GEMV on v_dot4_i32_i8
//   xq [M,K] i8, w [N,K] i8, x_s [M]|[] , w_s [N]|[]
// ---------------------------------------------------------------------------
gx_status gx906_w8a8_sdot4_gemv(const gx_tensor* xq, const gx_tensor* w,
                                const gx_tensor* x_s, const gx_tensor* w_s,
                                const gx_tensor* bias, int32_t out_dtype,
                                int64_t nthreads, gx_tensor* out);

gx_status gx906_w8a8_sdot4_gemm(const gx_tensor* xq, const gx_tensor* w,
                                const gx_tensor* x_s, const gx_tensor* w_s,
                                const gx_tensor* bias, int32_t out_dtype,
                                int64_t cfg, gx_tensor* out);

// Per-channel int8 prefill GEMM (S99), fp8 weights staged inside the op.
gx_status gx906_w8a8_sdot4_gemm_pc(const gx_tensor* xq, const gx_tensor* w,
                                   const gx_tensor* x_s, const gx_tensor* w_s,
                                   const gx_tensor* bias, int32_t out_dtype,
                                   int64_t cfg, gx_tensor* out);

// ---------------------------------------------------------------------------
// W8A8 FP8 (e4m3) emulated on the fp16 datapath with a 256-entry LDS LUT.
//   a [M,K] f8e4m3, w [N,K] f8e4m3, a_s/w_s fp32 per-tensor or per-row
//   lut: 256-entry fp16 table (see gx906_fp8_lut_fill)
// ---------------------------------------------------------------------------
gx_status gx906_fp8_emu_gemv(const gx_tensor* a, const gx_tensor* w,
                             const gx_tensor* a_s, const gx_tensor* w_s,
                             const gx_tensor* lut, const gx_tensor* bias,
                             int32_t out_dtype, int64_t nthreads,
                             gx_tensor* out);

gx_status gx906_fp8_emu_gemm(const gx_tensor* a, const gx_tensor* w,
                             const gx_tensor* a_s, const gx_tensor* w_s,
                             const gx_tensor* lut, const gx_tensor* bias,
                             int32_t out_dtype, int64_t cfg, gx_tensor* out);

// Block-scaled (DeepSeek-style) FP8: a_s [M, cdiv(K,block_k)],
// w_s [cdiv(N,block_n), cdiv(K,block_k)].
gx_status gx906_fp8_emu_gemv_block(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* a_s,
    const gx_tensor* w_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t nthreads, int64_t block_n, int64_t block_k,
    gx_tensor* out);

gx_status gx906_fp8_emu_gemm_block(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* a_s,
    const gx_tensor* w_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t cfg, int64_t block_n, int64_t block_k,
    gx_tensor* out);

// W8A16: activation stays fp16/bf16, weight stays fp8 + block scales.
gx_status gx906_fp8_emu_gemm_block_w8a16(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    gx_tensor* out);

// W8A16 M==1 down-projection with SiLU(gate)*up folded into staging.
gx_status gx906_fp8_emu_gemm_block_w8a16_smul(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t block_n, int64_t block_k, gx_tensor* out);

// Materialise one layer's fp8 weight * block scale into `out` ([N,K] fp16).
gx_status gx906_fp8_dequant_to_fp16_out(gx_tensor* out, const gx_tensor* wq,
                                        const gx_tensor* ws,
                                        const gx_tensor* lut, int64_t block_n,
                                        int64_t block_k);

// Activation quantization fused into the block-scaled path (S23 / O5).
gx_status gx906_fp8_emu_gemm_block_quant(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    float eps, gx_tensor* out);

// Same, plus the (x_q, x_s) witness used by the bit-exact gate.
gx_status gx906_fp8_emu_gemm_block_quant_x(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    float eps, gx_tensor* out, gx_tensor* x_q, gx_tensor* x_s);

// Stage an fp8 block-scaled checkpoint weight to per-channel int8 + scale.
gx_status gx906_fp8_stage_int8_pc(const gx_tensor* w, const gx_tensor* w_s,
                                  const gx_tensor* lut, int64_t block_n,
                                  int64_t block_k, gx_tensor* q,
                                  gx_tensor* scale);

// S99 prefill GEMM: fp8 weights staged to int8 inside the op.
gx_status gx906_fp8_emu_mm_int8_pc(
    const gx_tensor* xq, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* x_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t block_n, int64_t block_k, gx_tensor* out);

// Fill `lut` (256 fp16 values) with the fp8-e4m3 -> fp16 widening table.
gx_status gx906_fp8_lut_fill(gx_tensor* lut);

// ---------------------------------------------------------------------------
// fused RMSNorm family
// ---------------------------------------------------------------------------
gx_status gx906_rms_norm_gated(const gx_tensor* input, const gx_tensor* weight,
                               const gx_tensor* gate, float epsilon,
                               int32_t norm_before_gate, gx_tensor* out);

gx_status gx906_gemma_rms_norm(const gx_tensor* input, const gx_tensor* weight,
                               float epsilon, gx_tensor* out);

gx_status gx906_gemma_fused_add_rms_norm(const gx_tensor* input,
                                         const gx_tensor* residual,
                                         const gx_tensor* weight,
                                         float epsilon, gx_tensor* out,
                                         gx_tensor* residual_out);

// ---------------------------------------------------------------------------
// paged attention (legacy backend: [num_seqs, num_heads, head_size] query)
// ---------------------------------------------------------------------------
gx_status gx906_paged_attention(
    gx_tensor* out, gx_tensor* exp_sums, gx_tensor* max_logits,
    gx_tensor* tmp_out, const gx_tensor* query, const gx_tensor* key_cache,
    const gx_tensor* value_cache, int64_t num_kv_heads, double scale,
    const gx_tensor* block_tables, const gx_tensor* seq_lens,
    const gx_tensor* query_start_loc, int64_t block_size, int64_t max_seq_len,
    const gx_tensor* alibi_slopes, const char* kv_cache_dtype,
    const gx_tensor* k_scale, const gx_tensor* v_scale,
    const gx_tensor* fp8_out_scale, const char* mfma_type,
    int32_t use_interleaved_v_cache);

// ---------------------------------------------------------------------------
// gfx906 decode-attention inner loop (S53 fp16 / S81 int8 KV)
// segment buffers are shared with the caller's split-KV reducer.
// ---------------------------------------------------------------------------
gx_status gx906_attnd_decode_inner(
    gx_tensor* segm_output, gx_tensor* segm_max, gx_tensor* segm_expsum,
    const gx_tensor* query, const gx_tensor* key_cache,
    const gx_tensor* value_cache, const gx_tensor* block_tables,
    const gx_tensor* seq_lens, const gx_tensor* cu_query_lens, double scale,
    int64_t num_segments, int32_t delayed_merge, int64_t tps_hint,
    const gx_tensor* k_descale, const gx_tensor* v_descale);

gx_status gx906_attnd_decode_inner_kvint8(
    gx_tensor* segm_output, gx_tensor* segm_max, gx_tensor* segm_expsum,
    const gx_tensor* query, const gx_tensor* key_cache,
    const gx_tensor* value_cache, const gx_tensor* k_scale_cache,
    const gx_tensor* v_scale_cache, const gx_tensor* block_tables,
    const gx_tensor* seq_lens, const gx_tensor* cu_query_lens, double scale,
    int64_t num_segments, int32_t delayed_merge, int64_t tps_hint,
    int64_t seg_stride);

// ---------------------------------------------------------------------------
// Mamba / GatedDeltaNet decode kernels (S44/S46 production path)
// ---------------------------------------------------------------------------
gx_status gx906_causal_conv1d_decode_update(
    const gx_tensor* x, gx_tensor* conv_state, const gx_tensor* weight,
    const gx_tensor* bias, const gx_tensor* conv_state_indices,
    int64_t pad_slot_id, int32_t silu_activation, gx_tensor* out);

gx_status gx906_causal_conv1d_mtp_update(
    const gx_tensor* x, gx_tensor* conv_state, const gx_tensor* weight,
    const gx_tensor* bias, const gx_tensor* state_indices,
    const gx_tensor* cu_seqlens, const gx_tensor* num_accepted_tokens,
    int64_t pad_slot_id, int32_t silu_activation, gx_tensor* out);

gx_status gx906_fused_sigmoid_gating_delta_rule_decode(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, double beta, double threshold,
    double scale, int32_t use_qk_l2norm_in_kernel, gx_tensor* out);

gx_status gx906_fused_sigmoid_gating_delta_rule_prefill(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* cu_seqlens,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out);

gx_status gx906_fused_sigmoid_gating_delta_rule_indexed_decode(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out);

gx_status gx906_fused_sigmoid_gating_delta_rule_indexed_decode_kv_state(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out);

gx_status gx906_fused_sigmoid_gating_delta_rule_mtp_update(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    const gx_tensor* cu_seqlens, const gx_tensor* num_accepted_tokens,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out);

gx_status gx906_fused_recurrent_gated_delta_rule_packed_decode(
    const gx_tensor* mixed_qkv, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* A_log, const gx_tensor* dt_bias, gx_tensor* state,
    gx_tensor* out, const gx_tensor* state_indices, double scale,
    int32_t use_qk_l2norm_in_kernel, int32_t use_tiled_qk_head_mapping,
    int32_t use_transposed_state);

gx_status gx906_causal_conv1d_recurrent_gated_delta_rule_packed_decode(
    const gx_tensor* mixed_qkv, gx_tensor* conv_state,
    const gx_tensor* conv_weight, const gx_tensor* conv_bias,
    const gx_tensor* a, const gx_tensor* b, const gx_tensor* A_log,
    const gx_tensor* dt_bias, gx_tensor* state, gx_tensor* out,
    const gx_tensor* state_indices, int64_t pad_slot_id, double scale,
    int32_t silu_activation, int32_t use_qk_l2norm_in_kernel,
    int32_t use_tiled_qk_head_mapping, int32_t use_transposed_state);

gx_status gx906_causal_conv1d_recurrent_gated_delta_rule_ratio2_packed_decode(
    const gx_tensor* mixed_qkv, gx_tensor* conv_state,
    const gx_tensor* conv_weight, const gx_tensor* conv_bias,
    const gx_tensor* a, const gx_tensor* b, const gx_tensor* A_log,
    const gx_tensor* dt_bias, gx_tensor* state, gx_tensor* out,
    const gx_tensor* state_indices, int64_t pad_slot_id, double scale,
    int32_t silu_activation, int32_t use_qk_l2norm_in_kernel,
    int32_t use_tiled_qk_head_mapping, int32_t use_transposed_state);

#if defined(__cplusplus)
}  // extern "C"
#endif
