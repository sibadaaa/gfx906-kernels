// SPDX-License-Identifier: Apache-2.0
//
// C ABI implementation: descriptor marshalling + error/ownership handling.
//
// The math lives in the kernel translation units; this file only converts
// gx_tensor descriptors to gx::Tensor views, calls the C++ entry point, and
// hands ownership of the (library-allocated) output back to the caller.
#include "gfx906/gfx906.h"

#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "gfx906/detail/aten.hpp"
#include "gfx906/ops.hpp"

namespace {

using gx::Tensor;

thread_local std::string t_last_error;

gx_status fail(gx_status code, const std::string& msg) {
  t_last_error = msg;
  return code;
}

void clear_error() { t_last_error.clear(); }

Tensor view_of(const gx_tensor* t, const char* name) {
  if (t == nullptr) {
    throw gx::Error(std::string(name) + ": null tensor");
  }
  if (t->ndim < 0 || t->ndim > GX906_MAX_DIMS) {
    throw gx::Error(std::string(name) + ": bad ndim");
  }
  std::vector<int64_t> sizes(t->sizes, t->sizes + t->ndim);
  std::vector<int64_t> strides(t->strides, t->strides + t->ndim);
  int64_t numel = 1;
  for (int64_t s : sizes) numel *= s;
  if (t->data == nullptr && numel != 0) {
    throw gx::Error(std::string(name) + ": null data pointer");
  }
  return Tensor::view(t->data, std::move(sizes), std::move(strides),
                      static_cast<gx::ScalarType>(t->dtype), t->device);
}

std::optional<Tensor> opt_view(const gx_tensor* t, const char* name) {
  if (t == nullptr) return std::nullopt;
  return view_of(t, name);
}

void set_out(gx_tensor* dst, Tensor&& value) {
  if (dst == nullptr) {
    throw gx::Error("null output descriptor");
  }
  Tensor* owner = new Tensor(std::move(value));
  std::memset(dst, 0, sizeof(*dst));
  dst->data = owner->data_ptr();
  dst->ndim = static_cast<int32_t>(owner->dim());
  for (int32_t i = 0; i < dst->ndim; ++i) {
    dst->sizes[i] = owner->size(i);
    dst->strides[i] = owner->stride(i);
  }
  dst->dtype = static_cast<int32_t>(owner->scalar_type());
  dst->device = static_cast<int32_t>(owner->device().index());
  dst->owner = owner;
}

#define GX906_TRY_BEGIN() \
  try {                   \
    clear_error();

#define GX906_TRY_END()                                     \
  }                                                         \
  catch (const std::bad_alloc&) {                           \
    return fail(GX_ERR_OUT_OF_MEMORY, "out of device memory"); \
  }                                                         \
  catch (const gx::Error& e) {                              \
    return fail(GX_ERR_INVALID_ARGUMENT, e.what());         \
  }                                                         \
  catch (const std::exception& e) {                         \
    return fail(GX_ERR_INTERNAL, e.what());                 \
  }                                                         \
  return GX_OK;

}  // namespace

extern "C" {

int32_t gx906_abi_version(void) { return GX906_ABI_VERSION; }

const char* gx906_last_error(void) { return t_last_error.c_str(); }

void gx906_set_stream(void* hip_stream) {
  gx::stream::set(reinterpret_cast<hipStream_t>(hip_stream));
}

void* gx906_get_stream(void) { return reinterpret_cast<void*>(gx::stream::current()); }

void gx906_set_device(int32_t device) { gx::set_current_device(device); }

int32_t gx906_get_device(void) { return gx::current_device(); }

void gx906_set_allocator(gx_alloc_fn alloc, gx_free_fn free_fn, void* ctx) {
  gx::set_allocator(alloc, free_fn, ctx);
}

const char* gx906_build_id(void) {
#ifdef GX906_BUILD_ID
  return GX906_BUILD_ID;
#else
  return "gfx906-kernels dev";
#endif
}

void gx906_tensor_view(gx_tensor* t, void* data, const int64_t* sizes,
                       const int64_t* strides, int32_t ndim, int32_t dtype,
                       int32_t device) {
  std::memset(t, 0, sizeof(*t));
  t->data = data;
  t->ndim = ndim;
  t->dtype = dtype;
  t->device = device;
  int64_t stride = 1;
  for (int32_t i = ndim - 1; i >= 0; --i) {
    t->sizes[i] = sizes[i];
    if (strides != nullptr) {
      t->strides[i] = strides[i];
    } else {
      t->strides[i] = stride;
      stride *= sizes[i];
    }
  }
}

void gx906_tensor_release(gx_tensor* t) {
  if (t == nullptr) return;
  if (t->owner != nullptr) {
    delete static_cast<Tensor*>(t->owner);
  }
  std::memset(t, 0, sizeof(*t));
}

// ---------------------------------------------------------------------------
// linear
// ---------------------------------------------------------------------------

gx_status gx906_bf16_emu_gemv(const gx_tensor* x, const gx_tensor* w,
                              const gx_tensor* bias, int64_t nthreads,
                              int64_t route, int32_t range_safe,
                              gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::bf16_emu_gemv(view_of(x, "x"), view_of(w, "w"),
                                     opt_view(bias, "bias"), nthreads, route,
                                     range_safe != 0));
  GX906_TRY_END();
}

gx_status gx906_w8a8_sdot4_gemv(const gx_tensor* xq, const gx_tensor* w,
                                const gx_tensor* x_s, const gx_tensor* w_s,
                                const gx_tensor* bias, int32_t out_dtype,
                                int64_t nthreads, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::w8a8_sdot4_gemv(
                   view_of(xq, "xq"), view_of(w, "w"), view_of(x_s, "x_s"),
                   view_of(w_s, "w_s"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads));
  GX906_TRY_END();
}

gx_status gx906_w8a8_sdot4_gemm(const gx_tensor* xq, const gx_tensor* w,
                                const gx_tensor* x_s, const gx_tensor* w_s,
                                const gx_tensor* bias, int32_t out_dtype,
                                int64_t cfg, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::w8a8_sdot4_gemm(
                   view_of(xq, "xq"), view_of(w, "w"), view_of(x_s, "x_s"),
                   view_of(w_s, "w_s"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), cfg));
  GX906_TRY_END();
}

gx_status gx906_w8a8_sdot4_gemm_pc(const gx_tensor* xq, const gx_tensor* w,
                                   const gx_tensor* x_s, const gx_tensor* w_s,
                                   const gx_tensor* bias, int32_t out_dtype,
                                   int64_t cfg, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::w8a8_sdot4_gemm_pc(
                   view_of(xq, "xq"), view_of(w, "w"), view_of(x_s, "x_s"),
                   view_of(w_s, "w_s"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), cfg));
  GX906_TRY_END();
}

gx_status gx906_fp16_gemv_stream_m1(const gx_tensor* x, const gx_tensor* w,
                                    const gx_tensor* bias, int64_t nthreads,
                                    gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp16_gemv_stream_m1(view_of(x, "x"), view_of(w, "w"),
                                           opt_view(bias, "bias"), nthreads));
  GX906_TRY_END();
}

// ---------------------------------------------------------------------------
// fp8 emulation
// ---------------------------------------------------------------------------

gx_status gx906_fp8_emu_gemv(const gx_tensor* a, const gx_tensor* w,
                             const gx_tensor* a_s, const gx_tensor* w_s,
                             const gx_tensor* lut, const gx_tensor* bias,
                             int32_t out_dtype, int64_t nthreads,
                             gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemv(
                   view_of(a, "a"), view_of(w, "w"), view_of(a_s, "a_s"),
                   view_of(w_s, "w_s"), view_of(lut, "lut"),
                   opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm(const gx_tensor* a, const gx_tensor* w,
                             const gx_tensor* a_s, const gx_tensor* w_s,
                             const gx_tensor* lut, const gx_tensor* bias,
                             int32_t out_dtype, int64_t cfg, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemm(
                   view_of(a, "a"), view_of(w, "w"), view_of(a_s, "a_s"),
                   view_of(w_s, "w_s"), view_of(lut, "lut"),
                   opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), cfg));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemv_block(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* a_s,
    const gx_tensor* w_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t nthreads, int64_t block_n, int64_t block_k,
    gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemv_block(
                   view_of(a, "a"), view_of(w, "w"), view_of(a_s, "a_s"),
                   view_of(w_s, "w_s"), view_of(lut, "lut"),
                   opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads, block_n,
                   block_k));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm_block(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* a_s,
    const gx_tensor* w_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t cfg, int64_t block_n, int64_t block_k,
    gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemm_block(
                   view_of(a, "a"), view_of(w, "w"), view_of(a_s, "a_s"),
                   view_of(w_s, "w_s"), view_of(lut, "lut"),
                   opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), cfg, block_n,
                   block_k));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm_block_w8a16(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemm_block_w8a16(
                   view_of(a, "a"), view_of(w, "w"), view_of(w_s, "w_s"),
                   view_of(lut, "lut"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads, cfg,
                   block_n, block_k));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm_block_w8a16_smul(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t block_n, int64_t block_k, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemm_block_w8a16_smul(
                   view_of(a, "a"), view_of(w, "w"), view_of(w_s, "w_s"),
                   view_of(lut, "lut"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads, block_n,
                   block_k));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm_block_quant(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    float eps, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_gemm_block_quant(
                   view_of(a, "a"), view_of(w, "w"), view_of(w_s, "w_s"),
                   view_of(lut, "lut"), opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), nthreads, cfg,
                   block_n, block_k, eps));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_gemm_block_quant_x(
    const gx_tensor* a, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* lut, const gx_tensor* bias, int32_t out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k,
    float eps, gx_tensor* out, gx_tensor* x_q, gx_tensor* x_s) {
  GX906_TRY_BEGIN();
  auto res = gfx906::fp8_emu_gemm_block_quant_x(
      view_of(a, "a"), view_of(w, "w"), view_of(w_s, "w_s"),
      view_of(lut, "lut"), opt_view(bias, "bias"),
      static_cast<gx::ScalarType>(out_dtype), nthreads, cfg, block_n, block_k,
      eps);
  set_out(out, std::move(std::get<0>(res)));
  set_out(x_q, std::move(std::get<1>(res)));
  set_out(x_s, std::move(std::get<2>(res)));
  GX906_TRY_END();
}

gx_status gx906_fp8_stage_int8_pc(const gx_tensor* w, const gx_tensor* w_s,
                                  const gx_tensor* lut, int64_t block_n,
                                  int64_t block_k, gx_tensor* q,
                                  gx_tensor* scale) {
  GX906_TRY_BEGIN();
  auto res = gfx906::fp8_stage_int8_pc(view_of(w, "w"), view_of(w_s, "w_s"),
                                       view_of(lut, "lut"), block_n, block_k);
  set_out(q, std::move(std::get<0>(res)));
  set_out(scale, std::move(std::get<1>(res)));
  GX906_TRY_END();
}

gx_status gx906_fp8_emu_mm_int8_pc(
    const gx_tensor* xq, const gx_tensor* w, const gx_tensor* w_s,
    const gx_tensor* x_s, const gx_tensor* lut, const gx_tensor* bias,
    int32_t out_dtype, int64_t block_n, int64_t block_k, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fp8_emu_mm_int8_pc(
                   view_of(xq, "xq"), view_of(w, "w"), view_of(w_s, "w_s"),
                   view_of(x_s, "x_s"), view_of(lut, "lut"),
                   opt_view(bias, "bias"),
                   static_cast<gx::ScalarType>(out_dtype), block_n, block_k));
  GX906_TRY_END();
}

gx_status gx906_fp8_dequant_to_fp16_out(gx_tensor* out, const gx_tensor* wq,
                                        const gx_tensor* ws,
                                        const gx_tensor* lut, int64_t block_n,
                                        int64_t block_k) {
  GX906_TRY_BEGIN();
  gfx906::fp8_dequant_to_fp16_out(view_of(out, "out"), view_of(wq, "wq"),
                                  view_of(ws, "ws"), view_of(lut, "lut"),
                                  block_n, block_k);
  GX906_TRY_END();
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

gx_status gx906_rms_norm_gated(const gx_tensor* input, const gx_tensor* weight,
                               const gx_tensor* gate, float epsilon,
                               int32_t norm_before_gate, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::rms_norm_gated_gfx906(
                   view_of(input, "input"), view_of(weight, "weight"),
                   view_of(gate, "gate"), epsilon, norm_before_gate != 0));
  GX906_TRY_END();
}

gx_status gx906_gemma_rms_norm(const gx_tensor* input, const gx_tensor* weight,
                               float epsilon, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::gemma_rms_norm_gfx906(
                   view_of(input, "input"), view_of(weight, "weight"),
                   epsilon));
  GX906_TRY_END();
}

gx_status gx906_gemma_fused_add_rms_norm(const gx_tensor* input,
                                         const gx_tensor* residual,
                                         const gx_tensor* weight,
                                         float epsilon, gx_tensor* out,
                                         gx_tensor* residual_out) {
  GX906_TRY_BEGIN();
  auto outs = gfx906::gemma_fused_add_rms_norm_gfx906(
      view_of(input, "input"), view_of(residual, "residual"),
      view_of(weight, "weight"), epsilon);
  set_out(out, std::move(outs.at(0)));
  set_out(residual_out, std::move(outs.at(1)));
  GX906_TRY_END();
}

// ---------------------------------------------------------------------------
// fp8 widening table: byte i -> fp16 value of fp8 e4m3fn code i
// ---------------------------------------------------------------------------

gx_status gx906_fp8_lut_fill(gx_tensor* lut) {
  GX906_TRY_BEGIN();
  Tensor dst = view_of(lut, "lut");
  if (dst.numel() != 256 ||
      dst.scalar_type() != gx::ScalarType::Half) {
    return fail(GX_ERR_INVALID_SHAPE, "lut must be a 256-element fp16 tensor");
  }
  uint16_t host[256];
  for (int i = 0; i < 256; ++i) {
    gx::Float8_e4m3fn v(static_cast<uint8_t>(i), gx::Float8_e4m3fn::from_bits());
    const __half h = __float2half(static_cast<float>(v));
    std::memcpy(&host[i], &h, sizeof(uint16_t));
  }
  // Synchronous on purpose: `host` is a stack buffer and the table is a
  // one-time initialisation, so there is nothing to overlap with.
  if (hipMemcpy(dst.data_ptr(), host, sizeof(host), hipMemcpyHostToDevice) !=
      hipSuccess) {
    return fail(GX_ERR_HIP, "lut upload failed");
  }
  GX906_TRY_END();
}

// ---------------------------------------------------------------------------
// not-yet-wired families (attention / GDN) report a clean error instead of
// failing at link time; they are filled in as the kernel files land.
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
    int32_t use_interleaved_v_cache) {
  GX906_TRY_BEGIN();
  Tensor o = view_of(out, "out");
  Tensor es = view_of(exp_sums, "exp_sums");
  Tensor ml = view_of(max_logits, "max_logits");
  Tensor tmp = view_of(tmp_out, "tmp_out");
  Tensor q = view_of(query, "query");
  Tensor k = view_of(key_cache, "key_cache");
  Tensor v = view_of(value_cache, "value_cache");
  Tensor bt = view_of(block_tables, "block_tables");
  Tensor sl = view_of(seq_lens, "seq_lens");
  Tensor ks = view_of(k_scale, "k_scale");
  Tensor vs = view_of(v_scale, "v_scale");
  const std::string kv_dtype = kv_cache_dtype != nullptr ? kv_cache_dtype : "auto";
  const std::string mfma = mfma_type != nullptr ? mfma_type : "fp16";
  gfx906::paged_attention(
      o, es, ml, tmp, q, k, v, num_kv_heads, scale, bt, sl,
      opt_view(query_start_loc, "query_start_loc"), block_size, max_seq_len,
      opt_view(alibi_slopes, "alibi_slopes"), kv_dtype, ks, vs,
      opt_view(fp8_out_scale, "fp8_out_scale"), mfma,
      use_interleaved_v_cache != 0);
  GX906_TRY_END();
}

gx_status gx906_attnd_decode_inner(
    gx_tensor* segm_output, gx_tensor* segm_max, gx_tensor* segm_expsum,
    const gx_tensor* query, const gx_tensor* key_cache,
    const gx_tensor* value_cache, const gx_tensor* block_tables,
    const gx_tensor* seq_lens, const gx_tensor* cu_query_lens, double scale,
    int64_t num_segments, int32_t delayed_merge, int64_t tps_hint,
    const gx_tensor* k_descale, const gx_tensor* v_descale) {
  GX906_TRY_BEGIN();
  Tensor so = view_of(segm_output, "segm_output");
  Tensor sm = view_of(segm_max, "segm_max");
  Tensor se = view_of(segm_expsum, "segm_expsum");
  Tensor q = view_of(query, "query");
  Tensor k = view_of(key_cache, "key_cache");
  Tensor v = view_of(value_cache, "value_cache");
  Tensor bt = view_of(block_tables, "block_tables");
  Tensor sl = view_of(seq_lens, "seq_lens");
  Tensor cql = view_of(cu_query_lens, "cu_query_lens");
  gfx906::gfx906_attnd_decode_inner(so, sm, se, q, k, v, bt, sl, cql, scale,
                                    num_segments, delayed_merge != 0, tps_hint,
                                    opt_view(k_descale, "k_descale"),
                                    opt_view(v_descale, "v_descale"));
  GX906_TRY_END();
}

gx_status gx906_attnd_decode_inner_kvint8(
    gx_tensor* segm_output, gx_tensor* segm_max, gx_tensor* segm_expsum,
    const gx_tensor* query, const gx_tensor* key_cache,
    const gx_tensor* value_cache, const gx_tensor* k_scale_cache,
    const gx_tensor* v_scale_cache, const gx_tensor* block_tables,
    const gx_tensor* seq_lens, const gx_tensor* cu_query_lens, double scale,
    int64_t num_segments, int32_t delayed_merge, int64_t tps_hint,
    int64_t seg_stride) {
  GX906_TRY_BEGIN();
  Tensor so = view_of(segm_output, "segm_output");
  Tensor sm = view_of(segm_max, "segm_max");
  Tensor se = view_of(segm_expsum, "segm_expsum");
  Tensor q = view_of(query, "query");
  Tensor k = view_of(key_cache, "key_cache");
  Tensor v = view_of(value_cache, "value_cache");
  Tensor ksc = view_of(k_scale_cache, "k_scale_cache");
  Tensor vsc = view_of(v_scale_cache, "v_scale_cache");
  Tensor bt = view_of(block_tables, "block_tables");
  Tensor sl = view_of(seq_lens, "seq_lens");
  Tensor cql = view_of(cu_query_lens, "cu_query_lens");
  gfx906::gfx906_attnd_decode_inner_kvint8(
      so, sm, se, q, k, v, ksc, vsc, bt, sl, cql, scale, num_segments,
      delayed_merge != 0, tps_hint, seg_stride);
  GX906_TRY_END();
}

// ---------------------------------------------------------------------------
// mamba / gated delta rule
// ---------------------------------------------------------------------------

gx_status gx906_causal_conv1d_decode_update(
    const gx_tensor* x, gx_tensor* conv_state, const gx_tensor* weight,
    const gx_tensor* bias, const gx_tensor* conv_state_indices,
    int64_t pad_slot_id, int32_t silu_activation, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::causal_conv1d_gfx906_decode_update(
                   view_of(x, "x"), view_of(conv_state, "conv_state"),
                   view_of(weight, "weight"), opt_view(bias, "bias"),
                   opt_view(conv_state_indices, "conv_state_indices"),
                   pad_slot_id, silu_activation != 0));
  GX906_TRY_END();
}

gx_status gx906_causal_conv1d_mtp_update(
    const gx_tensor* x, gx_tensor* conv_state, const gx_tensor* weight,
    const gx_tensor* bias, const gx_tensor* state_indices,
    const gx_tensor* cu_seqlens, const gx_tensor* num_accepted_tokens,
    int64_t pad_slot_id, int32_t silu_activation, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::causal_conv1d_gfx906_mtp_update(
                   view_of(x, "x"), view_of(conv_state, "conv_state"),
                   view_of(weight, "weight"), opt_view(bias, "bias"),
                   view_of(state_indices, "state_indices"),
                   view_of(cu_seqlens, "cu_seqlens"),
                   view_of(num_accepted_tokens, "num_accepted_tokens"),
                   pad_slot_id, silu_activation != 0));
  GX906_TRY_END();
}

gx_status gx906_fused_sigmoid_gating_delta_rule_decode(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, double beta, double threshold,
    double scale, int32_t use_qk_l2norm_in_kernel, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fused_sigmoid_gating_delta_rule_gfx906_decode(
                   view_of(A_log, "A_log"), view_of(a, "a"), view_of(b, "b"),
                   view_of(dt_bias, "dt_bias"), view_of(q, "q"),
                   view_of(k, "k"), view_of(v, "v"), view_of(state, "state"),
                   beta, threshold, scale, use_qk_l2norm_in_kernel != 0));
  GX906_TRY_END();
}

gx_status gx906_fused_sigmoid_gating_delta_rule_prefill(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* cu_seqlens,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fused_sigmoid_gating_delta_rule_gfx906_prefill(
                   view_of(A_log, "A_log"), view_of(a, "a"), view_of(b, "b"),
                   view_of(dt_bias, "dt_bias"), view_of(q, "q"),
                   view_of(k, "k"), view_of(v, "v"), view_of(state, "state"),
                   view_of(cu_seqlens, "cu_seqlens"), beta, threshold, scale,
                   use_qk_l2norm_in_kernel != 0));
  GX906_TRY_END();
}

gx_status gx906_fused_sigmoid_gating_delta_rule_indexed_decode(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(out, gfx906::fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(
                   view_of(A_log, "A_log"), view_of(a, "a"), view_of(b, "b"),
                   view_of(dt_bias, "dt_bias"), view_of(q, "q"),
                   view_of(k, "k"), view_of(v, "v"), view_of(state, "state"),
                   view_of(state_indices, "state_indices"), beta, threshold,
                   scale, use_qk_l2norm_in_kernel != 0));
  GX906_TRY_END();
}

gx_status gx906_fused_sigmoid_gating_delta_rule_indexed_decode_kv_state(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out) {
  GX906_TRY_BEGIN();
  set_out(
      out, gfx906::fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state(
               view_of(A_log, "A_log"), view_of(a, "a"), view_of(b, "b"),
               view_of(dt_bias, "dt_bias"), view_of(q, "q"), view_of(k, "k"),
               view_of(v, "v"), view_of(state, "state"),
               view_of(state_indices, "state_indices"), beta, threshold, scale,
               use_qk_l2norm_in_kernel != 0));
  GX906_TRY_END();
}

gx_status gx906_fused_sigmoid_gating_delta_rule_mtp_update(
    const gx_tensor* A_log, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* dt_bias, const gx_tensor* q, const gx_tensor* k,
    const gx_tensor* v, gx_tensor* state, const gx_tensor* state_indices,
    const gx_tensor* cu_seqlens, const gx_tensor* num_accepted_tokens,
    double beta, double threshold, double scale,
    int32_t use_qk_l2norm_in_kernel, gx_tensor* out) {
  GX906_TRY_BEGIN();
  (void)(out, gfx906::fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
                   view_of(A_log, "A_log"), view_of(a, "a"), view_of(b, "b"),
                   view_of(dt_bias, "dt_bias"), view_of(q, "q"),
                   view_of(k, "k"), view_of(v, "v"), view_of(state, "state"),
                   view_of(state_indices, "state_indices"),
                   view_of(cu_seqlens, "cu_seqlens"),
                   view_of(num_accepted_tokens, "num_accepted_tokens"), beta,
                   threshold, scale, use_qk_l2norm_in_kernel != 0));;
  GX906_TRY_END();
}

gx_status gx906_fused_recurrent_gated_delta_rule_packed_decode(
    const gx_tensor* mixed_qkv, const gx_tensor* a, const gx_tensor* b,
    const gx_tensor* A_log, const gx_tensor* dt_bias, gx_tensor* state,
    gx_tensor* out, const gx_tensor* state_indices, double scale,
    int32_t use_qk_l2norm_in_kernel, int32_t use_tiled_qk_head_mapping,
    int32_t use_transposed_state) {
  GX906_TRY_BEGIN();
  (void)(out, gfx906::fused_recurrent_gated_delta_rule_gfx906_packed_decode(
                   view_of(mixed_qkv, "mixed_qkv"), view_of(a, "a"),
                   view_of(b, "b"), view_of(A_log, "A_log"),
                   view_of(dt_bias, "dt_bias"), view_of(state, "state"),
                   view_of(out, "out"), view_of(state_indices, "state_indices"),
                   scale, use_qk_l2norm_in_kernel != 0,
                   use_tiled_qk_head_mapping != 0,
                   use_transposed_state != 0));;
  GX906_TRY_END();
}

gx_status gx906_causal_conv1d_recurrent_gated_delta_rule_packed_decode(
    const gx_tensor* mixed_qkv, gx_tensor* conv_state,
    const gx_tensor* conv_weight, const gx_tensor* conv_bias,
    const gx_tensor* a, const gx_tensor* b, const gx_tensor* A_log,
    const gx_tensor* dt_bias, gx_tensor* state, gx_tensor* out,
    const gx_tensor* state_indices, int64_t pad_slot_id, double scale,
    int32_t silu_activation, int32_t use_qk_l2norm_in_kernel,
    int32_t use_tiled_qk_head_mapping, int32_t use_transposed_state) {
  GX906_TRY_BEGIN();
  (void)(out,
          gfx906::causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode(
              view_of(mixed_qkv, "mixed_qkv"),
              view_of(conv_state, "conv_state"),
              view_of(conv_weight, "conv_weight"),
              opt_view(conv_bias, "conv_bias"), view_of(a, "a"),
              view_of(b, "b"), view_of(A_log, "A_log"),
              view_of(dt_bias, "dt_bias"), view_of(state, "state"),
              view_of(out, "out"), view_of(state_indices, "state_indices"),
              pad_slot_id, scale, silu_activation != 0,
              use_qk_l2norm_in_kernel != 0, use_tiled_qk_head_mapping != 0,
              use_transposed_state != 0));;
  GX906_TRY_END();
}

gx_status gx906_causal_conv1d_recurrent_gated_delta_rule_ratio2_packed_decode(
    const gx_tensor* mixed_qkv, gx_tensor* conv_state,
    const gx_tensor* conv_weight, const gx_tensor* conv_bias,
    const gx_tensor* a, const gx_tensor* b, const gx_tensor* A_log,
    const gx_tensor* dt_bias, gx_tensor* state, gx_tensor* out,
    const gx_tensor* state_indices, int64_t pad_slot_id, double scale,
    int32_t silu_activation, int32_t use_qk_l2norm_in_kernel,
    int32_t use_tiled_qk_head_mapping, int32_t use_transposed_state) {
  GX906_TRY_BEGIN();
  set_out(
      out,
      gfx906::causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode(
          view_of(mixed_qkv, "mixed_qkv"), view_of(conv_state, "conv_state"),
          view_of(conv_weight, "conv_weight"),
          opt_view(conv_bias, "conv_bias"), view_of(a, "a"), view_of(b, "b"),
          view_of(A_log, "A_log"), view_of(dt_bias, "dt_bias"),
          view_of(state, "state"), view_of(out, "out"),
          view_of(state_indices, "state_indices"), pad_slot_id, scale,
          silu_activation != 0, use_qk_l2norm_in_kernel != 0,
          use_tiled_qk_head_mapping != 0, use_transposed_state != 0));
  GX906_TRY_END();
}

}  // extern "C"
