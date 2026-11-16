// SPDX-License-Identifier: Apache-2.0
//
// Optional torch adapter: exposes the C ABI as `torch.ops.gfx906_kernels.*`
// with the same schemas vLLM's `_rocm_C` uses, so a torch-based engine
// (vLLM, SGLang, ...) can switch by changing the op namespace.
//
// This file talks to the stable C ABI only; it does not include the HIP
// sources, so it builds with the host compiler (see tools/build_torch_ext.py).
//
// Buffers returned by the library come from torch's caching allocator: the
// adapter installs raw_alloc/raw_delete hooks at import time and wraps each
// output in at::from_blob with a deleter that drops the library reference.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>

#include <string>
#include <tuple>
#include <vector>

#include "gfx906/gfx906.h"

namespace {

void* torch_alloc(size_t bytes, void* /*ctx*/) {
  return c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
}

void torch_free(void* ptr, void* /*ctx*/) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

struct Initializer {
  Initializer() {
    gx906_set_allocator(torch_alloc, torch_free, nullptr);
  }
};

Initializer g_initializer;

gx_tensor desc(const at::Tensor& t) {
  gx_tensor d{};
  d.data = t.data_ptr();
  d.ndim = static_cast<int32_t>(t.dim());
  TORCH_CHECK(d.ndim <= GX906_MAX_DIMS, "gfx906: too many dims");
  d.dtype = static_cast<int32_t>(t.scalar_type());
  d.device = t.is_cuda() ? t.get_device() : -1;
  for (int32_t i = 0; i < d.ndim; ++i) {
    d.sizes[i] = t.size(i);
    d.strides[i] = t.stride(i);
  }
  return d;
}

const gx_tensor* desc_opt(const std::optional<at::Tensor>& t, gx_tensor* storage) {
  if (!t.has_value() || !t->defined()) return nullptr;
  *storage = desc(*t);
  return storage;
}

void check(gx_status status) {
  if (status != GX_OK) {
    TORCH_CHECK(false, "gfx906_kernels: ", gx906_last_error());
  }
}

at::Tensor from_gx(gx_tensor&& out, const at::TensorOptions& opts) {
  auto* holder = new gx_tensor(out);
  std::vector<int64_t> sizes(holder->sizes, holder->sizes + holder->ndim);
  void* data = holder->data;
  return at::from_blob(
      data, sizes,
      [holder](void*) {
        gx906_tensor_release(holder);
        delete holder;
      },
      opts);
}

struct Scoped {
  Scoped(const at::Tensor& t) : guard(t.device()) {
    gx906_set_stream(reinterpret_cast<void*>(c10::cuda::getCurrentCUDAStream().stream()));
    gx906_set_device(t.get_device());
  }
  c10::cuda::CUDAGuard guard;
};

at::Tensor bf16_emu_gemv(const at::Tensor& x, const at::Tensor& w,
                         const std::optional<at::Tensor>& bias,
                         int64_t nthreads, int64_t route, bool range_safe) {
  Scoped s(x);
  gx_tensor dx = desc(x), dw = desc(w), db{}, out{};
  const gx_tensor* pb = desc_opt(bias, &db);
  check(gx906_bf16_emu_gemv(&dx, &dw, pb, nthreads, route, range_safe, &out));
  return from_gx(std::move(out), x.options());
}

at::Tensor fp16_gemv_stream_m1(const at::Tensor& x, const at::Tensor& w,
                               const std::optional<at::Tensor>& bias,
                               int64_t nthreads) {
  Scoped s(x);
  gx_tensor dx = desc(x), dw = desc(w), db{}, out{};
  const gx_tensor* pb = desc_opt(bias, &db);
  check(gx906_fp16_gemv_stream_m1(&dx, &dw, pb, nthreads, &out));
  return from_gx(std::move(out), x.options());
}

at::Tensor w8a8_sdot4_gemv(const at::Tensor& xq, const at::Tensor& w,
                           const at::Tensor& x_s, const at::Tensor& w_s,
                           const std::optional<at::Tensor>& bias,
                           at::ScalarType out_dtype, int64_t nthreads) {
  Scoped s(xq);
  gx_tensor dxq = desc(xq), dw = desc(w), dxs = desc(x_s), dws = desc(w_s),
           db{}, out{};
  const gx_tensor* pb = desc_opt(bias, &db);
  check(gx906_w8a8_sdot4_gemv(&dxq, &dw, &dxs, &dws, pb,
                              static_cast<int32_t>(out_dtype), nthreads, &out));
  return from_gx(std::move(out), xq.options().dtype(
                                     static_cast<at::ScalarType>(out_dtype)));
}

at::Tensor w8a8_sdot4_gemm(const at::Tensor& xq, const at::Tensor& w,
                           const at::Tensor& x_s, const at::Tensor& w_s,
                           const std::optional<at::Tensor>& bias,
                           at::ScalarType out_dtype, int64_t cfg) {
  Scoped s(xq);
  gx_tensor dxq = desc(xq), dw = desc(w), dxs = desc(x_s), dws = desc(w_s),
           db{}, out{};
  const gx_tensor* pb = desc_opt(bias, &db);
  check(gx906_w8a8_sdot4_gemm(&dxq, &dw, &dxs, &dws, pb,
                              static_cast<int32_t>(out_dtype), cfg, &out));
  return from_gx(std::move(out), xq.options().dtype(
                                     static_cast<at::ScalarType>(out_dtype)));
}

at::Tensor w8a8_sdot4_gemm_pc(const at::Tensor& xq, const at::Tensor& w,
                              const at::Tensor& x_s, const at::Tensor& w_s,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype, int64_t cfg) {
  Scoped s(xq);
  gx_tensor dxq = desc(xq), dw = desc(w), dxs = desc(x_s), dws = desc(w_s),
           db{}, out{};
  const gx_tensor* pb = desc_opt(bias, &db);
  check(gx906_w8a8_sdot4_gemm_pc(&dxq, &dw, &dxs, &dws, pb,
                                 static_cast<int32_t>(out_dtype), cfg, &out));
  return from_gx(std::move(out), xq.options().dtype(
                                     static_cast<at::ScalarType>(out_dtype)));
}

at::Tensor rms_norm_gated(const at::Tensor& input, const at::Tensor& weight,
                          const at::Tensor& gate, double epsilon,
                          bool norm_before_gate) {
  Scoped s(input);
  gx_tensor di = desc(input), dw = desc(weight), dg = desc(gate), out{};
  check(gx906_rms_norm_gated(&di, &dw, &dg, static_cast<float>(epsilon),
                             norm_before_gate, &out));
  return from_gx(std::move(out), input.options());
}

at::Tensor gemma_rms_norm(const at::Tensor& input, const at::Tensor& weight,
                          double epsilon) {
  Scoped s(input);
  gx_tensor di = desc(input), dw = desc(weight), out{};
  check(gx906_gemma_rms_norm(&di, &dw, static_cast<float>(epsilon), &out));
  return from_gx(std::move(out), input.options());
}

std::vector<at::Tensor> gemma_fused_add_rms_norm(const at::Tensor& input,
                                                 const at::Tensor& residual,
                                                 const at::Tensor& weight,
                                                 double epsilon) {
  Scoped s(input);
  gx_tensor di = desc(input), dr = desc(residual), dw = desc(weight), out{},
           residual_out{};
  check(gx906_gemma_fused_add_rms_norm(&di, &dr, &dw,
                                       static_cast<float>(epsilon), &out,
                                       &residual_out));
  return {from_gx(std::move(out), input.options()),
          from_gx(std::move(residual_out), input.options().dtype(at::kFloat))};
}

void attnd_decode_inner(const at::Tensor& segm_output, const at::Tensor& segm_max,
                        const at::Tensor& segm_expsum, const at::Tensor& query,
                        const at::Tensor& key_cache, const at::Tensor& value_cache,
                        const at::Tensor& block_tables, const at::Tensor& seq_lens,
                        const at::Tensor& cu_query_lens, double scale,
                        int64_t num_segments, bool delayed_merge, int64_t tps_hint,
                        const std::optional<at::Tensor>& k_descale,
                        const std::optional<at::Tensor>& v_descale) {
  Scoped s(query);
  gx_tensor so = desc(segm_output), sm = desc(segm_max), se = desc(segm_expsum);
  gx_tensor q = desc(query), k = desc(key_cache), v = desc(value_cache);
  gx_tensor bt = desc(block_tables), sl = desc(seq_lens), cql = desc(cu_query_lens);
  gx_tensor kd{}, vd{};
  const gx_tensor* pkd = desc_opt(k_descale, &kd);
  const gx_tensor* pvd = desc_opt(v_descale, &vd);
  check(gx906_attnd_decode_inner(&so, &sm, &se, &q, &k, &v, &bt, &sl, &cql,
                                 scale, num_segments, delayed_merge, tps_hint,
                                 pkd, pvd));
}

void attnd_decode_inner_kvint8(
    const at::Tensor& segm_output, const at::Tensor& segm_max,
    const at::Tensor& segm_expsum, const at::Tensor& query,
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& k_scale_cache, const at::Tensor& v_scale_cache,
    const at::Tensor& block_tables, const at::Tensor& seq_lens,
    const at::Tensor& cu_query_lens, double scale, int64_t num_segments,
    bool delayed_merge, int64_t tps_hint, int64_t seg_stride) {
  Scoped s(query);
  gx_tensor so = desc(segm_output), sm = desc(segm_max), se = desc(segm_expsum);
  gx_tensor q = desc(query), k = desc(key_cache), v = desc(value_cache);
  gx_tensor ksc = desc(k_scale_cache), vsc = desc(v_scale_cache);
  gx_tensor bt = desc(block_tables), sl = desc(seq_lens), cql = desc(cu_query_lens);
  check(gx906_attnd_decode_inner_kvint8(&so, &sm, &se, &q, &k, &v, &ksc, &vsc,
                                        &bt, &sl, &cql, scale, num_segments,
                                        delayed_merge, tps_hint, seg_stride));
}

at::TensorOptions out_opts(const at::Tensor& like, at::ScalarType dtype) {
  return at::TensorOptions().dtype(dtype).device(like.device());
}

at::Tensor fp8_emu_gemv(const at::Tensor& a, const at::Tensor& w,
                        const at::Tensor& a_s, const at::Tensor& w_s,
                        const at::Tensor& lut,
                        const std::optional<at::Tensor>& bias,
                        at::ScalarType out_dtype, int64_t nthreads) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), das = desc(a_s), dws = desc(w_s),
           dl = desc(lut), db{}, out{};
  check(gx906_fp8_emu_gemv(&da, &dw, &das, &dws, &dl, desc_opt(bias, &db),
                           static_cast<int32_t>(out_dtype), nthreads, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_gemm(const at::Tensor& a, const at::Tensor& w,
                        const at::Tensor& a_s, const at::Tensor& w_s,
                        const at::Tensor& lut,
                        const std::optional<at::Tensor>& bias,
                        at::ScalarType out_dtype, int64_t cfg) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), das = desc(a_s), dws = desc(w_s),
           dl = desc(lut), db{}, out{};
  check(gx906_fp8_emu_gemm(&da, &dw, &das, &dws, &dl, desc_opt(bias, &db),
                           static_cast<int32_t>(out_dtype), cfg, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_gemv_block(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& a_s, const at::Tensor& w_s,
                              const at::Tensor& lut,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype, int64_t nthreads,
                              int64_t block_n, int64_t block_k) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), das = desc(a_s), dws = desc(w_s),
           dl = desc(lut), db{}, out{};
  check(gx906_fp8_emu_gemv_block(&da, &dw, &das, &dws, &dl,
                                 desc_opt(bias, &db),
                                 static_cast<int32_t>(out_dtype), nthreads,
                                 block_n, block_k, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_gemm_block(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& a_s, const at::Tensor& w_s,
                              const at::Tensor& lut,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype, int64_t cfg,
                              int64_t block_n, int64_t block_k) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), das = desc(a_s), dws = desc(w_s),
           dl = desc(lut), db{}, out{};
  check(gx906_fp8_emu_gemm_block(&da, &dw, &das, &dws, &dl,
                                 desc_opt(bias, &db),
                                 static_cast<int32_t>(out_dtype), cfg, block_n,
                                 block_k, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_gemm_block_w8a16(const at::Tensor& a, const at::Tensor& w,
                                    const at::Tensor& w_s, const at::Tensor& lut,
                                    const std::optional<at::Tensor>& bias,
                                    at::ScalarType out_dtype, int64_t nthreads,
                                    int64_t cfg, int64_t block_n,
                                    int64_t block_k) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), dws = desc(w_s), dl = desc(lut), db{},
           out{};
  check(gx906_fp8_emu_gemm_block_w8a16(
      &da, &dw, &dws, &dl, desc_opt(bias, &db), static_cast<int32_t>(out_dtype),
      nthreads, cfg, block_n, block_k, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_gemm_block_w8a16_smul(
    const at::Tensor& a, const at::Tensor& w, const at::Tensor& w_s,
    const at::Tensor& lut, const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, int64_t nthreads, int64_t block_n, int64_t block_k) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), dws = desc(w_s), dl = desc(lut), db{},
           out{};
  check(gx906_fp8_emu_gemm_block_w8a16_smul(
      &da, &dw, &dws, &dl, desc_opt(bias, &db), static_cast<int32_t>(out_dtype),
      nthreads, block_n, block_k, &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

at::Tensor fp8_emu_mm_int8_pc(const at::Tensor& xq, const at::Tensor& w,
                              const at::Tensor& w_s, const at::Tensor& x_s,
                              const at::Tensor& lut,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype, int64_t block_n,
                              int64_t block_k) {
  Scoped s(xq);
  gx_tensor dxq = desc(xq), dw = desc(w), dws = desc(w_s), dxs = desc(x_s),
           dl = desc(lut), db{}, out{};
  check(gx906_fp8_emu_mm_int8_pc(&dxq, &dw, &dws, &dxs, &dl,
                                 desc_opt(bias, &db),
                                 static_cast<int32_t>(out_dtype), block_n,
                                 block_k, &out));
  return from_gx(std::move(out), out_opts(xq, out_dtype));
}

void paged_attention(const at::Tensor& out, const at::Tensor& exp_sums,
                     const at::Tensor& max_logits, const at::Tensor& tmp_out,
                     const at::Tensor& query, const at::Tensor& key_cache,
                     const at::Tensor& value_cache, int64_t num_kv_heads,
                     double scale, const at::Tensor& block_tables,
                     const at::Tensor& seq_lens,
                     const std::optional<at::Tensor>& query_start_loc,
                     int64_t block_size, int64_t max_seq_len,
                     const std::optional<at::Tensor>& alibi_slopes,
                     const std::string& kv_cache_dtype,
                     const at::Tensor& k_scale, const at::Tensor& v_scale,
                     const std::optional<at::Tensor>& fp8_out_scale,
                     const std::string& mfma_type,
                     bool use_interleaved_v_cache) {
  Scoped s(query);
  gx_tensor dout = desc(out), des = desc(exp_sums), dml = desc(max_logits),
           dtm = desc(tmp_out), dq = desc(query), dk = desc(key_cache),
           dv = desc(value_cache), dbt = desc(block_tables),
           dsl = desc(seq_lens), dks = desc(k_scale), dvs = desc(v_scale);
  gx_tensor dqsl{}, dal{}, dfos{};
  check(gx906_paged_attention(
      &dout, &des, &dml, &dtm, &dq, &dk, &dv, num_kv_heads, scale, &dbt, &dsl,
      desc_opt(query_start_loc, &dqsl), block_size, max_seq_len,
      desc_opt(alibi_slopes, &dal), kv_cache_dtype.c_str(), &dks, &dvs,
      desc_opt(fp8_out_scale, &dfos), mfma_type.c_str(),
      use_interleaved_v_cache));
}

at::Tensor fp8_emu_gemm_block_quant(
    const at::Tensor& a, const at::Tensor& w, const at::Tensor& w_s,
    const at::Tensor& lut, const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, int64_t nthreads, int64_t cfg, int64_t block_n,
    int64_t block_k, double eps) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), dws = desc(w_s), dl = desc(lut), db{},
           out{};
  check(gx906_fp8_emu_gemm_block_quant(
      &da, &dw, &dws, &dl, desc_opt(bias, &db), static_cast<int32_t>(out_dtype),
      nthreads, cfg, block_n, block_k, static_cast<float>(eps), &out));
  return from_gx(std::move(out), out_opts(a, out_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_emu_gemm_block_quant_x(
    const at::Tensor& a, const at::Tensor& w, const at::Tensor& w_s,
    const at::Tensor& lut, const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, int64_t nthreads, int64_t cfg, int64_t block_n,
    int64_t block_k, double eps) {
  Scoped s(a);
  gx_tensor da = desc(a), dw = desc(w), dws = desc(w_s), dl = desc(lut), db{},
           out{}, xq{}, xs{};
  check(gx906_fp8_emu_gemm_block_quant_x(
      &da, &dw, &dws, &dl, desc_opt(bias, &db), static_cast<int32_t>(out_dtype),
      nthreads, cfg, block_n, block_k, static_cast<float>(eps), &out, &xq, &xs));
  return {from_gx(std::move(out), out_opts(a, out_dtype)),
          from_gx(std::move(xq), a.options().dtype(at::kByte)),
          from_gx(std::move(xs), a.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor> fp8_stage_int8_pc(const at::Tensor& w,
                                          const at::Tensor& w_s,
                                          const at::Tensor& lut,
                                          int64_t block_n, int64_t block_k) {
  Scoped s(w);
  gx_tensor dw = desc(w), dws = desc(w_s), dl = desc(lut), q{}, scale{};
  check(gx906_fp8_stage_int8_pc(&dw, &dws, &dl, block_n, block_k, &q, &scale));
  return {from_gx(std::move(q), w.options().dtype(at::kChar)),
          from_gx(std::move(scale), w_s.options())};
}

void fp8_dequant_to_fp16_out(const at::Tensor& out, const at::Tensor& wq,
                             const at::Tensor& ws, const at::Tensor& lut,
                             int64_t block_n, int64_t block_k) {
  Scoped s(wq);
  gx_tensor dout = desc(out), dwq = desc(wq), dws = desc(ws), dl = desc(lut);
  check(gx906_fp8_dequant_to_fp16_out(&dout, &dwq, &dws, &dl, block_n, block_k));
}

// --- Mamba / GatedDeltaNet -------------------------------------------------

at::Tensor conv1d_decode_update(const at::Tensor& x, at::Tensor& conv_state,
                                const at::Tensor& weight,
                                const std::optional<at::Tensor>& bias,
                                const std::optional<at::Tensor>& state_indices,
                                int64_t pad_slot_id, bool silu) {
  Scoped s(x);
  gx_tensor dx = desc(x), dcs = desc(conv_state), dw = desc(weight), db{},
           dsi{}, out{};
  check(gx906_causal_conv1d_decode_update(
      &dx, &dcs, &dw, desc_opt(bias, &db), desc_opt(state_indices, &dsi),
      pad_slot_id, silu, &out));
  return from_gx(std::move(out), x.options());
}

at::Tensor conv1d_mtp_update(const at::Tensor& x, at::Tensor& conv_state,
                             const at::Tensor& weight,
                             const std::optional<at::Tensor>& bias,
                             const at::Tensor& state_indices,
                             const at::Tensor& cu_seqlens,
                             const at::Tensor& num_accepted_tokens,
                             int64_t pad_slot_id, bool silu) {
  Scoped s(x);
  gx_tensor dx = desc(x), dcs = desc(conv_state), dw = desc(weight), db{},
           dsi = desc(state_indices), dcu = desc(cu_seqlens),
           dnt = desc(num_accepted_tokens), out{};
  check(gx906_causal_conv1d_mtp_update(&dx, &dcs, &dw, desc_opt(bias, &db),
                                       &dsi, &dcu, &dnt, pad_slot_id, silu,
                                       &out));
  return from_gx(std::move(out), x.options());
}

at::Tensor gdn_decode(const at::Tensor& A_log, const at::Tensor& a,
                      const at::Tensor& b, const at::Tensor& dt_bias,
                      const at::Tensor& q, const at::Tensor& k,
                      const at::Tensor& v, at::Tensor& state, double beta,
                      double threshold, double scale, bool use_l2norm) {
  Scoped s(q);
  gx_tensor dal = desc(A_log), da = desc(a), db = desc(b), ddt = desc(dt_bias),
           dq = desc(q), dk = desc(k), dv = desc(v), ds = desc(state), out{};
  check(gx906_fused_sigmoid_gating_delta_rule_decode(
      &dal, &da, &db, &ddt, &dq, &dk, &dv, &ds, beta, threshold, scale,
      use_l2norm, &out));
  return from_gx(std::move(out), q.options());
}

at::Tensor gdn_prefill(const at::Tensor& A_log, const at::Tensor& a,
                       const at::Tensor& b, const at::Tensor& dt_bias,
                       const at::Tensor& q, const at::Tensor& k,
                       const at::Tensor& v, at::Tensor& state,
                       const at::Tensor& cu_seqlens, double beta,
                       double threshold, double scale, bool use_l2norm) {
  Scoped s(q);
  gx_tensor dal = desc(A_log), da = desc(a), db = desc(b), ddt = desc(dt_bias),
           dq = desc(q), dk = desc(k), dv = desc(v), ds = desc(state),
           dcu = desc(cu_seqlens), out{};
  check(gx906_fused_sigmoid_gating_delta_rule_prefill(
      &dal, &da, &db, &ddt, &dq, &dk, &dv, &ds, &dcu, beta, threshold, scale,
      use_l2norm, &out));
  return from_gx(std::move(out), q.options());
}

at::Tensor gdn_indexed_decode(const at::Tensor& A_log, const at::Tensor& a,
                              const at::Tensor& b, const at::Tensor& dt_bias,
                              const at::Tensor& q, const at::Tensor& k,
                              const at::Tensor& v, at::Tensor& state,
                              const at::Tensor& state_indices, double beta,
                              double threshold, double scale, bool use_l2norm) {
  Scoped s(q);
  gx_tensor dal = desc(A_log), da = desc(a), db = desc(b), ddt = desc(dt_bias),
           dq = desc(q), dk = desc(k), dv = desc(v), ds = desc(state),
           dsi = desc(state_indices), out{};
  check(gx906_fused_sigmoid_gating_delta_rule_indexed_decode(
      &dal, &da, &db, &ddt, &dq, &dk, &dv, &ds, &dsi, beta, threshold, scale,
      use_l2norm, &out));
  return from_gx(std::move(out), q.options());
}

at::Tensor gdn_indexed_decode_kv_state(
    const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& dt_bias, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, at::Tensor& state, const at::Tensor& state_indices,
    double beta, double threshold, double scale, bool use_l2norm) {
  Scoped s(q);
  gx_tensor dal = desc(A_log), da = desc(a), db = desc(b), ddt = desc(dt_bias),
           dq = desc(q), dk = desc(k), dv = desc(v), ds = desc(state),
           dsi = desc(state_indices), out{};
  check(gx906_fused_sigmoid_gating_delta_rule_indexed_decode_kv_state(
      &dal, &da, &db, &ddt, &dq, &dk, &dv, &ds, &dsi, beta, threshold, scale,
      use_l2norm, &out));
  return from_gx(std::move(out), q.options());
}

at::Tensor gdn_mtp_update(const at::Tensor& A_log, const at::Tensor& a,
                          const at::Tensor& b, const at::Tensor& dt_bias,
                          const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, at::Tensor& state,
                          const at::Tensor& state_indices,
                          const at::Tensor& cu_seqlens,
                          const at::Tensor& num_accepted_tokens, double beta,
                          double threshold, double scale, bool use_l2norm) {
  Scoped s(q);
  gx_tensor dal = desc(A_log), da = desc(a), db = desc(b), ddt = desc(dt_bias),
           dq = desc(q), dk = desc(k), dv = desc(v), ds = desc(state),
           dsi = desc(state_indices), dcu = desc(cu_seqlens),
           dnt = desc(num_accepted_tokens), out{};
  check(gx906_fused_sigmoid_gating_delta_rule_mtp_update(
      &dal, &da, &db, &ddt, &dq, &dk, &dv, &ds, &dsi, &dcu, &dnt, beta,
      threshold, scale, use_l2norm, &out));
  return from_gx(std::move(out), q.options());
}

at::Tensor gdn_packed_decode(const at::Tensor& mixed_qkv, const at::Tensor& a,
                             const at::Tensor& b, const at::Tensor& A_log,
                             const at::Tensor& dt_bias, at::Tensor& state,
                             at::Tensor& out,
                             const at::Tensor& state_indices, double scale,
                             bool use_l2norm, bool use_tiled_head_map,
                             bool use_transposed_state) {
  Scoped s(mixed_qkv);
  gx_tensor dmq = desc(mixed_qkv), da = desc(a), db = desc(b),
           dal = desc(A_log), ddt = desc(dt_bias), ds = desc(state),
           dout = desc(out), dsi = desc(state_indices);
  check(gx906_fused_recurrent_gated_delta_rule_packed_decode(
      &dmq, &da, &db, &dal, &ddt, &ds, &dout, &dsi, scale, use_l2norm,
      use_tiled_head_map, use_transposed_state));
  return out;
}

at::Tensor conv_gdn_packed_decode(
    const at::Tensor& mixed_qkv, at::Tensor& conv_state,
    const at::Tensor& conv_weight, const std::optional<at::Tensor>& conv_bias,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, at::Tensor& state, at::Tensor& out,
    const at::Tensor& state_indices, int64_t pad_slot_id, double scale,
    bool silu, bool use_l2norm, bool use_tiled_head_map,
    bool use_transposed_state) {
  Scoped s(mixed_qkv);
  gx_tensor dmq = desc(mixed_qkv), dcs = desc(conv_state),
           dcw = desc(conv_weight), dcb{}, da = desc(a), db = desc(b),
           dal = desc(A_log), ddt = desc(dt_bias), ds = desc(state),
           dout = desc(out), dsi = desc(state_indices);
  check(gx906_causal_conv1d_recurrent_gated_delta_rule_packed_decode(
      &dmq, &dcs, &dcw, desc_opt(conv_bias, &dcb), &da, &db, &dal, &ddt, &ds,
      &dout, &dsi, pad_slot_id, scale, silu, use_l2norm, use_tiled_head_map,
      use_transposed_state));
  return out;
}

at::Tensor conv_gdn_ratio2_packed_decode(
    const at::Tensor& mixed_qkv, at::Tensor& conv_state,
    const at::Tensor& conv_weight, const std::optional<at::Tensor>& conv_bias,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, at::Tensor& state, at::Tensor& out,
    const at::Tensor& state_indices, int64_t pad_slot_id, double scale,
    bool silu, bool use_l2norm, bool use_tiled_head_map,
    bool use_transposed_state) {
  Scoped s(mixed_qkv);
  gx_tensor dmq = desc(mixed_qkv), dcs = desc(conv_state),
           dcw = desc(conv_weight), dcb{}, da = desc(a), db = desc(b),
           dal = desc(A_log), ddt = desc(dt_bias), ds = desc(state),
           dout = desc(out), dsi = desc(state_indices);
  check(gx906_causal_conv1d_recurrent_gated_delta_rule_ratio2_packed_decode(
      &dmq, &dcs, &dcw, desc_opt(conv_bias, &dcb), &da, &db, &dal, &ddt, &ds,
      &dout, &dsi, pad_slot_id, scale, silu, use_l2norm, use_tiled_head_map,
      use_transposed_state));
  return out;
}

}  // namespace

TORCH_LIBRARY(gfx906_kernels, m) {
  m.def(
      "bf16_emu_gemv(Tensor x, Tensor w, Tensor? bias, int nthreads, "
      "int route, bool range_safe) -> Tensor");
  m.def(
      "fp16_gemv_stream_m1(Tensor x, Tensor w, Tensor? bias, int nthreads) "
      "-> Tensor");
  m.def(
      "w8a8_sdot4_gemv(Tensor xq, Tensor w, Tensor x_s, Tensor w_s, "
      "Tensor? bias, ScalarType out_dtype, int nthreads) -> Tensor");
  m.def(
      "w8a8_sdot4_gemm(Tensor xq, Tensor w, Tensor x_s, Tensor w_s, "
      "Tensor? bias, ScalarType out_dtype, int cfg) -> Tensor");
  m.def(
      "w8a8_sdot4_gemm_pc(Tensor xq, Tensor w, Tensor x_s, Tensor w_s, "
      "Tensor? bias, ScalarType out_dtype, int cfg) -> Tensor");
  m.def(
      "rms_norm_gated_gfx906(Tensor input, Tensor weight, Tensor gate, "
      "float epsilon, bool norm_before_gate) -> Tensor");
  m.def("gemma_rms_norm_gfx906(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def(
      "gemma_fused_add_rms_norm_gfx906(Tensor input, Tensor residual, "
      "Tensor weight, float epsilon) -> Tensor[]");
  m.def(
      "fp8_emu_gemv(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int nthreads) -> Tensor");
  m.def(
      "fp8_emu_gemm(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int cfg) -> Tensor");
  m.def(
      "fp8_emu_gemv_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int nthreads, int block_n, "
      "int block_k) -> Tensor");
  m.def(
      "fp8_emu_gemm_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int cfg, int block_n, "
      "int block_k) -> Tensor");
  m.def(
      "fp8_emu_gemm_block_w8a16(Tensor a, Tensor w, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int nthreads, int cfg, int block_n, "
      "int block_k) -> Tensor");
  m.def(
      "fp8_emu_gemm_block_w8a16_smul(Tensor a, Tensor w, Tensor w_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int nthreads, int block_n, "
      "int block_k) -> Tensor");
  m.def(
      "fp8_emu_mm_int8_pc(Tensor xq, Tensor w, Tensor w_s, Tensor x_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int block_n, int block_k) "
      "-> Tensor");
  m.def(
      "fp8_emu_gemm_block_quant(Tensor a, Tensor w, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int nthreads, int cfg, int block_n, "
      "int block_k, float eps) -> Tensor");
  m.def(
      "fp8_emu_gemm_block_quant_x(Tensor a, Tensor w, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int nthreads, int cfg, int block_n, "
      "int block_k, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fp8_stage_int8_pc(Tensor w, Tensor w_s, Tensor lut, int block_n, "
      "int block_k) -> (Tensor, Tensor)");
  m.def(
      "fp8_dequant_to_fp16_out(Tensor(a!) out, Tensor wq, Tensor ws, "
      "Tensor lut, int block_n, int block_k) -> ()");
  m.def(
      "causal_conv1d_gfx906_decode_update(Tensor x, Tensor! conv_state, "
      "Tensor weight, Tensor? bias, Tensor? conv_state_indices, "
      "int pad_slot_id, bool silu_activation) -> Tensor");
  m.def(
      "causal_conv1d_gfx906_mtp_update(Tensor x, Tensor! conv_state, "
      "Tensor weight, Tensor? bias, Tensor state_indices, Tensor cu_seqlens, "
      "Tensor num_accepted_tokens, int pad_slot_id, bool silu_activation) "
      "-> Tensor");
  m.def(
      "fused_sigmoid_gating_delta_rule_gfx906_decode(Tensor A_log, Tensor a, "
      "Tensor b, Tensor dt_bias, Tensor q, Tensor k, Tensor v, "
      "Tensor! state, float beta, float threshold, float scale, "
      "bool use_qk_l2norm_in_kernel) -> Tensor");
  m.def(
      "fused_sigmoid_gating_delta_rule_gfx906_prefill(Tensor A_log, Tensor a, "
      "Tensor b, Tensor dt_bias, Tensor q, Tensor k, Tensor v, "
      "Tensor! state, Tensor cu_seqlens, float beta, float threshold, "
      "float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  m.def(
      "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(Tensor A_log, "
      "Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, Tensor v, "
      "Tensor! state, Tensor state_indices, float beta, float threshold, "
      "float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  m.def(
      "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor! state, Tensor state_indices, float beta, "
      "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  m.def(
      "fused_sigmoid_gating_delta_rule_gfx906_mtp_update(Tensor A_log, "
      "Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, Tensor v, "
      "Tensor! state, Tensor state_indices, Tensor cu_seqlens, "
      "Tensor num_accepted_tokens, float beta, float threshold, float scale, "
      "bool use_qk_l2norm_in_kernel) -> Tensor");
  m.def(
      "fused_recurrent_gated_delta_rule_gfx906_packed_decode(Tensor mixed_qkv, "
      "Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor! state, "
      "Tensor! out, Tensor state_indices, float scale, "
      "bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping, "
      "bool use_transposed_state) -> Tensor");
  m.def(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode("
      "Tensor mixed_qkv, Tensor! conv_state, Tensor conv_weight, "
      "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor! state, Tensor! out, Tensor state_indices, int pad_slot_id, "
      "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
      "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");
  m.def(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode("
      "Tensor mixed_qkv, Tensor! conv_state, Tensor conv_weight, "
      "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor! state, Tensor! out, Tensor state_indices, int pad_slot_id, "
      "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
      "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");
  m.def(
      "paged_attention(Tensor! out, Tensor exp_sums, Tensor max_logits, "
      "Tensor tmp_out, Tensor query, Tensor key_cache, Tensor value_cache, "
      "int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, "
      "Tensor? query_start_loc, int block_size, int max_seq_len, "
      "Tensor? alibi_slopes, str kv_cache_dtype, Tensor k_scale, "
      "Tensor v_scale, Tensor? fp8_out_scale, str mfma_type, "
      "bool use_interleaved_v_cache=False) -> ()");
  m.def(
      "gfx906_attnd_decode_inner(Tensor! segm_output, Tensor! segm_max, "
      "Tensor! segm_expsum, Tensor query, Tensor key_cache, "
      "Tensor value_cache, Tensor block_tables, Tensor seq_lens, "
      "Tensor cu_query_lens, float scale, int num_segments, "
      "bool delayed_merge, int tps_hint, Tensor? k_descale=None, "
      "Tensor? v_descale=None) -> ()");
  m.def(
      "gfx906_attnd_decode_inner_kvint8(Tensor! segm_output, Tensor! segm_max, "
      "Tensor! segm_expsum, Tensor query, Tensor key_cache, "
      "Tensor value_cache, Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor block_tables, Tensor seq_lens, Tensor cu_query_lens, "
      "float scale, int num_segments, bool delayed_merge, int tps_hint, "
      "int seg_stride=0) -> ()");
}

TORCH_LIBRARY_IMPL(gfx906_kernels, CUDA, m) {
  m.impl("bf16_emu_gemv", &bf16_emu_gemv);
  m.impl("fp16_gemv_stream_m1", &fp16_gemv_stream_m1);
  m.impl("w8a8_sdot4_gemv", &w8a8_sdot4_gemv);
  m.impl("w8a8_sdot4_gemm", &w8a8_sdot4_gemm);
  m.impl("w8a8_sdot4_gemm_pc", &w8a8_sdot4_gemm_pc);
  m.impl("rms_norm_gated_gfx906", &rms_norm_gated);
  m.impl("gemma_rms_norm_gfx906", &gemma_rms_norm);
  m.impl("gemma_fused_add_rms_norm_gfx906", &gemma_fused_add_rms_norm);
  m.impl("fp8_emu_gemv", &fp8_emu_gemv);
  m.impl("fp8_emu_gemm", &fp8_emu_gemm);
  m.impl("fp8_emu_gemv_block", &fp8_emu_gemv_block);
  m.impl("fp8_emu_gemm_block", &fp8_emu_gemm_block);
  m.impl("fp8_emu_gemm_block_w8a16", &fp8_emu_gemm_block_w8a16);
  m.impl("fp8_emu_gemm_block_w8a16_smul", &fp8_emu_gemm_block_w8a16_smul);
  m.impl("fp8_emu_mm_int8_pc", &fp8_emu_mm_int8_pc);
  m.impl("fp8_emu_gemm_block_quant", &fp8_emu_gemm_block_quant);
  m.impl("fp8_emu_gemm_block_quant_x", &fp8_emu_gemm_block_quant_x);
  m.impl("fp8_stage_int8_pc", &fp8_stage_int8_pc);
  m.impl("fp8_dequant_to_fp16_out", &fp8_dequant_to_fp16_out);
  m.impl("causal_conv1d_gfx906_decode_update", &conv1d_decode_update);
  m.impl("causal_conv1d_gfx906_mtp_update", &conv1d_mtp_update);
  m.impl("fused_sigmoid_gating_delta_rule_gfx906_decode", &gdn_decode);
  m.impl("fused_sigmoid_gating_delta_rule_gfx906_prefill", &gdn_prefill);
  m.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode",
         &gdn_indexed_decode);
  m.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state",
         &gdn_indexed_decode_kv_state);
  m.impl("fused_sigmoid_gating_delta_rule_gfx906_mtp_update", &gdn_mtp_update);
  m.impl("fused_recurrent_gated_delta_rule_gfx906_packed_decode",
         &gdn_packed_decode);
  m.impl("causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode",
         &conv_gdn_packed_decode);
  m.impl("causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode",
         &conv_gdn_ratio2_packed_decode);
  m.impl("paged_attention", &paged_attention);
  m.impl("gfx906_attnd_decode_inner", &attnd_decode_inner);
  m.impl("gfx906_attnd_decode_inner_kvint8", &attnd_decode_inner_kvint8);
}
