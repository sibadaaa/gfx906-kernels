// SPDX-License-Identifier: Apache-2.0
//
// gfx906 decode attention inner loop (S53 fp16 / S81 int8 KV).
#pragma once

#include <cstdint>
#include <optional>

#include "detail/aten.hpp"

namespace gfx906 {

using gx::Tensor;

// S53: gfx906 decode split-KV attention inner loop (fp16 KV).
void gfx906_attnd_decode_inner(
    Tensor& segm_output, Tensor& segm_max, Tensor& segm_expsum, Tensor& query,
    Tensor& key_cache, Tensor& value_cache, Tensor& block_tables,
    Tensor& seq_lens, Tensor& cu_query_lens, double scale,
    int64_t num_segments, bool delayed_merge, int64_t tps_hint,
    std::optional<Tensor> k_descale, std::optional<Tensor> v_descale);

// S81/S85: the int8_per_token_head sibling (int8 K/V + fp32 scale caches).
void gfx906_attnd_decode_inner_kvint8(
    Tensor& segm_output, Tensor& segm_max, Tensor& segm_expsum, Tensor& query,
    Tensor& key_cache, Tensor& value_cache, Tensor& k_scale_cache,
    Tensor& v_scale_cache, Tensor& block_tables, Tensor& seq_lens,
    Tensor& cu_query_lens, double scale, int64_t num_segments,
    bool delayed_merge, int64_t tps_hint, int64_t seg_stride);

}  // namespace gfx906
