// SPDX-License-Identifier: Apache-2.0
//
// Attention kernels (extracted from csrc/rocm).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "detail/aten.hpp"

namespace gfx906 {

using gx::Tensor;

void paged_attention(Tensor& out, Tensor& exp_sums, Tensor& max_logits,
                     Tensor& tmp_out, Tensor& query, Tensor& key_cache,
                     Tensor& value_cache, int64_t num_kv_heads, double scale,
                     Tensor& block_tables, Tensor& seq_lens,
                     const std::optional<Tensor>& query_start_loc,
                     int64_t block_size, int64_t max_seq_len,
                     const std::optional<Tensor>& alibi_slopes,
                     const std::string& kv_cache_dtype, Tensor& k_scale,
                     Tensor& v_scale,
                     const std::optional<Tensor>& fp8_out_scale,
                     const std::string& mfma_type,
                     bool use_interleaved_v_cache);

}  // namespace gfx906
