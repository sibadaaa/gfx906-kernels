// SPDX-License-Identifier: Apache-2.0
//
// Mamba kernels (extracted from csrc/mamba).
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "detail/aten.hpp"

namespace gfx906 {

using gx::Tensor;

// --- gfx906_decode_kernels.hip ---------------------------------------------
// Fused decode-path kernels for Mamba / GatedDeltaNet (conv1d state update,
// sigmoid-gated delta rule, packed decode).  Definitions live in
// src/kernels/gfx906_decode_kernels.hip.

Tensor causal_conv1d_gfx906_decode_update(
    Tensor x, Tensor conv_state, Tensor weight, std::optional<Tensor> bias,
    std::optional<Tensor> conv_state_indices, int64_t pad_slot_id,
    bool silu_activation);

Tensor causal_conv1d_gfx906_mtp_update(
    Tensor x, Tensor conv_state, Tensor weight, std::optional<Tensor> bias,
    Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens,
    int64_t pad_slot_id, bool silu_activation);

Tensor fused_sigmoid_gating_delta_rule_gfx906_decode(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k,
    Tensor v, Tensor state, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

Tensor fused_sigmoid_gating_delta_rule_gfx906_prefill(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k,
    Tensor v, Tensor state, Tensor cu_seqlens, double beta, double threshold,
    double scale, bool use_qk_l2norm_in_kernel);

Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k,
    Tensor v, Tensor state, Tensor state_indices, double beta, double threshold,
    double scale, bool use_qk_l2norm_in_kernel);

Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k,
    Tensor v, Tensor state, Tensor state_indices, double beta, double threshold,
    double scale, bool use_qk_l2norm_in_kernel);

Tensor fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k,
    Tensor v, Tensor state, Tensor state_indices, Tensor cu_seqlens,
    Tensor num_accepted_tokens, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

Tensor fused_recurrent_gated_delta_rule_gfx906_packed_decode(
    Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias,
    Tensor state, Tensor out, Tensor state_indices, double scale,
    bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping,
    bool use_transposed_state);

Tensor causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode(
    Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight,
    std::optional<Tensor> conv_bias, Tensor a, Tensor b, Tensor A_log,
    Tensor dt_bias, Tensor state, Tensor out, Tensor state_indices,
    int64_t pad_slot_id, double scale, bool silu_activation,
    bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping,
    bool use_transposed_state);

Tensor causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode(
    Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight,
    std::optional<Tensor> conv_bias, Tensor a, Tensor b, Tensor A_log,
    Tensor dt_bias, Tensor state, Tensor out, Tensor state_indices,
    int64_t pad_slot_id, double scale, bool silu_activation,
    bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping,
    bool use_transposed_state);

}  // namespace gfx906
