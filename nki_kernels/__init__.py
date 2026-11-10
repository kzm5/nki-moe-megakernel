# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom NKI kernels for the Qwen3.6-27B DeltaNet layers.

Stable public surface. Consumers import entrypoints from this package root
(``from ...nki_kernels import deltanet_fused_tkg_fwd``) and stay decoupled from
the internal module layout — add new kernels here as the megakernel grows.

Layout (see README.md for the full map):
- ``deltanet/components/`` reusable stage kernels (in_proj, conv, recurrence,
  norm_gate, out_proj) — the megakernel building blocks.
- ``deltanet/decode/``     token-generation (TKG) path; ``fused_layer`` is the
  assembled DeltaNet attention megakernel (current); ``recurrent`` is the
  per-token fallback.
- ``deltanet/prefill/``    context-encoding (CTE) path; ``chunked_step`` is the
  stable default; ``chunked_fused`` is a faster single-kernel variant that
  overflows fp32 for this checkpoint (gated off — see README).
- ``specs/``               design specs for individual kernels.
"""

import importlib

# Public name -> defining submodule; imported on first attribute access.
_LAZY_EXPORTS = {
    # Decode (TKG).
    "deltanet_fused_tkg_fwd": ".deltanet.decode.fused_layer",
    "deltanet_fused_tkg_fwd_state": ".deltanet.decode.fused_layer",
    "deltanet_attention_layer_state": ".deltanet.decode.fused_layer",
    "deltanet_recurrent_fwd": ".deltanet.decode.recurrent",
    "deltanet_recurrent_fwd_state": ".deltanet.decode.recurrent",
    # Prefill (CTE).
    "deltanet_chunk_step": ".deltanet.prefill.chunked_step",
    "deltanet_fused_chunked_fwd": ".deltanet.prefill.chunked_fused",
    "_make_lower_mask": ".deltanet.prefill.chunked_fused",
    "_make_lower_mask_diag": ".deltanet.prefill.chunked_fused",
    "_make_identity": ".deltanet.prefill.chunked_fused",
    # GQA (full-attention) decode/verify TKG megakernel.
    "gqa_fused_tkg_fwd": ".gqa",
    # MoE (post-attention FFN) fused-layer verify kernel.
    "moe_layer_fwd": ".moe",
    # LM head (final norm + vocab projection + per-rank greedy argmax), shared by both heads.
    "lm_head_fwd": ".lm_head",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "deltanet_fused_tkg_fwd",
    "deltanet_fused_tkg_fwd_state",
    "deltanet_attention_layer_state",
    "deltanet_recurrent_fwd",
    "deltanet_recurrent_fwd_state",
    "deltanet_chunk_step",
    "deltanet_fused_chunked_fwd",
    "_make_lower_mask",
    "_make_lower_mask_diag",
    "_make_identity",
    "gqa_fused_tkg_fwd",
    "moe_layer_fwd",
    "lm_head_fwd",
]
