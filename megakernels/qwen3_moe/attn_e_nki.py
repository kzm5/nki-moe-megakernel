"""Single-layer TKG attention kernel: the megakernel's attention stage as a standalone NKI call.

Computes (per decoder layer, T = S_tkg tokens):
    Y = AllReduce_TP(o_proj(attention(RoPE(QKnorm(QKV(RMSNorm(X; gpre)))), K_cache, V_cache)))
    K_cache / V_cache updated in place at position_ids[:, 0]

Tiled computation:
    Same stage sequence as one layer of transformer_qwen3_moe_speculative: shard-interleaved X load,
    vendored attention_block_tkg (fused input RMSNorm, QKV, pre-RoPE QK RMSNorm, RoPE, attention,
    KV cache update, output projection), _sb2sb_all_reduce_gather, HBM store.

Returns:
    (Y [B, S_tkg, H] bf16, K_cache, V_cache)  -- K/V returned so the in-place scatter is kept
"""
import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nki_kernels._legacy.attention import attention_block_tkg
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    D_HEAD, EPS, H, H0, H1, H1_SHARD, H2, N_PRGS, NUM_Q_HEADS_TP, SBM_SIZE_BYTES,
    _build_full_mask_hbm, _build_permuted_rope_hbm_from_pos,
)


def attn_e_tkg(X, Wqkv, Wo, Qn, Kn, Gpre, K_cache, V_cache, cos, sin, position_ids, replica_groups=None):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    _, n_prgs, prg_id = get_verified_program_sharding_info("attn_e_tkg", (0, 1), N_PRGS)
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None
    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("attn_e_tkg"))
    sbm.set_auto_alloc(True)
    S_FULL = K_cache.shape[-2]

    mask_full_hbm = _build_full_mask_hbm(position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg)
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(cos, B, S_tkg, D_HEAD, name="cos_perm_hbm")
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(sin, B, S_tkg, D_HEAD, name="sin_perm_hbm")

    x_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf, name="x_sb")
    nisa.dma_copy(
        dst=x_sb,
        src=X.reshape((BxS * H,)).ap(
            pattern=[[H2, H0], [H, BxS], [H0 * H2, N_PRGS], [1, H2]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    sbm.set_name_prefix("attn_")
    res = attention_block_tkg(
        X=x_sb.reshape((H0, BxS, H1)),
        X_hidden_dim_actual=H,
        rmsnorm_X_enabled=True, rmsnorm_X_eps=EPS, rmsnorm_X_gamma=Gpre.reshape((1, H)),
        W_qkv=Wqkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=Qn.reshape((1, D_HEAD)), rmsnorm_QK_pre_rope_W_K=Kn.reshape((1, D_HEAD)),
        cos=cos_perm_hbm, sin=sin_perm_hbm, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False, active_blocks_table=None, K_cache=K_cache, V_cache=V_cache,
        attention_mask=mask_full_hbm, sink=None,
        update_cache=True, kv_cache_update_idx=position_ids[:, :1],
        W_out=Wo, bias_out=None, quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=True, out_in_sb=True, sbm=sbm,
    )
    out_sb = res[0].reshape((H0, H1_SHARD * BxS))
    gathered_sb, _ = _sb2sb_all_reduce_gather(out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS)

    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        nisa.dma_copy(
            dst=Y.reshape((BxS * H,)).ap(
                pattern=[[H2, H0], [H, BxS], [H0 * H2, N_PRGS], [1, H2]], offset=0),
            src=gathered_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))
    return Y, res[1], res[2]


attn_e_tkg_jit = nki.jit(attn_e_tkg)
