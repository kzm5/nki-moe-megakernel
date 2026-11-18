"""Null TKG kernel: identity copy of a [B, S_tkg, H] tensor through SBUF, same layout, sharding and barrier as
attn_e_tkg's load and store. Measures the cost of one NKI call boundary per use.

Returns:
    Y [B, S_tkg, H], bit-identical to X
"""
import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import H, H0, H1, H2, N_PRGS


def null_tkg(X):
    B, S_tkg, _ = X.shape
    BxS = B * S_tkg
    _, _, prg_id = get_verified_program_sharding_info("null_tkg", (0, 1), N_PRGS)
    pattern = [[H2, H0], [H, BxS], [H0 * H2, N_PRGS], [1, H2]]
    x_sb = nl.ndarray((H0, BxS * H1), dtype=X.dtype, buffer=nl.sbuf, name="x_sb")
    nisa.dma_copy(dst=x_sb, src=X.reshape((BxS * H,)).ap(pattern=pattern, offset=0), dge_mode=nisa.dge_mode.hwdge)
    Y = nl.ndarray((B, S_tkg, H), dtype=X.dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        nisa.dma_copy(dst=Y.reshape((BxS * H,)).ap(pattern=pattern, offset=0), src=x_sb,
                      dge_mode=nisa.dge_mode.hwdge)
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))
    return Y


null_tkg_jit = nki.jit(null_tkg)
