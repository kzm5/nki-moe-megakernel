"""SwiGLU elementwise kernel: out = silu(gate) * up, bf16, shape [N, 128, F].

Computes:
    out[n] = silu(gate[n]) * up[n] for each [128, F] tile n

Usage:
    python swiglu_kernel.py   # CPU simulator check against a NumPy reference
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def swiglu(gate, up):
    N, _, F = gate.shape
    out = nl.ndarray((N, P, F), dtype=gate.dtype, buffer=nl.shared_hbm)
    act_hbm = nl.ndarray((N, P, F), dtype=gate.dtype, buffer=nl.shared_hbm)
    for n in nl.sequential_range(N):
        g = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=gate[n])
        a = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.activation(dst=a, op=nl.silu, data=g)
        nisa.dma_copy(dst=act_hbm[n], src=a)
        a2 = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a2, src=act_hbm[n])
        u = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=u, src=up[n])
        o = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o, data1=a2, data2=u, op=nl.multiply)
        nisa.dma_copy(dst=out[n], src=o)
    return out


def reference(gate, up):
    g = gate.astype(np.float32)
    return (g / (1 + np.exp(-g))) * up.astype(np.float32)


if __name__ == "__main__":
    import ml_dtypes
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((4, P, 512)).astype(ml_dtypes.bfloat16)
    up = rng.standard_normal((4, P, 512)).astype(ml_dtypes.bfloat16)
    out = np.asarray(nki.simulate(swiglu)(gate, up)).astype(np.float32)
    ref = reference(gate, up)
    print("max abs err vs fp32 reference", float(np.abs(out - ref).max()))
