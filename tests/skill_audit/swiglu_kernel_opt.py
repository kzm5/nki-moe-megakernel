"""SwiGLU elementwise kernel: out = silu(gate) * up, bf16, shape [N, 128, F].

Computes:
    out[n] = silu(gate[n]) * up[n] for each [128, F] tile n

Tiled computation:
    one [128, F] tile per n; silu on ScalarE, multiply on VectorE, all SBUF-resident
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128  # SBUF partition count


@nki.jit
def swiglu(gate, up):
    N, _, F = gate.shape
    out = nl.ndarray((N, P, F), dtype=gate.dtype, buffer=nl.shared_hbm)
    # Iterations are independent: lets DMA of tile n+1 overlap compute of tile n
    # and ScalarE (silu) pipeline against VectorE (multiply).
    for n in nl.affine_range(N):
        # Load: both inputs issued up front so the two DMAs run in parallel.
        g = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        u = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=gate[n])
        nisa.dma_copy(dst=u, src=up[n])

        # Compute: bf16 silu result stays in SBUF (same rounding point as an HBM round trip).
        a = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.activation(dst=a, op=nl.silu, data=g)
        o = nl.ndarray((P, F), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o, data1=a, data2=u, op=nl.multiply)

        # Store
        nisa.dma_copy(dst=out[n], src=o)
    return out
