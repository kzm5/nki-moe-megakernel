import sys, numpy as np, ml_dtypes, nki
sys.path.insert(0, ".")
from swiglu_kernel import swiglu as base, reference
from swiglu_kernel_opt import swiglu as opt

for shape, seed in [((4, 128, 512), 0), ((8, 128, 2048), 1), ((3, 128, 96), 2)]:
    rng = np.random.default_rng(seed)
    g = (rng.standard_normal(shape) * 4).astype(ml_dtypes.bfloat16)
    u = (rng.standard_normal(shape) * 4).astype(ml_dtypes.bfloat16)
    a = np.asarray(nki.simulate(base)(g, u))
    b = np.asarray(nki.simulate(opt)(g, u))
    same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint16), b.view(np.uint16))
    err = float(np.abs(b.astype(np.float32) - reference(g, u)).max())
    print(f"{shape} dtype={b.dtype} bit_identical={same} mismatches={int((a.view(np.uint16)!=b.view(np.uint16)).sum())} max_abs_err_vs_fp32_ref={err:.4g}")
