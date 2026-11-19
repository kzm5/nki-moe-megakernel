"""On-device check of the skill-audit kernels: bf16 bit-identity of swiglu_kernel_opt vs swiglu_kernel, and
wall time per call (torch-xla, single NeuronCore).

Usage:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=0 python device_check.py
"""
import importlib.util, json, os, sys, time
import torch
import torch_xla.core.xla_model as xm

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.swiglu


def main():
    dev = xm.xla_device()
    res = {}
    for shape in [(4, 128, 512), (8, 128, 2048)]:
        g = torch.Generator().manual_seed(0)
        gate = (torch.randn(shape, generator=g) * 4).bfloat16()
        up = (torch.randn(shape, generator=g) * 4).bfloat16()
        gd, ud = gate.to(dev), up.to(dev)
        outs, times = {}, {}
        for name in ("swiglu_kernel", "swiglu_kernel_opt"):
            k = load(name)
            o = k(gd, ud)
            xm.mark_step(); xm.wait_device_ops()
            outs[name] = o.cpu()
            t0 = time.time()
            for _ in range(50):
                o = k(gd, ud)
                xm.mark_step()
            xm.wait_device_ops()
            times[name] = (time.time() - t0) / 50 * 1e6
        a, b = outs["swiglu_kernel"], outs["swiglu_kernel_opt"]
        ref = torch.nn.functional.silu(gate.float()).bfloat16().float() * up.float()
        res[str(shape)] = {"dtype": str(a.dtype), "bit_identical": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
                           "mismatches": int((a.view(torch.int16) != b.view(torch.int16)).sum()),
                           "max_abs_vs_ref_orig": float((a.float() - ref).abs().max()),
                           "us_per_call_orig": times["swiglu_kernel"], "us_per_call_opt": times["swiglu_kernel_opt"]}
        print(shape, json.dumps(res[str(shape)]), flush=True)
    json.dump(res, open(f"{HERE}/device_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
