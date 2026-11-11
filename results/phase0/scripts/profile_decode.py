"""Profile single decode (TKG) steps. Usage: profile_decode.py KIND COMPILED_DIR OUT_DIR SEQ_LEN
Two inspect windows (device profile mode "model", NEFF + NTFF dumped): TKG call 6 (smallest bucket)
and a call at position SEQ_LEN-4 (largest bucket)."""
import ctypes, glob, os, sys, time
kind, cdir, out, seq = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.expanduser("~/runs/val"))
import common as C  # noqa: E402
C.FLAGS = ["--seq-len", seq, "--on-device-sampling", "--top-k", "1", "--do-sample", "",
           "--output-logits", "--skip-compile"]  # no pre-shard: weights shard on load
import torch  # noqa: E402
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter  # noqa: E402

model, tok, gen_cfg = C.load(kind, cdir)
try:
    os.makedirs(os.path.join(out, "neffs"), exist_ok=True)
    for i, spmd in enumerate(model.traced_model.nxd_model.models.token_generation_model.models):
        p = os.path.join(out, "neffs", f"tkg_bucket{i}.neff")
        print(f"NEFF_DUMP bucket{i} ok={spmd.save_neff(p)} {os.path.getsize(p) if os.path.exists(p) else 0}", flush=True)
except Exception as ex:  # keep profiling even if the dump path changes
    print(f"NEFF_DUMP_FAILED {type(ex).__name__}: {ex}", flush=True)
libpath = next(l.split()[-1] for l in open("/proc/self/maps") if "libnrt.so" in l)
nrt = ctypes.CDLL(libpath)
enc = tok([C.prompts()[0]], return_tensors="pt")
plen = enc.input_ids.shape[1]
n_new = int(seq) - plen - 2
# TKG call k (1-based) consumes the token at position plen + k - 1
windows = {6: "small_bucket", int(seq) - 4 - plen + 1: "large_bucket"}
print(f"LIBNRT {libpath} plen={plen} n_new={n_new} windows={windows}", flush=True)
tkg = model.token_generation_model
orig = tkg.forward
calls = {"n": 0}


def wrapped(*a, **k):
    calls["n"] += 1
    tag = windows.get(calls["n"])
    if tag is None:
        return orig(*a, **k)
    os.environ.update({"NEURON_RT_INSPECT_ENABLE": "1", "NEURON_RT_INSPECT_DEVICE_PROFILE": "model",
                       "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
                       "NEURON_RT_INSPECT_OUTPUT_DIR": os.path.join(out, tag)})
    print(f"INSPECT_BEGIN {tag} call={calls['n']} rc={nrt.nrt_inspect_begin()}", flush=True)
    t0 = time.perf_counter()
    r = orig(*a, **k)
    print(f"PROFILED {tag} host_ms={(time.perf_counter() - t0) * 1e3:.2f}", flush=True)
    print(f"INSPECT_STOP {tag} rc={nrt.nrt_inspect_stop()}", flush=True)
    return r


tkg.forward = wrapped
adapter = HuggingFaceGenerationAdapter(model)
model.reset()
with torch.inference_mode():
    adapter.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=n_new,
                     min_new_tokens=n_new, do_sample=False, generation_config=gen_cfg)
time.sleep(5)
tot = 0
for f in sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True)):
    if os.path.isfile(f):
        s = os.path.getsize(f); tot += s
        print(f"FILE {s / 1e6:10.1f} MB {f}")
print(f"TKG_CALLS {calls['n']} TOTAL_MB {tot / 1e6:.1f}")
print("PROFILE_DONE")
