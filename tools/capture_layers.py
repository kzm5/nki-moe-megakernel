"""Per-layer decode tensors from an NxDI build compiled with tensor capture (teacher-forced).

Computes:
    Compiles (if COMPILED_DIR has no model.pt) and loads KIND with tensor_capture_config capturing, per layer,
    output and input of input_layernorm, self_attn.o_proj and mlp; then teacher-forces each target prompt
    on its reference tokens and keeps the captured tensors at the target decode steps.

Returns (OUT/cap_pNN.pt):
    {"positions": [p...], "tensors": {p: [n_capture] tensors}, "names": capture names, "logits": {p: [V]}}

Usage:
    capture_layers.py KIND COMPILED_DIR OUT --targets NN:P,... [--ref-short DIR] [--ref-long DIR]
"""
import argparse, importlib, os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/runs/val"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from capture_tf import Forcer  # noqa: E402

N_LAYERS = 48
MODULES = [f"layers.{i}.{m}" for i in range(N_LAYERS) for m in ("input_layernorm", "self_attn.o_proj", "mlp")]
NAMES = [f"{m}.{io}" for m in MODULES for io in ("out", "in")]


def load(kind, cdir):
    extra, mod = C.KINDS[kind]
    flags = ["--seq-len", "768", "--on-device-sampling", "--top-k", "1", "--do-sample", "", "--output-logits"]
    if os.path.exists(f"{cdir}/model.pt"):
        flags.append("--skip-compile")
    sys.argv = ["main.py", "--model", "qwen3_moe", "--platform-target", "trn2",
                "--model-path", os.path.expanduser("~/qwen-30b-a3b/hf_model"), "--compiled-model-path", cdir,
                "--tp-degree", "4", "--logical-nc-config", "2"] + flags + extra
    args = C.M.parse_args()
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.platform_target
    args.prompts, args.batch_size, args.max_length, args.draft_model_cls = ["x"], 1, args.seq_len, None
    args.tensor_capture_config = {"modules_to_capture": MODULES, "capture_inputs": True}
    cls = getattr(importlib.import_module(mod), "NeuronQwen3MoeForCausalLM")
    t0 = time.time()
    model, tok, gen_cfg = C.M.prepare_inference(cls, args)
    print(f"LOADED {kind} {time.time() - t0:.0f}s capture={model.neuron_config.tensor_capture_config is not None}", flush=True)
    return model, tok, gen_cfg


class CapForcer(Forcer):
    def __init__(self, adapter, model):
        super().__init__(adapter, model)
        self.want, self.got = set(), {}

    def __call__(self, *a, **k):
        step = self.step
        out = super().__call__(*a, **k)
        if step in self.want:
            self.got[step] = ([t.detach().float().cpu().clone() for t in out.captured_tensors],
                              out.logits[0, -1].float().cpu().clone())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--ref-short", default=os.path.expanduser("~/runs/val"))
    ap.add_argument("--ref-long", default=os.path.expanduser("~/runs/val_long"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = {}
    for t in a.targets.split(","):
        nn, p = t.split(":")
        want.setdefault(int(nn), []).append(int(p))
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    model, tok, gen_cfg = load(a.kind, a.cdir)
    adapter = HuggingFaceGenerationAdapter(model)
    f = CapForcer(adapter, model)
    for nn, ps in sorted(want.items()):
        rf = f"{a.ref_short if nn < 20 else a.ref_long}/expected_p{nn:02d}.pt"
        e = torch.load(rf)
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        f.want, f.got = set(ps), {}
        f.run(adapter, gen_cfg, e["input_ids"], n, ref=e["sequences"][0, plen:plen + n])
        tens = {p: f.got[p][0] for p in ps}
        torch.save({"positions": ps, "tensors": tens, "names": NAMES, "logits": {p: f.got[p][1] for p in ps},
                    "shapes": {p: [tuple(t.shape) for t in tens[p]] for p in ps}}, f"{a.out}/cap_p{nn:02d}.pt")
        print(f"CAPTURED p{nn:02d} positions={ps} n_tensors={len(tens[ps[0]])} shape0={tuple(tens[ps[0]][0].shape)}", flush=True)
    print("CAPTURE_LAYERS_DONE", flush=True)


if __name__ == "__main__":
    main()
