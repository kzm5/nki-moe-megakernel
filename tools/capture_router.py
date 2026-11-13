"""Teacher-forced capture of each layer's router decision and MoE output, from the model's own MoE modules.

Computes:
    Compiles (if COMPILED_DIR has no model.pt) and loads KIND with return_router_logits (and, for XLA,
    return_expert_index); MoE.forward is wrapped at runtime to register its router outputs as manual
    capture tensors, and each layer's mlp output is module-captured. Then every reference prompt is
    teacher-forced and the captures at every step are kept (last row = current token).

Returns (OUT/router_pNN.pt):
    router_logits [n, 48, 128] (bf16 values as returned), expert_index [n, 48, 8] (XLA only, else None),
    moe_out [n, 48, 2048], scores [n, V], valid [n] (False where the step returned capture fallbacks)

Note:
    The fused NKI MoE path does not return expert indices (NxD returns an empty placeholder), so for
    library MoE the kernel's top-8 must be derived from its returned router logits.

Usage:
    capture_router.py KIND COMPILED_DIR OUT [--ref-short ~/runs/val] [--ref-long ~/runs/val_long]
"""
import argparse, glob, importlib, os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/runs/val"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from capture_tf import Forcer  # noqa: E402

N_LAYERS = 48
MODULES = [f"layers.{i}.mlp" for i in range(N_LAYERS)]


def patch_moe(with_index):
    from neuronx_distributed.modules.moe import model as moe_model
    from neuronx_distributed.utils.tensor_capture.api import register_tensor
    orig, order = moe_model.MoE.forward, {}

    def forward(self, *a, **k):
        out = orig(self, *a, **k)
        i = order.setdefault(id(self), len(order))
        register_tensor(f"L{i:02d}.router_logits", out[1])
        if with_index:
            register_tensor(f"L{i:02d}.expert_index", out[2])
        return out
    moe_model.MoE.forward = forward


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
    with_index = kind == "xla"
    args.return_router_logits, args.return_expert_index = True, with_index
    args.tensor_capture_config = {"modules_to_capture": MODULES,
                                  "max_intermediate_tensors": N_LAYERS * (2 if with_index else 1)}
    patch_moe(with_index)
    cls = getattr(importlib.import_module(mod), "NeuronQwen3MoeForCausalLM")
    t0 = time.time()
    model, tok, gen_cfg = C.M.prepare_inference(cls, args)
    print(f"LOADED {kind} {time.time() - t0:.0f}s", flush=True)
    return model, tok, gen_cfg, with_index


class RouterForcer(Forcer):
    def __init__(self, adapter, model):
        super().__init__(adapter, model)
        self.got = []

    def __call__(self, *a, **k):
        out = super().__call__(*a, **k)
        self.got.append([t.detach().cpu().clone() for t in out.captured_tensors])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--ref-short", default=os.path.expanduser("~/runs/val"))
    ap.add_argument("--ref-long", default=os.path.expanduser("~/runs/val_long"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    model, tok, gen_cfg, with_index = load(a.kind, a.cdir)
    adapter = HuggingFaceGenerationAdapter(model)
    f = RouterForcer(adapter, model)
    L = N_LAYERS
    for rf in sorted(glob.glob(f"{a.ref_short}/expected_p*.pt")) + sorted(glob.glob(f"{a.ref_long}/expected_p*.pt")):
        e = torch.load(rf)
        nn = int(os.path.basename(rf)[len("expected_p"):-3])
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        f.got = []
        _, scores = f.run(adapter, gen_cfg, e["input_ids"], n, ref=e["sequences"][0, plen:plen + n])
        rl = torch.zeros(n, L, 128); mo = torch.zeros(n, L, 2048)
        ei = torch.zeros(n, L, 8, dtype=torch.long) if with_index else None
        valid = torch.zeros(n, dtype=torch.bool)
        for s, caps in enumerate(f.got[:n]):
            mod, man = caps[:L], caps[L:]
            if mod[0].numel() <= 1 or not man or man[0].numel() <= 1:
                continue  # capture fallback in this bucket
            valid[s] = True
            k = 2 if with_index else 1  # manual tensors are registered per layer: logits[, index]
            for i in range(L):
                mo[s, i] = mod[i].reshape(-1, 2048)[-1].float()
                rl[s, i] = man[k * i].reshape(-1, 128)[-1].float()
                if with_index:
                    ei[s, i] = man[k * i + 1].reshape(-1, 8)[-1].long()
        torch.save({"router_logits": rl, "expert_index": ei, "moe_out": mo, "scores": scores[:, 0], "valid": valid,
                    "plen": plen, "n_captures": len(f.got[0])}, f"{a.out}/router_p{nn:02d}.pt")
        print(f"ROUTER p{nn:02d} plen={plen} n={n} valid={int(valid.sum())}/{n} n_captures={len(f.got[0])}", flush=True)
    print("CAPTURE_ROUTER_DONE", flush=True)


if __name__ == "__main__":
    main()
