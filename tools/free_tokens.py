"""Free-running greedy pass per prompt: emitted (on-device sampled) tokens vs argmax of the returned scores.

Computes (per prompt, fresh process, prompts in order):
    first step where the emitted token differs from XLA's reference token
    first step where the emitted token differs from the argmax of the build's own scores at that step
    first step where argmax(scores) differs from XLA's token (the day-1 "first divergence" definition)

Usage:
    free_tokens.py KIND COMPILED_DIR OUT.json [--ref ~/runs/val]
"""
import argparse, glob, json, os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/runs/val"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from capture_tf import Forcer  # noqa: E402


def first(mask):
    nz = mask.nonzero()
    return int(nz[0, 0]) if len(nz) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--ref", default=os.path.expanduser("~/runs/val"))
    a = ap.parse_args()
    if not os.path.exists(f"{a.cdir}/weights/tp0_sharded_checkpoint.safetensors"):
        C.FLAGS = [x for x in C.FLAGS if x != "--save-sharded-checkpoint"]
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    model, tok, gen_cfg = C.load(a.kind, a.cdir)
    adapter = HuggingFaceGenerationAdapter(model)
    f = Forcer(adapter, model)
    rows = []
    for rf in sorted(glob.glob(f"{a.ref}/expected_p*.pt")):
        e = torch.load(rf)
        i = int(os.path.basename(rf)[len("expected_p"):-3])
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        seq, lg = f.run(adapter, gen_cfg, e["input_ids"], n)
        emitted, ref = seq[0, plen:plen + n], e["sequences"][0, plen:plen + n]
        am = lg[:, 0].argmax(-1)
        s = {"prompt": i, "emitted_vs_xla": first(emitted != ref), "emitted_vs_own_argmax": first(emitted != am),
             "argmax_vs_xla": first(am != ref)}
        j = s["emitted_vs_own_argmax"]
        if j >= 0:
            top2 = lg[j, 0].float().topk(2)
            s.update(at=j, emitted=int(emitted[j]), own_argmax=int(am[j]), top2_values=[float(v) for v in top2.values],
                     value_of_emitted=float(lg[j, 0, emitted[j]]), xla_token=int(ref[j]))
        rows.append(s)
        print("FREE", json.dumps(s), flush=True)
        torch.save({"sequences": seq, "logits": lg}, a.out.replace(".json", f"_p{i:02d}.pt"))
    json.dump(rows, open(a.out, "w"), indent=1)
    print("FREE_TOKENS_DONE", flush=True)


if __name__ == "__main__":
    main()
