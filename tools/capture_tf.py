"""Teacher-forced decode capture on a loaded NxDI build (one model per process).

Computes:
    tf:     prefill on the prompt, then decode steps that consume the reference token at every step
            (the on-device sampled token is overwritten in the adapter's forward); saves the per-step
            output scores, same generate() path as the free-running captures
    free:   free-running greedy capture of extra prompts (reference tokens for later tf runs)
    single: at each gross position p >= 1, prefill on prompt + ref[:p-1], then exactly one decode step
            on ref[p-1]; compares the step's output with the tf output at p

Returns (under --out):
    tf_<tag>/tf_pNN.pt       {"logits": [N, 1, V], "sequences": ref sequence}
    free/expected_pNN.pt     same format as ~/runs/val/expected_pNN.pt
    single.json              per gross position: KL(golden||single), KL(golden||tf), max|single - tf|

Usage (precedence: --free, then each --tf, then --single):
    capture_tf.py KIND COMPILED_DIR OUT [--free PROMPTS.json --free-new 64 --free-start 20]
                  [--tf TAG=REF_DIR ...] [--single TAG=GOLDEN_DIR=REF_DIR ...]
"""
import argparse, glob, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/runs/val"))
import common as C  # noqa: E402

GROSS_KL, XLA_KL = 0.2, 0.01


class Forcer:
    """Wraps adapter.forward; replaces the sampled token with ref[step] when a reference is set."""

    def __init__(self, adapter, model):
        self.orig, self.ref, self.step, self.model = adapter.forward, None, 0, model
        adapter.forward = self

    def __call__(self, *a, **k):
        out = self.orig(*a, **k)
        if self.ref is not None:
            t = torch.full_like(out.tokens, int(self.ref[self.step]))
            out["tokens"] = t
            out.tokens = t
        self.step += 1
        return out

    def run(self, adapter, gen_cfg, ids, n_new, ref=None):
        self.ref, self.step = ref, 0
        attn = torch.ones_like(ids)
        self.model.reset()
        with torch.inference_mode():
            o = adapter.generate(input_ids=ids, attention_mask=attn, max_new_tokens=n_new, min_new_tokens=n_new,
                                 do_sample=False, return_dict_in_generate=True, output_scores=True,
                                 generation_config=gen_cfg)
        self.ref = None
        return o.sequences, torch.stack(o.scores).float()


def kl(g, b):
    keep = torch.isfinite(g) & torch.isfinite(b)
    lg, lb = F.log_softmax(g[keep].double(), -1), F.log_softmax(b[keep].double(), -1)
    return float((lg.exp() * (lg - lb)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--free")
    ap.add_argument("--free-new", type=int, default=64)
    ap.add_argument("--free-start", type=int, default=20)
    ap.add_argument("--tf", nargs="*", default=[])
    ap.add_argument("--single", nargs="*", default=[])
    ap.add_argument("--max-single", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    if not os.path.exists(f"{a.cdir}/weights/tp0_sharded_checkpoint.safetensors"):
        C.FLAGS = [x for x in C.FLAGS if x != "--save-sharded-checkpoint"]  # shard on load
    model, tok, gen_cfg = C.load(a.kind, a.cdir)
    adapter = HuggingFaceGenerationAdapter(model)
    f = Forcer(adapter, model)

    if a.free:
        os.makedirs(f"{a.out}/free", exist_ok=True)
        for j, p in enumerate(json.load(open(a.free))):
            ids = tok([p], return_tensors="pt").input_ids
            seq, lg = f.run(adapter, gen_cfg, ids, a.free_new)
            i = a.free_start + j
            torch.save({"prompt": p, "input_ids": ids, "attention_mask": torch.ones_like(ids), "sequences": seq,
                        "logits": lg}, f"{a.out}/free/expected_p{i:02d}.pt")
            print(f"FREE p{i:02d} plen={ids.shape[1]} n_new={lg.shape[0]}", flush=True)

    for spec in a.tf:
        tag, rdir = spec.split("=", 1)
        od = f"{a.out}/tf_{tag}"
        os.makedirs(od, exist_ok=True)
        for rf in sorted(glob.glob(f"{rdir}/expected_p*.pt")):
            t0 = time.time()
            e = torch.load(rf)
            plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
            ref = e["sequences"][0, plen:plen + n]
            seq, lg = f.run(adapter, gen_cfg, e["input_ids"], n, ref=ref)
            assert torch.equal(seq[0, plen:], ref), "forced tokens not applied"
            name = os.path.basename(rf).replace("expected_", "tf_")
            torch.save({"logits": lg, "sequences": seq}, f"{od}/{name}")
            print(f"TF {tag} {name} plen={plen} n={n} {time.time() - t0:.1f}s", flush=True)

    rows = []
    for spec in a.single:
        tag, gdir, rdir = spec.split("=", 2)
        tfdir = f"{a.out}/tf_{tag}"
        cands = []
        for rf in sorted(glob.glob(f"{rdir}/expected_p*.pt")):
            nn = os.path.basename(rf)[len("expected_p"):-3]
            gf, tf = f"{gdir}/golden_p{nn}.pt", f"{tfdir}/tf_p{nn}.pt"
            if not (os.path.exists(gf) and os.path.exists(tf)):
                continue
            e, g, t = torch.load(rf), torch.load(gf)["logits"], torch.load(tf)["logits"][:, 0]
            x = e["logits"][:, 0].float()
            for p in range(1, min(len(g), len(t))):
                kb = kl(g[p], t[p])
                if kb >= GROSS_KL and kl(g[p], x[p]) < XLA_KL:
                    cands.append((kb, rf, int(nn), p))
        for kb, rf, i, p in sorted(cands, reverse=True)[: a.max_single]:
            e = torch.load(rf)
            plen = e["input_ids"].shape[1]
            seqp = e["sequences"][:, : plen + p - 1]
            ref = e["sequences"][0, plen + p - 1: plen + p]
            _, lg = f.run(adapter, gen_cfg, seqp, 2, ref=torch.cat([ref, ref]))
            s = lg[1, 0]
            g = torch.load(f"{gdir}/golden_p{i:02d}.pt")["logits"][p]
            t = torch.load(f"{tfdir}/tf_p{i:02d}.pt")["logits"][p, 0]
            keep = torch.isfinite(s) & torch.isfinite(t)
            rows.append({"prompt": i, "pos": p, "kl_golden_tf": kb, "kl_golden_single": kl(g, s),
                         "max_abs_single_vs_tf": float((s[keep] - t[keep]).abs().max()),
                         "single_top1": int(s.argmax()), "tf_top1": int(t.argmax()), "golden_top1": int(g.argmax())})
            print("SINGLE", json.dumps(rows[-1]), flush=True)
    if a.single:
        json.dump(rows, open(f"{a.out}/single.json", "w"), indent=1)
    print("CAPTURE_TF_DONE", flush=True)


if __name__ == "__main__":
    main()
