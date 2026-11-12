"""Does a build's teacher-forced output on a target prompt depend on the generate() calls run before it?

Computes (one process, one build), for each (history prompt H, target prompt T, position P) pair and each history:
    run the history, then teacher-force T and record KL(golden||build) at P, the number of gross positions
    in T, and whether T's output is bit-identical to the fresh run
Histories (in this order, each followed by a teacher-forced T):
    fresh            nothing
    tf_H             teacher-forced H (one generate, prompt prefill + 128 decode steps)
    free_H           free-running H from its prompt (day-1 first pass)
    prefill_long_H   one generate with prefill on H's prompt + 126 reference tokens, 1 decode step
    validate_H       day-1 validate.py sequence on H: logit_validation at repo tol_map and at rtol 1e-2
    again            nothing (does the effect persist into a second T run?)

Usage:
    history_probe.py KIND COMPILED_DIR OUT.json --pairs H:T:P,...
"""
import argparse, ast, json, os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/runs/val"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from capture_tf import Forcer, kl, GROSS_KL, XLA_KL  # noqa: E402

H = os.path.expanduser("~")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    if not os.path.exists(f"{a.cdir}/weights/tp0_sharded_checkpoint.safetensors"):
        C.FLAGS = [x for x in C.FLAGS if x != "--save-sharded-checkpoint"]
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    from torch_neuronx.testing.validation import logit_validation
    model, tok, gen_cfg = C.load(a.kind, a.cdir)
    adapter = HuggingFaceGenerationAdapter(model)
    f = Forcer(adapter, model)
    E = lambda i: torch.load(f"{H}/runs/val/expected_p{i:02d}.pt")
    G = lambda i: torch.load(f"{H}/runs/golden_v2/golden_p{i:02d}.pt")["logits"]

    def tf(i):
        e = E(i)
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        return f.run(adapter, gen_cfg, e["input_ids"], n, ref=e["sequences"][0, plen:plen + n])[1][:, 0]

    def validate(i):
        e = E(i)
        exp = e["logits"]
        plen, total = e["input_ids"].shape[1], e["input_ids"].shape[1] + exp.shape[0]

        def gfn(ids):
            n = ids.shape[1]
            model.reset()
            with torch.inference_mode():
                o = adapter.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=total - n,
                                     min_new_tokens=total - n, do_sample=False, return_dict_in_generate=True,
                                     output_scores=True, generation_config=gen_cfg)
            return torch.stack(o.scores)
        for tol in (C.REPO_TOL, C.RTOL1E2):
            try:
                logit_validation(input_ids=e["input_ids"], generate_fn=gfn, expected_logits=exp,
                                 tol_map=ast.literal_eval(tol), divergence_difference_tol=0.001)
            except Exception as ex:  # same handling as validate.py
                print(f"logit_validation exception {type(ex).__name__}", flush=True)

    histories = {
        "fresh": lambda h: None,
        "tf_H": lambda h: tf(h),
        "free_H": lambda h: f.run(adapter, gen_cfg, E(h)["input_ids"], E(h)["logits"].shape[0]),
        "prefill_long_H": lambda h: f.run(adapter, gen_cfg, E(h)["sequences"][:, :E(h)["input_ids"].shape[1] + 126], 2),
        "validate_H": validate,
        "again": lambda h: None,
    }
    rows = []
    for spec in a.pairs.split(","):
        h, t, p = map(int, spec.split(":"))
        g, x = G(t), E(t)["logits"][:, 0].float()
        base = None
        for name, fn in histories.items():
            t0 = time.time()
            fn(h)
            b = tf(t)
            n = min(len(b), len(g))
            gross = [q for q in range(n) if kl(g[q], b[q]) >= GROSS_KL and kl(g[q], x[q]) < XLA_KL]
            if base is None:
                base = b.clone()
            row = {"history_prompt": h, "target": t, "pos": p, "history": name, "kl_at_pos": kl(g[p], b[p]),
                   "n_gross": len(gross), "gross_positions": gross, "bit_identical_to_fresh": bool(torch.equal(b, base)),
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print("PROBE", json.dumps(row), flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print("HISTORY_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
