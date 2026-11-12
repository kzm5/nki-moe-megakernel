"""Why did day-1 first divergences look like gross errors? Audit the step before each one.

Computes (per build, per day-1 first divergence p from results_<build>.json):
    whether the build's free-running emitted token at any step q < p differs from XLA's emitted token
    (i.e. the contexts were no longer identical at p), and whether that step was an exact top-2 tie in
    the build's scores, in XLA's scores, or neither
Also counts exact top-2 ties in XLA's own scores and where XLA's emitted token != its own argmax.

Usage:
    tie_audit.py [--val ~/runs/val] [--free ~/runs/replay] [--out ~/runs/triage/tie_audit.json]
"""
import argparse, json, os
import torch

H = os.path.expanduser("~")


def tie(v):
    t = v.float().topk(2).values
    return bool(t[0] == t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=f"{H}/runs/val")
    ap.add_argument("--free", default=f"{H}/runs/replay")
    ap.add_argument("--triage", default=f"{H}/runs/triage/triage.json")
    ap.add_argument("--out", default=f"{H}/runs/triage/tie_audit.json")
    a = ap.parse_args()
    cls = {(r["build"], r["prompt"], r["pos"]): r["class"] for r in json.load(open(a.triage))["rows"]}
    xla_ties = xla_tok_ne_argmax = 0
    X = {}
    for i in range(20):
        e = torch.load(f"{a.val}/expected_p{i:02d}.pt")
        plen = e["input_ids"].shape[1]
        lg, tok = e["logits"][:, 0], e["sequences"][0, plen:plen + e["logits"].shape[0]]
        X[i] = (lg, tok)
        xla_ties += sum(tie(lg[q]) for q in range(len(lg)))
        xla_tok_ne_argmax += int((lg.argmax(-1) != tok).sum())
    rows = []
    for b in ("libmoe", "nki"):
        fr = f"{a.free}/free_{b}.json"
        if not os.path.exists(fr):
            continue
        for r in json.load(open(f"{a.val}/results_{b}.json")):
            i, p = r["prompt_idx"], r["first_divergence"]
            if p < 0:
                continue
            d = torch.load(f"{a.free}/free_{b}_p{i:02d}.pt")
            plen = X[i][1].shape[0]
            em = d["sequences"][0, -d["logits"].shape[0]:]
            lg = d["logits"][:, 0]
            xl, xt = X[i]
            diff = (em[:p] != xt[:p]).nonzero()
            q = int(diff[0, 0]) if len(diff) else -1
            rows.append({"build": b, "prompt": i, "day1_first_div": p, "day1_class": cls.get((b, i, p)),
                         "token_divergence_before_p": q, "contexts_identical_at_p": q < 0,
                         "tie_in_build_at_q": tie(lg[q]) if q >= 0 else None,
                         "tie_in_xla_at_q": tie(xl[q]) if q >= 0 else None,
                         "build_emitted_ne_own_argmax_at_q": bool(em[q] != lg[q].argmax()) if q >= 0 else None,
                         "xla_emitted_ne_own_argmax_at_q": bool(xt[q] != xl[q].argmax()) if q >= 0 else None})
    summ = {"xla_exact_top2_ties": xla_ties, "xla_emitted_ne_argmax": xla_tok_ne_argmax}
    for b in ("libmoe", "nki"):
        for c in ("gross", "benign"):
            rr = [r for r in rows if r["build"] == b and r["day1_class"] == c]
            if rr:
                summ[f"{b}_{c}"] = {"n": len(rr), "contexts_identical": sum(r["contexts_identical_at_p"] for r in rr),
                                    "token_diverged_earlier": sum(not r["contexts_identical_at_p"] for r in rr),
                                    "of_which_tie_in_build": sum(bool(r["tie_in_build_at_q"]) for r in rr),
                                    "of_which_tie_in_xla": sum(bool(r["tie_in_xla_at_q"]) for r in rr)}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
