"""Triage of first divergences against the fp32 golden (CPU, existing captures).

Computes (per build, per first-divergence position p):
    shift test: KL(build@p || golden@q) for q in {p-1, p, p+1}
    router margin: golden 8th-9th router value gap, min over the 48 layers, at the step that produces p
    absolute position of the consumed token (plen + p - 1) and its TKG bucket; near-edge flag
Classes: gross = KL(golden||build) >= 0.2 and KL(golden||XLA) < 0.01; benign otherwise.
Random baseline: router margin over all generated positions of all prompts.

Usage:
    triage_gross.py [--val ~/runs/val] [--golden ~/runs/golden_v2] [--out ~/runs/triage/triage.json]
"""
import argparse, json, os
import torch
import torch.nn.functional as F

BUCKETS = [128, 256, 512, 768]
EDGE = 2
GROSS_KL, XLA_KL = 0.2, 0.01


def kl(p_logits, q_logits):
    keep = torch.isfinite(p_logits) & torch.isfinite(q_logits)
    lp, lq = F.log_softmax(p_logits[keep].double(), -1), F.log_softmax(q_logits[keep].double(), -1)
    return float((lp.exp() * (lp - lq)).sum())


def bucket(pos):
    return next(b for b in BUCKETS if pos < b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.expanduser("~/runs/val"))
    ap.add_argument("--golden", default=os.path.expanduser("~/runs/golden_v2"))
    ap.add_argument("--out", default=os.path.expanduser("~/runs/triage/triage.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    margins, G, X, plens = [], {}, {}, {}
    for i in range(20):
        g = torch.load(f"{a.golden}/golden_p{i:02d}.pt")
        e = torch.load(f"{a.val}/expected_p{i:02d}.pt")
        plens[i] = e["input_ids"].shape[1]
        G[i], X[i] = g["logits"], e["logits"][:, 0].float()
        rt = g["router_top"]  # [L, T, 9], T = plen + 127, row t processes token t
        m = (rt[:, :, 7] - rt[:, :, 8]).min(0).values  # min over layers
        margins.append(m[plens[i] - 1:])  # rows producing generated positions 0..127
    allm = torch.cat(margins)
    rows = []
    for k in ["libmoe", "nki"]:
        for r in json.load(open(f"{a.val}/results_{k}.json")):
            i, p = r["prompt_idx"], r["first_divergence"]
            if p < 0:
                continue
            b = torch.load(f"{a.val}/actual_{k}_p{i:02d}.pt")[p, 0].float()
            g, x = G[i], X[i]
            kbg = kl(g[p], b)
            shift = {d: (kl(b, g[p + d]) if 0 <= p + d < len(g) else None) for d in (-1, 0, 1)}
            best = min((v, d) for d, v in shift.items() if v is not None)[1]
            pos = plens[i] + p - 1  # token consumed at this step (p = 0 is the prefill's last token)
            m = float(margins[i][p])
            gross = kbg >= GROSS_KL and kl(g[p], x[p]) < XLA_KL
            near = any(abs(pos - e_) <= EDGE or abs(pos + 1 - e_) <= EDGE for e_ in BUCKETS[:-1])
            rows.append({"build": k, "prompt": i, "pos": p, "class": "gross" if gross else "benign",
                         "kl_golden_build": kbg, "kl_build_golden_m1": shift[-1], "kl_build_golden_0": shift[0],
                         "kl_build_golden_p1": shift[1], "best_shift": best, "router_min_margin": m,
                         "router_margin_pctile": float((allm < m).double().mean()), "abs_pos": pos,
                         "step": "prefill" if p == 0 else "decode", "tkg_bucket": bucket(pos) if p else None,
                         "near_bucket_edge": bool(near and p > 0)})
    summ = {"router_margin_all_positions": {"n": int(allm.numel()), "median": float(allm.median()),
                                            "p10": float(allm.quantile(0.1)), "p90": float(allm.quantile(0.9))}}
    for c in ["gross", "benign"]:
        rr = [r for r in rows if r["class"] == c]
        if not rr:
            continue
        ms = torch.tensor([r["router_min_margin"] for r in rr])
        summ[c] = {"n": len(rr), "router_margin_median": float(ms.median()),
                   "router_margin_pctile_median": float(torch.tensor([r["router_margin_pctile"] for r in rr]).median()),
                   "best_shift_counts": {str(d): sum(r["best_shift"] == d for r in rr) for d in (-1, 0, 1)},
                   "near_bucket_edge": sum(r["near_bucket_edge"] for r in rr)}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for r in sorted(rows, key=lambda r: -r["kl_golden_build"]):
        print(f"{r['build']:6s} p{r['prompt']:02d}@{r['pos']:3d} {r['class']:6s} KL(g||b)={r['kl_golden_build']:7.3f} "
              f"KL(b||g[p-1,p,p+1])={r['kl_build_golden_m1'] if r['kl_build_golden_m1'] is None else round(r['kl_build_golden_m1'], 3)},"
              f"{r['kl_build_golden_0']:.3f},{r['kl_build_golden_p1'] if r['kl_build_golden_p1'] is None else round(r['kl_build_golden_p1'], 3)} "
              f"best={r['best_shift']:+d} margin={r['router_min_margin']:.4f} (pct {r['router_margin_pctile']:.2f}) "
              f"abs={r['abs_pos']} bucket={r['tkg_bucket']} edge={r['near_bucket_edge']}")


if __name__ == "__main__":
    main()
