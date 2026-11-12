"""Teacher-forced census: every generated position of every prompt, per build, against fp32 golden.

Computes (per build, per position):
    KL(golden||build), KL(golden||XLA), rel-L2(build vs golden), top-1(build == golden)
    gross = KL(golden||build) >= 0.2 and KL(golden||XLA) < 0.01
Reports: gross rate (short / long prompts), KL histogram, gross positions shared by builds, rho over all
positions with and without gross positions, max per-position KL, determinism (in-process repeat pass,
cross-process vs day-1 free-running captures before first divergence), XLA tf self-check, single-step summary.

Usage:
    census.py --tf-root ~/runs/tf --builds libmoe nki [--out ~/runs/tf/census.json]
"""
import argparse, glob, json, os
import torch
import torch.nn.functional as F

GROSS_KL, XLA_KL = 0.2, 0.01
BINS = [0, 1e-3, 1e-2, 0.05, 0.2, 1, 5, float("inf")]
H = os.path.expanduser("~")


def per_pos(b, g):
    keep = torch.isfinite(b).all(0) & torch.isfinite(g).all(0)
    b, g = b[:, keep].double(), g[:, keep].double()
    lg = F.log_softmax(g, -1)
    return {"kl": (lg.exp() * (lg - F.log_softmax(b, -1))).sum(-1), "rel_l2": (b - g).norm(dim=-1) / g.norm(dim=-1),
            "top1": (b.argmax(-1) == g.argmax(-1))}


def prompts(short_ref, long_ref, short_g, long_g):
    out = []
    for rdir, gdir, kind in ((short_ref, short_g, "short"), (long_ref, long_g, "long")):
        for rf in sorted(glob.glob(f"{rdir}/expected_p*.pt")):
            nn = os.path.basename(rf)[len("expected_p"):-3]
            if os.path.exists(f"{gdir}/golden_p{nn}.pt"):
                out.append((int(nn), kind, rf, f"{gdir}/golden_p{nn}.pt"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-root", default=f"{H}/runs/tf")
    ap.add_argument("--builds", nargs="+", default=["libmoe", "nki"])
    ap.add_argument("--short-ref", default=f"{H}/runs/val")
    ap.add_argument("--long-ref", default=f"{H}/runs/val_long")
    ap.add_argument("--short-golden", default=f"{H}/runs/golden_v2")
    ap.add_argument("--long-golden", default=f"{H}/runs/golden_long")
    ap.add_argument("--out", default=f"{H}/runs/tf/census.json")
    a = ap.parse_args()
    P = prompts(a.short_ref, a.long_ref, a.short_golden, a.long_golden)
    res = {"n_prompts": len(P), "xla_self_check": {}, "builds": {}}
    # XLA self-check: teacher-forced == free-running, bit for bit
    for tag, kind in (("main", "short"), ("long", "long")):
        ok = tot = 0
        for i, k, rf, _ in P:
            f = f"{a.tf_root}/xla/tf_{tag}/tf_p{i:02d}.pt"
            if k != kind or not os.path.exists(f):
                continue
            tot += 1
            ok += int(torch.equal(torch.load(f)["logits"], torch.load(rf)["logits"]))
        res["xla_self_check"][kind] = f"{ok}/{tot} bit-identical"
    X = {i: torch.load(rf)["logits"][:, 0].float() for i, _, rf, _ in P}
    G = {i: torch.load(gf)["logits"] for i, _, _, gf in P}
    gross_sets = {}
    for bname in a.builds:
        rows, kls, xkls, rb, rx = [], [], [], [], []
        fdiv = {r["prompt_idx"]: r["first_divergence"] for r in json.load(open(f"{a.short_ref}/results_{bname}.json"))}
        det_in = [0, 0]; det_x = [0, 0, 0]
        for i, kind, rf, gf in P:
            tag = "main" if kind == "short" else "long"
            f = f"{a.tf_root}/{bname}/tf_{tag}/tf_p{i:02d}.pt"
            if not os.path.exists(f):
                continue
            b = torch.load(f)["logits"][:, 0].float()
            g, x = G[i], X[i]
            n = min(len(b), len(g), len(x))
            mb, mx = per_pos(b[:n], g[:n]), per_pos(x[:n], g[:n])
            for p in range(n):
                gross = bool(mb["kl"][p] >= GROSS_KL and mx["kl"][p] < XLA_KL)
                rows.append({"prompt": i, "kind": kind, "pos": p, "kl": float(mb["kl"][p]), "xla_kl": float(mx["kl"][p]),
                             "rel_l2": float(mb["rel_l2"][p]), "xla_rel_l2": float(mx["rel_l2"][p]),
                             "top1": bool(mb["top1"][p]), "gross": gross})
            if kind == "short":
                rep = f"{a.tf_root}/{bname}/tf_repeat/tf_p{i:02d}.pt"
                if os.path.exists(rep):
                    det_in[1] += 1
                    det_in[0] += int(torch.equal(torch.load(rep)["logits"], torch.load(f)["logits"]))
                act = f"{a.short_ref}/actual_{bname}_p{i:02d}.pt"
                fd = fdiv.get(i, -1)
                m_ = n if fd < 0 else fd + 1  # positions up to and incl. first divergence share the context
                if os.path.exists(act):
                    d1 = torch.load(act)[:m_, 0]
                    same = (b[:m_].half() == d1) | (torch.isinf(b[:m_].half()) & torch.isinf(d1))
                    det_x[0] += int(same.all(-1).sum()); det_x[1] += m_; det_x[2] += int(same.all())
        if not rows:
            continue
        kl_t = torch.tensor([r["kl"] for r in rows])
        gs = {(r["prompt"], r["pos"]) for r in rows if r["gross"]}
        gross_sets[bname] = gs
        rl, xrl = torch.tensor([r["rel_l2"] for r in rows]), torch.tensor([r["xla_rel_l2"] for r in rows])
        mask = torch.tensor([not r["gross"] for r in rows])
        s = {"n_pos": len(rows), "n_gross": len(gs), "gross_rate": len(gs) / len(rows)}
        for kind in ("short", "long"):
            rr = [r for r in rows if r["kind"] == kind]
            if rr:
                s[f"gross_{kind}"] = f"{sum(r['gross'] for r in rr)}/{len(rr)}"
        s["kl_histogram"] = {f"[{lo:g},{hi:g})": int(((kl_t >= lo) & (kl_t < hi)).sum()) for lo, hi in zip(BINS, BINS[1:])}
        xk = torch.tensor([r["xla_kl"] for r in rows])
        s["xla_kl_histogram"] = {f"[{lo:g},{hi:g})": int(((xk >= lo) & (xk < hi)).sum()) for lo, hi in zip(BINS, BINS[1:])}
        s.update(rho_all=float(rl.mean() / xrl.mean()), rho_without_gross=float(rl[mask].mean() / xrl[mask].mean()),
                 kl_mean=float(kl_t.mean()), kl_mean_without_gross=float(kl_t[mask].mean()),
                 xla_kl_mean=float(xk.mean()), max_kl=float(kl_t.max()),
                 max_kl_at=[(r["prompt"], r["pos"]) for r in rows if r["kl"] == float(kl_t.max())][0],
                 top1_golden=float(torch.tensor([r["top1"] for r in rows]).double().mean()),
                 determinism_in_process=f"{det_in[0]}/{det_in[1]} short prompts bit-identical (repeat pass)",
                 determinism_cross_process=f"{det_x[2]}/20 prompts, {det_x[0]}/{det_x[1]} positions identical to day-1 free-running (fp16) up to first divergence",
                 gross_positions=sorted(gs, key=lambda t: -next(r["kl"] for r in rows if (r["prompt"], r["pos"]) == t)))
        sj = f"{a.tf_root}/{bname}/single.json"
        if os.path.exists(sj):
            sg = json.load(open(sj))
            s["single_step"] = {"n": len(sg), "still_gross": sum(r["kl_golden_single"] >= GROSS_KL for r in sg),
                                "same_top1_as_tf": sum(r["single_top1"] == r["tf_top1"] for r in sg),
                                "max_abs_single_vs_tf_median": float(torch.tensor([r["max_abs_single_vs_tf"] for r in sg]).median()) if sg else None}
        res["builds"][bname] = s
        json.dump(rows, open(f"{a.tf_root}/{bname}/census_rows.json", "w"))
    if len(gross_sets) == 2:
        x, y = list(gross_sets.values())
        res["shared_gross"] = sorted(x & y)
        res["n_shared_gross"] = len(x & y)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v if k != "builds" else {b: {kk: vv for kk, vv in s.items() if kk != "gross_positions"} for b, s in v.items()}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
