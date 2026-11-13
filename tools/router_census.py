"""Router decisions vs fp32 golden at every teacher-forced decode position, per layer.

Computes (per build, per layer L, per position p; golden row = plen - 1 + p):
    flip = top-8 expert set != golden top-8 set
    flip rate per layer and overall; NKI - XLA flip rate in pp (positions valid in both)
    golden 8th-9th router-logit gap at every flip; rank AUC of (-gap) as a flip predictor; flip rate by gap decile
    share of each layer's MoE-output squared error (vs golden) carried by flipped positions
    validity per TKG bucket (capture fallbacks excluded)
Expert sets: XLA uses its returned expert_index; library MoE uses topk(returned router logits, 8) (the fused
kernel returns no indices). XLA's topk(logits) vs its expert_index is reported as a check of that method.

Usage:
    router_census.py --builds libmoe=~/runs/router/libmoe xla=~/runs/router/xla [--moe-golden ~/runs/router/golden]
"""
import argparse, glob, json, os
import torch

H = os.path.expanduser("~")
BUCKETS = [128, 256, 512, 768]
N_LAYERS = 48


def bucket(pos):
    return next(b for b in BUCKETS if pos < b)


def auc(score, label):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = torch.cat([pos, neg])
    r = allv.argsort().argsort().double() + 1
    # average ranks for ties
    for v in allv.unique():
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--golden-short", default=f"{H}/runs/golden_v2")
    ap.add_argument("--golden-long", default=f"{H}/runs/golden_long")
    ap.add_argument("--moe-golden", default=f"{H}/runs/router/golden")
    ap.add_argument("--census", default=f"{H}/runs/tf")
    ap.add_argument("--out", default=f"{H}/runs/router/router_census.json")
    a = ap.parse_args()
    builds = dict(b.split("=", 1) for b in a.builds)
    rows = {b: [] for b in builds}  # per (prompt, pos): flips [48], valid, gap [48], err [48], err_flip...
    check = {"xla_topk_logits_eq_index": [0, 0]}
    per_bucket = {b: {} for b in builds}
    files = sorted(glob.glob(f"{next(iter(builds.values()))}/router_p*.pt"))
    for rf in files:
        nn = int(os.path.basename(rf)[len("router_p"):-3])
        gdir = a.golden_short if nn < 20 else a.golden_long
        g = torch.load(f"{gdir}/golden_p{nn:02d}.pt")
        gt, gi = g["router_top"].float(), g["router_ids"].long()  # [48, T, 9]
        mg = f"{a.moe_golden}/dump_p{nn:02d}.pt"
        gmoe = torch.load(mg)["moe_out"] if os.path.exists(mg) else None  # [48, n, 2048]
        for b, d in builds.items():
            c = torch.load(f"{d}/router_p{nn:02d}.pt")
            plen, valid = c["plen"], c["valid"]
            n = len(valid)
            for p in range(n):
                bk = bucket(plen + p - 1) if p > 0 else "cte"
                vb = per_bucket[b].setdefault(str(bk), [0, 0])
                vb[1] += 1
                if not valid[p]:
                    continue
                vb[0] += 1
                row = plen - 1 + p
                gset = gi[:, row, :8]
                gap = gt[:, row, 7] - gt[:, row, 8]
                if c["expert_index"] is not None:
                    ids = c["expert_index"][p]
                    tk = c["router_logits"][p].topk(8).indices
                    for L in range(N_LAYERS):
                        check["xla_topk_logits_eq_index"][1] += 1
                        check["xla_topk_logits_eq_index"][0] += int(set(tk[L].tolist()) == set(ids[L].tolist()))
                else:
                    ids = c["router_logits"][p].topk(8).indices
                flips = torch.tensor([set(ids[L].tolist()) != set(gset[L].tolist()) for L in range(N_LAYERS)])
                n_missing = torch.tensor([8 - len(set(ids[L].tolist()) & set(gset[L].tolist())) for L in range(N_LAYERS)])
                err = ((c["moe_out"][p] - gmoe[:, p]).pow(2).sum(-1)) if gmoe is not None else None
                rows[b].append({"prompt": nn, "pos": p, "flips": flips, "n_missing": n_missing, "gap": gap, "err": err})
    res = {"check": {k: f"{v[0]}/{v[1]}" for k, v in check.items()}, "validity_by_bucket": per_bucket, "builds": {}}
    key = lambda r: (r["prompt"], r["pos"])
    common = set.intersection(*[set(map(key, rows[b])) for b in builds]) if len(builds) > 1 else set(map(key, rows[next(iter(builds))]))
    for b in builds:
        rr = [r for r in rows[b] if key(r) in common]
        if not rr:
            continue
        F = torch.stack([r["flips"] for r in rr]).double()  # [P, 48]
        G = torch.stack([r["gap"] for r in rr])
        s = {"n_positions": len(rr), "flip_rate_overall_pct": float(F.mean() * 100),
             "flip_rate_per_layer_pct": [round(float(x) * 100, 3) for x in F.mean(0)],
             "positions_with_any_flip_pct": float((F.sum(1) > 0).double().mean() * 100),
             "flips_total": int(F.sum()), "gap_at_flips_median": float(G[F.bool()].median()) if F.sum() else None,
             "gap_all_median": float(G.median()),
             "auc_neg_gap_predicts_flip": auc(-G.flatten(), F.bool().flatten())}
        dec = torch.quantile(G.flatten(), torch.linspace(0, 1, 11))
        s["flip_rate_by_gap_decile_pct"] = [round(float(F.flatten()[(G.flatten() >= lo) & (G.flatten() <= hi)].mean() * 100), 3)
                                            for lo, hi in zip(dec[:-1], dec[1:])]
        if rr[0]["err"] is not None:
            E = torch.stack([r["err"] for r in rr])  # [P, 48]
            share = (E * F).sum(0) / E.sum(0)
            s["moe_err_share_flipped_per_layer"] = [round(float(x), 3) for x in share]
            s["moe_err_share_flipped_overall"] = float((E * F).sum() / E.sum())
            s["flipped_fraction_of_layer_positions"] = float(F.mean())
        res["builds"][b] = s
    if "libmoe" in res["builds"] and "xla" in res["builds"]:
        lb, xb = res["builds"]["libmoe"], res["builds"]["xla"]
        res["nki_minus_xla_pp"] = {"overall": lb["flip_rate_overall_pct"] - xb["flip_rate_overall_pct"],
                                   "per_layer": [round(x - y, 3) for x, y in zip(lb["flip_rate_per_layer_pct"], xb["flip_rate_per_layer_pct"])],
                                   "gate_pp": 0.5}
        res["nki_minus_xla_pp"]["max_layer"] = max(range(N_LAYERS), key=lambda L: res["nki_minus_xla_pp"]["per_layer"][L])
        res["nki_minus_xla_pp"]["layers_over_gate"] = [L for L, v in enumerate(res["nki_minus_xla_pp"]["per_layer"]) if v > 0.5]
    # worst-KL position from the day-3 census
    cr = json.load(open(f"{a.census}/libmoe/census_rows.json"))
    w = max(cr, key=lambda r: r["kl"])
    wpos = {"prompt": w["prompt"], "pos": w["pos"], "kl_libmoe": w["kl"], "kl_xla": w["xla_kl"]}
    for b in builds:
        m = [r for r in rows[b] if key(r) == (w["prompt"], w["pos"])]
        if m:
            wpos[f"{b}_flip_layers"] = [L for L in range(N_LAYERS) if m[0]["flips"][L]]
            wpos["golden_min_gap"] = float(m[0]["gap"].min())
            wpos["golden_min_gap_layer"] = int(m[0]["gap"].argmin())
    res["worst_kl_position"] = wpos
    json.dump(res, open(a.out, "w"), indent=1)
    out = {k: v for k, v in res.items() if k != "builds"}
    out["builds"] = {b: {k: v for k, v in s.items() if "per_layer" not in k} for b, s in res["builds"].items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
