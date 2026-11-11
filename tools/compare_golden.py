"""Compare saved build logits against golden logits, position by position.

Computes (per build, per prompt, over positions before the build's first divergence from XLA):
    rel_l2 = ||b - g|| / ||g||, max_abs, KL(g || b), top-1 agreement with golden argmax
    rho    = mean rel_l2(build) / mean rel_l2(XLA) on the same positions (also rho_kl)

Note:
    Vocab columns that are -inf in any compared tensor (EOS under min_new_tokens) are dropped.
    XLA's positions are all n_new (its tokens define the teacher-forced context).
"""
import argparse, json, os
import torch
import torch.nn.functional as F

SPOTS = [(1, 15), (6, 4)]  # (prompt, position) where both NKI builds picked the same off-ranked token


def metrics(b, g):
    keep = torch.isfinite(b).all(0) & torch.isfinite(g).all(0)
    b, g = b[:, keep].double(), g[:, keep].double()
    lg, lb = F.log_softmax(g, -1), F.log_softmax(b, -1)
    return {"rel_l2": ((b - g).norm(dim=-1) / g.norm(dim=-1)), "max_abs": (b - g).abs().max(-1).values,
            "kl": (lg.exp() * (lg - lb)).sum(-1), "top1": (b.argmax(-1) == g.argmax(-1)).double()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.expanduser("~/runs/val"))
    ap.add_argument("--golden", default=os.path.expanduser("~/runs/golden"))
    ap.add_argument("--builds", nargs="+", default=["libmoe", "nki"])
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    ap.add_argument("--out", default=os.path.expanduser("~/runs/golden/compare.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    fdiv = {k: {r["prompt_idx"]: r["first_divergence"] for r in json.load(open(f"{a.val}/results_{k}.json"))}
            for k in a.builds}
    rows, spots = [], []
    acc = {k: {m: [] for m in ("rel_l2", "max_abs", "kl", "top1", "xla_rel_l2", "xla_kl")} for k in ["xla"] + a.builds}
    i = 0
    while os.path.exists(f"{a.golden}/golden_p{i:02d}.pt"):
        g = torch.load(f"{a.golden}/golden_p{i:02d}.pt")["logits"]
        e = torch.load(f"{a.val}/expected_p{i:02d}.pt")
        x = e["logits"][:, 0].float()
        n = min(g.shape[0], x.shape[0])
        mx = metrics(x[:n], g[:n])
        row = {"prompt": i, "xla": {m: float(v.mean()) for m, v in mx.items()}}
        for m in ("rel_l2", "max_abs", "kl", "top1"):
            acc["xla"][m].append(mx[m])
        for k in a.builds:
            b = torch.load(f"{a.val}/actual_{k}_p{i:02d}.pt")[:, 0].float()
            fd = fdiv[k][i]
            m_ = n if fd < 0 else min(fd, n)
            if m_ == 0:
                row[k] = {"n_pos": 0}
                continue
            mb = metrics(b[:m_], g[:m_])
            r = {mm: float(v.mean()) for mm, v in mb.items()}
            r.update(n_pos=m_, first_div=fd, max_abs=float(mb["max_abs"].max()),
                     rho=float(mb["rel_l2"].mean() / mx["rel_l2"][:m_].mean()),
                     rho_kl=float(mb["kl"].mean() / mx["kl"][:m_].mean()))
            row[k] = r
            for mm in ("rel_l2", "max_abs", "kl", "top1"):
                acc[k][mm].append(mb[mm])
            acc[k]["xla_rel_l2"].append(mx["rel_l2"][:m_])
            acc[k]["xla_kl"].append(mx["kl"][:m_])
        for (pi, pos) in SPOTS:
            if pi != i:
                continue
            gp = g[pos].float()
            order = gp.argsort(descending=True)
            rank = {int(t): int(r_) for r_, t in enumerate(order[:2000])}
            seq = e["sequences"][0]
            plen = e["input_ids"].shape[1]
            xla_tok = int(seq[plen + pos])
            nki_tok = int(torch.load(f"{a.val}/actual_nki_p{i:02d}.pt")[pos, 0].float().argmax())
            lib_tok = int(torch.load(f"{a.val}/actual_libmoe_p{i:02d}.pt")[pos, 0].float().argmax())
            sp = {"prompt": pi, "pos": pos,
                  "golden_top5": [(tok.decode([int(t)]), float(gp[t])) for t in order[:5]],
                  "xla_token": (tok.decode([xla_tok]), rank.get(xla_tok, ">2000"), float(gp[xla_tok])),
                  "libmoe_token": (tok.decode([lib_tok]), rank.get(lib_tok, ">2000"), float(gp[lib_tok])),
                  "nki_token": (tok.decode([nki_tok]), rank.get(nki_tok, ">2000"), float(gp[nki_tok]))}
            spots.append(sp)
        rows.append(row)
        i += 1
    summary = {}
    for k, d in acc.items():
        if not d["rel_l2"]:
            continue
        cat = {m: torch.cat(v) for m, v in d.items() if v}
        s = {"n_pos": int(cat["rel_l2"].numel()), "rel_l2_mean": float(cat["rel_l2"].mean()),
             "rel_l2_median": float(cat["rel_l2"].median()), "max_abs_max": float(cat["max_abs"].max()),
             "max_abs_mean": float(cat["max_abs"].mean()), "kl_mean": float(cat["kl"].mean()),
             "top1_golden": float(cat["top1"].mean())}
        if k != "xla":
            s["rho"] = float(cat["rel_l2"].mean() / cat["xla_rel_l2"].mean())
            s["rho_kl"] = float(cat["kl"].mean() / cat["xla_kl"].mean())
            s["xla_rel_l2_same_pos"] = float(cat["xla_rel_l2"].mean())
            s["xla_kl_same_pos"] = float(cat["xla_kl"].mean())
        summary[k] = s
    json.dump({"summary": summary, "per_prompt": rows, "spots": spots}, open(a.out, "w"), indent=1)
    print(json.dumps({"summary": summary, "spots": spots}, indent=1))


if __name__ == "__main__":
    main()
