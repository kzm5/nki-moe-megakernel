"""Summarize validation: pre-divergence metrics and divergence tie-gaps. Usage: analyze.py KIND..."""
import json, sys, torch
V = "/home/ubuntu/runs/val"
out = {}
for kind in sys.argv[1:]:
    R = json.load(open(f"{V}/results_{kind}.json"))
    rows = []
    for r in R:
        i, d = r["prompt_idx"], r["first_divergence"]
        e = torch.load(f"{V}/expected_p{i:02d}.pt")["logits"][:, 0]
        a = torch.load(f"{V}/actual_{kind}_p{i:02d}.pt").float()[:, 0]
        n = e.shape[0] if d < 0 else d
        pp = r["per_pos"]
        pre_mad = max(pp["max_abs_diff"][:n]) if n else 0.0
        pre_kl = sum(pp["kl"][:n]) / n if n else 0.0
        pre_kl_max = max(pp["kl"][:n]) if n else 0.0
        gap = None
        if d >= 0:
            v = e[d].topk(2).values
            gap = float(v[0] - v[1])
            a_choice_rank = int((e[d] > e[d, a[d].argmax()]).sum())  # rank of actual's pick in expected
        rows.append({"p": i, "first_div": d, "agree_until": n, "pre_div_max_abs": round(pre_mad, 3),
                     "pre_div_kl_mean": pre_kl, "pre_div_kl_max": pre_kl_max,
                     "exp_top1_top2_gap_at_div": None if gap is None else round(gap, 4),
                     "actual_pick_rank_in_expected": None if d < 0 else a_choice_rank,
                     "pass_repo": r["pass_repo_tol"], "pass_rtol_1e-2": r["pass_rtol_1e-2"]})
    out[kind] = rows
    divs = [x["first_div"] for x in rows]
    print(f"== {kind}: pass_repo {sum(x['pass_repo'] for x in rows)}/20, pass_rtol1e-2 {sum(x['pass_rtol_1e-2'] for x in rows)}/20, "
          f"no-divergence {sum(d < 0 for d in divs)}/20, median first_div {sorted(d if d >= 0 else 128 for d in divs)[10]}")
    print(" p  first_div  pre_maxabs  pre_KLmean  pre_KLmax  gap@div  pick_rank")
    for x in rows:
        print(f"{x['p']:2d}  {x['first_div']:9d}  {x['pre_div_max_abs']:10.3f}  {x['pre_div_kl_mean']:10.2e}  {x['pre_div_kl_max']:9.2e}  "
              f"{str(x['exp_top1_top2_gap_at_div']):>7}  {str(x['actual_pick_rank_in_expected']):>9}")
json.dump(out, open(f"{V}/summary.json", "w"), indent=1)
