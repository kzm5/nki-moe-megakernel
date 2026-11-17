"""Frozen correctness validator: build -> teacher-forced capture over the corpus -> one JSON of spec rows.

Computes (cand vs fp32 golden; ratios vs the same campaign's XLA: --base = XLA capture build for op_boundary
and router, --base-logits = production XLA build for logits_tf, worst_pos and perplexity):
    op_boundary   per layer, per boundary (attn, moe, resid): pooled rel-L2 vs golden; rho = cand / base;
                  gate median_L rho <= rho_median_max and max_L rho <= rho_layer_max (per boundary)
    router        tie-tolerant top-8 flip rate vs golden; gate cand - base <= router_pp_max (overall)
    logits_tf     mean KL(golden || cand) over all corpus positions; gate <= kl_ratio_max * base; top-1 reported
    worst_pos     max KL; count(cand KL > gross_kl and base KL < base_clean_kl); gate count == 0
    perplexity    exp(mean NLL) on the passages; gate |cand / base - 1| <= ppl_rel_max
    run_to_run    in-process repeat (and --prev cross-process) bit-identity; non-blocking, documented

Returns:
    REPORT.json  {"rows": {...}, "pass": bool, "hashes": {...}, "frozen_ok": bool, ...}

Usage (precedence: run, then score):
    validate.py run KIND CDIR OUT [--capture] [--mutant mN] [--base DIR] [--prev DIR]
    validate.py score --cand DIR --base DIR [--prev DIR] --out REPORT.json
    validate.py hashes
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
H = os.path.expanduser("~")
P1 = f"{H}/runs/p1"
SPEC = f"{ROOT}/results/phase1/spec.json"
CORPUS = f"{ROOT}/results/phase1/corpus/corpus.json"
FROZEN = f"{ROOT}/results/phase1/frozen.json"
GOLDEN = f"{P1}/golden"
REFS = f"{P1}/refs"
VALIDATOR_FILES = ["validate.py", "device.py", "golden.py", "mutants.py", "corpus.py"]
BOUNDARIES = ("attn", "moe", "resid")
PY = [f"{H}/venv-neuron-229/bin/python"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def current_hashes():
    g = sorted(glob.glob(f"{GOLDEN}/*.pt"))
    r = sorted(glob.glob(f"{REFS}/expected_p*.pt"))
    man = lambda fs: hashlib.sha256("".join(f"{os.path.basename(f)} {sha(f)}\n" for f in fs).encode()).hexdigest()
    return {"validator": {f: sha(f"{HERE}/{f}") for f in VALIDATOR_FILES}, "spec": sha(SPEC), "corpus": sha(CORPUS),
            "golden_manifest": man(g), "refs_manifest": man(r), "n_golden_files": len(g), "n_refs": len(r)}


def kl(g, b):
    keep = torch.isfinite(g) & torch.isfinite(b)
    lg = F.log_softmax(g.double().masked_fill(~keep, -math.inf), -1)
    lb = F.log_softmax(b.double().masked_fill(~keep, -math.inf), -1)
    t = lg.exp() * (lg - lb)
    return t.masked_fill(~keep, 0).sum(-1)


def valid_top8(logits, ids):
    m = torch.zeros(logits.numel(), dtype=torch.bool)
    m[ids.long()] = True
    return bool(logits[m].min() >= logits[~m].max())


def logits_row(corpus, d):
    out = {}
    for p in corpus["prompts"]:
        nn = p["id"][1:]
        g = torch.load(f"{GOLDEN}/golden_p{nn}.pt")["logits"]
        c = torch.load(f"{d}/tf/tf_p{nn}.pt")["scores"].float()
        out[p["id"]] = {"kl": kl(g, c), "top1": (c.argmax(-1) == g.argmax(-1))}
    return out


def flips(corpus, d):
    """[P, 48] tie-tolerant flips vs golden top-8, NaN where the capture fell back."""
    rows = []
    for p in corpus["prompts"]:
        nn = p["id"][1:]
        gi = torch.load(f"{GOLDEN}/golden_p{nn}.pt")["router_ids"].long()  # [48, n, 9]
        c = torch.load(f"{d}/cap/cap_p{nn}.pt")
        lg, ei, ok = c["router_logits"], c["expert_index"], c["valid"]
        n = len(ok)
        fl = torch.full((n, 48), float("nan"))
        for s in range(n):
            if not ok[s]:
                continue
            for L in range(48):
                gset = gi[L, s, :8]
                if ei is None:
                    fl[s, L] = float(not valid_top8(lg[s, L], gset))
                else:
                    ids = ei[s, L]
                    same = set(ids.tolist()) == set(gset.tolist())
                    tie = valid_top8(lg[s, L], ids) and valid_top8(lg[s, L], gset)
                    fl[s, L] = float(not (same or tie))
        rows.append(fl)
    return torch.cat(rows)


def ppl(corpus, d):
    nll = torch.cat([torch.load(f"{d}/ppl_{w['id']}.pt" if d == GOLDEN else f"{d}/ppl/ppl_{w['id']}.pt")["nll"]
                     for w in corpus["ppl"]])
    return float(torch.exp(nll.mean())), int(nll.numel())


def score(cand, base, prev, out, base_logits=None):
    base_logits = base_logits or base
    spec = json.load(open(SPEC))
    th = spec["thresholds"]
    corpus = json.load(open(CORPUS))
    hashes = current_hashes()
    frozen = json.load(open(FROZEN)) if os.path.exists(FROZEN) else None
    frozen_ok = frozen is not None and all(frozen.get(k) == hashes[k] for k in
                                           ("validator", "spec", "corpus", "golden_manifest", "refs_manifest"))
    rows = {}
    # op boundary
    if os.path.exists(f"{cand}/stats.pt") and os.path.exists(f"{base}/stats.pt"):
        sc, sb = torch.load(f"{cand}/stats.pt"), torch.load(f"{base}/stats.pt")
        r = {"pass": True, "per_boundary": {}}
        for bd in BOUNDARIES:
            ok = ~(torch.isnan(sc[f"{bd}.err2"]) | torch.isnan(sb[f"{bd}.err2"]))
            z = lambda t: torch.where(ok, t, torch.zeros_like(t)).sum(0)
            rel_c = (z(sc[f"{bd}.err2"]) / z(sc[f"{bd}.gold2"])).sqrt()
            rel_b = (z(sb[f"{bd}.err2"]) / z(sb[f"{bd}.gold2"])).sqrt()
            rho = rel_c / rel_b
            cos = z(sc[f"{bd}.dot"]) / (z(sc[f"{bd}.cand2"]).sqrt() * z(sc[f"{bd}.gold2"]).sqrt())
            med, mx = float(rho.median()), float(rho.max())
            ok_b = med <= th["rho_median_max"] and mx <= th["rho_layer_max"]
            r["per_boundary"][bd] = {
                "pass": ok_b, "rho_median": med, "rho_max": mx, "rho_max_layer": int(rho.argmax()),
                "layers_over_max": [L for L in range(48) if rho[L] > th["rho_layer_max"]],
                "rho_per_layer": [round(float(x), 4) for x in rho], "rel_l2_cand": [float(x) for x in rel_c],
                "rel_l2_base": [float(x) for x in rel_b], "cosine_cand": [float(x) for x in cos],
                "maxabs_cand_max": float(torch.where(ok, sc[f"{bd}.maxabs"], torch.zeros(1)).max()),
                "n_positions": int(ok[:, 0].sum())}
            r["pass"] &= ok_b
        rows["op_boundary"] = r
    else:
        rows["op_boundary"] = {"pass": None, "status": "not_measurable", "why": "no per-layer capture"}
    # router
    if os.path.exists(f"{cand}/cap") and os.path.exists(f"{base}/cap"):
        fc, fb = flips(corpus, cand), flips(corpus, base)
        ok = ~(torch.isnan(fc) | torch.isnan(fb))
        rc = float(fc[ok].mean() * 100)
        rb = float(fb[ok].mean() * 100)
        okL = ok.float().sum(0)
        pc = torch.where(ok, fc, torch.zeros(1)).sum(0) / okL * 100
        pb = torch.where(ok, fb, torch.zeros(1)).sum(0) / okL * 100
        d = pc - pb
        rows["router"] = {"pass": rc - rb <= th["router_pp_max"], "flip_pct_cand": rc, "flip_pct_base": rb,
                          "delta_pp": rc - rb, "delta_pp_max_layer": float(d.max()), "max_layer": int(d.argmax()),
                          "layers_over_gate": [L for L in range(48) if d[L] > th["router_pp_max"]],
                          "flip_pct_per_layer_cand": [round(float(x), 3) for x in pc],
                          "delta_pp_per_layer": [round(float(x), 3) for x in d], "n_layer_positions": int(ok.sum())}
    else:
        rows["router"] = {"pass": None, "status": "not_measurable", "why": "no router capture"}
    # logits
    lc, lb = logits_row(corpus, cand), logits_row(corpus, base_logits)
    kc = torch.cat([lc[k]["kl"] for k in lc])
    kb = torch.cat([lb[k]["kl"] for k in lc])
    t1 = torch.cat([lc[k]["top1"] for k in lc]).float()
    ratio = float(kc.mean() / kb.mean()) if float(kb.mean()) > 0 else (1.0 if float(kc.mean()) == 0 else math.inf)
    rows["logits_tf"] = {"pass": ratio <= th["kl_ratio_max"], "kl_mean_cand": float(kc.mean()),
                         "kl_mean_base": float(kb.mean()), "kl_ratio": ratio, "top1_golden_cand": float(t1.mean()),
                         "top1_golden_base": float(torch.cat([lb[k]["top1"] for k in lb]).float().mean()),
                         "n_positions": int(kc.numel())}
    names = [(k, i) for k in lc for i in range(len(lc[k]["kl"]))]
    gross = ((kc > th["gross_kl"]) & (kb < th["base_clean_kl"])).nonzero().flatten().tolist()
    w = int(kc.argmax())
    rows["worst_pos"] = {"pass": len(gross) == 0, "max_kl": float(kc.max()), "max_kl_at": names[w],
                         "base_kl_there": float(kb[w]), "gross_count": len(gross),
                         "gross_at": [names[i] for i in gross][:50],
                         "kl_hist_edges": [1e-3, 1e-2, 0.05, 0.2, 1.0],
                         "kl_hist_cand": torch.bucketize(kc, torch.tensor([1e-3, 1e-2, 0.05, 0.2, 1.0]).double())
                                         .bincount(minlength=6).tolist()}
    # perplexity
    pc_, n = ppl(corpus, cand)
    pb_, _ = ppl(corpus, base_logits)
    pg_, _ = ppl(corpus, GOLDEN) if os.path.exists(f"{GOLDEN}/ppl_w00.pt") else (None, 0)
    rel = pc_ / pb_ - 1
    rows["perplexity"] = {"pass": abs(rel) <= th["ppl_rel_max"], "ppl_cand": pc_, "ppl_base": pb_, "ppl_golden": pg_,
                          "rel_delta": rel, "n_tokens": n}
    # run-to-run
    dc = json.load(open(f"{cand}/device.json"))
    r2r = {"in_process": dc.get("repeat")}
    ident = dc.get("repeat") is not None and not dc["repeat"]["differ"]
    if prev:
        dp = json.load(open(f"{prev}/device.json"))
        diff = [k for k in dc["tf_sha256"] if dp["tf_sha256"].get(k) != dc["tf_sha256"][k]]
        r2r["cross_process"] = {"prev": prev, "identical": len(dc["tf_sha256"]) - len(diff), "differ": diff}
        ident = ident and not diff
    r2r["pass"] = True if ident else None
    r2r["status"] = "bit_identical" if ident else "documented"
    rows["run_to_run"] = r2r
    blocking = [k for k in ("op_boundary", "router", "logits_tf", "worst_pos", "perplexity")]
    measured = [k for k in blocking if rows[k]["pass"] is not None]
    rep = {"cand": cand, "base": base, "base_logits": base_logits, "cand_device": {k: dc[k] for k in ("kind", "cdir", "capture", "mutant")},
           "pass": all(rows[k]["pass"] for k in measured),
           "rows_failed": [k for k in measured if not rows[k]["pass"]],
           "rows_not_measurable": [k for k in blocking if rows[k]["pass"] is None],
           "rows": rows, "thresholds": th, "hashes": hashes, "frozen_ok": frozen_ok,
           "scored_utc": time.strftime("%FT%TZ", time.gmtime())}
    json.dump(rep, open(out, "w"), indent=1)
    summ = {k: rep[k] for k in ("pass", "rows_failed", "rows_not_measurable", "frozen_ok")}
    summ["key"] = {"rho_median": {b: rows["op_boundary"].get("per_boundary", {}).get(b, {}).get("rho_median")
                                  for b in BOUNDARIES},
                   "router_delta_pp": rows["router"].get("delta_pp"), "kl_ratio": ratio,
                   "gross": len(gross), "ppl_rel": rel, "r2r": r2r["status"]}
    print("VALIDATE", json.dumps(summ), flush=True)
    return rep


def run(a):
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(f"{a.out}/device.json"):
        cmd = PY + [f"{HERE}/device.py", a.kind, a.cdir, a.out, "--corpus", CORPUS, "--refs", REFS,
                    "--tf", "--ppl", "--repeat"] + (["--capture"] if a.capture else []) + \
              (["--mutant", a.mutant] if a.mutant else [])
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    if a.capture and not os.path.exists(f"{a.out}/stats.pt"):
        cmd = PY + [f"{HERE}/golden.py", "--corpus", CORPUS, "--refs", REFS, "--out", f"{a.out}/golden_tmp",
                    "--stats", f"cand={a.out}"]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return score(a.out, a.base, a.prev, f"{a.out}/report.json", a.base_logits)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("kind")
    r.add_argument("cdir")
    r.add_argument("out")
    r.add_argument("--capture", action="store_true")
    r.add_argument("--mutant", default="")
    r.add_argument("--base", default=f"{P1}/val/xla_base")
    r.add_argument("--prev")
    r.add_argument("--base-logits", default=f"{P1}/val/xla_plain")
    s = sub.add_parser("score")
    s.add_argument("--cand", required=True)
    s.add_argument("--base", default=f"{P1}/val/xla_base")
    s.add_argument("--prev")
    s.add_argument("--base-logits", default=f"{P1}/val/xla_plain")
    s.add_argument("--out", required=True)
    sub.add_parser("hashes")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "score":
        score(a.cand, a.base, a.prev, a.out, a.base_logits)
    else:
        print(json.dumps(current_hashes(), indent=1))


if __name__ == "__main__":
    main()
