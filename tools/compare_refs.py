"""Bit-for-bit comparison of free-running reference captures (sequences and logits).

Usage:
    compare_refs.py NEW_DIR OLD_DIR [OLD_DIR ...] --out result.json
"""
import argparse, glob, json, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for od in a.old:
        for f in sorted(glob.glob(f"{od}/expected_p*.pt")):
            n = f"{a.new}/{os.path.basename(f)}"
            if not os.path.exists(n):
                continue
            o, c = torch.load(f), torch.load(n)
            lo, lc = o["logits"].float(), c["logits"].float()
            same_seq = torch.equal(o["sequences"], c["sequences"])
            same_lg = lo.shape == lc.shape and torch.equal(lo, lc)
            r = {"file": os.path.basename(f), "old_dir": od, "sequences_identical": same_seq,
                 "logits_identical": same_lg}
            if not same_lg and lo.shape == lc.shape:
                k = torch.isfinite(lo) & torch.isfinite(lc)
                r["max_abs_logit_diff"] = float((lo[k] - lc[k]).abs().max())
                r["first_step_diff"] = int(((lo - lc).abs().nan_to_num(0).amax(-1) > 0).nonzero()[0, 0])
            rows.append(r)
    res = {"n": len(rows), "all_identical": all(r["sequences_identical"] and r["logits_identical"] for r in rows),
           "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}),
          [r["file"] for r in rows if not (r["sequences_identical"] and r["logits_identical"])])


if __name__ == "__main__":
    main()
