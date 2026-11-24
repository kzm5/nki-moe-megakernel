"""Supplementary long-context scoring: the frozen validator v1.1 rows on the long corpus, via the frozen modules unchanged.

Computes:
    validate.score() from the frozen validate.py with its CORPUS / GOLDEN / REFS constants pointed at the long corpus,
    its golden and its XLA references; frozen_long_ok = the frozen validator files hash as in frozen.json and the
    long corpus / golden / refs hash as in frozen_long.json.

Usage:
    validate_long.py --cand DIR --base-logits DIR --out REPORT.json [--freeze]
    --freeze: write frozen_long.json from the current long corpus, golden and refs (once, before any scoring)
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.expanduser("~/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator")
LONG = {"CORPUS": os.path.join(HERE, "..", "results/phase1/corpus_long/corpus.json"),
        "GOLDEN": os.path.expanduser("~/runs/p1/golden_long2048"), "REFS": os.path.expanduser("~/runs/p1/refs_long2048")}
FROZEN_LONG = os.path.join(HERE, "..", "results/phase1/frozen_long.json")

sys.path.insert(0, FROZEN_DIR)
import validate as V  # noqa: E402

for k, v in LONG.items():
    setattr(V, k, os.path.abspath(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand")
    ap.add_argument("--base-logits")
    ap.add_argument("--out")
    ap.add_argument("--spec", default=V.DEFAULT_SPEC)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    h = V.current_hashes(a.spec)
    if a.freeze:
        assert not os.path.exists(FROZEN_LONG), "already frozen"
        json.dump({k: h[k] for k in ("corpus", "golden_manifest", "refs_manifest", "n_golden_files", "n_refs")},
                  open(FROZEN_LONG, "w"), indent=1)
        print("FROZEN_LONG", json.dumps(h, indent=1))
        return
    rep = V.score(a.cand, a.cand, None, a.out, a.base_logits, a.spec)
    fz, fl = json.load(open(V.FROZEN)), json.load(open(FROZEN_LONG))
    rep["frozen_long_ok"] = (fz.get("validator") == h["validator"] and h["spec"] in fz.get("specs", {}).values()
                             and all(fl[k] == h[k] for k in ("corpus", "golden_manifest", "refs_manifest")))
    rep["wrapper"] = {"file": os.path.abspath(__file__), "frozen_dir": FROZEN_DIR, **LONG}
    rep["supplementary"] = "long-context corpus (results/phase1/corpus_long); not the v1 validator corpus"
    json.dump(rep, open(a.out, "w"), indent=1)
    print("VALIDATE_LONG", json.dumps({"pass": rep["pass"], "frozen_long_ok": rep["frozen_long_ok"],
                                       "rows_failed": rep["rows_failed"]}), flush=True)


if __name__ == "__main__":
    main()
