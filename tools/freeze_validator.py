"""Freeze the validator: record SHA-256 of validator code, spec, corpus, refs and golden; make them read-only.

Returns:
    results/phase1/frozen.json  {"frozen_utc", "paths", <validate.current_hashes()>, "files": {path: sha256}}
    results/phase1/FROZEN.md    the same, human-readable

Usage:
    freeze_validator.py [--no-chmod]
"""
import argparse, glob, json, os, stat, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools/validator")
import validate as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chmod", action="store_true")
    a = ap.parse_args()
    paths = [V.HERE, V.SPEC, V.CORPUS, V.GOLDEN, V.REFS]
    files = sorted([f"{V.HERE}/{f}" for f in V.VALIDATOR_FILES] + [V.SPEC, V.CORPUS]
                   + glob.glob(f"{V.GOLDEN}/*.pt") + glob.glob(f"{V.REFS}/expected_p*.pt"))
    h = V.current_hashes()
    rec = {"frozen_utc": time.strftime("%FT%TZ", time.gmtime()), "paths": paths, **h,
           "files": {f: V.sha(f) for f in files}}
    json.dump(rec, open(V.FROZEN, "w"), indent=1)
    md = ["# Frozen validator", "", f"Frozen {rec['frozen_utc']}. Every validator report and ledger entry carries "
          "these hashes; `validate.py` sets `frozen_ok` only when they match `frozen.json`.", "",
          "| Item | SHA-256 |", "|---|---|"]
    md += [f"| `tools/validator/{k}` | `{v}` |" for k, v in h["validator"].items()]
    md += [f"| `results/phase1/spec.json` | `{h['spec']}` |", f"| `results/phase1/corpus/corpus.json` | `{h['corpus']}` |",
           f"| golden manifest ({h['n_golden_files']} files in `~/runs/p1/golden`) | `{h['golden_manifest']}` |",
           f"| refs manifest ({h['n_refs']} files in `~/runs/p1/refs`) | `{h['refs_manifest']}` |", "",
           "Read-only paths (chmod a-w; PreToolUse hook `tools/validator/guard_hook.py` denies edits):", ""]
    md += [f"- `{p}`" for p in paths]
    md += ["", "Per-file hashes: `results/phase1/frozen.json`."]
    open(f"{ROOT}/results/phase1/FROZEN.md", "w").write("\n".join(md) + "\n")
    if not a.no_chmod:
        for f in files:
            os.chmod(f, os.stat(f).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for d in (V.HERE, V.GOLDEN, V.REFS, os.path.dirname(V.CORPUS)):
            os.chmod(d, os.stat(d).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({k: rec[k] for k in ("frozen_utc", "spec", "corpus", "golden_manifest", "refs_manifest")}))


if __name__ == "__main__":
    main()
