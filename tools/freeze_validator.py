"""Freeze the validator: record SHA-256 of validator code, every spec version, corpus, refs and golden; make them
read-only, including the freeze records themselves.

Returns:
    results/phase1/frozen.json  {"frozen_utc", "paths", "specs": {version: sha256}, <validate.current_hashes()>,
                                 "files": {path: sha256}}
    results/phase1/FROZEN.md    the same, human-readable

Usage:
    freeze_validator.py [--no-chmod]
"""
import argparse, glob, json, os, stat, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools/validator")
import validate as V  # noqa: E402

R = f"{ROOT}/results/phase1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chmod", action="store_true")
    a = ap.parse_args()
    records = sorted(glob.glob(f"{R}/frozen*.json")) + [V.FROZEN] + sorted(glob.glob(f"{R}/FROZEN*.md"))
    records = sorted(set(records + [f"{R}/FROZEN.md"]))
    paths = [V.HERE, *V.SPECS.values(), V.CORPUS, V.GOLDEN, V.REFS, *records]
    files = sorted([f"{V.HERE}/{f}" for f in V.VALIDATOR_FILES] + list(V.SPECS.values()) + [V.CORPUS]
                   + glob.glob(f"{V.GOLDEN}/*.pt") + glob.glob(f"{V.REFS}/expected_p*.pt"))
    h = V.current_hashes()
    specs = {ver: V.sha(p) for ver, p in V.SPECS.items()}
    rec = {"frozen_utc": time.strftime("%FT%TZ", time.gmtime()), "paths": paths, "specs": specs,
           "default_spec": V.DEFAULT_SPEC, **h, "files": {f: V.sha(f) for f in files}}
    json.dump(rec, open(V.FROZEN, "w"), indent=1)
    md = ["# Frozen validator", "", f"Frozen {rec['frozen_utc']} (default spec {V.DEFAULT_SPEC}). Every validator report "
          "and ledger entry carries these hashes; `validate.py` sets `frozen_ok` only when they match `frozen.json`. "
          "Earlier freeze: `FROZEN_v1.md` / `frozen_v1.json`.", "", "| Item | SHA-256 |", "|---|---|"]
    md += [f"| `tools/validator/{k}` | `{v}` |" for k, v in h["validator"].items()]
    md += [f"| `results/phase1/{os.path.basename(V.SPECS[ver])}` (v{ver}) | `{s}` |" for ver, s in specs.items()]
    md += [f"| `results/phase1/corpus/corpus.json` | `{h['corpus']}` |",
           f"| golden manifest ({h['n_golden_files']} files in `~/runs/p1/golden`) | `{h['golden_manifest']}` |",
           f"| refs manifest ({h['n_refs']} files in `~/runs/p1/refs`) | `{h['refs_manifest']}` |", "",
           "Read-only paths (chmod a-w; PreToolUse hook `tools/validator/guard_hook.py` denies edits):", ""]
    md += [f"- `{p}`" for p in paths]
    md += ["", "Per-file hashes: `results/phase1/frozen.json`."]
    open(f"{R}/FROZEN.md", "w").write("\n".join(md) + "\n")
    if not a.no_chmod:
        for f in files + [p for p in records if os.path.exists(p)]:
            os.chmod(f, os.stat(f).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for d in (V.HERE, V.GOLDEN, V.REFS, os.path.dirname(V.CORPUS)):
            os.chmod(d, os.stat(d).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({k: rec[k] for k in ("frozen_utc", "specs", "corpus", "golden_manifest", "refs_manifest")}))


if __name__ == "__main__":
    main()
