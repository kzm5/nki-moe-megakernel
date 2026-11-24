"""Exit 0 iff the repo's validator files, spec, corpus, golden and refs hash as recorded in results/phase1/frozen.json.

Usage:
    check_frozen.py REPO
"""
import json, os, sys

repo = sys.argv[1]
sys.path.insert(0, f"{repo}/tools/validator")
import validate as V  # noqa: E402

h = V.current_hashes()
fz = json.load(open(V.FROZEN))
bad = [k for k in ("validator", "corpus", "golden_manifest", "refs_manifest") if fz.get(k) != h[k]]
if h["spec"] not in fz.get("specs", {}).values():
    bad.append("spec")
print("frozen_ok" if not bad else f"mismatch: {bad}")
sys.exit(1 if bad else 0)
