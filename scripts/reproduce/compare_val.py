"""Compare a reproduced validator capture with a reference capture of the same config: per-prompt tf score hashes
(device.json) and the report's verdict and key rows.

Usage:
    compare_val.py REF_DIR NEW_DIR
"""
import json, os, sys

ref, new = sys.argv[1:3]
a, b = json.load(open(f"{ref}/device.json")), json.load(open(f"{new}/device.json"))
ha, hb = a.get("tf_sha256", {}), b.get("tf_sha256", {})
same = [k for k in ha if hb.get(k) == ha[k]]
line = f"COMPARE {os.path.basename(new)}: tf bit-identical {len(same)}/{len(ha)} prompts"
for d, tag in ((ref, "ref"), (new, "new")):
    p = f"{d}/report.json"
    if os.path.exists(p):
        r = json.load(open(p))
        lt = r["rows"]["logits_tf"]
        line += f"; {tag} pass={r['pass']} kl_ratio={lt['kl_ratio']:.4f} ppl={r['rows']['perplexity']['ppl_cand']:.4f}"
print(line)
sys.exit(0 if len(same) == len(ha) and ha else 1)
