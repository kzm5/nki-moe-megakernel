"""Four fixed long prompts (520-595 tokens) cut from the model's own README and LICENSE. Usage: make_long_prompts.py OUT.json"""
import hashlib, json, os, sys
from transformers import AutoTokenizer
mp = os.path.expanduser("~/qwen-30b-a3b/hf_model")
tok = AutoTokenizer.from_pretrained(mp)
readme, lic = open(f"{mp}/README.md").read(), open(f"{mp}/LICENSE").read()
srcs = [(readme, 0), (readme, 6000), (lic, 0), (lic, 5000)]
out = []
for (txt, off), n in zip(srcs, [520, 545, 570, 595]):
    ids = tok(txt[off:], add_special_tokens=False).input_ids[:n]
    p = tok.decode(ids)
    assert 500 <= len(tok(p).input_ids) <= 600, len(tok(p).input_ids)
    out.append(p)
json.dump(out, open(sys.argv[1], "w"))
print([len(tok(p).input_ids) for p in out], hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16])
