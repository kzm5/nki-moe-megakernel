"""Build the supplementary long-context corpus (deterministic; same inputs -> same bytes).

Computes:
    prompts  q00-q15, 64 new tokens each, prompt lengths spread evenly over 1,400-1,900 tokens (total <= 2048):
             8 plain continuations and 4 instruction prompts on wikitext-2-raw-v1 test articles (CC BY-SA 3.0)
             not used by the v1 corpus, 4 continuations of CPython 3.12 standard-library source (PSF-2.0)
    ppl      4 passages of PPL_LEN tokens from further unused wikitext articles

Returns:
    OUT/corpus.json  same schema as the v1 corpus (validator/corpus.py)

Usage:
    corpus_long.py --out results/phase1/corpus_long
"""
import argparse, hashlib, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validator"))
import corpus as C1  # noqa: E402  (frozen; helpers only)

SEED = 1124
MIN_PLEN, MAX_PLEN, N_NEW, MAX_TOTAL = 1400, 1900, 64, 2048
N_PPL, PPL_LEN = 4, 1920
STDLIB = "/usr/lib/python3.12"
CODE_FILES = ["textwrap.py", "fractions.py", "statistics.py", "heapq.py", "bisect.py", "difflib.py", "shlex.py",
              "calendar.py", "string.py", "graphlib.py"]
INSTR = ["Summarize the text above in three sentences.", "What are the two most important facts in this passage?",
         "List the people, places and dates mentioned above.", "Write a one-paragraph summary for a general audience."]
KINDS = ["wiki", "code", "wiki", "wiki_instr"] * 4  # 8 wiki, 4 code, 4 wiki_instr, interleaved across lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", default=f"{C1.H}/runs/p1/src_wikitext/test.txt")
    ap.add_argument("--v1", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results/phase1/corpus/corpus.json"))
    ap.add_argument("--model-path", default=f"{C1.H}/qwen-30b-a3b/hf_model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    raw = open(a.wikitext, encoding="utf-8").read()
    arts = C1.articles(raw)
    v1 = json.load(open(a.v1))
    used_titles = {w["article"] for w in v1["ppl"]} | {p["source"].split(": ", 1)[1] for p in v1["prompts"]
                                                       if p["source"].startswith("wikitext")}
    rng = random.Random(SEED)
    order = [i for i in range(len(arts)) if arts[i][0] not in used_titles]
    rng.shuffle(order)
    targets = [round(MIN_PLEN + k * (MAX_PLEN - MIN_PLEN) / 15) for k in range(16)]
    code = sorted(CODE_FILES)
    rng.shuffle(code)
    prompts, n_instr = [], 0
    for j, (target, kind) in enumerate(zip(targets, KINDS)):
        if kind == "code":
            for f in list(code):
                text = open(f"{STDLIB}/{f}", encoding="utf-8").read()
                ids = tok(text).input_ids
                code.remove(f)
                if len(ids) >= target:
                    break
            else:
                raise SystemExit(f"no stdlib file long enough for {target}")
            prompt, src = tok.decode(ids[:target]), f"CPython 3.12 Lib/{f} (PSF-2.0)"
        else:
            instr = INSTR[n_instr] if kind == "wiki_instr" else None
            suffix = f"\n\n{instr}\n" if instr else ""
            nsuf = len(tok(suffix).input_ids) if instr else 0
            for i in order:
                ids = tok(C1.clean(arts[i][1])).input_ids
                if len(ids) >= target:
                    order.remove(i)
                    break
            else:
                raise SystemExit(f"no article long enough for {target}")
            cut = tok.decode(ids[:target - nsuf])
            cut = cut[:cut.rfind(" ")] if instr else cut
            prompt, src = cut + suffix, f"wikitext-2-raw-v1 test: {arts[i][0]}"
            n_instr += bool(instr)
        prompts.append({"id": f"q{j:02d}", "prompt": prompt, "n_new": N_NEW, "source": src, "kind": kind})
    ppl = []
    for i in list(order):
        ids = tok(arts[i][1].strip(), add_special_tokens=False).input_ids
        if len(ids) >= PPL_LEN and len(ppl) < N_PPL:
            ppl.append({"id": f"w{len(ppl):02d}", "article": arts[i][0], "ids": ids[:PPL_LEN]})
            order.remove(i)
    assert len(ppl) == N_PPL, len(ppl)
    for p in prompts:
        p["plen"] = len(tok([p["prompt"]]).input_ids[0])
        assert MIN_PLEN - 8 <= p["plen"] <= MAX_PLEN and p["plen"] + p["n_new"] <= MAX_TOTAL, (p["id"], p["plen"])
    meta = {"seed": SEED, "max_total": MAX_TOTAL, "n_prompts": len(prompts), "n_ppl": len(ppl), "ppl_len": PPL_LEN,
            "supplementary": True, "excludes_v1_articles": sorted(used_titles),
            "wikitext_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "wikitext_source": "Salesforce/wikitext wikitext-2-raw-v1 test (CC BY-SA 3.0)",
            "code_source": "CPython 3.12 standard library, /usr/lib/python3.12 (PSF License 2.0)",
            "code_sha256": {p["source"]: hashlib.sha256(open(f"{STDLIB}/{p['source'].split('Lib/')[1].split(' ')[0]}", "rb")
                                                        .read()).hexdigest() for p in prompts if p["kind"] == "code"}}
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/corpus.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "prompts": prompts, "ppl": ppl}, f, ensure_ascii=False, indent=1)
    print(json.dumps({"plens": [p["plen"] for p in prompts], "kinds": [p["kind"] for p in prompts],
                      "sources": [p["source"] for p in prompts], "ppl": [w["article"] for w in ppl]},
                     ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
