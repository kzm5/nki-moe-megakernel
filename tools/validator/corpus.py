"""Build the frozen validation corpus (deterministic; same inputs -> same bytes).

Computes:
    prompts  p00-p19 day-1 short prompts (n_new 128), p20-p23 day-3 long prompts (n_new 64),
             p24-p47 hand-written mixed-domain prompts, p48-p63 wikitext-2 test excerpts cut to target
             token lengths; plen + n_new <= MAX_TOTAL
    ppl      N_PPL passages of PPL_LEN tokens from distinct wikitext-2-raw test articles

Returns:
    OUT/corpus.json  {"prompts": [{"id", "prompt", "n_new", "plen", "source"}], "ppl": [{"id", "article", "ids"}],
                      "meta": {...}}

Usage:
    corpus.py --wikitext ~/runs/p1/src_wikitext/test.txt --out results/phase1/corpus
"""
import argparse, hashlib, json, os, random, re

SEED = 1117
MAX_TOTAL = 768
N_PPL, PPL_LEN = 16, 512
H = os.path.expanduser("~")

HAND = [  # (prompt, n_new)
    ("Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.", 128),
    ("¿Cuáles son las principales diferencias entre el clima mediterráneo y el clima continental?", 128),
    ("Explique en quelques phrases pourquoi le ciel est bleu.", 128),
    ("请用三句话介绍长城的历史。", 128),
    ("User: Hi, I'm planning a trip to Japan in April. Any advice?\nAssistant:", 128),
    ("Prove that the square root of 2 is irrational.", 128),
    ("Solve for x: 3x + 7 = 2x - 5. Show each step.", 128),
    ("The quick brown fox", 128),
    ("Give a numbered list of the planets in the solar system with one fact about each.", 128),
    ("#include <stdio.h>\n\nint main(void) {\n", 128),
    ("Describe the water cycle to a ten-year-old.", 128),
    ("Kannst du mir erklären, wie eine Wärmepumpe funktioniert?", 128),
    ("Write a limerick about a cat who learns to code.", 128),
    ("What are the pros and cons of remote work? Answer as a bulleted list.", 128),
    ("Explain what the following JavaScript function does and point out any bugs.\n\n"
     "function debounce(fn, wait) {\n  let timer;\n  return function (...args) {\n    clearTimeout(timer);\n"
     "    timer = setTimeout(() => fn(args), wait);\n  };\n}\n\nconst log = debounce((msg) => console.log(msg), 200);\n"
     "window.addEventListener('resize', () => log('resized'));\n", 128),
    ("A store sells notebooks for $3.50 each and pens for $1.25 each. Maria buys 4 notebooks and some pens, "
     "and pays with a $20 bill. She receives $2.25 in change. A second customer buys twice as many pens as Maria "
     "and half as many notebooks. How many pens did Maria buy, and how much did the second customer pay? "
     "Explain your reasoning step by step.", 128),
    ("The following is a conversation between a librarian and a student.\n\n"
     "Student: Hi, I'm looking for books about the history of navigation.\n"
     "Librarian: Sure. Are you interested in a particular period, like the age of sail, or modern methods?\n"
     "Student: Mostly how sailors found longitude before GPS.\n"
     "Librarian: Then you'll want to start with the story of the marine chronometer.\n"
     "Student: What's that?\nLibrarian:", 128),
    ("Ingredients for a simple tomato soup:\n1. 2 tablespoons olive oil\n2. 1 onion, chopped\n"
     "3. 3 cloves garlic, minced\n4. 800 g canned tomatoes\n5. 500 ml vegetable stock\n6. Salt and pepper\n\n"
     "Steps:\n1. Heat the oil in a large pot over medium heat.\n2.", 128),
    ("{\n  \"name\": \"inventory-service\",\n  \"version\": \"1.4.2\",\n  \"dependencies\": {\n"
     "    \"express\": \"^4.18.2\",\n    \"pg\": \"^8.11.0\"\n  },\n  \"scripts\": {\n"
     "    \"start\": \"node server.js\",\n    \"test\": \"jest\"\n  }\n}\n\n"
     "Add a lint script using eslint and a dev dependency for it, and explain each change.", 128),
    ("CREATE TABLE customers (id INT PRIMARY KEY, name TEXT, country TEXT);\n"
     "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customers(id), total NUMERIC, "
     "placed_at DATE);\n\nWrite a SQL query that returns, for each country, the number of customers who placed "
     "at least one order in 2023 and the average order total for those customers, sorted by the average.", 128),
    ("The lighthouse keeper had not seen another person in forty-one days when the boat appeared on the horizon. "
     "At first she thought it was a trick of the light, the way the sea sometimes painted shapes at dusk, but the "
     "shape grew larger and steadier, and soon she could hear the engine coughing over the waves.", 128),
    ("Dear team,\n\nThe quarterly review meeting has been moved from Thursday to Friday at 10 am because the "
     "conference room is being renovated. Please update your calendars and send me your slides by Wednesday "
     "evening so I can merge them into one deck.\n\nThanks,\nPriya\n\nWrite a short, polite reply confirming "
     "attendance and asking whether remote participation is possible.", 128),
    ("#!/bin/bash\n# Back up every .conf file under /etc to /backup with a date suffix\n"
     "for f in $(find /etc -name *.conf); do\n  cp $f /backup/$(basename $f).$(date +%F)\ndone\n\n"
     "This script breaks on some file names and can fail silently. Explain why and rewrite it safely.", 128),
    ("| City | Population (millions) | Area (km²) |\n|---|---|---|\n| Tokyo | 13.9 | 2,194 |\n"
     "| Delhi | 16.8 | 1,484 |\n| Shanghai | 24.9 | 6,341 |\n| São Paulo | 12.3 | 1,521 |\n"
     "| Mexico City | 9.2 | 1,485 |\n\nWhich city in the table has the highest population density? "
     "Show the calculation for each city.", 128),
]
# wikitext excerpts: (target prompt tokens, n_new, instruction or None for plain continuation)
WIKI = [(260, 96, "Summarize the text above in three sentences."), (290, 96, None),
        (320, 96, "What are the two most important facts in this passage?"), (350, 96, None),
        (380, 96, "Write a one-paragraph summary for a general audience."), (410, 96, None),
        (430, 96, "List the people, places and dates mentioned above."), (450, 96, None),
        (500, 88, "Summarize the text above."), (530, 88, None), (560, 80, "What is this passage mainly about?"),
        (590, 80, None), (620, 72, "Summarize the text above in two sentences."), (640, 72, None),
        (665, 64, "List three key points from the passage."), (690, 64, None)]


def articles(text):
    """Split wikitext into (title, body) on top-level ' = Title = ' headings."""
    parts = re.split(r"\n = ([^=\n][^\n]*?) = \n", "\n" + text)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def clean(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r"\n *=+ ([^=\n]+?) =+ *\n", r"\n\1\n", s)
    s = re.sub(r" ([.,;:!?)'])", r"\1", s).replace("( ", "(")
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", default=f"{H}/runs/p1/src_wikitext/test.txt")
    ap.add_argument("--day1", default=f"{H}/runs/val/prompts.json")
    ap.add_argument("--long", default=f"{H}/runs/val_long/prompts_long.json")
    ap.add_argument("--model-path", default=f"{H}/qwen-30b-a3b/hf_model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    raw = open(a.wikitext, encoding="utf-8").read()
    arts = articles(raw)
    rng = random.Random(SEED)
    order = list(range(len(arts)))
    rng.shuffle(order)
    ppl, used = [], set()
    for i in order:  # perplexity passages: raw text (standard wikitext format), first PPL_LEN tokens of the body
        ids = tok(arts[i][1].strip(), add_special_tokens=False).input_ids
        if len(ids) >= PPL_LEN and len(ppl) < N_PPL:
            ppl.append({"id": f"w{len(ppl):02d}", "article": arts[i][0], "ids": ids[:PPL_LEN]})
            used.add(i)
    prompts = [{"id": f"p{i:02d}", "prompt": p, "n_new": 128, "source": "day1"}
               for i, p in enumerate(json.load(open(a.day1)))]
    prompts += [{"id": f"p{20 + i:02d}", "prompt": p, "n_new": 64, "source": "day3_long"}
                for i, p in enumerate(json.load(open(a.long)))]
    prompts += [{"id": f"p{24 + i:02d}", "prompt": p, "n_new": n, "source": "hand"} for i, (p, n) in enumerate(HAND)]
    rest = [i for i in order if i not in used]
    for j, (target, n_new, instr) in enumerate(WIKI):
        suffix = f"\n\n{instr}\n" if instr else ""
        nsuf = len(tok(suffix).input_ids) if instr else 0
        for i in rest:  # next unused article long enough for this target
            body = clean(arts[i][1])
            ids = tok(body).input_ids
            if len(ids) >= target:
                rest.remove(i)
                break
        else:
            raise SystemExit(f"no article long enough for target {target}")
        cut = tok.decode(ids[: target - nsuf])
        cut = cut[: cut.rfind(" ")] if instr else cut  # instruction prompts end at a word boundary
        prompts.append({"id": f"p{48 + j:02d}", "prompt": cut + suffix, "n_new": n_new,
                        "source": f"wikitext-2-raw-v1 test: {arts[i][0]}"})
    for p in prompts:
        p["plen"] = len(tok([p["prompt"]]).input_ids[0])
        assert p["plen"] + p["n_new"] <= MAX_TOTAL and 64 <= p["n_new"] <= 128, p["id"]
    meta = {"seed": SEED, "max_total": MAX_TOTAL, "n_prompts": len(prompts), "n_ppl": len(ppl), "ppl_len": PPL_LEN,
            "wikitext_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "wikitext_source": "Salesforce/wikitext wikitext-2-raw-v1 test (CC BY-SA 3.0), datasets-server rows"}
    os.makedirs(a.out, exist_ok=True)
    out = f"{a.out}/corpus.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "prompts": prompts, "ppl": ppl}, f, ensure_ascii=False, indent=1)
    print(json.dumps({**meta, "plens": [p["plen"] for p in prompts],
                      "ppl_articles": [p["article"] for p in ppl]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
