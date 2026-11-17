"""fp32 CPU golden for the validator corpus, plus per-layer error statistics of device captures.

Computes:
    One layer-by-layer fp32 forward (stock HF Qwen3MoeDecoderLayer, one layer resident at a time) over every
    corpus prompt (reference tokens) and every perplexity passage. Per layer, at every generated position p
    (golden row plen - 1 + p) and for each --stats build:
        boundaries  attn = self_attn output, moe = mlp output, resid = layer output
        err2 = ||cand - gold||^2, gold2 = ||gold||^2, dot = <cand, gold>, cand2 = ||cand||^2, maxabs
    Golden per-layer tensors are kept only at --sample seeded positions.

Returns (under --out):
    golden_pNN.pt    {"logits" [n,V] fp32, "router_top" [48,n,9], "router_ids" [48,n,9]}   (--write-golden)
    ppl_wNN.pt       {"nll" [L-1] fp64, "argmax"}                                          (--write-golden)
    layers.pt        {"index" [(prompt, pos)], "attn"/"moe"/"resid" [S,48,H] fp32}         (--write-golden)
    DEVICE_OUT/stats.pt  {"index" [(prompt, pos)], "<boundary>.<stat>" [P,48]}             (per --stats)

Usage:
    golden.py --corpus corpus.json --refs REFS --out DIR [--write-golden] [--stats NAME=DEVICE_OUT ...]
"""
import argparse, gc, json, os, random, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding)

H = os.path.expanduser("~")
SAMPLE_SEED = 1117
BOUNDARIES = ("attn", "moe", "resid")
STATS = ("err2", "gold2", "dot", "cand2", "maxabs")


class Weights:
    def __init__(self, path):
        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def load(self, prefix, dtype=torch.float32):
        by_file = {}
        for k, f in self.map.items():
            if k.startswith(prefix):
                by_file.setdefault(f, []).append(k)
        out = {}
        for f, keys in by_file.items():
            with safe_open(os.path.join(self.path, f), framework="pt") as h:
                for k in keys:
                    out[k[len(prefix):]] = h.get_tensor(k).to(dtype)
        return out


def build(cls, sd, *args):
    with torch.device("meta"):
        m = cls(*args)
    m.load_state_dict(sd, assign=True, strict=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=f"{H}/qwen-30b-a3b/hf_model")
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--stats", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    corpus = json.load(open(a.corpus))
    seqs, rows, names = [], [], []  # rows: golden rows of generated positions 0..n-1
    for p in corpus["prompts"]:
        e = torch.load(f"{a.refs}/expected_p{p['id'][1:]}.pt")
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        assert plen == p["plen"] and n == p["n_new"], p["id"]
        seqs.append(e["sequences"][:, : plen + n - 1])
        rows.append(torch.arange(plen - 1, plen + n - 1))
        names.append(p["id"])
    n_prompts = len(seqs)
    for w in corpus["ppl"]:
        seqs.append(torch.tensor([w["ids"]])[:, :-1])
        rows.append(torch.arange(len(w["ids"]) - 1))
        names.append(w["id"])
    index = [(j, int(q)) for j in range(n_prompts) for q in range(len(rows[j]))]
    rng = random.Random(SAMPLE_SEED)
    sample = sorted(rng.sample(range(len(index)), min(a.sample, len(index)))) if a.write_golden else []
    sample_by_seq = {}
    for s in sample:
        sample_by_seq.setdefault(index[s][0], []).append(index[s][1])

    builds = {}
    for spec in a.stats:
        name, d = spec.split("=", 1)
        caps = []
        for p in corpus["prompts"]:
            c = torch.load(f"{d}/cap/cap_p{p['id'][1:]}.pt")
            caps.append(c)
        builds[name] = {"caps": caps, "dir": d, **{f"{b}.{s}": torch.zeros(len(index), 48, dtype=torch.float64)
                                         for b in BOUNDARIES for s in STATS}}
    offs = [0]
    for j in range(n_prompts):
        offs.append(offs[-1] + len(rows[j]))

    cfg = AutoConfig.from_pretrained(a.model_path)
    cfg._attn_implementation = "sdpa"
    W = Weights(a.model_path)
    emb = W.load("model.embed_tokens.")["weight"]
    hs = [emb[s[0]].unsqueeze(0) for s in seqs]
    del emb
    rope = Qwen3MoeRotaryEmbedding(cfg)
    ctx = []
    for h in hs:
        T = h.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        mask = create_causal_mask(config=cfg, input_embeds=h, attention_mask=None, cache_position=torch.arange(T),
                                  past_key_values=None, position_ids=pos)
        ctx.append((pos, mask, rope(h, pos)))
    tops = [[] for _ in range(n_prompts)]
    dumps = {b: torch.zeros(len(sample), 48, cfg.hidden_size) for b in BOUNDARIES} if sample else {}
    t_all = time.time()
    with torch.inference_mode():
        for li in range(cfg.num_hidden_layers):
            t0 = time.time()
            layer = build(Qwen3MoeDecoderLayer, W.load(f"model.layers.{li}."), cfg, li)
            rec = {}
            hooks = [layer.mlp.gate.register_forward_hook(lambda m, i, o: rec.__setitem__("router", o)),
                     layer.self_attn.register_forward_hook(lambda m, i, o: rec.__setitem__("attn", o[0])),
                     layer.mlp.register_forward_hook(lambda m, i, o: rec.__setitem__("moe", o[0]))]
            for j, h in enumerate(hs):
                pos, mask, pe = ctx[j]
                out = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
                hs[j] = out
                if j >= n_prompts:
                    continue
                r = rows[j]
                v, ix = rec["router"].view(h.shape[1], -1)[r].float().topk(9, dim=-1)
                tops[j].append((v, ix.to(torch.int16)))
                g = {"attn": rec["attn"][0, r], "moe": rec["moe"].reshape(h.shape[1], -1)[r], "resid": out[0, r]}
                for b, st in builds.items():
                    c = st["caps"][j]
                    ok = c["valid"]
                    for bd in BOUNDARIES:
                        x, y = c[bd][:, li].float(), g[bd]
                        sl = slice(offs[j], offs[j + 1])
                        st[f"{bd}.err2"][sl, li] = (x - y).double().pow(2).sum(-1)
                        st[f"{bd}.gold2"][sl, li] = y.double().pow(2).sum(-1)
                        st[f"{bd}.dot"][sl, li] = (x.double() * y.double()).sum(-1)
                        st[f"{bd}.cand2"][sl, li] = x.double().pow(2).sum(-1)
                        st[f"{bd}.maxabs"][sl, li] = (x - y).abs().amax(-1).double()
                        for s in STATS:
                            st[f"{bd}.{s}"][sl, li][~ok] = float("nan")
                for q in sample_by_seq.get(j, []):
                    k = sample.index(next(s for s in sample if index[s] == (j, q)))
                    for bd in BOUNDARIES:
                        dumps[bd][k, li] = g[bd][q]
            for hk in hooks:
                hk.remove()
            del layer, rec
            gc.collect()
            print(f"layer {li:2d} {time.time() - t0:6.1f}s", flush=True)
        norm = build(Qwen3MoeRMSNorm, W.load("model.norm."), cfg.hidden_size, cfg.rms_norm_eps)
        hs = [norm(h) for h in hs]
        if a.write_golden:
            w = W.load("lm_head.")["weight"]
            for j, (h, r) in enumerate(zip(hs, rows)):
                lg = (h[0, r] @ w.T).float()
                if j < n_prompts:
                    torch.save({"logits": lg, "router_top": torch.stack([v for v, _ in tops[j]]),
                                "router_ids": torch.stack([i for _, i in tops[j]])},
                               f"{a.out}/golden_p{names[j][1:]}.pt")
                else:
                    tgt = seqs[j][0, 1:].tolist() + [corpus["ppl"][j - n_prompts]["ids"][-1]]
                    lp = F.log_softmax(lg.double(), -1)
                    nll = -lp.gather(1, torch.tensor(tgt)[:, None])[:, 0]
                    torch.save({"nll": nll, "argmax": lg.argmax(-1)}, f"{a.out}/ppl_{names[j]}.pt")
            torch.save({"index": [index[s] for s in sample], "prompt_ids": [names[index[s][0]] for s in sample],
                        **dumps}, f"{a.out}/layers.pt")
    for b, st in builds.items():
        st.pop("caps")
        d = st.pop("dir")
        torch.save({"index": index, "prompt_ids": names[:n_prompts], **st}, f"{d}/stats.pt")
    print(f"GOLDEN_DONE prompts={n_prompts} ppl={len(seqs) - n_prompts} builds={list(builds)} "
          f"{time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
