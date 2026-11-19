"""Router precision emulated on CPU in a bf16 model, scored against the frozen fp32 golden.

Computes (weights, activations and residual stream in bf16; HF Qwen3MoeDecoderLayer, one layer resident at a time):
    bf16logits  router logits rounded to bf16, then softmax / top-k in fp32
    fp32        bf16 router input and weights, fp32 logits, softmax and top-k (NxDI router_config.dtype=float32)
    fp32in      as fp32, but the router input is the post-attention RMSNorm output before its bf16 rounding
Against the golden: KL(golden || variant) and top-1 at every corpus position, tie-tolerant router flips (validator
rule) and strict flips per layer, perplexity on the ppl passages, residual rel-L2 per layer at the sampled positions.

Usage:
    router_emul.py --variants bf16logits fp32 fp32in --out ~/runs/p1/t2/router_emul.json
"""
import argparse, gc, json, math, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validator"))
import golden as G  # noqa: E402
import validate as V  # noqa: E402

BF = torch.bfloat16
STATE = {"v": "fp32", "norm32": None, "logits": None}


def norm_forward(self, x):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    STATE["norm32"] = self.weight.float() * h
    return self.weight * h.to(x.dtype)


def moe_forward(self, x):
    b, s, d = x.shape
    x = x.view(-1, d)
    v = STATE["v"]
    if v == "bf16logits":
        logits = self.gate(x).float()
    else:
        xin = STATE["norm32"].view(-1, d) if v == "fp32in" else x.float()
        logits = xin @ self.gate.weight.float().T
    STATE["logits"] = logits
    rw, sel = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), self.top_k, dim=-1)
    if self.norm_topk_prob:
        rw /= rw.sum(-1, keepdim=True)
    rw = rw.to(x.dtype)
    out = torch.zeros_like(x)
    for e in sel.unique().tolist():
        tok, slot = torch.where(sel == e)
        out.index_add_(0, tok, self.experts[e](x[tok]) * rw[tok, slot, None])
    return out.view(b, s, d), logits


def flip_rows(lg, gi):
    """lg [n,E] candidate logits, gi [n,9] golden ids -> (tie-tolerant, strict) flips [n]."""
    ids = lg.topk(8, -1).indices
    tol, strict = torch.zeros(len(lg)), torch.zeros(len(lg))
    for s in range(len(lg)):
        gset = gi[s, :8]
        same = set(ids[s].tolist()) == set(gset.tolist())
        tie = V.valid_top8(lg[s], ids[s]) and V.valid_top8(lg[s], gset)
        tol[s], strict[s] = float(not (same or tie)), float(not same)
    return tol, strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bf16logits", "fp32", "fp32in"])
    ap.add_argument("--model-path", default=os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-layers", type=int, default=48)
    ap.add_argument("--max-prompts", type=int, default=0, help="smoke test: first N prompts, one ppl passage")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    M.Qwen3MoeRMSNorm.forward = norm_forward
    M.Qwen3MoeSparseMoeBlock.forward = moe_forward
    corpus = json.load(open(V.CORPUS))
    if a.max_prompts:
        corpus = {"prompts": corpus["prompts"][:a.max_prompts], "ppl": corpus["ppl"][:1]}
    cfg = AutoConfig.from_pretrained(a.model_path)
    cfg._attn_implementation = "sdpa"
    W = G.Weights(a.model_path)

    seqs, rows, gids = [], [], []
    for p in corpus["prompts"]:
        e = torch.load(f"{V.REFS}/expected_p{p['id'][1:]}.pt")
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        seqs.append(e["sequences"][:, :plen + n - 1])
        rows.append(torch.arange(plen - 1, plen + n - 1))
        gids.append(torch.load(f"{V.GOLDEN}/golden_p{p['id'][1:]}.pt")["router_ids"].long())
    n_prompts = len(seqs)
    for w in corpus["ppl"]:
        seqs.append(torch.tensor([w["ids"]])[:, :-1])
        rows.append(torch.arange(len(w["ids"]) - 1))
    lay = torch.load(f"{V.GOLDEN}/layers.pt")
    samp = {}
    for k, (j, q) in enumerate(lay["index"]):
        samp.setdefault(j, []).append((k, q))

    emb = W.load("model.embed_tokens.", BF)["weight"]
    hs = {v: [emb[s[0]].unsqueeze(0) for s in seqs] for v in a.variants}
    del emb
    rope = M.Qwen3MoeRotaryEmbedding(cfg)
    ctx = []
    for h in hs[a.variants[0]]:
        T = h.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        mask = create_causal_mask(config=cfg, input_embeds=h, attention_mask=None, cache_position=torch.arange(T),
                                  past_key_values=None, position_ids=pos)
        ctx.append((pos, mask, rope(h, pos)))
    st = {v: {"tol": torch.zeros(48), "strict": torch.zeros(48), "err2": torch.zeros(48, dtype=torch.float64),
              "gold2": torch.zeros(48, dtype=torch.float64)} for v in a.variants}
    n_rows = sum(len(rows[j]) for j in range(n_prompts))
    t0 = time.time()
    with torch.inference_mode():
        for li in range(a.max_layers):
            layer = G.build(M.Qwen3MoeDecoderLayer, W.load(f"model.layers.{li}.", BF), cfg, li)
            for v in a.variants:
                STATE["v"] = v
                for j, h in enumerate(hs[v]):
                    pos, mask, pe = ctx[j]
                    hs[v][j] = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
                    if j >= n_prompts:
                        continue
                    tol, strict = flip_rows(STATE["logits"][rows[j]], gids[j][li])
                    st[v]["tol"][li] += tol.sum()
                    st[v]["strict"][li] += strict.sum()
                    for k, q in samp.get(j, []):
                        x, g = hs[v][j][0, rows[j][q]].double(), lay["resid"][k, li].double()
                        st[v]["err2"][li] += (x - g).pow(2).sum()
                        st[v]["gold2"][li] += g.pow(2).sum()
            del layer
            gc.collect()
            print(f"layer {li:2d} {time.time() - t0:.0f}s " + " ".join(
                f"{v}:{st[v]['tol'][li] / n_rows * 100:.2f}%" for v in a.variants), flush=True)
        norm = G.build(M.Qwen3MoeRMSNorm, W.load("model.norm.", BF), cfg.hidden_size, cfg.rms_norm_eps)
        wl = W.load("lm_head.", BF)["weight"]
        res = {"n_positions": n_rows, "results": []}
        for v in a.variants:
            kls, top1, nll = [], [], []
            for j in range(len(seqs)):
                lg = (norm(hs[v][j])[0, rows[j]] @ wl.T).float()
                if j < n_prompts:
                    g = torch.load(f"{V.GOLDEN}/golden_p{corpus['prompts'][j]['id'][1:]}.pt")["logits"]
                    kls.append(V.kl(g, lg))
                    top1.append(lg.argmax(-1) == g.argmax(-1))
                else:
                    tgt = seqs[j][0, 1:].tolist() + [corpus["ppl"][j - n_prompts]["ids"][-1]]
                    nll.append(-F.log_softmax(lg.double(), -1).gather(1, torch.tensor(tgt)[:, None])[:, 0])
            kl, s = torch.cat(kls), st[v]
            res["results"].append({
                "variant": v, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                "top1_golden": float(torch.cat(top1).float().mean()), "ppl": math.exp(float(torch.cat(nll).mean())),
                "flip_pct": float(s["tol"].sum() / (n_rows * 48) * 100),
                "flip_pct_strict": float(s["strict"].sum() / (n_rows * 48) * 100),
                "flip_pct_per_layer": [round(float(x / n_rows * 100), 3) for x in s["tol"]],
                "resid_rel_l2": [float(x) for x in (s["err2"] / s["gold2"]).sqrt()]})
            r = res["results"][-1]
            print("ROUTER_EMUL", json.dumps({k: r[k] for k in ("variant", "kl_mean", "top1_golden", "ppl", "flip_pct",
                                                               "flip_pct_strict")}), flush=True)
        res["seconds"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
