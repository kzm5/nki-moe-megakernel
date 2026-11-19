"""H2.2 emulated in the CPU golden: accumulation precision of the expert GEMMs, measured against the fp32 golden.

Computes (everything fp32 except the expert MLPs' gate/up/down projections):
    bf16in   inputs and weights rounded to bf16, fp32 accumulation, outputs rounded to bf16 (PSUM-fp32 hardware)
    bf16acc  as bf16in, but the K dimension is accumulated in bf16 across CHUNK-wide partial products
Against the frozen golden: KL(golden || variant) of the logits at every corpus position, top-1 agreement, and
residual rel-L2 per layer at the 512 sampled positions of ~/runs/p1/golden/layers.pt.

Usage:
    h22_accum.py --variants bf16in bf16acc --out ~/runs/p1/h22/h22.json
"""
import argparse, gc, json, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validator"))
import golden as G  # noqa: E402
import validate as V  # noqa: E402

CHUNK = 128
MODE = {"v": "fp32"}


def lin(x, w):
    if MODE["v"] == "fp32":
        return x @ w.T
    xb, wb = x.bfloat16().float(), w.bfloat16().float()
    if MODE["v"] == "bf16in":
        return (xb @ wb.T).bfloat16().float()
    acc = None
    for k in range(0, xb.shape[-1], CHUNK):
        p = (xb[..., k:k + CHUNK] @ wb[:, k:k + CHUNK].T).bfloat16()
        acc = p if acc is None else (acc + p).bfloat16()
    return acc.float()


def mlp_forward(self, x):
    h = self.act_fn(lin(x, self.gate_proj.weight)) * lin(x, self.up_proj.weight)
    if MODE["v"] != "fp32":
        h = h.bfloat16().float()
    return lin(h, self.down_proj.weight)


def run(variant, corpus, model_path):
    MODE["v"] = variant
    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = "sdpa"
    W = G.Weights(model_path)
    seqs, rows = [], []
    for p in corpus["prompts"]:
        e = torch.load(f"{V.REFS}/expected_p{p['id'][1:]}.pt")
        plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
        seqs.append(e["sequences"][:, :plen + n - 1])
        rows.append(torch.arange(plen - 1, plen + n - 1))
    lay = torch.load(f"{V.GOLDEN}/layers.pt")
    samp = {}
    for k, (j, q) in enumerate(lay["index"]):
        samp.setdefault(j, []).append((k, q))
    emb = W.load("model.embed_tokens.")["weight"]
    hs = [emb[s[0]].unsqueeze(0) for s in seqs]
    del emb
    rope = M.Qwen3MoeRotaryEmbedding(cfg)
    ctx = []
    for h in hs:
        T = h.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        mask = create_causal_mask(config=cfg, input_embeds=h, attention_mask=None, cache_position=torch.arange(T),
                                  past_key_values=None, position_ids=pos)
        ctx.append((pos, mask, rope(h, pos)))
    err2 = torch.zeros(48, dtype=torch.float64)
    gold2 = torch.zeros(48, dtype=torch.float64)
    t0 = time.time()
    with torch.inference_mode():
        for li in range(cfg.num_hidden_layers):
            layer = G.build(M.Qwen3MoeDecoderLayer, W.load(f"model.layers.{li}."), cfg, li)
            for j, h in enumerate(hs):
                pos, mask, pe = ctx[j]
                hs[j] = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
                for k, q in samp.get(j, []):
                    x, g = hs[j][0, rows[j][q]].double(), lay["resid"][k, li].double()
                    err2[li] += (x - g).pow(2).sum()
                    gold2[li] += g.pow(2).sum()
            del layer
            gc.collect()
            print(f"{variant} layer {li:2d} {time.time() - t0:.0f}s", flush=True)
        norm = G.build(M.Qwen3MoeRMSNorm, W.load("model.norm."), cfg.hidden_size, cfg.rms_norm_eps)
        w = W.load("lm_head.")["weight"]
        kls, top1 = [], []
        for j, p in enumerate(corpus["prompts"]):
            lg = (norm(hs[j])[0, rows[j]] @ w.T).float()
            g = torch.load(f"{V.GOLDEN}/golden_p{p['id'][1:]}.pt")["logits"]
            kls.append(V.kl(g, lg))
            top1.append(lg.argmax(-1) == g.argmax(-1))
    kl = torch.cat(kls)
    return {"variant": variant, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1_golden": float(torch.cat(top1).float().mean()), "n_positions": int(kl.numel()),
            "resid_rel_l2": [float(x) for x in (err2 / gold2).sqrt()], "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bf16in", "bf16acc"])
    ap.add_argument("--model-path", default=os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    M.Qwen3MoeMLP.forward = mlp_forward
    corpus = json.load(open(V.CORPUS))
    res = {"chunk": CHUNK, "results": []}
    for v in a.variants:
        res["results"].append(run(v, corpus, a.model_path))
        r = res["results"][-1]
        print("H22", json.dumps({k: r[k] for k in ("variant", "kl_mean", "kl_max", "top1_golden")}),
              "resid L0/L24/L47", [round(r["resid_rel_l2"][L], 5) for L in (0, 24, 47)], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
