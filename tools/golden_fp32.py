"""Layer-by-layer CPU reference forward for Qwen3-30B-A3B (teacher-forced, no KV cache).

Computes:
    logits = lm_head(norm(L47(...L0(embed(ids)))))
    using the stock HF Qwen3MoeDecoderLayer, one layer resident at a time.

Tiled computation:
    For each layer: load its weights from the safetensors shards, cast to --dtype, run every
    sequence through it, free it. Peak RAM is about one fp32 layer (~2.5 GB) plus embed/lm_head.

Returns (per input, --out/<name>.pt):
    logits      [n_rows, V] fp32, rows = positions plen-1 .. plen+n_new-2 (the logits that
                predict each generated token), or all positions with --all-positions
    router_top  [L, T, 9] fp32 top-9 router logits; router_ids [L, T, 9] int16

Note:
    --check-hf loads the stock HF model in bf16 (~61 GB RAM) and asserts the layerwise bf16
    logits match it on one prompt; run it only while nothing else is using host RAM.
"""
import argparse, gc, glob, json, os, time
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}


class Weights:
    def __init__(self, path):
        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def load(self, prefix, dtype):
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


DUMP_KEYS = ("layer_in", "attn_out", "moe_in", "router", "moe_out", "layer_out")


@torch.inference_mode()
def forward_layerwise(model_path, seqs, dtype, attn_impl="sdpa", log=print, dump=None, dump_keys=DUMP_KEYS):
    """seqs: list of [1, T] int64. Returns final-norm hidden states, router tops per sequence and,
    for dump = {seq_index: [rows]}, per-layer tensors at those rows ({seq_index: {key: [L, rows, ...]}})."""
    dump = dump or {}
    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = attn_impl
    W = Weights(model_path)
    emb = W.load("model.embed_tokens.", dtype)["weight"]
    hs = [emb[s[0]].unsqueeze(0) for s in seqs]
    del emb
    rope = Qwen3MoeRotaryEmbedding(cfg)
    ctx = []
    for h in hs:
        T = h.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        mask = create_causal_mask(config=cfg, input_embeds=h, attention_mask=None,
                                  cache_position=torch.arange(T), past_key_values=None, position_ids=pos)
        ctx.append((pos, mask, rope(h, pos)))
    tops = [[] for _ in seqs]
    dumps = {j: {k: [] for k in dump_keys} for j in dump}
    for li in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = build(Qwen3MoeDecoderLayer, W.load(f"model.layers.{li}.", dtype), cfg, li)
        rec = {}
        hooks = [layer.mlp.gate.register_forward_hook(lambda m, i, o: rec.__setitem__("router", o)),
                 layer.self_attn.register_forward_hook(lambda m, i, o: rec.__setitem__("attn_out", o[0])),
                 layer.mlp.register_forward_hook(lambda m, i, o: rec.update(moe_in=i[0], moe_out=o[0]))]
        for j, h in enumerate(hs):
            pos, mask, pe = ctx[j]
            out = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
            v, ix = rec["router"].float().topk(9, dim=-1)
            tops[j].append((v, ix.to(torch.int16)))
            if j in dump:
                r = dump[j]
                got = {"layer_in": h[0, r], "attn_out": rec["attn_out"][0, r], "moe_in": rec["moe_in"][0, r],
                       "router": rec["router"].view(h.shape[1], -1)[r], "moe_out": rec["moe_out"][0, r],
                       "layer_out": out[0, r]}
                for k in dump_keys:
                    dumps[j][k].append(got[k].float().clone())
            hs[j] = out
        for hk in hooks:
            hk.remove()
        del layer, rec
        gc.collect()
        log(f"layer {li:2d} {time.time() - t0:6.1f}s")
    norm = build(Qwen3MoeRMSNorm, W.load("model.norm.", dtype), cfg.hidden_size, cfg.rms_norm_eps)
    hs = [norm(h) for h in hs]
    dumps = {j: {k: torch.stack(v) for k, v in d.items()} for j, d in dumps.items()}
    return (hs, tops, W, dumps) if dump else (hs, tops, W)


@torch.inference_mode()
def lm_head(W, dtype, hs, rows):
    w = W.load("lm_head.", dtype)["weight"]
    return [(h[0, r] @ w.T).float() for h, r in zip(hs, rows)]


def check_hf(model_path, prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok([prompt], return_tensors="pt").input_ids
    hs, _, W = forward_layerwise(model_path, [ids], torch.bfloat16)
    ours = lm_head(W, torch.bfloat16, hs, [slice(None)])[0]
    del hs
    gc.collect()
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        ref = m(ids).logits[0].float()
    d = (ours - ref).abs()
    res = {"prompt": prompt, "T": ids.shape[1], "max_abs": float(d.max()), "bit_identical": bool(torch.equal(ours, ref)),
           "top1_agree": float((ours.argmax(-1) == ref.argmax(-1)).float().mean()),
           "rel_l2": float((ours - ref).norm() / ref.norm())}
    print("CHECK_HF", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    ap.add_argument("--dtype", choices=DTYPES, default="fp32")
    ap.add_argument("--inputs", nargs="*", default=[], help="expected_pNN.pt files (input_ids + sequences)")
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--out", default=os.path.expanduser("~/runs/golden"))
    ap.add_argument("--all-positions", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--check-hf", metavar="PROMPT")
    ap.add_argument("--dump", default="", help="NN:P,... — per-layer tensors at prompt NN, generated position P (or all)")
    ap.add_argument("--dump-keys", default=",".join(DUMP_KEYS))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.check_hf:
        check_hf(a.model_path, a.check_hf)
        return
    os.makedirs(a.out, exist_ok=True)
    seqs, rows, names, plens = [], [], [], []
    want = {}
    for t in filter(None, a.dump.split(",")):
        nn, p = t.split(":")
        want.setdefault(int(nn), []).append(p if p == "all" else int(p))
    n_news = []
    for p in sorted(sum((glob.glob(x) for x in a.inputs), [])):
        e = torch.load(p)
        plen = e["input_ids"].shape[1]
        n_new = e["logits"].shape[0] if "logits" in e else a.n_new
        seqs.append(e["sequences"][:, : plen + n_new - 1])
        rows.append(slice(None) if a.all_positions else slice(plen - 1, plen + n_new - 1))
        names.append(os.path.basename(p).replace("expected_", "golden_"))
        plens.append(plen)
        n_news.append(n_new)
    dump = {}
    for j, n in enumerate(names):
        nn = int(n[len("golden_p"):-3])
        if nn in want:
            want[nn] = list(range(n_news[j])) if "all" in want[nn] else want[nn]
            dump[j] = [plens[j] - 1 + p for p in want[nn]]  # row whose output predicts generated position p
    t0 = time.time()
    dt = DTYPES[a.dtype]
    r = forward_layerwise(a.model_path, seqs, dt, log=lambda s: print(s, flush=True), dump=dump,
                          dump_keys=tuple(a.dump_keys.split(",")))
    hs, tops, W = r[:3]
    for j, d in (r[3].items() if dump else []):
        nn = int(names[j][len("golden_p"):-3])
        torch.save({"dtype": a.dtype, "positions": want[nn], "rows": dump[j], **d},
                   os.path.join(a.out, f"dump_p{nn:02d}.pt"))
        print(f"DUMP p{nn:02d} positions={want[nn]}", flush=True)
    logits = lm_head(W, dt, hs, rows)
    for n, s, lg, tp in zip(names, seqs, logits, tops):
        torch.save({"dtype": a.dtype, "input_ids": s, "logits": lg,
                    "router_top": torch.stack([v for v, _ in tp]), "router_ids": torch.stack([i for _, i in tp])},
                   os.path.join(a.out, n))
    print(f"GOLDEN_DONE n={len(seqs)} tokens={sum(s.shape[1] for s in seqs)} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
