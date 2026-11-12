"""Per-layer error vs fp32 golden at target decode positions, from tensor-capture builds.

Computes (per target, per layer, per build):
    rel-L2 vs golden of: residual into the layer, attention-block output, MoE input after the
    post-attention RMSNorm, MoE-block output; ratio build/XLA when an XLA capture exists;
    router top-8 recomputed on CPU from the build's normalized MoE input with the HF gate weights
    (fused NKI path: raw residual captured, normalized on CPU), vs golden's top-8
Reports the first layer where any ratio exceeds --jump (default 10x) and which tensor departs first.

Usage:
    bisect_compare.py --golden ~/runs/bisect/golden --builds libmoe=~/runs/bisect/libmoe [xla=~/runs/bisect/xla] [--out OUT.json]
"""
import argparse, glob, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_fp32 import Weights  # noqa: E402

N_LAYERS = 48
H = os.path.expanduser("~")


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm())


def device_layers(cap, p, mlp_in_normalized, norm):
    """mlp.in is the post-attention-RMSNorm output for XLA, the raw residual for the fused NKI MoE path."""
    t, names = cap["tensors"][p], cap["names"]
    d = {n: x.reshape(-1, x.shape[-1])[-1] for n, x in zip(names, t)}  # last row = current token
    out = []
    for i in range(N_LAYERS):
        x = d[f"layers.{i}.mlp.in"]
        out.append({"resid_in": d[f"layers.{i}.input_layernorm.in"], "attn_out": d[f"layers.{i}.self_attn.o_proj.out"],
                    "moe_in": x if mlp_in_normalized else norm(x, i), "moe_out": d[f"layers.{i}.mlp.out"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=f"{H}/runs/bisect/golden")
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--model-path", default=f"{H}/qwen-30b-a3b/hf_model")
    ap.add_argument("--jump", type=float, default=10.0)
    ap.add_argument("--out", default=f"{H}/runs/bisect/bisect.json")
    a = ap.parse_args()
    builds = dict(b.split("=", 1) for b in a.builds)
    W = Weights(a.model_path)
    eps = json.load(open(f"{a.model_path}/config.json"))["rms_norm_eps"]
    norm_w = [W.load(f"model.layers.{i}.post_attention_layernorm.", torch.float32)["weight"] for i in range(N_LAYERS)]
    gate_w = [W.load(f"model.layers.{i}.mlp.gate.", torch.float32)["weight"] for i in range(N_LAYERS)]

    def norm(x, i):
        x = x.double()
        return x * torch.rsqrt(x.pow(2).mean() + eps) * norm_w[i].double()

    def router_top8(h, i):
        return set((gate_w[i].double() @ h.double()).topk(8).indices.tolist())

    res = []
    for gf in sorted(glob.glob(f"{a.golden}/dump_p*.pt")):
        g = torch.load(gf)
        nn = int(os.path.basename(gf)[len("dump_p"):-3])
        for j, p in enumerate(g["positions"]):
            gl = [{"resid_in": g["layer_in"][i, j], "attn_out": g["attn_out"][i, j], "moe_in": g["moe_in"][i, j],
                   "moe_out": g["moe_out"][i, j], "top8": set(g["router"][i, j].topk(8).indices.tolist())}
                  for i in range(N_LAYERS)]
            row = {"prompt": nn, "pos": p, "layers": []}
            dev = {}
            for b, d in builds.items():
                cf = f"{d}/cap_p{nn:02d}.pt"
                if os.path.exists(cf):
                    cap = torch.load(cf)
                    if p in cap["tensors"] and cap["tensors"][p][0].numel() > 1:  # (1,) = capture fallback
                        dev[b] = device_layers(cap, p, b == "xla", norm)
            if not dev:
                continue
            for i in range(N_LAYERS):
                L = {"layer": i}
                for b, dl in dev.items():
                    for k in ("resid_in", "attn_out", "moe_in", "moe_out"):
                        L[f"{b}.{k}"] = rel(dl[i][k], gl[i][k])
                    L[f"{b}.router_top8_match"] = len(router_top8(dl[i]["moe_in"], i) & gl[i]["top8"])
                if "xla" in dev:
                    for b in dev:
                        if b != "xla":
                            for k in ("resid_in", "attn_out", "moe_in", "moe_out"):
                                L[f"{b}/xla.{k}"] = L[f"{b}.{k}"] / max(L[f"xla.{k}"], 1e-12)
                row["layers"].append(L)
            first = None
            for L in row["layers"]:
                hits = [k for k in L if "/xla." in k and L[k] > a.jump]
                if hits:
                    first = {"layer": L["layer"], "tensors": hits}
                    break
            row["first_jump"] = first
            res.append(row)
            msg = {b: [round(row["layers"][i][f"{b}.resid_in"], 4) for i in (0, 12, 24, 36, 47)] for b in dev}
            print(f"p{nn:02d}@{p} resid_in rel-L2 at layers 0/12/24/36/47: {msg} first_jump={first}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
