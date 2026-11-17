"""Device-side capture for the validator (one build per process).

Computes (in order, each optional):
    free    free-running greedy decode of corpus prompts -> reference files (day-1 format)
    tf      teacher-forced decode of every corpus prompt on the reference tokens; per-step scores, and with
            --capture per layer: attention output, MoE output, layer output (residual), router logits [, ids]
    ppl     teacher-forced decode of each perplexity passage from its first token; per-token NLL and argmax
    repeat  a second tf pass in the same process; bit-identity of scores per prompt

Returns (under OUT):
    free/expected_pNN.pt   {"prompt", "input_ids", "attention_mask", "sequences", "logits" [n,1,V]}
    tf/tf_pNN.pt           {"scores" [n,V] (bf16 when lossless, else fp32), "sequences", "plen"}
    cap/cap_pNN.pt         {"attn", "moe", "resid" [n,48,H] bf16, "router_logits" [n,48,E], "expert_index"
                            [n,48,8] or None, "valid" [n]}
    ppl/ppl_wNN.pt         {"nll" [L-1] fp64, "argmax" [L-1]}
    device.json            build, flags, mutant, per-prompt timings, repeat result

Usage:
    device.py KIND CDIR OUT --corpus corpus.json [--refs DIR] [--free all|missing] [--tf] [--ppl]
              [--repeat] [--capture] [--mutant mN] [--seq-len 768]
"""
import argparse, glob, hashlib, importlib, json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MK_DIR", os.path.dirname(os.path.dirname(HERE)))
H = os.path.expanduser("~")
N_LAYERS = 48
KINDS = {  # kind -> (extra main.py flags, model module)
    "xla": ([], "megakernels.qwen3_moe.qwen"),
    "libmoe": (["--moe-fused-nki-kernel-enabled"], "megakernels.qwen3_moe.qwen"),
    "nki": (["--enable-nki"], "megakernels.qwen3_moe.qwen_with_megakernel"),
    "E": (["--moe-fused-nki-kernel-enabled"], "megakernels.qwen3_moe.qwen_attn_e"),
}
WITH_INDEX = {"xla"}  # builds whose MoE returns expert indices


def capture_modules():
    return [m for i in range(N_LAYERS) for m in (f"layers.{i}.self_attn", f"layers.{i}.mlp", f"layers.{i}")]


def patch_router_capture(with_index):
    from neuronx_distributed.modules.moe import model as moe_model
    from neuronx_distributed.utils.tensor_capture.api import register_tensor
    orig = moe_model.MoE.forward

    def forward(self, *a, **k):
        out = orig(self, *a, **k)
        register_tensor("router_logits", out[1])
        if with_index:
            register_tensor("expert_index", out[2])
        return out
    moe_model.MoE.forward = forward


def keep_capture_config_on_clear():
    """NxDI clears the registry after each trace, which also resets its config; keep the config."""
    from neuronx_distributed.utils.tensor_capture.registry import CapturedModelInfo, TensorRegistry

    def clear(self):
        mi = self.model_info
        new = CapturedModelInfo(mi.modules_to_capture, mi.max_tensors, mi.capture_inputs)
        new.hooks = mi.hooks
        self.model_info = new
    TensorRegistry.clear = clear


def load(kind, cdir, capture, mutant, seq_len):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import main as M
    extra, mod = KINDS[kind]
    flags = ["--seq-len", str(seq_len), "--on-device-sampling", "--top-k", "1", "--do-sample", "", "--output-logits"]
    if os.path.exists(f"{cdir}/model.pt"):
        flags.append("--skip-compile")
    sys.argv = ["main.py", "--model", "qwen3_moe", "--platform-target", "trn2",
                "--model-path", f"{H}/qwen-30b-a3b/hf_model", "--compiled-model-path", cdir,
                "--tp-degree", "4", "--logical-nc-config", "2"] + flags + extra
    args = M.parse_args()
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.platform_target
    args.prompts, args.batch_size, args.max_length, args.draft_model_cls = ["x"], 1, args.seq_len, None
    with_index = kind in WITH_INDEX
    if capture:
        args.return_router_logits, args.return_expert_index = True, with_index
        args.tensor_capture_config = {"modules_to_capture": capture_modules(),
                                      "max_intermediate_tensors": N_LAYERS * (2 if with_index else 1)}
        patch_router_capture(with_index)
        keep_capture_config_on_clear()
    marker = f"{cdir}/validator_mutant.txt"
    if mutant:
        sys.path.insert(0, HERE)
        import mutants
        mutants.apply(mutant)
    if os.path.exists(marker):
        assert open(marker).read().strip() == (mutant or ""), "build was compiled with a different mutant"
    cls = getattr(importlib.import_module(mod), "NeuronQwen3MoeForCausalLM")
    t0 = time.time()
    model, tok, gen_cfg = M.prepare_inference(cls, args)
    os.makedirs(cdir, exist_ok=True)
    if not os.path.exists(marker):
        open(marker, "w").write(mutant or "")
    print(f"LOADED {kind} {time.time() - t0:.0f}s", flush=True)
    assert model.neuron_config.output_logits
    return model, tok, gen_cfg, with_index


class Forcer:
    """Wraps adapter.forward; replaces the sampled token with ref[step] when a reference is set."""

    def __init__(self, adapter, model, keep_captures):
        self.orig, self.ref, self.step, self.model = adapter.forward, None, 0, model
        self.keep, self.got = keep_captures, []
        adapter.forward = self

    def __call__(self, *a, **k):
        out = self.orig(*a, **k)
        if self.ref is not None:
            t = torch.full_like(out.tokens, int(self.ref[self.step]))
            out["tokens"] = t
            out.tokens = t
        if self.keep:
            self.got.append([x.detach().cpu() for x in out.captured_tensors])
        self.step += 1
        return out

    def run(self, adapter, gen_cfg, ids, n_new, ref=None):
        self.ref, self.step, self.got = ref, 0, []
        self.model.reset()
        with torch.inference_mode():
            o = adapter.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                 min_new_tokens=n_new, do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, generation_config=gen_cfg)
        self.ref = None
        return o.sequences, torch.stack(o.scores).float()


def compact(x):
    """bf16 if the values are exactly bf16-representable, else fp32."""
    b = x.bfloat16()
    return b if torch.equal(b.float(), x) else x


def last_row(t, d, plen):
    """Row of the current token: plen - 1 on the (right-padded) prefill step, else the only row."""
    t = t.reshape(-1, d)
    return t[plen - 1] if t.shape[0] > 1 else t[0]


def unpack_captures(got, n, with_index, plen):
    L, M = N_LAYERS, 3 * N_LAYERS
    hid = None
    res = {"valid": torch.zeros(n, dtype=torch.bool)}
    for s, caps in enumerate(got[:n]):
        mod, man = caps[:M], caps[M:]
        if len(mod) < M or mod[0].numel() <= 1 or not man or man[0].numel() <= 1:
            continue
        if hid is None:
            hid = mod[0].shape[-1]
            E = man[0].shape[-1]
            for k in ("attn", "moe", "resid"):
                res[k] = torch.zeros(n, L, hid, dtype=torch.bfloat16)
            res["router_logits"] = torch.zeros(n, L, E)
            res["expert_index"] = torch.zeros(n, L, 8, dtype=torch.int16) if with_index else None
        res["valid"][s] = True
        k = 2 if with_index else 1
        for i in range(L):
            res["attn"][s, i] = last_row(mod[3 * i], hid, plen).bfloat16()
            res["moe"][s, i] = last_row(mod[3 * i + 1], hid, plen).bfloat16()
            res["resid"][s, i] = last_row(mod[3 * i + 2], hid, plen).bfloat16()
            res["router_logits"][s, i] = last_row(man[k * i], E, plen).float()
            if with_index:
                res["expert_index"][s, i] = last_row(man[k * i + 1], 8, plen).to(torch.int16)
    for k in ("attn", "moe", "resid", "router_logits", "expert_index"):
        res.setdefault(k, None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("cdir")
    ap.add_argument("out")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--refs", help="dir with expected_pNN.pt (for --tf, and for --free missing)")
    ap.add_argument("--free", choices=["all", "missing"])
    ap.add_argument("--only", default="", help="comma-separated prompt ids to restrict to")
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--mutant", default="")
    ap.add_argument("--seq-len", type=int, default=768)
    a = ap.parse_args()
    corpus = json.load(open(a.corpus))
    only = set(filter(None, a.only.split(",")))
    prompts = [p for p in corpus["prompts"] if not only or p["id"] in only]
    os.makedirs(a.out, exist_ok=True)
    info = {"kind": a.kind, "cdir": a.cdir, "capture": a.capture, "mutant": a.mutant, "corpus": a.corpus,
            "corpus_sha256": hashlib.sha256(open(a.corpus, "rb").read()).hexdigest(), "refs": a.refs,
            "started_utc": time.strftime("%FT%TZ", time.gmtime()), "timing_s": {}}
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    model, tok, gen_cfg, with_index = load(a.kind, a.cdir, a.capture, a.mutant, a.seq_len)
    adapter = HuggingFaceGenerationAdapter(model)
    f = Forcer(adapter, model, a.capture)

    if a.free:
        od = f"{a.out}/free"
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        for p in prompts:
            nn = p["id"][1:]
            if a.free == "missing" and a.refs and os.path.exists(f"{a.refs}/expected_p{nn}.pt"):
                continue
            ids = tok([p["prompt"]], return_tensors="pt").input_ids
            assert ids.shape[1] == p["plen"], (p["id"], ids.shape)
            seq, lg = f.run(adapter, gen_cfg, ids, p["n_new"])
            torch.save({"prompt": p["prompt"], "input_ids": ids, "attention_mask": torch.ones_like(ids),
                        "sequences": seq, "logits": lg}, f"{od}/expected_p{nn}.pt")
            print(f"FREE {p['id']} plen={ids.shape[1]} n={lg.shape[0]}", flush=True)
        info["timing_s"]["free"] = round(time.time() - t0, 1)

    def tf_pass(tag):
        od = f"{a.out}/{tag}"
        if tag == "tf":
            os.makedirs(od, exist_ok=True)
        if a.capture and tag == "tf":
            os.makedirs(f"{a.out}/cap", exist_ok=True)
        hashes = {}
        for p in prompts:
            nn = p["id"][1:]
            e = torch.load(f"{a.refs}/expected_p{nn}.pt")
            plen, n = e["input_ids"].shape[1], e["logits"].shape[0]
            ref = e["sequences"][0, plen:plen + n]
            t0 = time.time()
            seq, lg = f.run(adapter, gen_cfg, e["input_ids"], n, ref=ref)
            assert torch.equal(seq[0, plen:], ref), "forced tokens not applied"
            sc = compact(lg[:, 0])
            hashes[p["id"]] = hashlib.sha256(sc.float().numpy().tobytes()).hexdigest()
            if tag == "tf":
                torch.save({"scores": sc, "sequences": seq, "plen": plen}, f"{od}/tf_p{nn}.pt")
            if a.capture and tag == "tf":
                c = unpack_captures(f.got, n, with_index, plen)
                c["plen"] = plen
                torch.save(c, f"{a.out}/cap/cap_p{nn}.pt")
                print(f"CAP {p['id']} valid={int(c['valid'].sum())}/{n}", flush=True)
            print(f"TF {tag} {p['id']} plen={plen} n={n} {time.time() - t0:.1f}s", flush=True)
        return hashes

    if a.tf:
        t0 = time.time()
        info["tf_sha256"] = tf_pass("tf")
        info["timing_s"]["tf"] = round(time.time() - t0, 1)
    if a.ppl:
        od = f"{a.out}/ppl"
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        for w in corpus["ppl"]:
            ids = torch.tensor([w["ids"]])
            n = ids.shape[1] - 1
            _, lg = f.run(adapter, gen_cfg, ids[:, :1], n, ref=ids[0, 1:])
            lp = F.log_softmax(lg[:, 0].double(), -1)
            tgt = ids[0, 1:]
            nll = -lp.gather(1, tgt[:, None])[:, 0]
            torch.save({"nll": nll, "argmax": lg[:, 0].argmax(-1)}, f"{od}/ppl_{w['id']}.pt")
            print(f"PPL {w['id']} n={n} mean_nll={float(nll.mean()):.4f}", flush=True)
        info["timing_s"]["ppl"] = round(time.time() - t0, 1)
    if a.repeat and a.tf:
        t0 = time.time()
        h2 = tf_pass("repeat")
        same = {k: h2[k] == info["tf_sha256"][k] for k in h2}
        info["repeat"] = {"identical": sum(same.values()), "n": len(same),
                          "differ": [k for k, v in same.items() if not v]}
        info["timing_s"]["repeat"] = round(time.time() - t0, 1)
        print("REPEAT", json.dumps(info["repeat"]), flush=True)
    info["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    json.dump(info, open(f"{a.out}/device.json", "w"), indent=1)
    print("DEVICE_DONE", flush=True)


if __name__ == "__main__":
    main()
