"""Replay of validate.py writing to OUT_DIR (day-1 files untouched). Usage: replay_validate.py KIND COMPILED_DIR OUT_DIR; PRESHARD=0 shards on load"""
import ast, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch
import torch.nn.functional as F
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from torch_neuronx.testing.validation import logit_validation

kind, cdir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
if os.environ.get("PRESHARD", "1") == "0":
    C.FLAGS = [x for x in C.FLAGS if x != "--save-sharded-checkpoint"]
model, tok, gen_cfg = C.load(kind, cdir)
adapter = HuggingFaceGenerationAdapter(model)
val_dir = os.path.expanduser("~/runs/val")
results = []


def make_generate_fn(attn_full, total_len, capture):
    def generate_fn(input_ids):
        n = input_ids.shape[1]
        model.reset()
        with torch.inference_mode():
            o = adapter.generate(input_ids=input_ids, attention_mask=attn_full[:, :n],
                                 max_new_tokens=total_len - n, min_new_tokens=total_len - n,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, generation_config=gen_cfg)
        lg = torch.stack(o.scores)
        if not capture:
            capture.append(lg.float().clone())
        return lg
    return generate_fn


for i, _ in enumerate(C.prompts()):
    t0 = time.time()
    e = torch.load(f"{val_dir}/expected_p{i:02d}.pt")
    exp = e["logits"]  # [N, 1, V]
    plen = e["input_ids"].shape[1]
    total = plen + exp.shape[0]
    attn_full = torch.cat([e["attention_mask"], torch.ones((1, exp.shape[0]), dtype=e["attention_mask"].dtype)], dim=1)
    row = {"prompt_idx": i, "prompt_len": plen}
    first_pass = []
    for name, tol in (("repo_tol", C.REPO_TOL), ("rtol_1e-2", C.RTOL1E2)):
        gfn = make_generate_fn(attn_full, total, first_pass)
        try:
            passed, _, msg = logit_validation(input_ids=e["input_ids"], generate_fn=gfn,
                                              expected_logits=exp, tol_map=ast.literal_eval(tol),
                                              divergence_difference_tol=0.001)
        except Exception as ex:  # record, don't abort the sweep
            passed, msg = False, f"EXCEPTION {type(ex).__name__}: {ex}"
        row[f"pass_{name}"] = bool(passed)
        row[f"msg_{name}"] = str(msg)[-600:]
    act = first_pass[0][: exp.shape[0]]  # free-running first pass, [N, 1, V]
    n = min(act.shape[0], exp.shape[0])
    a, x = act[:n, 0].clamp(min=-1e4), exp[:n, 0].clamp(min=-1e4)  # EOS -inf from min_new_tokens
    top1 = (a.argmax(-1) == x.argmax(-1))
    kl = F.kl_div(F.log_softmax(a, -1), F.log_softmax(x, -1), log_target=True, reduction="none").sum(-1)
    mad = (a - x).abs().max(-1).values
    div = int((~top1).nonzero()[0, 0]) if (~top1).any() else -1
    row.update({"first_divergence": div, "top1_agree_frac": float(top1.float().mean()),
                "max_abs_diff_max": float(mad.max()), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                "per_pos": {"max_abs_diff": mad.tolist(), "top1": top1.int().tolist(), "kl": kl.tolist()}})
    torch.save(act.half(), f"{out_dir}/actual_{kind}_p{i:02d}.pt")
    results.append(row)
    print(f"VALIDATED {kind} p{i:02d} pass_repo={row['pass_repo_tol']} pass_rtol1e-2={row['pass_rtol_1e-2']} "
          f"first_div={div} top1={row['top1_agree_frac']:.3f} maxabs={row['max_abs_diff_max']:.3f} "
          f"kl_mean={row['kl_mean']:.2e} {time.time()-t0:.1f}s", flush=True)
    json.dump(results, open(f"{out_dir}/results_{kind}.json", "w"))
print("VALIDATE_DONE")
