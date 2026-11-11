"""Greedy logits for the 20 validation prompts. Usage: capture_to.py KIND COMPILED_DIR OUT_DIR"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter  # noqa: E402

kind, cdir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
model, tok, gen_cfg = C.load(kind, cdir)
adapter = HuggingFaceGenerationAdapter(model)
for i, p in enumerate(C.prompts()):
    t0 = time.time()
    enc = tok([p], padding=True, return_tensors="pt")
    model.reset()
    with torch.inference_mode():
        o = adapter.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=C.N_NEW, min_new_tokens=C.N_NEW, do_sample=False,
                             return_dict_in_generate=True, output_scores=True, generation_config=gen_cfg)
    logits = torch.stack(o.scores).float()
    torch.save({"prompt": p, "input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
                "sequences": o.sequences, "logits": logits}, f"{out_dir}/expected_p{i:02d}.pt")
    print(f"CAPTURED p{i:02d} {time.time() - t0:.1f}s", flush=True)
print("CAPTURE_DONE")
