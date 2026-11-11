"""Process A: greedy expected logits from the XLA build. Usage: capture.py COMPILED_DIR"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

model, tok, gen_cfg = C.load("xla", sys.argv[1])
adapter = HuggingFaceGenerationAdapter(model)
out_dir = os.path.expanduser("~/runs/val")
for i, p in enumerate(C.prompts()):
    t0 = time.time()
    enc = tok([p], padding=True, return_tensors="pt")
    model.reset()
    with torch.inference_mode():
        o = adapter.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=C.N_NEW, min_new_tokens=C.N_NEW, do_sample=False,
                             return_dict_in_generate=True, output_scores=True,
                             generation_config=gen_cfg)
    logits = torch.stack(o.scores).float()  # [N_NEW, 1, V]
    torch.save({"prompt": p, "input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
                "sequences": o.sequences, "logits": logits}, f"{out_dir}/expected_p{i:02d}.pt")
    print(f"CAPTURED p{i:02d} prompt_len={enc.input_ids.shape[1]} logits={tuple(logits.shape)} "
          f"{time.time()-t0:.1f}s", flush=True)
print("CAPTURE_DONE")
