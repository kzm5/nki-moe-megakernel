"""Capture device profiles for a few decode steps. Usage: profile_tkg.py KIND COMPILED_DIR OUT_DIR"""
import os, sys
kind, cdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = out
sys.path.insert(0, os.path.expanduser("~/runs/val"))
import common as C  # noqa: E402
import torch  # noqa: E402
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter  # noqa: E402

model, tok, gen_cfg = C.load(kind, cdir)
adapter = HuggingFaceGenerationAdapter(model)
enc = tok([C.prompts()[0]], return_tensors="pt")
model.reset()
with torch.inference_mode():
    adapter.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=4,
                     min_new_tokens=4, do_sample=False, generation_config=gen_cfg)
print("PROFILE_DONE")
