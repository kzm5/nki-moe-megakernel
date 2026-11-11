"""Dump TKG bucket NEFFs from an NxD model.pt. Usage: neffdump.py MODEL_PT OUT_DIR"""
import os, sys, torch, torch_neuronx
m = torch.jit.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for i, s in enumerate(m.nxd_model.models.token_generation_model.models):
    p = f"{out}/tkg_bucket{i}.neff"
    print(i, s.save_neff(p), os.path.getsize(p), flush=True)
