#!/usr/bin/env bash
# Frozen validator on C's pre-sharded build twice — MK_LOAD_PRESHARDED=1 and sharding on load — then compare outputs.
# Usage: check_preshard_hook.sh
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3
VAL=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator/validate.py
CD=/home/ubuntu/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy
V=/home/ubuntu/runs/p1/val
exec 9> ~/runs/device.lock; flock 9
for mode in hook shard; do
  for f in ~/qwen-30b-a3b/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
  env=""; [ $mode = hook ] && env="MK_LOAD_PRESHARDED=1"
  env MK_DIR=$WT $env ~/runs/run_py.sh p1_q3_presh_$mode $VAL run nki $CD $V/C_presh_$mode
done
~/venv-neuron-229/bin/python - <<'EOF'
import glob, os, re, torch
V = "/home/ubuntu/runs/p1/val"
a, b = f"{V}/C_presh_hook", f"{V}/C_presh_shard"
tf = sorted(glob.glob(f"{a}/tf/tf_p*.pt"))
same = sum(torch.equal(torch.load(f)["scores"], torch.load(f.replace(a, b))["scores"]) for f in tf)
pp = sorted(glob.glob(f"{a}/ppl/ppl_*.pt"))
psame = sum(all(torch.equal(x, y) for x, y in zip(torch.load(f).values(), torch.load(f.replace(a, b)).values())
                if torch.is_tensor(x)) for f in pp)
for m in ("hook", "shard"):
    t = open(os.path.expanduser(f"~/runs/p1_q3_presh_{m}.log"), errors="replace").read()
    ld = re.findall(r"LOADED \w+ (\d+)s", t)
    pre = "Loading presharded checkpoints" in t
    sh = re.findall(r"Done Sharding weights in ([\d.]+)", t)
    print(f"{m}: LOADED {ld} presharded_path={pre} sharding_s={sh}")
print(f"tf bit-identical {same}/{len(tf)}; ppl bit-identical {psame}/{len(pp)}")
EOF
