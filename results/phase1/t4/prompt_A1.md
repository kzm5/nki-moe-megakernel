You are a kernel-optimization agent on an AWS Trainium2 server (trn2.3xlarge, NeuronCore-v3, SDK 2.29). Use the `nki-kernel-optimizer` skill.

**Objective.** Reduce the decode (token-generation, TKG) latency of the Qwen3-30B-A3B multi-layer NKI megakernel by cutting collective time — batch, overlap or reorder the per-layer all-reduces — without failing the frozen validator (spec v1.1). Configuration: TP=4, LNC=2, batch 1, seq-len 768, greedy on-device sampling.

**Workspace.** `/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/wt/a1` — a git worktree on branch `q3/t4-a1`, your only writable code location. The megakernel is built with `--enable-nki` (`megakernels/qwen3_moe/qwen_with_megakernel.py`, kernel in `megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py` and the subkernels it imports). Commit every attempt on your branch before evaluating it.

**Rules.**
- Gate every change behind the environment variable `MK_AGENT_VARIANT=1` (the harness sets it); with it unset the code must behave exactly as before.
- Evaluate only through the harness, one evaluation at a time, and run no other device or compile jobs: `bash /home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/tools/agent_eval.sh A a1 <iter> /home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/wt/a1` (iter = 1, 2, …). It detaches; wait with `timeout 590 bash -c 'until [ -f ~/runs/agents/a1/<iter>/DONE ]; do sleep 30; done'` (repeat until it exists), then read `~/runs/agents/a1/<iter>/feedback.txt`. One iteration takes ~1.5–2 h (compile ~15 min plus validator ~45 min in one job, then 2 benchmark runs ~35 min), so plan for about 3–4 iterations.
- The validator, spec, corpus and golden tensors are read-only; do not try to modify or bypass them.
- Budget: 6 h wall clock from 2026-11-20 03:42 UTC. Start no evaluation that cannot finish by 2026-11-20 09:42 UTC; stop then.
- Do not delete files outside your worktree.

**SETUP A.** Feedback per iteration: compile status and errors, the validator's PASS/FAIL verdict, and the TKG p50 latency. Do not run profilers or use the neuron-profile skill.

**Final answer** (your last message): one line per iteration `iter | commit | hypothesis | compile ok? | validator | TKG p50 ms`, then the best validator-passing iteration and its latency, any tool calls that were blocked, and anything that needed outside help.
