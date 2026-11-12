# Day-3 queue state (source of truth; see QUEUE_DAY3.md for the queue)

Updated: 2026-11-11 23:35 ET (04:35 UTC Thu). Deadlines ET: no compile after 16:00, no device job after 18:00, wrap-up done by 19:30, hard stop 20:00.

## Environment
- Worktree: `~/nki-moe-megakernel/.claude/worktrees/session-rules` (branch `worktree-session-rules`), push with `git push mine HEAD:phase0/trn2-setup`.
- Device jobs run `main.py` from the main checkout `~/nki-moe-megakernel` (same `main.py`).
- Venv `~/venv-neuron-229`; python wrapper `~/.claude/jobs/c953e81d/tmp/py.sh` (sources activate-neuron.sh).
- Ledger `~/runs/ledger.jsonl` (helpers `~/runs/ledger_event.py`, `~/runs/ledger_bench.sh`), copy to `results/phase0/ledger.jsonl`.

## Builds (all in `~/qwen-30b-a3b/`); pre-sharded `weights/` present?
| Build | weights/ | Use |
|---|---|---|
| traced_xla_trn2_229_s768_greedy | no (deleted after noise floor) | XLA reference |
| traced_xlamoe_trn2_229_s768_greedy | no | library MoE |
| traced_nki_trn2_229_s768_greedy | no | megakernel |
| traced_cfgD_trn2_229_s{768,2048}_greedy | no | config D |
| traced_{libmoe,nki}_trn2_229_s{256,512,1024,2048,4096}_greedy_sweep | no | sweep |
All loads shard on load (~10–40 min, peak RSS ~121 GiB). Never delete `~/runs/golden_v2`, `~/runs/val`, ledger, `results/`.

## Data
- Golden fp32 (20 short prompts): `~/runs/golden` (v1, used for comparisons) and `~/runs/golden_v2` (router top-9 `[48,T,9]`).
- Day-1 free-running captures: `~/runs/val/expected_pNN.pt` (XLA), `~/runs/val/actual_{libmoe,nki}_pNN.pt` (fp16), `results_{libmoe,nki}.json`.
- Long prompts: `~/runs/val_long/prompts_long.json` (4 prompts, 520/545/570/595 tokens, sha256 761e9a4f26ef4d25…).

## Tasks
| Task | Status | Commands / paths | Next step |
|---|---|---|---|
| T0 triage | **done** 23:30 ET | `tools/triage_gross.py` → `~/runs/triage/triage.json`; FINDINGS "Day 3, T0" | – |
| T1.1 capture_tf.py | **done, validated** | `tools/capture_tf.py`; wrapper `~/runs/run_tf.sh NAME KIND CDIR OUT args…` | – |
| T1.2–1.3 XLA self-check + long prompts | **done** 00:00 ET | `run_tf.sh tf_xla xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy ~/runs/tf/xla --free ~/runs/val_long/prompts_long.json --tf main=$HOME/runs/val long=$HOME/runs/tf/xla/free`. **TF == free-running bit-identical 20/20 short, 4/4 long.** Long refs copied to `~/runs/val_long/expected_p20..23.pt` | – |
| T1.3b golden long | **running** 00:00 ET | `run_cpu.sh golden_long tools/golden_fp32.py --dtype fp32 --inputs ~/runs/val_long/expected_p*.pt --out ~/runs/golden_long` | then libMoE process |
| T1.4–1.5 NKI census | pending | per build: `run_tf.sh tf_<b> <b> <cdir> ~/runs/tf/<b> --tf main=$HOME/runs/val long=$HOME/runs/val_long repeat=$HOME/runs/val --single main=$HOME/runs/golden_v2=$HOME/runs/val long=$HOME/runs/golden_long=$HOME/runs/val_long` | libMoE first, then megakernel |
| T1 census | pending | `tools/census.py` | |
| T2 bisect | pending | | |
| T3 profile D | optional | | |
| T4 wrap-up | pending | SUMMARY_DAY3.md, RESUME.md, backup, push | start by 18:00 ET |
