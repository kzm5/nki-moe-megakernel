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
| T1.3b golden long | **done** 23:58 ET | `run_cpu.sh golden_long tools/golden_fp32.py --dtype fp32 --inputs ~/runs/val_long/expected_p*.pt --out ~/runs/golden_long` (68 s) | – |
| T1.4–1.5 NKI census | **libMoE running** (started 23:59 ET, log `~/runs/tf_libmoe.log`, out `~/runs/tf/libmoe`); megakernel next (`~/runs/tf/nki`) | per build: `run_tf.sh tf_<b> <b> <cdir> ~/runs/tf/<b> --tf main=$HOME/runs/val long=$HOME/runs/val_long repeat=$HOME/runs/val --single main=$HOME/runs/golden_v2=$HOME/runs/val long=$HOME/runs/golden_long=$HOME/runs/val_long` | libMoE first, then megakernel |
| T1 census | **done** 00:30 ET: both builds 0/2816 gross; ρ 0.909 / 0.935; repeat 20/20 bit-identical both; all day-1 gross positions golden-like today (`~/runs/tf/census.json`; FINDINGS "Day 3, T1") | `tools/census.py --builds libmoe nki` | – |
| T1.6 replay (added: explains non-reproduction) | **done** 01:00 ET: 20/20 prompts bit-identical to day-1 incl. all gross errors (shard on load) → R2 dropped; errors deterministic, history-dependent (FINDINGS "replaying the day-1 validation sequence") | `~/runs/replay/libmoe_onload` | – |
| T1.7 history probe (added) | **done** 01:20 ET: 12/12 conditions bit-identical to fresh; history does NOT change teacher-forced output (`~/runs/replay/history_libmoe.json`) → "history-dependent" was wrong; difference is free-running vs teacher-forced | – | – |
| T1.8 free-running token test (added) | libMoE **done** 01:32 ET; megakernel running (`~/runs/free_nki.log`) | `run_py.sh free_<b> tools/free_tokens.py <b> <build> ~/runs/replay/free_<b>.json`; `tools/tie_audit.py` → `~/runs/triage/tie_audit.json` | **libMoE: 7/7 day-1 gross had a token divergence 1+ steps earlier at an exact top-2 tie (5 build-side sampler vs argmax, 2 XLA-side); 11/12 benign had identical contexts → day-2 "gross single-step errors" were an analysis artifact (contexts differed)** | `run … tools/history_probe.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy ~/runs/replay/history_libmoe.json --pairs 0:1:15,5:6:4` | which prior call poisons the next prompt; feeds T2 (bisect needs the failure reproduced) | R1: `PRESHARD=0 run … ~/runs/val/replay_validate.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy ~/runs/replay/libmoe_onload`; R2 if R1 clean: `preshard.py libmoe …` (52 min, 58 GB) then `PRESHARD=1 … ~/runs/replay/libmoe_preshard`, delete shards after | compare actual_* with day-1 `~/runs/val/actual_libmoe_*` bit-for-bit |
| T2 bisect | **done** 04:00 ET: first >10× departures are single-layer MoE outputs at near-tie routers (p01@15 L7, p03@92 L10, p17@41 L15); attention never departs first (FINDINGS "Day 3, T2"; `~/runs/bisect/bisect.json`). Megakernel per-layer capture not possible (one kernel) | (1) golden dump: `golden_fp32.py --dump NN:P,… --inputs … --out ~/runs/bisect/golden` → `dump_pNN.pt` (layer_in, attn_out, moe_in, router[128], moe_out, layer_out per layer). (2) device: `tools/capture_layers.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_capture ~/runs/bisect/libmoe --targets NN:P,…` (compiles with tensor_capture_config: per layer input_layernorm / self_attn.o_proj / mlp, outputs+inputs = 288 tensors); XLA same into `traced_xla_trn2_229_s768_capture` if time | No position fails under identical context, so T2 becomes a per-layer error-vs-depth profile (lib MoE vs XLA vs golden) at targets 01:15,06:4,07:45,00:6,17:41,23:4,03:92,15:86 (named + largest census KL; p17@41 = largest NKI-specific, shared). Order: golden dump (CPU) → libMoE capture build (compile+run) → XLA capture build if time; compile must start before 16:00 ET |
| T3 profile D | **done** 04:20 ET: +26 µs/layer vs lib MoE, mostly DMA (+20–23), spills 2.2–2.6× (FINDINGS "Day 3, T3") | `run_prof.sh prof_cfgD_s768 cfgD ~/qwen-30b-a3b/traced_cfgD_trn2_229_s768_greedy 768`; then `tools/profile_table.py` with dumped NEFFs | compare vs lib MoE s768 table |
| T4 wrap-up | **in progress** 04:20 ET | SUMMARY_DAY3.md, RESUME.md, backup, push | finish, then leave machine idle |
