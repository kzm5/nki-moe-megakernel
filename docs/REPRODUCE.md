# Reproducing the trn2 results

Qwen3-30B-A3B decode on trn2.3xlarge (1 Trainium2 chip, TP=4, LNC=2, bs=1, greedy on-device sampling), SDK 2.29.
One script, one config file:

```bash
scripts/reproduce/reproduce.sh env        # check instance, packages, pins, weights, RAM/swap/disk, frozen hashes
scripts/reproduce/reproduce.sh build      # compile every config x length in config.sh (skips existing builds)
scripts/reproduce/reproduce.sh validate   # frozen validator v1.1 rows per build (XLA baseline first off 768)
scripts/reproduce/reproduce.sh bench      # N interleaved rounds, one process per run; summary vs the ledger
scripts/reproduce/reproduce.sh figures    # results/final/ from the ledger and reports (tools/make_figures.py)
```

Edit `scripts/reproduce/config.sh` (or pass another file as the second argument): `CONFIGS` (A B C E Eh V),
`LENGTHS` (multiples of 256), `N`, paths, and optional `BUILD_<cfg>_<len>=<dir>` to reuse an existing build.
Set `REF_VAL=<dir>` to compare each validator capture with a reference capture (bit-identity per prompt).

| Config | What | Flags / env |
|---|---|---|
| A | XLA reference | – |
| B | XLA + SDK library MoE kernel | `--moe-fused-nki-kernel-enabled` |
| C | multi-layer megakernel | `--enable-nki` |
| V | C with all-gather + local fp32 sum per all-reduce | `--enable-nki`, `MK_AGENT_VARIANT=1` |
| E | megakernel attention unfused + library MoE | `--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e` |
| Eh | E with the mask/RoPE build hoisted to once per step | as E + `ATTN_E_HOIST=1`; source only on branch `q3/t3-fusion` (`EH_DIR`) |

Every run uses `--on-device-sampling --top-k 1 --do-sample '' --output-logits`.

## Starting point 1: the Phase 0/1 AMI
Everything is installed. Pre-read the lazily restored volume first, or the first loads crawl at ~12 MB/s:
`bash tools/preread.sh ~/qwen-30b-a3b/hf_model ~/runs/p1/golden ~/runs/p1/refs <build dirs>`.
Then `reproduce.sh env`. Builds from the semester are under `~/qwen-30b-a3b/` (map them with `BUILD_*`);
validator data in `~/runs/p1/{refs,golden,val}`; the run ledger in `~/runs/ledger.jsonl` and `results/phase1/ledger.jsonl`.

## Starting point 2: a fresh Deep Learning AMI (Neuron, Ubuntu)
Not run end to end on a fresh DLAMI: these steps are this instance's setup, in order.
1. Neuron apt packages: runtime-lib 2.32.31, collectives 2.32.28, dkms 2.28.0, and **aws-neuronx-tools 2.30.10**
   (hold it: `apt-mark hold aws-neuronx-tools`; older tools cannot read NTFF v7 profiles).
2. SDK 2.29 venv at `~/venv-neuron-229`: neuronx-cc 2.24.5133, torch 2.9.0, torch-neuronx 2.9.0.2.13,
   neuronx-distributed 0.18.27753, neuronx-distributed-inference 0.9.17334, transformers 4.57.6. Add
   `activate-neuron.sh` (copy from this repo's history or: activate, add `/opt/aws/neuron/bin` to PATH,
   `PJRT_DEVICE=NEURON`, `NEURON_LIBRARY_PATH=$(libneuronpjrt-path)`). Do not use SDK 2.30: its bundled nkilib
   breaks the legacy MoE kernels.
3. nkilib fork: `git submodule update --init nki_kernels/nki-library` (branch `2.29_release`, commit `ddbca35`),
   then `pip install -e nki_kernels/nki-library` in the venv.
4. Weights: `Qwen/Qwen3-30B-A3B` from Hugging Face (bf16 safetensors, 57 GB) into `~/qwen-30b-a3b/hf_model`.
5. 64 GB swapfile; `vm.swappiness=10`.
6. Validator data: the corpus and spec are in the repo (`results/phase1/corpus/`, `spec_v1.1.json`, hashes in
   `frozen.json`); the XLA references and fp32 golden are not (several GB). Regenerate them:
   build A at 768, then `tools/validator/device.py xla <A build> ~/runs/p1/val/xla_plain --corpus
   results/phase1/corpus/corpus.json --refs ~/runs/p1/val/xla_plain/free --free all --tf --ppl --repeat`,
   copy `free/expected_p*.pt` to `~/runs/p1/refs`, and run `tools/validator/golden.py --corpus … --refs
   ~/runs/p1/refs --out ~/runs/p1/golden --write-golden` (CPU, ~10 min, 12 GiB). XLA is bit-identical across
   processes and restores (Q1 T1), so the regenerated refs should hash as frozen (untested); `reproduce.sh env` checks.

## Expected cost (measured on this instance)
| Step | Time | Disk | Host RAM peak |
|---|---|---|---|
| Compile one build (768–4096) | 15–25 min (megakernel TKG 931 s at 768; XLA 2048 1,375 s) | 1.7–2.7 GB per build; compile cache grows ~1–5 GB per build (92 GB after the semester) | ~121 GiB of 124 |
| Load a build (shard on load) | 9–35 min, depending on page cache | – | ~121 GiB |
| Load a pre-sharded build (`MK_LOAD_PRESHARDED=1` / `--save-sharded-checkpoint --skip-sharding`) | ~2.5 min | 59 GB per build | lower |
| One benchmark run (`bench`, incl. load) | 20–40 min | small | ~121 GiB |
| One validator capture (`validate`, 768) | 17–50 min | 4.2 GB | ~121 GiB |
| fp32 golden (CPU) | ~10 min | ~1.5 GB | 12 GiB |

Only one compile or device job at a time: two loads exceed host RAM. `reproduce.sh` takes `~/runs/device.lock`.

## Pitfalls found this semester
- `--do-sample ""`: main.py parses `--do-sample` with `type=bool`, so any non-empty string (even `False`) is True.
- `--seq-len` must be a multiple of 256 at LNC=2.
- RAM: a load peaks at ~121 of 124 GiB. Keep the 64 GB swap, but avoid swapping: the root EBS volume is
  IOPS-capped and swapped or cold pages load at ~12 MB/s. Pre-warm the safetensors into the page cache
  (`dd … of=/dev/null`, ~870 MB/s; done by `reproduce.sh` before every load).
- aws-neuronx-tools must be 2.30 (held) to read NTFF v7 profiles.
- Git: pull with `git pull --no-rebase` (merge commits are fine); a rebase cannot rewrite the read-only frozen
  validator files. `*.json` is gitignored, so result JSONs need `git add -f`.
- A guard hook protects the frozen validator paths; it matches basenames (`validator`, `golden`, `refs`,
  `corpus.json`) anywhere in a shell command next to `cp`/`rm`/`>`/`tee`. Put such commands in a script file.
- Router flip rate is **tie-tolerant**: a top-8 set that differs from the golden only by experts tied at the
  boundary is not a flip (validator `router` row).
- The validator's run-to-run row is **non-blocking** ("documented"): a non-deterministic build can pass v1.1.
  Non-deterministic builds were reported but never counted as eligible results.
- The frozen corpus tops out at 768 tokens, so builds compiled at 2048/4096 are validated only at positions < 768.
  The supplementary long corpus (`results/phase1/corpus_long/`, `tools/validate_long.py`) covers 1,400–1,959.
- Day-to-day drift (up to ~0.15 ms) exceeds within-session spread: compare configs only within one interleaved session.
- The first round after a long device job can run high for every config (+0.2–0.3 ms seen in Q4 T2); report
  within-round deltas.

## trn3
The trn3 day was shelved (no capacity). Its queue and bring-up script are kept: `docs/queues/TRN3.md`,
`scripts/trn3_bringup.{sh,md}`.
