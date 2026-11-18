# T11 skill audit (in progress)

## (a) nki-kernel-optimizer (trn2) on a planted kernel — CPU part done 2026-11-17 14:40 ET

Kernel: `tests/skill_audit/swiglu_kernel.py` (out = silu(gate) · up, bf16, [N, 128, F]) with two planted flaws:
(1) the silu result is stored to an HBM scratch and read straight back; (2) the tile loop is `sequential_range`
although iterations are independent. A fresh subagent ran the skill with no hint, CPU only (device busy), and wrote
`tests/skill_audit/swiglu_kernel_opt.py` + `check_bitident.py`.

| | Result |
|---|---|
| Planted flaw 1 (HBM round trip) | **found** (Measured): removed; HBM traffic 5 → 3 tile-volumes per tile |
| Planted flaw 2 (sequential loop) | **found** (Measured): changed to `affine_range` |
| Extra change | both input loads hoisted to the top of the loop body |
| Output bit-identical | yes under `nki.simulate` at 3 shapes (Measured) — weak evidence: the simulator computes and returns fp32 and does not round at bf16 intermediates (verified independently: max error vs an fp32 reference is exactly 0) |
| Speed-up | not measured (device busy); on-device benchmark + bf16 bit-identity pending |
| Followed the skill's loop | no: profiling, benchmarking and the N-plan rounds all need the device |

Errors in the skill found by the audit (each verified against the installed SDK 2.29 / NKI 0.3, Measured):
- `references/trn2-architecture.md:107,119` recommends `dge_mode=nisa.dge_mode.hw_dge`; the enum members are `unknown, swdge, hwdge, none` → `AttributeError`.
- `references/nki-syntax-quickref.md` shows `nki.jit(kernel, platform_target="trn2")`; NKI 0.3 rejects it ("deprecated, use NEURON_PLATFORM_TARGET_OVERRIDE").
- `SKILL.md:16`, `references/templates.md:201` activate `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`; this project pins `~/venv-neuron-229` (SDK 2.29, fork 2.29_release).
- The correctness harness has no `nki.simulate` path and does not warn that the simulator skips bf16 rounding.
- Playbook examples use Beta-1 syntax (`nl.load` / value-returning `nisa.activation`); flagged in the text but easy to copy.

trn3-only facts still stated by the trn2 skill:
- `references/nki-syntax-quickref.md:147,199` lists `nisa.nc_matmul_mx` / `nisa.quantize_mx` with no trn2 caveat (only `common-pitfalls.md` says MX matmul is trn3-only, NCC_IBIR530).
- `SKILL.md` header "Hardware: AWS Trainium 2/3" and quick rules dominated by MXFP scale layout / MoE specifics.
- Verified correct: trn2 SBUF 28 MiB = 128 × 224 KiB (toolchain `total_available_sbuf_size` = 212,984 B/partition on trn2 after the reserved region; trn3 245,752 B ≈ 32 MiB); PSUM 2 MiB. "3 TB/s HBM" and "8 NeuronCores" are per device; the table does not say that an LNC=2 logical core is two physical cores.

## (b) neuron-profile skill vs tools/profile_table.py — done 2026-11-18 18:00 ET

Input: the T9 NTFFs of config E (bucket 128, 4 ranks; `~/runs/prof/p1_prof_E_s768`). A fresh subagent ran the skill blind (CPU only); scratch in the job tmp dir.

| Per layer (÷48), least-skewed rank | Skill | `profile_table.py` | Agree |
|---|---|---|---|
| Rank chosen | vnc_1 (min total_time; spread = CC wait from staggered starts) | min total_time | yes |
| Total device time | 11.102 ms / 231.3 µs | 11.10 ms / 231.3 µs | yes |
| Sync / tensor / vector / scalar / gpsimd (µs) | 62.0 / 64.9 / 58.4 / 37.0 / 57.7 | 62.0 / 64.9 / 58.4 / 37.0 / 57.7 | yes |
| DMA active / CC (µs) | 142.7 / 25.1 | 142.7 / 25.1 | yes |
| DMA packets / avg bytes | 44,169 / 753 | 44,169 / 753 | yes |
| Spill reload / save (B per graph) | 360,320 / 135,176 | 360,320 / 135,176 | yes |

Both read the same `summary-json`, so agreement is expected; the skill additionally gave a CC-trigger steady-state layer time (220.4 µs mean, vs 231.3 incl. prologue/tail) and a source-file attribution (gate_up 68.6, down 39.8, interleave_copy 18.8, qkv 15.2, router 13.3, attention_block 12.2 µs/layer wall; 73.6 µs/layer with no source location, mostly GpSimd semaphores).

Errors in the skill (verified against the skill text; values from the subagent's measurement):
- Units: `references/kernel_source_attribution.md:35–36` and `reading_the_profiler.md:156–158` say JSON `timestamp`/`duration`/`evt_wait_time` are picoseconds; with tools 2.30.10 they are **ns** (max timestamp 11,102,153 for an 11.1 ms step). `show-session`'s "CYCLES" column holds ps and its EXEC TIME (NS) is 0; `--show-trace` timestamps are ps.
- `SKILL.md:152`, `reading_the_profiler.md:160`: "1 cycle ≈ 1 ns" — measured ≈ 1.2 GHz (13.3 M cycles in 11.1 ms).
- `SKILL.md:114`, `layer_latency.md:13` hardcode "Collectives trace for CC-core 8"; this rank has CC-cores 12–15.
- "Layer 0 is slow (~511 µs)" did not hold (233 µs); 101 CC ops ≠ 2×48 (1 extra at start, 3 at end).
- `SKILL.md:56`, `kernel_source_attribution.md:12`: "~1 GB per 250k instructions" — 289 MB for 205k with `--ignore-dma-trace --ignore-event-trace --ignore-nc-buf-usage`.
- Summing `evt_wait_time` per line ("top lines by stall") gives hundreds of ms for an 11 ms step — not wall-clock attribution; `*_percent` fields are fractions; `--show-dma` has no size column; `layer_summary` entries nest.

Missed by the skill: multiple per-rank NTFFs and the rank-skew effect on `cc_op_time`; LNC=2 (one NTFF = two physical cores; summary engine times are their union, counts their sum); wall-clock attribution by merging intervals per source; the large no-source bucket.

## Verdicts (T11)
- **nki-kernel-optimizer (trn2):** right on the planted case (found both flaws, kept simulated output identical); wrong in 3 checked API/environment facts (`hw_dge`, `platform_target=`, venv path); misses that `nki.simulate` skips bf16 rounding; still carries trn3-only MX ops in its quick reference. On-device bit-identity/speed of the optimized kernel not yet measured.
- **neuron-profile:** numbers right (identical to our tool); method guidance wrong on units, clock, CC-core id and output size, and silent on multi-rank skew and LNC=2 — the two things that decide which numbers to trust on this box.
