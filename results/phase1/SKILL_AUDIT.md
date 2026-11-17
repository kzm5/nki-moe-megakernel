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

## (b) neuron-profile vs tools/profile_table.py — pending (no Phase 0 NTFF survives; uses the T9 profile of config E)
