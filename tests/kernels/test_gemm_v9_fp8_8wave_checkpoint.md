# FP8 8-Wave ScaleA GEMM Checkpoint

Recorded: 2026-08-09

## Scope

- FlyDSL kernel: `tests/kernels/test_gemm_v9_fp8_8wave.py`
- Reference pipeline: `/mywork/pyhip/src/contrib/gemm_fp8.py`
- Target: gfx950, FP8 E4M3FN inputs, FP32 accumulation, BF16 output
- Tile: `256 x 256 x 128`, 512 threads / 8 waves
- Optimized path: `with_scale=True`, ScaleA only, ScaleA group size 128

## Current Implementation

- ScaleA is loaded into two ping-pong LDS buffers with 32-bit accesses.
- The main loop uses a one-fragment FIFO and follows the pyhip four-phase pipeline.
- Each scaled main-loop slot is one side-effect inline-assembly block:

  ```asm
  v_fma_f32
  v_fma_f32
  v_fma_f32
  v_fma_f32
  v_mfma_f32_16x16x128_f8f6f4
  ```

- MFMA operand order remains B * A.
- All five inline-assembly outputs use early-clobber `=&v`. This is required to
  prevent an output from overwriting an A/B source before the MFMA reads it.
- The scaled main loop bypasses `sched_group_barrier`; the tail still uses
  `fx.fma`.
- The no-scale path is not part of this optimization work.

## Validated Results

Correctness at `M=N=256, K=8192`:

- `torch.isclose(rtol=0.016, atol=1e-5)`: all elements pass.
- `calc_diff(vs f32 ref)`: approximately `0.000002`.
- `calc_diff(vs bf16 ref)`: approximately `0.000000`.
- Maximum absolute error: approximately `0.031`, consistent with BF16 output
  rounding.

Resources:

- ISA next-free VGPR: 252.
- ISA next-free SGPR: 96.
- Private segment size: 0.
- Scratch load/store instructions: 0.
- No register spill.
- Occupancy: 2 waves/SIMD, same as pyhip.

Performance at `M=N=K=8192`:

- Current FlyDSL inline-assembly path: approximately `504.7 us`, 2178.6 TFLOPS.
- Previous FlyDSL `fx.fma` group-3 baseline: approximately `514.3 us`.
- Current inline assembly improves the FlyDSL baseline by approximately 1.9%.
- Same-machine pyhip result: approximately `396.4 us`, 2774.0 TFLOPS.
- FlyDSL remains approximately 27% slower by latency.

## Pipeline Comparison

The FlyDSL and pyhip pipelines are semantically aligned:

1. Prologue loads the current ScaleA and four A/B quadrants.
2. `vmcnt(4)` and a barrier make the first LDS buffer ready.
3. The second ScaleA/At/Bl/Br set is prefetched, followed by `vmcnt(5)`.
4. Phase 0 reads Bl, At, and upper ScaleA; it prefetches the next Ab.
5. Phase 1 reads Br and prefetches the next At.
6. Phase 2 reads Ab and lower ScaleA; it prefetches the next Bl.
7. Phase 3 prefetches the next Br and ScaleA, then waits at `vmcnt(5)`.
8. Both consume the previous MFMA partial before writing the current partial.

No ScaleA parity, LDS ping-pong, FIFO, or A/B operand-order defect remains.

## Profile Comparison

The following counters use one matching `256 x 256 x 8192` dispatch. The pyhip
reference also reads ScaleB and computes ScaleA * ScaleB; FlyDSL is A-only. This
makes the result stronger: pyhip performs more scale work but is still faster.

| Counter | FlyDSL | pyhip | Observation |
| --- | ---: | ---: | --- |
| `SQ_WAVE_CYCLES` | 433,547 | 374,100 | FlyDSL +15.9% |
| `SQ_WAIT_ANY` | 190,704 | 87,184 | FlyDSL 2.19x |
| `SQ_WAIT_INST_ANY` | 114,837 | 146,580 | Different wait composition |
| `SQ_WAIT_INST_LDS` | 18,584 | 36,673 | LDS wait is not FlyDSL's gap |
| `SQ_INSTS_MFMA` | 16,384 | 16,384 | Identical MFMA work |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 524,288 | 524,288 | Identical MFMA busy time |
| `SQ_VALU_MFMA_COEXEC_CYCLES` | 236,028 | 277,296 | FlyDSL -14.9% |
| MFMA coexec ratio | 45.0% | 52.9% | Main identified gap |
| `SQ_INSTS_VMEM` | 4,864 | 4,872 | Essentially identical |
| `SQ_INSTS_LDS` | 14,336 | 17,408 | FlyDSL performs fewer LDS ops |
| `SQ_INSTS_VALU` | 85,144 | 100,064 | FlyDSL performs fewer VALU ops |
| `SQ_LDS_BANK_CONFLICT` | 0 | 0 | Bank conflicts excluded |
| `SQ_INSTS_BRANCH` | 288 | 24 | FlyDSL keeps a runtime K loop |
| `SQ_INSTS_SALU` | 12,160 | 13,520 | Branch overhead is secondary |

The approximately 15.9% wave-cycle excess closely tracks the hot-loop gap.
Memory traffic, LDS conflicts, spills, and occupancy have been excluded. The
remaining loss is primarily VALU/MFMA issue efficiency and sequencer wait bubbles.

## Important Assembly Difference

FlyDSL fixes four scalar FMA instructions immediately before each MFMA. The
current pyhip assembly instead lowers the four accumulation lanes to:

```asm
v_pk_fma_f32
v_fmac_f32
v_fmac_f32
v_mfma_f32_16x16x128_f8f6f4
```

Pyhip additionally emits ScaleA * ScaleB multiplies but still reaches 52.9%
MFMA/VALU coexecution. Its packed/scalar mix and cross-slot register scheduling
are the leading explanations for the remaining issue-efficiency gap.

## Rejected Experiments

- Split multiply/add was slower than explicit `fx.fma`.
- Scheduler group 2 improved interleaving but caused spill.
- Group 3 was spill-free but only reached approximately 514.3 us.
- Swapping MFMA operands improved one broken inline-assembly experiment but
  violated required B * A semantics and was reverted.
- Removing apparently redundant source-level `lgkmcnt(0)` waits had no effect:
  the backend restored dependency waits, counter values did not improve, and
  performance remained approximately 503.7 us. The experiment was reverted.
- A rocprofiler active/FIFO counter combination aborted inside rocprofiler; do
  not treat that failed collection as kernel evidence.

## Next Experiments

1. Reproduce pyhip's `v_pk_fma_f32 + 2 x v_fmac_f32 + MFMA` sequence in the
   scaled main loop while preserving B * A and early-clobber correctness.
2. If per-slot inline assembly prevents packed operands or cross-slot register
   reuse, place all eight slots of one compute phase in a single assembly block.
3. After every scheduling change, require strong accuracy, BF16-reference
   comparison, zero scratch, and ISA inspection before benchmarking.
4. Recollect `SQ_WAIT_ANY`, `SQ_WAVE_CYCLES`, MFMA busy, and MFMA coexec on the
   same `256 x 256 x 8192` dispatch, then benchmark `8192^3`.

## Useful Artifacts

- Validated FlyDSL ISA:
  `/tmp/flydsl_scaled_inline_ba_ec/gemm_kernel_0/21_final_isa.s`
- FlyDSL wait counters: `/tmp/prof_gap_fly_wait/`
- Pyhip wait counters: `/tmp/prof_gap_pyhip_wait/`
- FlyDSL execution counters: `/tmp/prof_gap_fly_exec/`
- Pyhip execution counters: `/tmp/prof_gap_pyhip_exec/`
- FlyDSL instruction-mix counters: `/tmp/prof_gap_fly_mix/`
- Pyhip instruction-mix counters: `/tmp/prof_gap_pyhip_mix/`
- Pyhip cached assembly: `/root/.pyhip/`

Files under `/tmp` are not persistent across reboot. The numeric results above
are the durable record.