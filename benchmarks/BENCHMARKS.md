# Benchmarks — PyPTO-FlashKDA

All numbers: Ascend 910C (`a2a3`), driver 26.0.rc1, median effective latency (µs),
100 rounds / 5 warmup, idle dies only. Raw evidence: `benchmarks/evidence/`.

## 1. Operator contract suite (20 cases)

| Kernel | Holdout aggregate | Holdout min | vs unfused same-DSL decomposition | vs Ascend C (geomean) |
|---|---|---|---|---|
| `flash_kda/kernels/recurrent_fwd.py` | 52.21 µs | 11.29 µs | +67.8% | — |
| sister build, same contract | 52.90 µs | 11.35 µs | +67.4% | **0.878×** |

The Ascend C reference numbers come from the official implementation (CANN 9.2.0,
20/20 correctness). Note that the reference suite lists a 10 µs floor value (not
measured) on 17/20 cases; the geometric mean above is computed against the measured
Ascend C latencies. Optimization is ongoing; the target is geomean ≥ 1.0×.

## 2. Head-to-head vs the official PyPTO implementation (GLM shapes)

Subject: pypto-lib upstream main `a18654d`, `models/glm5_3_flash/kda.py` (the official
PyPTO KDA). Same card, same bench protocol, both sides validated correct.

| Scenario | Official kda.py | This library | Speedup |
|---|---|---|---|
| decode (T=1/request, H=4 D=128 bf16 l2norm) | 97.2 µs (5 requests) | 23.35 µs (B=6, more work) | **4.16×** |
| prefill (211 tokens) | 337.6 µs (ragged multi-request) | 292.0 µs (B=1 T=211 single sequence, recurrent path) | **1.16×** |

Protocol note: the official decode test is the pure KDA recurrence step
(q/k/v/decay/beta/state → output+state), same semantics as the contract. The prefill
comparison slightly favors us (single sequence, no ragged overhead) — but we ran the
*recurrent* path against their *chunked* path. The current release kernel is faster
than the build used for these GLM-shape numbers, so treat them as conservative lower
bounds. Full record: `benchmarks/evidence/glm-kda-h2h.md`.

## 3. Backward

| Kernel | Holdout latency | vs unfused same-DSL decomposition |
|---|---|---|
| `flash_kda/kernels/bwd.py` | 85.12 µs | +90.4% |

Backward oracle: differentiable torch reference, self-verified with FP64
`torch.autograd.gradcheck`.

## Measurement discipline

- Every shipped kernel passes a sealed holdout with dual gates (accuracy + latency);
  candidates that are faster but incorrect are recorded as FAIL and never shipped.
- `benchmarks/evidence/`: `*.holdout.json` (raw holdout events) and `*.record.json`
  (decision-level measurement records).
