# Head-to-head vs the official PyPTO GLM KDA

Date: 2026-11-09. Subject: pypto-lib upstream main `a18654d`,
`models/glm5_3_flash/kda.py` (the official PyPTO KDA implementation).
Ascend 910C (`a2a3`), `PYPTO_BENCH=1`, 100 rounds / 5 warmup, median effective µs.
Both sides validated correct.

## Official numbers

| Component | Shape | Median µs | Validate |
|---|---|---|---|
| decode | state (6,4,128,128) fp32 / out (5,4,128) bf16 | 97.2 (min 94.3, max 100.9) | PASS |
| prefill | state (6,4,128,128) / out (211,4,128) bf16 (ragged 211 tokens) | 337.6 (min 330.2, max 345.4) | PASS |

## Results

| Scenario | Official kda.py | This library | Speedup |
|---|---|---|---|
| decode (T=1/request, H=4 D=128 bf16 l2norm) | 97.2 µs (5 requests) | **23.35 µs** (B=6 T=1, equal-or-more work) | **4.16×** |
| prefill (211 tokens) | 337.6 µs (ragged multi-request) | **292.0 µs** (B=1 T=211 single sequence, recurrent path) | **1.16×** |

## Methodology notes

- The official decode test is the pure KDA recurrence step
  (q/k/v/decay/beta/state → output+state) — no conv/gating extras — so the decode
  comparison is same-semantics against our contract.
- The prefill comparison slightly favors us (single sequence, no ragged overhead),
  but note we ran the **recurrent** path against the official **chunked** prefill
  path; a dedicated chunked path is on the roadmap.
- The official GLM shapes (H=4, D=128, TP configuration) differ from the 20-case
  contract suite; numbers are only comparable within the same shape.
- The current release kernel is newer than the build measured here, so these numbers
  are conservative lower bounds.
