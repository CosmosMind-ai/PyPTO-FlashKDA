# PyPTO-FlashKDA

**English** | [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> Part of the **CosmosMind PyPTO kernel series** for Ascend — all kernels driven by [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1).

Kimi Delta Attention (KDA) kernels for **Ascend 910C**, written in
[PyPTO](https://github.com/hw-native-sys/pypto) — the Ascend-native counterpart of
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) (CUDA/CUTLASS, SM90+).

KDA is the core recurrent operator of [Kimi Linear](https://arxiv.org/abs/2510.26692)
(Eq. 1). This library provides forward and backward implementations for Ascend 910C,
with every release gated by a sealed correctness holdout.

## Status

v0.1.0 — recurrent forward and full backward pass the complete operator contract
(20/20 scored cases plus boundary-semantics suites; see `tests/`). A standalone
chunked-parallel forward path for long-sequence prefill is on the roadmap.

## Highlights

- **Faster than the official PyPTO implementation.** Head-to-head against pypto-lib's
  in-repo GLM KDA kernel (`models/glm5_3_flash/kda.py`, upstream main `a18654d`), same
  card, same protocol: **4.16× on decode, 1.16× on prefill** — and the prefill number
  is our *recurrent* path against their *chunked* path.
- **Closing on hand-written Ascend C.** On the 20-case KDA operator contract suite,
  geometric-mean latency relative to the official Ascend C implementation is
  **0.878×** and still improving.
- **Forward and backward.** Full gradients (`dq, dk, dv, dg, dbeta, dinitial_state`)
  validated against an FP64-gradcheck-verified autograd oracle — a first for the
  Ascend PyPTO ecosystem, which previously had no KDA backward at all.
- **Measured, sealed evidence.** Every number below comes from a build that passed a
  sealed holdout with dual gates (accuracy + latency). Raw measurement records ship in
  `benchmarks/evidence/`.

## Performance

Ascend 910C (`a2a3`), driver 26.0.rc1, median effective latency, 100 rounds / 5 warmup.
Full protocol and per-case tables: `benchmarks/BENCHMARKS.md`.

| Component | Latency | Reference point |
|---|---|---|
| Forward (20-case contract suite) | 52.21 µs | +67.8% vs unfused same-DSL decomposition; 0.878× geomean vs Ascend C |
| Forward (GLM decode shape) | 23.35 µs vs 97.2 µs official | **4.16×** |
| Forward (GLM prefill, 211 tokens) | 292.0 µs vs 337.6 µs official | **1.16×** |
| Backward | 85.12 µs | +90.4% vs unfused same-DSL decomposition |

## Installation

```bash
pip install -e .
```

Requirements: Ascend 910C (`a2a3`), CANN ≥ 9.0, `pypto`, `pypto-lib` (its `golden`
package backs the validated launch path), `torch`.

## Quickstart

```python
import torch
import flash_kda

B, T, H, Dk, Dv = 1, 64, 8, 128, 128
q = torch.randn(B, T, H, Dk, dtype=torch.bfloat16)
k = torch.randn(B, T, H, Dk, dtype=torch.bfloat16)
v = torch.randn(B, T, H, Dv, dtype=torch.bfloat16) * 0.5
g = -torch.rand(B, T, H, Dk) * 5.0      # per-token log decay, fp32
beta = torch.rand(B, T, H)               # already-activated write strength, fp32
s0 = torch.zeros(B, H, Dk, Dv)           # zero tensor = no history

y, final_state = flash_kda.fwd(q, k, v, g, beta, s0, mode="auto")
# y: [B,T,H,Dv] (same dtype as q); final_state: [B,H,Dk,Dv] fp32

dy = torch.randn_like(y.float())
dfs = torch.randn(B, H, Dk, Dv)
dq, dk, dv, dg, dbeta, ds0 = flash_kda.bwd(dy, dfs, q, k, v, g, beta, s0)
```

## Testing

```bash
python tests/test_fwd_contract.py --device 0   # 20-case contract vs torch oracle
python tests/test_bwd.py --device 0            # FP64 gradcheck oracle + device comparison
```

## Roadmap

- [ ] Standalone chunked forward path (the main lever for long-sequence prefill),
      with chunk/recurrent numerical-equivalence and segment-continuity suites
- [ ] Variable-length (`cu_seqlens`) support — prerequisite for fla-ecosystem dispatch
- [ ] Zero-overhead direct launcher (the current API launches through the validating
      `golden` harness; see `flash_kda/api.py`)
- [ ] Dispatch PR into `fla.ops.kda` (fla-org/flash-linear-attention auto-dispatch)
- [ ] Geometric mean ≥ 1.0× vs Ascend C (currently 0.878×, optimization in progress)

## Acknowledgements

The operator contract and the 20-case evaluation suite are aligned with the official
Ascend C implementation of KimiDeltaAttention; the math is Kimi Linear
(arXiv 2510.26692) Eq. 1. Interface semantics are aligned with MoonshotAI/FlashKDA so
that fla-ecosystem dispatch can route to either backend.

## License

MIT. See `LICENSE`.
