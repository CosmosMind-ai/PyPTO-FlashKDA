# KDA Operator Contract (PyPTO-FlashKDA)

Source: the operator contract is aligned with the official Ascend C
implementation. Upstream math: Kimi Linear (arXiv 2510.26692), Eq. (1).

## Mathematical definition

With normalized query/key q_t, k_t (raw inputs when `use_qk_l2norm=False`):

```
Hbar_t = diag(exp(g_t)) @ H_{t-1}
r_t    = v_t - k_t^T @ Hbar_t
H_t    = Hbar_t + beta_t * k_t r_t^T
y_t    = scale * q_t^T @ H_t
```

Normalization is exactly `x / sqrt(sum(x²) + 1e-6)`.

**Four non-negotiable semantics:**

1. **Decay first, then residual** — `r_t` is computed against the *decayed* state.
2. **Output reads the updated state** — `y_t` uses H_t, not Hbar_t or H_{t-1}.
3. **Epsilon inside the square root** — `sqrt(sum(x²) + 1e-6)`, never `sqrt(sum(x²)) + 1e-6`.
4. **scale multiplies q only** — never k or the output.

`final_state` always participates in accuracy comparison; checking `y` alone is not a pass.

## Interface

```
fwd(q [B,T,H,Dk], k [B,T,H,Dk], v [B,T,H,Dv], g [B,T,H,Dk] fp32,
    beta [B,T,H] fp32, initial_state [B,H,Dk,Dv] fp32,
    use_qk_l2norm=True, scale=-1.0)
  -> (y [B,T,H,Dv] (dtype of q), final_state [B,H,Dk,Dv] fp32)
```

- `scale <= 0` means `Dk^{-1/2}`; a positive value is an explicit scale applied to q only
- Support domain: B ≥ 1, T ∈ [1, 512], H ∈ [1, 16], Dk/Dv ∈ [1, 128]
- Rectangular state (Dk ≠ Dv) and unaligned dims are supported
- `initial_state` is mandatory and read-only (bitwise-unchanged test coverage);
  pass a zero tensor when there is no history
- `g` is the per-token natural-log decay (not a cumulative quantity);
  `beta` is already activated (no sigmoid inside the operator)

## Accuracy gates

| Output | fp32 | fp16 | bf16 |
|---|---|---|---|
| `y` | rtol 1e-4 / atol 1e-5 | 3e-3 / 3e-4 | 2e-2 / 2e-3 |
| `final_state` | rtol 2e-4 / atol 2e-5 | same | same |

## Backward

No authoritative upstream contract exists; this library defines it per autograd convention:

```
bwd(dy, dfinal_state, q, k, v, g, beta, initial_state, ...)
  -> (dq, dk, dv, dg, dbeta, dinitial_state)
```

`dg`/`dbeta`/`dinitial_state` are fp32; the rest follow their forward inputs' dtypes.
The oracle is self-hosted: the differentiable torch reference (`tests/reference.py`) is
first verified with FP64 `torch.autograd.gradcheck`, then used as the backward baseline.
The `dfinal_state` gradient path is covered.
