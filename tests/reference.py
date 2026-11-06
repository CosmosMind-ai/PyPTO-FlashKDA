"""Torch reference for the KDA contract (docs/CONTRACT.md §4.1).

Self-contained: no pypto/golden imports. Used by the test-suite as the golden
oracle; the FP64 instantiation doubles as the gradcheck oracle for backward.
"""
import torch

EPS = 1e-6  # l2norm epsilon, inside the sqrt


def kda_ref(q, k, v, g, beta, s0, use_qk_l2norm=True, scale=-1.0, dtype=torch.float32):
    """Contract reference, computed in `dtype` (fp32 oracle, fp64 for gradcheck).

    q, k: [B,T,H,Dk]; v: [B,T,H,Dv]; g: [B,T,H,Dk] fp32; beta: [B,T,H] fp32;
    s0: [B,H,Dk,Dv] fp32. Returns (y [B,T,H,Dv], final_state [B,H,Dk,Dv]).
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    if scale <= 0:
        scale = Dk ** -0.5
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    gf, bf = g.to(dtype), beta.to(dtype)
    Hs = s0.to(dtype).clone()
    y = torch.empty(B, T, H, Dv, dtype=dtype)
    for t in range(T):
        qt, kt = qf[:, t], kf[:, t]
        if use_qk_l2norm:
            qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + EPS)
            kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + EPS)
        Hb = torch.exp(gf[:, t]).unsqueeze(-1) * Hs               # decay first
        r = vf[:, t] - (kt.unsqueeze(-1) * Hb).sum(dim=2)         # residual on decayed state
        Hs = Hb + bf[:, t].reshape(B, H, 1, 1) * kt.unsqueeze(-1) * r.unsqueeze(2)
        y[:, t] = scale * (qt.unsqueeze(-1) * Hs).sum(dim=2)      # output from updated state
    return y, Hs


def gen_inputs(case):
    """Contract §3.1 input distributions, seeded; per-case specials from §4.3."""
    B, T, H, Dk, Dv = case["B"], case["T"], case["H"], case["Dk"], case["Dv"]
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[case["dtype"]]
    l2norm = bool(case.get("use_qk_l2norm", True))
    gen = torch.Generator().manual_seed(int(case.get("seed", 0)) * 131071 + int(case.get("id", 0)))

    def u(shape, lo, hi):
        return torch.rand(shape, generator=gen, dtype=torch.float32) * (hi - lo) + lo

    qk_lo, qk_hi = (-1.0, 1.0) if l2norm else (-0.05, 0.05)
    q, k = u([B, T, H, Dk], qk_lo, qk_hi), u([B, T, H, Dk], qk_lo, qk_hi)
    v = u([B, T, H, Dv], -0.5, 0.5)
    g = u([B, T, H, Dk], -5.0, 0.0)
    beta = u([B, T, H], 0.0, 1.0)
    s0 = u([B, H, Dk, Dv], -0.5, 0.5)

    special = case.get("special")
    if special == "near_zero_qk":
        q, k = u([B, T, H, Dk], -3e-4, 3e-4), u([B, T, H, Dk], -3e-4, 3e-4)
    elif special == "strong_decay":
        g = u([B, T, H, Dk], -5.0, -1.0)
    elif special == "no_decay_explicit_scale":
        g = torch.zeros(B, T, H, Dk, dtype=torch.float32)
    elif special == "zero_qk":
        q = torch.zeros(B, T, H, Dk, dtype=torch.float32)
        k = torch.zeros(B, T, H, Dk, dtype=torch.float32)
        g = torch.zeros(B, T, H, Dk, dtype=torch.float32)
        beta = torch.ones(B, T, H, dtype=torch.float32)
    elif special is not None:
        raise ValueError(f"unknown special {special!r}")

    return {"q": q.to(dt), "k": k.to(dt), "v": v.to(dt),
            "g": g, "beta": beta, "initial_state": s0}


# Contract §4.2 tolerances: y by dtype; final_state always fp32 at fp32 thresholds
Y_TOL = {"fp32": (1e-4, 1e-5), "fp16": (3e-3, 3e-4), "bf16": (2e-2, 2e-3)}
FS_RTOL, FS_ATOL = 2e-4, 2e-5
