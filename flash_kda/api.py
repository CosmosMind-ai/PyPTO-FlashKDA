"""Python API for PyPTO-FlashKDA.

Signature follows the KDA contract (docs/CONTRACT.md):

    fwd(q, k, v, g, beta, initial_state, use_qk_l2norm=True, scale=-1.0, mode="auto")
        -> (y, final_state)
    bwd(dy, dfinal_state, q, k, v, g, beta, initial_state, ...)
        -> (dq, dk, dv, dg, dbeta, dinitial_state)

Launch path: v0.1.0 dispatches through the public `golden` harness
(pypto-lib) with `save_data=True` and reads the outputs back from the run's
work directory. This is the *validated* execution path — the same one the
kernels were verified with — at the cost of a host-side tensor round-trip per
call. A zero-overhead direct launcher (pypto.runtime.device_runner) is on the
roadmap; for throughput-sensitive integration use the kernels under
`flash_kda/kernels/` directly with your own launcher.
"""
import tempfile
from pathlib import Path

import torch

from .kernels import bwd as _bwd_kernel
from .kernels import recurrent_fwd as _fwd_kernel

_DTYPE_NAME = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def _case_from(q, k, v, use_qk_l2norm, scale):
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    if k.shape != q.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != q shape {tuple(q.shape)}")
    if v.shape[:3] != (B, T, H):
        raise ValueError(f"v shape {tuple(v.shape)} incompatible with q {tuple(q.shape)}")
    if q.dtype not in _DTYPE_NAME:
        raise TypeError(f"unsupported dtype {q.dtype}; expected fp32/fp16/bf16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q/k/v must share the same dtype")
    return {
        "B": B, "T": T, "H": H, "Dk": Dk, "Dv": Dv,
        "dtype": _DTYPE_NAME[q.dtype],
        "use_qk_l2norm": bool(use_qk_l2norm),
        "scale": float(scale),
    }


def _bind_inputs(specs, tensors):
    """Point each input spec at the caller's real tensor."""
    for spec in specs:
        if spec.name in tensors:
            t = tensors[spec.name]
            if list(spec.shape) != list(t.shape):
                raise ValueError(f"{spec.name}: expected shape {spec.shape}, got {list(t.shape)}")
            spec.init_value = t.contiguous()
    return specs


def _launch(fn, specs, out_names, rtol, atol, golden_fn, device, validate):
    from golden import run

    work = Path(tempfile.mkdtemp(prefix="flashkda_"))
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn if validate else None,
        config=dict(backend_type="a2a3", device_id=device),
        rtol=rtol,
        atol=atol,
        save_data=True,
    )
    if validate and not result.passed:
        if result.error:
            raise RuntimeError(result.error)
        raise RuntimeError("golden validation failed")
    outs = []
    for name in out_names:
        path = work / "data" / "out" / f"{name}.pt"
        if not path.exists():
            raise RuntimeError(f"output {name!r} not produced (missing {path})")
        outs.append(torch.load(path, weights_only=True))
    return outs


def fwd(q, k, v, g, beta, initial_state, use_qk_l2norm=True, scale=-1.0,
        mode="auto", device=0, validate=False):
    """KDA forward. Returns (y, final_state); final_state is always fp32.

    mode: "recurrent" | "chunk" | "auto" — the v0.1.0 champion kernel selects
    its internal path by T, so all three are currently equivalent; an explicit
    standalone chunk kernel is on the roadmap.
    """
    if mode not in ("recurrent", "chunk", "auto"):
        raise ValueError(f"unknown mode {mode!r}")
    case = _case_from(q, k, v, use_qk_l2norm, scale)
    fn, specs, golden_fn, rtol, atol = _fwd_kernel.build(case)
    _bind_inputs(specs, {"q": q, "k": k, "v": v, "g": g, "beta": beta,
                         "initial_state": initial_state})
    y, final_state = _launch(fn, specs, ["y", "final_state"], rtol, atol,
                             golden_fn, device, validate)
    return y, final_state


def bwd(dy, dfinal_state, q, k, v, g, beta, initial_state,
        use_qk_l2norm=True, scale=-1.0, device=0, validate=False):
    """KDA backward. Returns (dq, dk, dv, dg, dbeta, dinitial_state)."""
    case = _case_from(q, k, v, use_qk_l2norm, scale)
    fn, specs, golden_fn, rtol, atol = _bwd_kernel.build(case)
    _bind_inputs(specs, {"q": q, "k": k, "v": v, "g": g, "beta": beta,
                         "initial_state": initial_state,
                         "dy": dy, "dfinal_state": dfinal_state})
    return tuple(_launch(fn, specs,
                         ["dq", "dk", "dv", "dg", "dbeta", "dinitial_state"],
                         rtol, atol, golden_fn, device, validate))


def available(shape=None, dtype=None):
    """Capability query. v0.1.0 support domain: B>=1, T>=1, H>=1, 1<=Dk,Dv<=128."""
    if shape is None:
        return True
    B, T, H, Dk, Dv = shape
    return B >= 1 and T >= 1 and H >= 1 and 1 <= Dk <= 128 and 1 <= Dv <= 128
