"""KDA backward test — FP64 gradcheck oracle + device comparison.

1. The torch reference (tests/reference.py, differentiable) is self-verified
   with torch.autograd.gradcheck in FP64.
2. The PyPTO backward kernel's gradients are compared against autograd on the
   reference on a real 910C device.

    python tests/test_bwd.py --device 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference import gen_inputs, kda_ref  # noqa: E402

from flash_kda.kernels import bwd as bwd_kernel  # noqa: E402

BWD_CASES = [
    {"id": 101, "B": 1, "T": 8,  "H": 2, "Dk": 16,  "Dv": 16,  "dtype": "fp32"},
    {"id": 102, "B": 2, "T": 32, "H": 4, "Dk": 64,  "Dv": 32,  "dtype": "fp32"},
    {"id": 103, "B": 1, "T": 64, "H": 4, "Dk": 64,  "Dv": 64,  "dtype": "bf16"},
    {"id": 104, "B": 1, "T": 17, "H": 2, "Dk": 32,  "Dv": 32,  "dtype": "fp16"},
]


def _self_test_gradcheck():
    """The reference itself must be differentiable-correct in FP64."""
    case = {"B": 1, "T": 4, "H": 1, "Dk": 8, "Dv": 8, "dtype": "fp32", "seed": 7, "id": 0}
    inp = gen_inputs(case)
    args = [inp["q"].double().requires_grad_(True), inp["k"].double().requires_grad_(True),
            inp["v"].double().requires_grad_(True), inp["g"].double().requires_grad_(True),
            inp["beta"].double().requires_grad_(True),
            inp["initial_state"].double().requires_grad_(True)]

    def f(q, k, v, g, beta, s0):
        y, fs = kda_ref(q, k, v, g, beta, s0, True, -1.0, dtype=torch.float64)
        return y, fs

    ok = torch.autograd.gradcheck(f, tuple(args), raise_exception=False)
    print(f"reference FP64 gradcheck: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()

    if not _self_test_gradcheck():
        sys.exit(1)

    from golden import run

    n_pass = 0
    for case in BWD_CASES:
        case = dict(case, seed=args.seed)
        inputs = gen_inputs(case)
        gen = torch.Generator().manual_seed(args.seed + case["id"])
        dy = (torch.rand(inputs["v"].shape, generator=gen) * 0.5).to(inputs["v"].dtype)
        dfs = torch.rand(inputs["initial_state"].shape, generator=gen) * 0.5

        # autograd oracle (fp32 compute, fp64-verified reference above)
        qa = {k: v.detach().float().requires_grad_(True) for k, v in inputs.items()}
        y, fs = kda_ref(qa["q"], qa["k"], qa["v"], qa["g"], qa["beta"], qa["initial_state"],
                        bool(case.get("use_qk_l2norm", True)), float(case.get("scale", -1.0)),
                        dtype=torch.float32)
        torch.autograd.backward([y, fs], [dy.float(), dfs])
        want = {k: qa[k].grad for k in ("q", "k", "v", "g", "beta", "initial_state")}

        fn, specs, _golden, rtol, atol = bwd_kernel.build(case)
        bind = dict(inputs, dy=dy, dfinal_state=dfs)
        for spec in specs:
            if spec.name in bind:
                spec.init_value = bind[spec.name].contiguous()

        def oracle(scratch, want=want, case=case):
            for name, out in (("dq", "q"), ("dk", "k"), ("dv", "v"), ("dg", "g"),
                              ("dbeta", "beta"), ("dinitial_state", "initial_state")):
                scratch[name][:] = want[out].to(scratch[name].dtype)

        result = run(fn=fn, specs=specs, golden_fn=oracle,
                     config=dict(backend_type="a2a3", device_id=args.device),
                     rtol=rtol, atol=atol)
        status = "PASS" if result.passed else f"FAIL ({result.error})"
        n_pass += bool(result.passed)
        print(f"case {case['id']}  T={case['T']:4d} {case['dtype']:4s}  {status}")
    print(f"\n{n_pass}/{len(BWD_CASES)} backward cases passed")
    sys.exit(0 if n_pass == len(BWD_CASES) else 1)


if __name__ == "__main__":
    main()
