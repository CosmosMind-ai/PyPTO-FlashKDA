"""KDA forward contract suite — the 20 scored cases of the KimiDeltaAttention
operator contract (aligned with the official Ascend C implementation), run
against the torch oracle in tests/reference.py on a real 910C device.

    python tests/test_fwd_contract.py --device 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference import FS_ATOL, FS_RTOL, Y_TOL, gen_inputs, kda_ref  # noqa: E402

from flash_kda.kernels import recurrent_fwd  # noqa: E402

CASES = [
    {"id": 1,  "B": 1, "T": 64,  "H": 8,  "Dk": 128, "Dv": 128, "dtype": "fp16"},
    {"id": 2,  "B": 1, "T": 128, "H": 16, "Dk": 128, "Dv": 128, "dtype": "bf16"},
    {"id": 3,  "B": 4, "T": 1,   "H": 16, "Dk": 128, "Dv": 128, "dtype": "bf16"},
    {"id": 4,  "B": 2, "T": 64,  "H": 8,  "Dk": 128, "Dv": 128, "dtype": "fp16"},
    {"id": 5,  "B": 1, "T": 1,   "H": 1,  "Dk": 4,   "Dv": 3,   "dtype": "fp32"},
    {"id": 6,  "B": 4, "T": 32,  "H": 4,  "Dk": 64,  "Dv": 32,  "dtype": "fp16"},
    {"id": 7,  "B": 1, "T": 512, "H": 4,  "Dk": 64,  "Dv": 64,  "dtype": "bf16"},
    {"id": 8,  "B": 1, "T": 128, "H": 8,  "Dk": 64,  "Dv": 48,  "dtype": "fp16"},
    {"id": 9,  "B": 1, "T": 63,  "H": 3,  "Dk": 63,  "Dv": 31,  "dtype": "fp16"},
    {"id": 10, "B": 1, "T": 65,  "H": 5,  "Dk": 64,  "Dv": 33,  "dtype": "bf16"},
    {"id": 11, "B": 2, "T": 129, "H": 2,  "Dk": 33,  "Dv": 17,  "dtype": "fp32", "use_qk_l2norm": False},
    {"id": 12, "B": 1, "T": 96,  "H": 4,  "Dk": 64,  "Dv": 64,  "dtype": "fp32"},
    {"id": 13, "B": 1, "T": 17,  "H": 2,  "Dk": 32,  "Dv": 32,  "dtype": "fp16", "special": "near_zero_qk"},
    {"id": 14, "B": 1, "T": 127, "H": 4,  "Dk": 64,  "Dv": 64,  "dtype": "bf16", "special": "strong_decay"},
    {"id": 15, "B": 1, "T": 31,  "H": 2,  "Dk": 16,  "Dv": 16,  "dtype": "fp32", "use_qk_l2norm": False},
    {"id": 16, "B": 1, "T": 32,  "H": 4,  "Dk": 64,  "Dv": 64,  "dtype": "fp16",
     "special": "no_decay_explicit_scale", "scale": 0.5},
    {"id": 17, "B": 1, "T": 8,   "H": 2,  "Dk": 16,  "Dv": 16,  "dtype": "bf16"},
    {"id": 18, "B": 1, "T": 7,   "H": 2,  "Dk": 16,  "Dv": 16,  "dtype": "fp32", "special": "zero_qk"},
    {"id": 19, "B": 1, "T": 17,  "H": 3,  "Dk": 32,  "Dv": 32,  "dtype": "fp16"},
    {"id": 20, "B": 1, "T": 256, "H": 16, "Dk": 128, "Dv": 128, "dtype": "bf16"},
]


def final_state_cmp(actual, expected, **_kw):
    ok = torch.allclose(actual.float(), expected.float(), rtol=FS_RTOL, atol=FS_ATOL)
    return ok, "" if ok else f"final_state mismatch (rtol={FS_RTOL} atol={FS_ATOL})"


def initial_state_cmp(actual, expected, **_kw):
    ok = bool(torch.equal(actual, expected))
    return ok, "" if ok else "initial_state MUTATED by kernel (read-only contract)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--cases", type=str, default="")
    args = ap.parse_args()

    from golden import run

    cases = CASES if not args.cases else [c for c in CASES if str(c["id"]) in args.cases.split(",")]
    n_pass = 0
    for case in cases:
        case = dict(case, seed=args.seed)
        inputs = gen_inputs(case)
        fn, specs, _golden, rtol, atol = recurrent_fwd.build(case)
        rtol, atol = Y_TOL[case["dtype"]]  # contract tolerances, not the kernel's
        for spec in specs:
            if spec.name in inputs:
                spec.init_value = inputs[spec.name].contiguous()

        def oracle(scratch, case=case):
            y, fs = kda_ref(scratch["q"], scratch["k"], scratch["v"], scratch["g"],
                            scratch["beta"], scratch["initial_state"],
                            bool(case.get("use_qk_l2norm", True)),
                            float(case.get("scale", -1.0)), dtype=torch.float32)
            scratch["y"][:] = y.to(scratch["y"].dtype)
            scratch["final_state"][:] = fs

        result = run(fn=fn, specs=specs, golden_fn=oracle,
                     config=dict(backend_type="a2a3", device_id=args.device),
                     rtol=rtol, atol=atol,
                     compare_fn={"final_state": final_state_cmp,
                                 "initial_state": initial_state_cmp})
        status = "PASS" if result.passed else f"FAIL ({result.error})"
        n_pass += bool(result.passed)
        print(f"case {case['id']:2d}  T={case['T']:4d} {case['dtype']:4s}  {status}")
    print(f"\n{n_pass}/{len(cases)} cases passed")
    sys.exit(0 if n_pass == len(cases) else 1)


if __name__ == "__main__":
    main()
