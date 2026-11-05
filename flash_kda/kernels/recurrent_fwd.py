"""FP32 recurrent KDA with joint exact q/k normalization handoffs."""
import math

import torch
from golden import TensorSpec

import pypto.language as pl

PL_DTYPES = {"fp32": (torch.float32, pl.FP32), "fp16": (torch.float16, pl.FP16),
             "bf16": (torch.bfloat16, pl.BF16)}

# informational only — the sealed kda_harness.py owns the verdict tolerances (contract 4.2)
RTOL = 3e-3
ATOL = 3e-4

PIPELINE_STAGES = 2
RECIPROCAL_REFINEMENT = 0

# {CO}/{CC} = cast-open/cast-close for the q/k/v upcast, {YO}/{YC} for the y downcast
# (empty for fp32 cases: the @pl.jit compiler rejects same-dtype casts).
_KERNEL = '''
@pl.jit
def kda_recurrent(
    q: pl.Tensor[[B, T, H, DK], PL_DTYPE],
    k: pl.Tensor[[B, T, H, DK], PL_DTYPE],
    v: pl.Tensor[[B, T, H, DV], PL_DTYPE],
    g: pl.Tensor[[B, T, H, DK], pl.FP32],
    beta: pl.Tensor[[B, T, H], pl.FP32],
    initial_state: pl.InOut[pl.Tensor[[B, H, DK, DV], pl.FP32]],
    y: pl.Out[pl.Tensor[[B, T, H, DV], PL_DTYPE]],
    final_state: pl.Out[pl.Tensor[[B, H, DK, DV], pl.FP32]],
):
    q2 = pl.reshape(q, [B * T * H, DK])
    k2 = pl.reshape(k, [B * T * H, DK])
    v2 = pl.reshape(v, [B * T * H, DV])
    g2 = pl.reshape(g, [B * T * H, DK])
    b1 = pl.reshape(beta, [B * T * H, 1])
    s2 = pl.reshape(initial_state, [B * H * DK, DV])
    y2 = pl.reshape(y, [B * T * H, DV])
    f2 = pl.reshape(final_state, [B * H * DK, DV])
    for b in pl.parallel(0, B, 1):
        for h in pl.parallel(0, H, 1):
            for cb in pl.parallel(0, DVP, CH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kda_bh"):
                    bh = b * H + h
                    s0 = {LDS}
                    state = s0
                    w8 = {NORM_CARRIER}
                    contraction_scratch = pl.create_tile([DKP, CH], dtype=pl.FP32)
                    {NORM_SCRATCH}
                    for t, (state,) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state,)):
                        row = (b * T + t) * H + h
                        qt = {CO}{LDQ}{CC}
                        kt = {CO}{LDK}{CC}
                        vt = {CO}{LDV}{CC}
                        {L2NORM}
                        gt = pl.reshape({LDG}, [DKP, 1])
                        decay = pl.exp(gt)
                        state = pl.row_expand_mul(state, decay)                # decay first
                        kcol = pl.reshape(kt, [DKP, 1])
                        tmp = pl.row_expand_mul(state, kcol)
                        kh = pl.col_sum(tmp, contraction_scratch)               # k^T Hbar, [1, CH]
                        r = pl.sub(vt, kh)
                        bt = pl.read(b1, [row, 0])                             # scalar beta_t
                        kb = pl.mul(kcol, bt)                                  # beta_t * k_t
                        upd = pl.col_expand(state, r)                          # broadcast r rows
                        upd = pl.row_expand_mul(upd, kb)                       # outer(kb, r)
                        state = pl.add(state, upd)                             # rank-1 update
                        qcol = pl.reshape(qt, [DKP, 1])
                        tmp = pl.row_expand_mul(state, qcol)
                        yt = pl.mul(pl.col_sum(tmp, contraction_scratch), SCALE)
                        yv = {YO}yt{YC}
                        {YV}
                        pl.store(yv, [row, cb], y2)
                        {CARRY_YIELD}
                    {CARRY_RESULT}
                    {FS}
                    pl.store(state, [bh * DK, cb], f2)
    return y, final_state
'''

_SPMD_KERNEL = '\n@pl.jit\ndef kda_recurrent(\n    q: pl.Tensor[[B, T, H, DK], PL_DTYPE],\n    k: pl.Tensor[[B, T, H, DK], PL_DTYPE],\n    v: pl.Tensor[[B, T, H, DV], PL_DTYPE],\n    g: pl.Tensor[[B, T, H, DK], pl.FP32],\n    beta: pl.Tensor[[B, T, H], pl.FP32],\n    initial_state: pl.InOut[pl.Tensor[[B, H, DK, DV], pl.FP32]],\n    y: pl.Out[pl.Tensor[[B, T, H, DV], PL_DTYPE]],\n    final_state: pl.Out[pl.Tensor[[B, H, DK, DV], pl.FP32]],\n):\n    q2 = pl.reshape(q, [B * T * H, DK])\n    k2 = pl.reshape(k, [B * T * H, DK])\n    v2 = pl.reshape(v, [B * T * H, DV])\n    g2 = pl.reshape(g, [B * T * H, DK])\n    b1 = pl.reshape(beta, [B * T * H, 1])\n    s2 = pl.reshape(initial_state, [B * H * DK, DV])\n    y2 = pl.reshape(y, [B * T * H, DV])\n    f2 = pl.reshape(final_state, [B * H * DK, DV])\n    for owner in pl.spmd(B * H * VALUE_CHUNKS, name_hint="kda_owner"):\n        value_chunk = owner // (B * H)\n        batch_head = owner % (B * H)\n        b = batch_head // H\n        h = batch_head % H\n        cb = value_chunk * CH\n        bh = b * H + h\n        s0 = {LDS}\n        state = s0\n        w8 = {NORM_CARRIER}\n        contraction_scratch = pl.create_tile([DKP, CH], dtype=pl.FP32)\n        {NORM_SCRATCH}\n        for t, (state,) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state,)):\n            row = (b * T + t) * H + h\n            qt = {CO}{LDQ}{CC}\n            kt = {CO}{LDK}{CC}\n            vt = {CO}{LDV}{CC}\n            {L2NORM}\n            gt = pl.reshape({LDG}, [DKP, 1])\n            decay = pl.exp(gt)\n            state = pl.row_expand_mul(state, decay)                # decay first\n            kcol = pl.reshape(kt, [DKP, 1])\n            tmp = pl.row_expand_mul(state, kcol)\n            kh = pl.col_sum(tmp, contraction_scratch)               # k^T Hbar, [1, CH]\n            r = pl.sub(vt, kh)\n            bt = pl.read(b1, [row, 0])                             # scalar beta_t\n            kb = pl.mul(kcol, bt)                                  # beta_t * k_t\n            upd = pl.col_expand(state, r)                          # broadcast r rows\n            upd = pl.row_expand_mul(upd, kb)                       # outer(kb, r)\n            state = pl.add(state, upd)                             # rank-1 update\n            qcol = pl.reshape(qt, [DKP, 1])\n            tmp = pl.row_expand_mul(state, qcol)\n            yt = pl.mul(pl.col_sum(tmp, contraction_scratch), SCALE)\n            yv = {YO}yt{YC}\n            {YV}\n            pl.store(yv, [row, cb], y2)\n            {CARRY_YIELD}\n        {CARRY_RESULT}\n        {FS}\n        pl.store(state, [bh * DK, cb], f2)\n    return y, final_state\n'


_L2NORM = 'qsq = pl.mul(qt, qt)\n                        ksq = pl.mul(kt, kt)\n                        qps = pl.col_sum(pl.reshape(qsq, [DKP // 8, 8]), norm_scratch)\n                        kps = pl.col_sum(pl.reshape(ksq, [DKP // 8, 8]), k_norm_scratch)\n                        qsum = pl.read(qps, [0, 0]) + pl.read(qps, [0, 1]) + pl.read(qps, [0, 2]) + pl.read(qps, [0, 3]) + pl.read(qps, [0, 4]) + pl.read(qps, [0, 5]) + pl.read(qps, [0, 6]) + pl.read(qps, [0, 7])\n                        ksum = pl.read(kps, [0, 0]) + pl.read(kps, [0, 1]) + pl.read(kps, [0, 2]) + pl.read(kps, [0, 3]) + pl.read(kps, [0, 4]) + pl.read(kps, [0, 5]) + pl.read(kps, [0, 6]) + pl.read(kps, [0, 7])\n                        pl.write(w8, [0, 0], qsum + 1e-6)\n                        pl.write(w8, [0, 1], ksum + 1e-6)\n                        denominators = pl.sqrt(w8)\n                        qinv = 1.0 / pl.read(denominators, [0, 0])\n                        kinv = 1.0 / pl.read(denominators, [0, 1])\n                        qt = pl.mul(qt, qinv)\n                        kt = pl.mul(kt, kinv)'


def _packed_norm(source):
    lines = source.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == "qsq = pl.mul(qt, qt)"]
    if len(starts) != 1:
        raise ValueError("Expected one parent normalization chain")
    start = starts[0]
    end = next(index for index in range(start, len(lines)) if lines[index].strip() == "kt = pl.mul(kt, kinv)")
    prefix = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    normalization = [
        "qk_rows = pl.create_tile([8, DKP], dtype=pl.FP32)",
        "qk_rows = pl.assemble(qk_rows, pl.tile.slice(qt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [0, 0])",
        "qk_rows = pl.assemble(qk_rows, pl.tile.slice(kt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [1, 0])",
        "qk_rows = pl.set_validshape(qk_rows, 2, DKP)",
        "qk_square = pl.mul(qk_rows, qk_rows)",
        "qk_total = pl.reshape(pl.row_sum(qk_square, norm_scratch), [1, 8])",
        "qk_total = pl.set_validshape(qk_total, 1, 2)",
        "qk_total = pl.fillpad(qk_total, 0)",
        "qk_total = pl.set_validshape(qk_total, 1, 8)",
        "qk_denominator = pl.sqrt(pl.add(qk_total, 1e-6))",
        "qk_inverse = pl.reshape(pl.div(norm_ones, qk_denominator), [8, 1])",
        "qk_inverse = pl.set_validshape(qk_inverse, 2, 1)",
        "qk_normalized = pl.row_expand_mul(qk_rows, qk_inverse)",
        "qt = pl.tile.slice(qk_normalized, [1, DKP], [0, 0], pad_value=pl.PadValue.null)",
        "kt = pl.tile.slice(qk_normalized, [1, DKP], [1, 0], pad_value=pl.PadValue.null)",
    ]
    lines[start:end + 1] = [prefix + line for line in normalization]
    source = "\n".join(lines) + "\n"
    carrier = "w8 = pl.full([1, 8], dtype=pl.FP32, value=1.0)"
    if source.count(carrier) != 1:
        raise ValueError("Expected one parent normalization carrier")
    source = source.replace(carrier, "norm_ones = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)")
    source = source.replace("norm_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)", "norm_scratch = pl.create_tile([8, DKP], dtype=pl.FP32)", 1)
    source = "\n".join(line for line in source.splitlines() if "k_norm_scratch =" not in line) + "\n"
    if source.count("pl.read(") != 1 or "pl.write(" in source:
        raise ValueError("Normalization scalar boundary survived transformation")
    return source


def _epoch_coordinates(source, tokens):
    lines = source.splitlines()
    header = "for t, (state, factor) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state, factor)):"
    starts = [index for index, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1:
        raise ValueError("Expected exactly one coordinate pipeline")
    start = starts[0]
    prefix = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    reset = next(index for index in range(start + 1, len(lines)) if lines[index].strip() == "if (t + 1) % 8 == 0 or t == T - 1:")
    end = next(index for index in range(reset + 1, len(lines)) if lines[index].strip() == "state = state_out")
    body = [line[len(prefix) + 4:] for line in lines[start + 1:reset]]
    if any("yield_" in line for line in body):
        raise ValueError("Unexpected yield inside numerical token body")
    rendered = [
        "for epoch, (state,) in pl.range(T // 8, init_values=(state,)):",
        "    factor = pl.tile.full([1, DKP], dtype=pl.FP32, value=1.0)",
        "    for local_t, (state, factor) in pl.pipeline(8, stage=PIPELINE_STAGES, init_values=(state, factor)):",
        "        t = epoch * 8 + local_t",
    ]
    rendered.extend("        " + line for line in body)
    rendered.extend([
        "        state_out, factor_out = pl.yield_(state, factor_next)",
        "    state = pl.row_expand_mul(state_out, pl.reshape(factor_out, [DKP, 1]))",
        "    epoch_out = pl.yield_(state)",
        "state = epoch_out",
    ])
    if tokens % 8:
        import re
        assigned = {match.group(1) for line in body if (match := re.match(r"^([A-Za-z_][A-Za-z_0-9]*) =", line))}
        assigned.add("factor")
        names = re.compile(r"(?<![A-Za-z_0-9])(" + "|".join(sorted(assigned, key=len, reverse=True)) + r")(?![A-Za-z_0-9])")
        tail_body = [names.sub(lambda match: "tail_" + match.group(1), line) for line in body]
        rendered.extend([
            "tail_factor_initial = pl.tile.full([1, DKP], dtype=pl.FP32, value=1.0)",
            "for local_t, (tail_state, tail_factor) in pl.pipeline(T % 8, stage=PIPELINE_STAGES, init_values=(state, tail_factor_initial)):",
            "    t = (T // 8) * 8 + local_t",
        ])
        rendered.extend("    " + line for line in tail_body)
        rendered.extend([
            "    tail_state_out, tail_factor_out = pl.yield_(tail_state, tail_factor_next)",
            "state = pl.row_expand_mul(tail_state_out, pl.reshape(tail_factor_out, [DKP, 1]))",
        ])
    lines[start:end + 1] = [prefix + line for line in rendered]
    initializer = prefix + "factor = pl.tile.full([1, DKP], dtype=pl.FP32, value=1.0)"
    if lines[:start].count(initializer) != 1:
        raise ValueError("Expected one inherited factor initializer")
    lines.remove(initializer)
    return "\n".join(lines) + "\n"


def _parent_build(shape):
    B, T, H = shape["B"], shape["T"], shape["H"]
    DK, DV = shape["Dk"], shape["Dv"]
    L2NORM = bool(shape.get("use_qk_l2norm", True))
    SCALE = float(shape.get("scale", -1.0))
    if SCALE <= 0:
        SCALE = 1.0 / math.sqrt(DK)
    TORCH_DTYPE, PL_DTYPE = PL_DTYPES[shape["dtype"]]
    need_cast = PL_DTYPE != pl.FP32
    align = 8 if PL_DTYPE == pl.FP32 else 16  # 32-byte tile rows
    DKP = -(-DK // align) * align
    DVP = -(-DV // align) * align
    # UB budget: chunk Dv so the state tile stays <= Dk x 64 fp32 (32 KB). Chunking only happens
    # for DV = 128 (all such contract cases are exactly divisible by 64), so every chunk is
    # fully valid there; single-chunk cases carry the pad in CV.
    CH = 64 if DVP > 64 else DVP
    CV = CH if DVP > 64 else DV
    VALUE_CHUNKS = (DVP + CH - 1) // CH

    def cast_pair(target):
        return ("pl.cast(", ", target_type=pl.%s)" % target) if need_cast else ("", "")

    down = {pl.FP32: "FP32", pl.FP16: "FP16", pl.BF16: "BF16"}[PL_DTYPE]
    co, cc = cast_pair("FP32")
    yo, yc = cast_pair(down)

    def load_expr(tensor, offsets, shape, valid):
        """GM load, padding the row width only when the case is actually unaligned. fillpad zeroes
        the pad; set_validshape then re-marks the whole tile valid — a leftover pad tag poisons
        tile.assemble/subview downstream on this stack."""
        rows, cols = shape.strip("[]").split(", ")
        base = "pl.load(%s, %s, %s" % (tensor, offsets, shape)
        if valid is None:
            return base + ")"
        return ("pl.set_validshape(pl.fillpad(" + base + ", valid_shape=%s), 0), %s, %s)"
                % (valid, rows, cols))

    dk_pad = DKP != DK
    dynamic_cv = DVP > CH and DV % CH != 0
    cv_pad = CV != CH or dynamic_cv
    st_pad = dk_pad or cv_pad
    lds = load_expr("s2", "[bh * DK, cb]", "[DKP, CH]", "[DK, CV]" if st_pad else None)
    ldq = load_expr("q2", "[row, 0]", "[1, DKP]", "[1, DK]" if dk_pad else None)
    ldk = load_expr("k2", "[row, 0]", "[1, DKP]", "[1, DK]" if dk_pad else None)
    ldv = load_expr("v2", "[row, cb]", "[1, CH]", "[1, CV]" if cv_pad else None)
    ldg = load_expr("g2", "[row, 0]", "[1, DKP]", "[1, DK]" if dk_pad else None)
    src = ((_KERNEL if B * H * VALUE_CHUNKS == 1 else _SPMD_KERNEL)
           .replace("{CARRY_YIELD}", "state_out = pl.yield_(state)" if T >= 4 else "pass")
           .replace("{CARRY_RESULT}", "state = state_out" if T >= 4 else "pass")
           .replace("{CO}", co).replace("{CC}", cc)
           .replace("{YO}", yo).replace("{YC}", yc)
           .replace("{YV}", "yv = pl.set_validshape(yv, 1, CV)" if cv_pad else "pass")
           .replace("{FS}", "state = pl.set_validshape(state, DK, CV)" if st_pad else "pass")
           .replace("{LDS}", lds).replace("{LDQ}", ldq).replace("{LDK}", ldk)
           .replace("{LDV}", ldv).replace("{LDG}", ldg)
           .replace("{L2NORM}", (_L2NORM.replace("\n                        ", "\n            ") if B * H * VALUE_CHUNKS > 1 else _L2NORM) if L2NORM else "pass")
           .replace("{NORM_CARRIER}", "pl.full([1, 8], dtype=pl.FP32, value=1.0)" if L2NORM else "pl.create_tile([1, 8], dtype=pl.FP32)")
           .replace("{NORM_SCRATCH}", ('norm_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)\n                    k_norm_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)'.replace("\n                    ", "\n        ") if B * H * VALUE_CHUNKS > 1 else 'norm_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)\n                    k_norm_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)') if L2NORM else "pass"))
    if T < 4:
        src = src.replace(
            "for t, (state,) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state,)):",
            "for t in pl.range(T):",
        )
    if T >= 16 and 64 <= DKP <= 128:
        lines = src.splitlines()
        transformed = []
        for line in lines:
            indent = line[:len(line) - len(line.lstrip())]
            text = line.strip()
            if text == "state = s0":
                transformed.extend([line, indent + "factor = pl.tile.full([1, DKP], dtype=pl.FP32, value=1.0)"])
            elif text == "for t, (state,) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state,)):":
                transformed.append(indent + "for t, (state, factor) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state, factor)):")
            elif text == "gt = pl.reshape(" + ldg + ", [DKP, 1])":
                transformed.extend([indent + "gt = " + ldg, indent + "factor_next = pl.mul(factor, pl.exp(gt))", indent + "inverse = pl.recip(factor_next)"])
                if RECIPROCAL_REFINEMENT:
                    transformed.extend([indent + "correction = pl.sub(2.0, pl.mul(factor_next, inverse))", indent + "inverse = pl.mul(inverse, correction)"])
            elif text == "decay = pl.exp(gt)" or text.startswith("state = pl.row_expand_mul(state, decay)"):
                continue
            elif text == "kcol = pl.reshape(kt, [DKP, 1])":
                transformed.extend([indent + "kcol = pl.reshape(pl.mul(kt, factor_next), [DKP, 1])", indent + "inverse_k = pl.reshape(pl.mul(kt, inverse), [DKP, 1])"])
            elif text.startswith("kb = pl.mul(kcol, bt)"):
                transformed.append(indent + "kb = pl.mul(inverse_k, bt)")
            elif text == "qcol = pl.reshape(qt, [DKP, 1])":
                transformed.append(indent + "qcol = pl.reshape(pl.mul(qt, factor_next), [DKP, 1])")
            elif text == "state_out = pl.yield_(state)":
                transformed.extend([indent + "if (t + 1) % 8 == 0 or t == T - 1:", indent + "    state = pl.row_expand_mul(state, pl.reshape(factor_next, [DKP, 1]))", indent + "    factor_next = pl.tile.full([1, DKP], dtype=pl.FP32, value=1.0)", indent + "state_out, factor_out = pl.yield_(state, factor_next)"])
            elif text == "state = state_out":
                transformed.append(indent + "state = state_out")
            else:
                transformed.append(line)
        src = "\n".join(transformed) + "\n"
    if L2NORM and (T >= 16 or B * H * VALUE_CHUNKS > 1):
        src = _packed_norm(src)
    if T >= 16 and 64 <= DKP <= 128:
        src = _epoch_coordinates(src, T)
    if dynamic_cv:
        src = src.replace("CV", "pl.min(CH, DV - cb)")
    REPLAY_INTERVALS = _interval_count(B, T, H, VALUE_CHUNKS)
    if REPLAY_INTERVALS > 1:
        src = _replay_source(src, T, DKP)
    ns = {"pl": pl, "B": B, "T": T, "H": H, "DK": DK, "DV": DV, "DKP": DKP, "DVP": DVP,
          "CH": CH, "CV": CV, "VALUE_CHUNKS": VALUE_CHUNKS, "PL_DTYPE": PL_DTYPE, "SCALE": SCALE, "PIPELINE_STAGES": PIPELINE_STAGES, "REPLAY_INTERVALS": REPLAY_INTERVALS, "REPLAY_UNITS": (T + 7) // 8, "STATE_OWNERS": B * H * VALUE_CHUNKS}
    # pl.jit parses via inspect.getsource: give the exec'd code a linecache-backed filename
    import linecache
    fname = "kda_exact_prefix_replay_screen_admission_%s.py" % __import__("hashlib").sha256(src.encode()).hexdigest()
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec"), ns)
    kda_recurrent = ns["kda_recurrent"]

    def specs():
        return [
            TensorSpec("q", [B, T, H, DK], TORCH_DTYPE, init_value=torch.randn),
            TensorSpec("k", [B, T, H, DK], TORCH_DTYPE, init_value=torch.randn),
            TensorSpec("v", [B, T, H, DV], TORCH_DTYPE, init_value=torch.randn),
            TensorSpec("g", [B, T, H, DK], torch.float32, init_value=torch.randn),
            TensorSpec("beta", [B, T, H], torch.float32, init_value=torch.rand),
            TensorSpec("initial_state", [B, H, DK, DV], torch.float32, init_value=torch.randn),
            TensorSpec("y", [B, T, H, DV], TORCH_DTYPE),
            TensorSpec("final_state", [B, H, DK, DV], torch.float32),
        ]

    def golden(t):
        # minimal in-place reference for devkit smoke tests only; the sealed harness judges
        # correctness with its own contract-4.1 oracle
        dtype = t["q"].dtype
        qf, kf, vf = t["q"].float(), t["k"].float(), t["v"].float()
        state = t["initial_state"].float().clone()
        ys = []
        for ti in range(T):
            qt, kt = qf[:, ti], kf[:, ti]
            if L2NORM:
                qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            hb = torch.exp(t["g"][:, ti]).unsqueeze(-1) * state
            r = vf[:, ti] - (kt.unsqueeze(-1) * hb).sum(dim=2)
            state = hb + t["beta"][:, ti].reshape(B, H, 1, 1) * kt.unsqueeze(-1) * r.unsqueeze(2)
            ys.append(SCALE * (qt.unsqueeze(-1) * state).sum(dim=2))
        t["y"][:] = torch.stack(ys, dim=1).to(dtype)
        t["final_state"][:] = state

    return kda_recurrent, specs(), golden, RTOL, ATOL


INTERVAL_CAP = 8
MINIMUM_SEQUENCE_LENGTH = 64
MAXIMUM_TOTAL_OWNERS = 32
MAXIMUM_STATE_OWNERS = 16


def _interval_count(batch, tokens, heads, value_chunks):
    owners = batch * heads * value_chunks
    if tokens < MINIMUM_SEQUENCE_LENGTH or owners == 1 or owners > MAXIMUM_STATE_OWNERS:
        return 1
    limit = min(INTERVAL_CAP, MAXIMUM_TOTAL_OWNERS // owners, (tokens + 7) // 8)
    return 1 << (limit.bit_length() - 1) if limit >= 2 else 1


def _replay_source(source, tokens, key_padding):
    lines = source.splitlines()
    owner_header = 'for owner in pl.spmd(B * H * VALUE_CHUNKS, name_hint="kda_owner"):'
    owner_start = next(index for index, line in enumerate(lines) if line.strip() == owner_header)
    indent = lines[owner_start][:len(lines[owner_start]) - len(lines[owner_start].lstrip())]
    lines[owner_start] = indent + 'for temporal_owner in pl.spmd(STATE_OWNERS * REPLAY_INTERVALS, name_hint="kda_replay_owner"):'
    lines[owner_start + 1:owner_start + 1] = [indent + "    " + line for line in (
        "interval = temporal_owner // STATE_OWNERS",
        "owner = temporal_owner % STATE_OWNERS",
        "interval_start = (interval * REPLAY_UNITS // REPLAY_INTERVALS) * 8",
        "interval_end_unit = (interval + 1) * REPLAY_UNITS // REPLAY_INTERVALS",
        "interval_epoch_end = pl.min(interval_end_unit, T // 8)",
        "interval_token_end = pl.min(interval_end_unit * 8, T)",
    )]
    coordinate = tokens >= 16 and 64 <= key_padding <= 128
    if coordinate:
        header = "for epoch, (state,) in pl.range(T // 8, init_values=(state,)):"
        substitute = "for epoch, (state,) in pl.range(interval_epoch_end, init_values=(state,)):"
    else:
        header = "for t, (state,) in pl.pipeline(T, stage=PIPELINE_STAGES, init_values=(state,)):"
        substitute = "for t, (state,) in pl.pipeline(interval_token_end, stage=PIPELINE_STAGES, init_values=(state,)):"
    matches = [index for index, line in enumerate(lines) if line.strip() == header]
    if len(matches) != 1:
        raise ValueError("Expected one absolute parent recurrence loop")
    index = matches[0]
    lines[index] = lines[index].replace(header, substitute)
    projection_starts = [index for index, line in enumerate(lines) if line.strip().startswith(("qcol = ", "tail_qcol = "))]
    expected = 2 if coordinate and tokens % 8 else 1
    if len(projection_starts) != expected:
        raise ValueError("Unexpected parent output projection count")
    for start in reversed(projection_starts):
        end = next(index for index in range(start, len(lines)) if lines[index].strip().startswith("pl.store(") and lines[index].strip().endswith(", y2)"))
        prefix = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
        lines[start:end + 1] = [prefix + "if t >= interval_start:"] + ["    " + line for line in lines[start:end + 1]]
    if coordinate and tokens % 8:
        start = next(index for index, line in enumerate(lines) if line.strip().startswith("tail_factor_initial = "))
        end = next(index for index in range(start, len(lines)) if lines[index].strip() == "state = pl.row_expand_mul(tail_state_out, pl.reshape(tail_factor_out, [DKP, 1]))")
        prefix = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
        lines[start:end + 1] = [prefix + "if interval == REPLAY_INTERVALS - 1:"] + ["    " + line for line in lines[start:end + 1]]
    final = next(index for index, line in enumerate(lines) if line.strip() == "pl.store(state, [bh * DK, cb], f2)")
    start = final - 1
    if lines[start].strip() != "pass" and not lines[start].strip().startswith("state = pl.set_validshape("):
        raise ValueError("Unexpected final-state validity chain")
    prefix = lines[final][:len(lines[final]) - len(lines[final].lstrip())]
    final_lines = [line.replace("state = pl.set_validshape(", "final_tile = pl.set_validshape(") for line in lines[start:final + 1]]
    final_lines[0] = prefix + "final_tile = state" if final_lines[0].strip() == "pass" else final_lines[0]
    final_lines[-1] = final_lines[-1].replace("pl.store(state,", "pl.store(final_tile,")
    lines[start:final + 1] = [prefix + "if interval == REPLAY_INTERVALS - 1:"] + ["    " + line for line in final_lines]
    return "\n".join(lines) + "\n"


import hashlib
import linecache

CONSUMER_MODE = "four"
CHUNK_THRESHOLD = 64
CHUNK_WIDTH = 16
PHYSICAL_WIDTH = 16
PREPARATION_MODE = "simd"
PREP_CHUNKS_PER_TASK = 2


def _indented(lines, depth):
    return "\n".join(" " * depth + line if line else "" for line in lines.splitlines())

def _sum_eight(name):
    return " + ".join("pl.read(%s, [0, %d])" % (name, column) for column in range(8))

def _padded_load(tensor, offsets, rows, width, valid_rows, valid_width):
    return "pl.set_validshape(pl.fillpad(pl.load(%s, %s, [%s, %s], valid_shape=[%s, %s]), 0), %s, %s)" % (tensor, offsets, rows, width, valid_rows, valid_width, rows, width)

def _normalization():
    return "\n".join([
        "qk_rows = pl.create_tile([8, DKP], dtype=pl.FP32)",
        "qk_rows = pl.assemble(qk_rows, pl.tile.slice(qt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [0, 0])",
        "qk_rows = pl.assemble(qk_rows, pl.tile.slice(kt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [1, 0])",
        "qk_rows = pl.set_validshape(qk_rows, 2, DKP)",
        "qk_square = pl.mul(qk_rows, qk_rows)",
        "qk_total = pl.reshape(pl.row_sum(qk_square, norm_scratch), [1, 8])",
        "qk_total = pl.set_validshape(qk_total, 1, 2)",
        "qk_total = pl.fillpad(qk_total, 0)",
        "qk_total = pl.set_validshape(qk_total, 1, 8)",
        "qk_denominator = pl.sqrt(pl.add(qk_total, 1e-6))",
        "qk_inverse = pl.reshape(pl.div(norm_ones, qk_denominator), [8, 1])",
        "qk_inverse = pl.set_validshape(qk_inverse, 2, 1)",
        "qk_normalized = pl.row_expand_mul(qk_rows, qk_inverse)",
        "qt = pl.tile.slice(qk_normalized, [1, DKP], [0, 0], pad_value=pl.PadValue.null)",
        "kt = pl.tile.slice(qk_normalized, [1, DKP], [1, 0], pad_value=pl.PadValue.null)",
    ])

def _product(prefix, left, right):
    return "\n".join([
        ("%s_lmat = %s" % (prefix, left) if left in ("fk", "fq", "inverse_top", "inverse_bottom", "lower_block") else "%s_lmat = pl.tile.move(%s, target_memory=pl.MemorySpace.Mat)" % (prefix, left)),
        "%s_left = pl.tile.move(%s_lmat, target_memory=pl.MemorySpace.Left)" % (prefix, prefix),
        "%s_rmat = pl.tile.move(%s, target_memory=pl.MemorySpace.Mat)" % (prefix, right),
        "%s_right = pl.tile.move(%s_rmat, target_memory=pl.MemorySpace.Right)" % (prefix, prefix),
        "%s_acc = pl.matmul(%s_left, %s_right)" % (prefix, prefix, prefix),
        "%s = pl.tile.move(%s_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)" % (prefix, prefix),
    ])

_PREPARE = '\nhead = chunk_task // NC\nchunk = chunk_task % NC\nbatch_index = head // H\nhead_index = head % H\nbase = chunk_task * C\nnorm_ones = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)\nnorm_scratch = pl.create_tile([8, DKP], dtype=pl.FP32)\npair_scratch = pl.create_tile([DKP // 8, 8], dtype=pl.FP32)\nqrows = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nkrows = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nsrows = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nvrows = pl.tile.full([C, DVP], dtype=pl.FP32, value=0.0)\nbrows = pl.tile.full([1, C], dtype=pl.FP32, value=0.0)\ncumulative = pl.tile.full([1, DKP], dtype=pl.FP32, value=0.0)\nfor token, (qrows, krows, srows, vrows, cumulative) in pl.range(CW, init_values=(qrows, krows, srows, vrows, cumulative)):\n    token_index = chunk * CW + token\n    row = (batch_index * T + pl.min(token_index, T - 1)) * H + head_index\n    qt = __Q_LOAD__\n    kt = __K_LOAD__\n    vt = __V_LOAD__\n    gt = __G_LOAD__\n__NORMALIZATION__\n    token_beta = pl.read(b1, [row, 0])\n    if token_index >= T:\n        qt = pl.mul(qt, 0.0)\n        kt = pl.mul(kt, 0.0)\n        vt = pl.mul(vt, 0.0)\n        gt = pl.mul(gt, 0.0)\n        token_beta = 0.0\n    cumulative = pl.add(cumulative, gt)\n    qrows = pl.assemble(qrows, pl.tile.slice(qt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token, 0])\n    krows = pl.assemble(krows, pl.tile.slice(kt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token, 0])\n    srows = pl.assemble(srows, pl.tile.slice(cumulative, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token, 0])\n    vrows = pl.assemble(vrows, pl.tile.slice(vt, [1, DVP], [0, 0], pad_value=pl.PadValue.null), [token, 0])\n    pl.write(brows, [0, token], token_beta)\n    qdone, kdone, sdone, vdone, cumulative_done = pl.yield_(qrows, krows, srows, vrows, cumulative)\nqrows = qdone\nkrows = kdone\nsrows = sdone\nvrows = vdone\nlast_log = pl.tile.slice(srows, [1, DKP], [CW - 1, 0], pad_value=pl.PadValue.null)\nlast_decay = pl.exp(last_log)\nlow = pl.tile.full([C, C], dtype=pl.FP32, value=0.0)\noutput_coeff = pl.tile.full([C, C], dtype=pl.FP32, value=0.0)\nfor target in pl.range(CW):\n    si = pl.tile.slice(srows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    ki = pl.tile.slice(krows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    qi = pl.tile.slice(qrows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    for source_token in pl.range(target + 1):\n        sj = pl.tile.slice(srows, [1, DKP], [source_token, 0], pad_value=pl.PadValue.null)\n        kj = pl.tile.slice(krows, [1, DKP], [source_token, 0], pad_value=pl.PadValue.null)\n        causal_decay = pl.exp(pl.sub(si, sj))\n        decayed_key = pl.mul(kj, causal_decay)\n        key_product = pl.mul(ki, decayed_key)\n        query_product = pl.mul(qi, decayed_key)\n        key_parts = pl.col_sum(pl.reshape(key_product, [DKP // 8, 8]), pair_scratch)\n        key_dot = __KEY_SUM__\n        query_parts = pl.col_sum(pl.reshape(query_product, [DKP // 8, 8]), pair_scratch)\n        query_dot = __QUERY_SUM__\n        source_beta = pl.read(brows, [0, source_token])\n        if source_token < target:\n            pl.write(low, [target, source_token], source_beta * key_dot)\n        pl.write(output_coeff, [target, source_token], source_beta * query_dot)\nsolved_w = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nsolved_u = pl.tile.full([C, DVP], dtype=pl.FP32, value=0.0)\nqcoeff = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nbcoeff = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)\nfor target, (solved_w, solved_u, qcoeff, bcoeff) in pl.range(CW, init_values=(solved_w, solved_u, qcoeff, bcoeff)):\n    si = pl.tile.slice(srows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    ki = pl.tile.slice(krows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    qi = pl.tile.slice(qrows, [1, DKP], [target, 0], pad_value=pl.PadValue.null)\n    wr = pl.mul(ki, pl.exp(si))\n    ur = pl.tile.slice(vrows, [1, DVP], [target, 0], pad_value=pl.PadValue.null)\n    for previous, (wr, ur) in pl.range(target, init_values=(wr, ur)):\n        coefficient = pl.read(low, [target, previous])\n        previous_w = pl.tile.slice(solved_w, [1, DKP], [previous, 0], pad_value=pl.PadValue.null)\n        previous_u = pl.tile.slice(solved_u, [1, DVP], [previous, 0], pad_value=pl.PadValue.null)\n        wr = pl.sub(wr, pl.mul(previous_w, coefficient))\n        ur = pl.sub(ur, pl.mul(previous_u, coefficient))\n        wr_done, ur_done = pl.yield_(wr, ur)\n    solved_w = pl.assemble(solved_w, wr_done, [target, 0])\n    solved_u = pl.assemble(solved_u, ur_done, [target, 0])\n    qcoeff = pl.assemble(qcoeff, pl.mul(qi, pl.exp(si)), [target, 0])\n    end_key = pl.mul(pl.mul(ki, pl.exp(pl.sub(last_log, si))), pl.read(brows, [0, target]))\n    bcoeff = pl.assemble(bcoeff, end_key, [target, 0])\n    wdone, udone, qcoeff_done, bcoeff_done = pl.yield_(solved_w, solved_u, qcoeff, bcoeff)\nprojection_base = chunk_task * (2 * C)\naction_base = chunk_task * (C + DKP)\npl.store(wdone, [projection_base, 0], prepared_p)\npl.store(qcoeff_done, [projection_base + C, 0], prepared_p)\npl.store(udone, [base, 0], prepared_u)\npl.store(output_coeff, [action_base, 0], prepared_j)\nend_keys = pl.transpose(bcoeff_done, 0, 1)\npl.store(end_keys, [action_base + C, 0], prepared_j)\npl.store(last_decay, [chunk_task, 0], prepared_d)\n'

_SCAN_PREFIX = '''
value_chunk = owner // (B * H)
head = owner % (B * H)
batch_index = head // H
head_index = head % H
cb = value_chunk * CH
cv = pl.min(CH, DV - cb)
state = __STATE_LOAD__
for chunk, (state,) in pl.range(NC, init_values=(state,)):
    chunk_task = head * NC + chunk
    base = chunk_task * C
    projection_base = chunk_task * (2 * C)
    action_base = chunk_task * (C + DKP)
'''

_SCATTER = '''
output_tile = chunk_y
output_tile = __Y_CAST__
valid_tokens = pl.min(CW, T - chunk * CW)
pl.store(pl.set_validshape(output_tile, valid_tokens, cv), [batch_index * T + chunk * CW, head_index * DV + cb], y_block)
'''

def _slice(name, source, rows, cols, row=0, column=0):
    return "%s = pl.tile.slice(%s, [%s, %s], [%s, %s], pad_value=pl.PadValue.null)" % (name, source, rows, cols, row, column)

def _consumer():
    body = "\n".join([
        "packed_p = pl.load(prepared_p, [projection_base, 0], [2 * C, DKP], target_memory=pl.MemorySpace.Mat)",
        "projection_left = pl.tile.move(packed_p, target_memory=pl.MemorySpace.Left)",
        "state_mat = pl.tile.move(state, target_memory=pl.MemorySpace.Mat)",
        "projection_right = pl.tile.move(state_mat, target_memory=pl.MemorySpace.Right)",
        "projection_acc = pl.matmul(projection_left, projection_right)",
        "uc = " + _padded_load("prepared_u", "[base, cb]", "C", "CH", "C", "cv"),
        "dc = pl.load(prepared_d, [chunk_task, 0], [1, DKP])",
        "projection = pl.tile.move(projection_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)",
        _slice("wh", "projection", "C", "CH"),
        "residual = pl.sub(uc, wh)",
        _slice("qh_view", "projection", "C", "CH", "C"),
        "qh = pl.mul(qh_view, 1.0)",
        "packed_j = pl.load(prepared_j, [action_base, 0], [C + DKP, C], target_memory=pl.MemorySpace.Mat)",
        "action_left = pl.tile.move(packed_j, target_memory=pl.MemorySpace.Left)",
        "residual_mat = pl.tile.move(residual, target_memory=pl.MemorySpace.Mat)",
        "action_right = pl.tile.move(residual_mat, target_memory=pl.MemorySpace.Right)",
        "action_acc = pl.matmul(action_left, action_right)",
        "action = pl.tile.move(action_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)",
        _slice("mr", "action", "C", "CH"),
        "chunk_y = pl.mul(pl.add(qh, mr), SCALE)",
        _slice("br", "action", "DKP", "CH", "C"),
        "state = pl.add(pl.row_expand_mul(state, pl.reshape(dc, [DKP, 1])), br)",
    ])
    return _SCAN_PREFIX + _indented(body + "\n" + _SCATTER + "\nstate_done = pl.yield_(state)", 4) + "\nstate = state_done\npl.store(pl.set_validshape(state, DK, cv), [head * DK, cb], f2)\n"



_single_chunk_consumer = _consumer


def _consumer():
    original = _single_chunk_consumer()
    prefix, remaining = original.split("for chunk, (state,) in pl.range(NC, init_values=(state,)):\n", 1)
    original_loop, suffix = remaining.split("\nstate = state_done\n", 1)
    chunk_body = "\n".join(line[4:] for line in original_loop.splitlines()[4:-1])
    paired_body = chunk_body
    replacements = (
        ("packed_p = pl.load(prepared_p, [projection_base, 0], [2 * C, DKP], target_memory=pl.MemorySpace.Mat)\nprojection_left = pl.tile.move(packed_p, target_memory=pl.MemorySpace.Left)",
         "projection_left = pl.tile.extract(pair_p, lane * 2 * C, 0, [2 * C, DKP], target_memory=pl.MemorySpace.Left)"),
        ("uc = " + _padded_load("prepared_u", "[base, cb]", "C", "CH", "C", "cv"),
         "__PAIR_U_LOAD__\nuc = pl.tile.slice(pair_u, [C, CH], [lane * C, 0], pad_value=pl.PadValue.null)"),
        ("dc = pl.load(prepared_d, [chunk_task, 0], [1, DKP])",
         "__PAIR_D_LOAD__\ndc = pl.tile.slice(pair_d, [1, DKP], [lane, 0], pad_value=pl.PadValue.null)"),
        ("packed_j = pl.load(prepared_j, [action_base, 0], [C + DKP, C], target_memory=pl.MemorySpace.Mat)\naction_left = pl.tile.move(packed_j, target_memory=pl.MemorySpace.Left)",
         "__PAIR_J_LOAD__\naction_left = pl.tile.extract(pair_j, lane * (C + DKP), 0, [C + DKP, C], target_memory=pl.MemorySpace.Left)"),
    )
    for before, after in replacements:
        if paired_body.count(before) != 1:
            raise ValueError("Pinned consumer site changed: " + before)
        paired_body = paired_body.replace(before, after, 1)
    lines = [prefix.rstrip(),
             "chunk = 0",
             "chunk_task = head * NC + chunk",
             "chunk_origin = chunk_task",
             "for pair, (state,) in pl.range(NC // 2, init_values=(state,)):",
             "    chunk_task = chunk_origin + pair * 2",
             "    pair_p = pl.load(prepared_p, [chunk_task * 2 * C, 0], [4 * C, DKP], target_memory=pl.MemorySpace.Mat)"]
    pair_loads = {
        "__PAIR_U_LOAD__": "pair_u = " + _padded_load("prepared_u", "[(chunk_origin + pair * 2) * C, cb]", "2 * C", "CH", "2 * C", "cv"),
        "__PAIR_D_LOAD__": "pair_d = pl.load(prepared_d, [chunk_origin + pair * 2, 0], [2, DKP])",
        "__PAIR_J_LOAD__": "pair_j = pl.load(prepared_j, [(chunk_origin + pair * 2) * (C + DKP), 0], [2 * (C + DKP), C], target_memory=pl.MemorySpace.Mat)",
    }
    for lane in range(2):
        lines.append("    chunk = pair * 2 + %d" % lane)
        lane_body = paired_body.replace("lane", str(lane))
        for marker, load in pair_loads.items():
            lane_body = lane_body.replace(marker, load if lane == 0 else "")
        lines.append(_indented(lane_body, 4))
    lines.extend(["    pair_done = pl.yield_(state)",
                  "state = pair_done",
                  "if NC % 2 != 0:",
                  "    chunk = NC - 1",
                  "    chunk_task = chunk_origin + chunk",
                  "    base = chunk_task * C",
                  "    projection_base = chunk_task * (2 * C)",
                  "    action_base = chunk_task * (C + DKP)",
                  _indented(chunk_body, 4),
                  suffix.rstrip()])
    return "\n".join(lines) + "\n"



_out_of_place_pair_consumer = _consumer


def _consumer():
    source = _out_of_place_pair_consumer()
    before = "pair_u = pl.set_validshape(pl.fillpad("
    after = "pair_u = pl.set_validshape(pl.tile.fillpad_inplace("
    if source.count(before) != 1:
        raise ValueError("Pinned bulk-U padding site changed")
    return source.replace(before, after, 1)


def _chunk_source(case):
    dtype = case["dtype"]
    prepare = _PREPARE if PREPARATION_MODE == "scalar" else _simd_prepare(case)
    for marker, tensor, width, logical in (("__Q_LOAD__", "q2", "DKP", "DK"), ("__K_LOAD__", "k2", "DKP", "DK"), ("__V_LOAD__", "v2", "DVP", "DV"), ("__G_LOAD__", "g2", "DKP", "DK")):
        expression = _padded_load(tensor, "[row, 0]", "1", width, "1", logical)
        if tensor != "g2" and dtype != "fp32":
            expression = "pl.cast(%s, target_type=pl.FP32)" % expression
        prepare = prepare.replace(marker, expression)
    prepare = prepare.replace("__NORMALIZATION__", _indented(_normalization(), 8 if PREPARATION_MODE == "simd" else 4) if case.get("use_qk_l2norm", True) else ("        pass" if PREPARATION_MODE == "simd" else "    pass"))
    prepare = prepare.replace("__KEY_SUM__", _sum_eight("key_parts")).replace("__QUERY_SUM__", _sum_eight("query_parts"))
    scan = _consumer().replace("__STATE_LOAD__", _padded_load("s2", "[head * DK, cb]", "DKP", "CH", "DK", "cv"))
    scan = scan.replace("__Y_CAST__", "output_tile" if dtype == "fp32" else "pl.cast(output_tile, target_type=PL_DTYPE, mode='rint')")
    header = _SPMD_KERNEL.split("    for owner in pl.spmd", 1)[0]
    shapes = [("prepared_p", "[B * H * NC * 2 * C, DKP]"), ("prepared_j", "[B * H * NC * (C + DKP), C]"), ("prepared_u", "[B * H * NC * C, DVP]"), ("prepared_d", "[B * H * NC, DKP]")]
    buffers = "\n".join("    %s = pl.create_tensor(%s, dtype=pl.FP32)" % item for item in shapes)
    if PREPARATION_MODE == "scalar":
        body = "head = group_task // NG\ngroup_index = group_task % NG\nfor group_lane in pl.range(G):\n    chunk = group_index * G + group_lane\n    if chunk < NC:\n        chunk_task = head * NC + chunk\n" + _indented(prepare.replace("head = chunk_task // NC\nchunk = chunk_task % NC\n", ""), 8)
    else:
        body = prepare
    launch = "    with pl.spmd(B * H * NG, name_hint='kda_independent_chunk_prepare') as input_tid:\n        group_task = pl.tile.get_block_idx()\n" + _indented(body, 8)
    dependency = "input_tid"
    launch += "\n    for owner in pl.spmd(B * H * VALUE_CHUNKS, name_hint='kda_four_product_owner', deps=[%s], optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=2)]):\n" % dependency + _indented(scan, 8)
    header += "    q_block = pl.reshape(q, [B * T, H * DK])\n    k_block = pl.reshape(k, [B * T, H * DK])\n    v_block = pl.reshape(v, [B * T, H * DV])\n    g_block = pl.reshape(g, [B * T, H * DK])\n    beta_block = pl.reshape(beta, [B * T, H])\n"
    return header + "    y_block = pl.reshape(y, [B * T, H * DV])\n" + buffers + "\n" + launch + "\n    return y, final_state\n"

def build(case):
    parent_result = _parent_build(case)
    if case["T"] < CHUNK_THRESHOLD:
        return parent_result
    batch, time, heads = case["B"], case["T"], case["H"]
    dk, dv = case["Dk"], case["Dv"]
    dkp, dvp = ((dk + 15) // 16) * 16, ((dv + 15) // 16) * 16
    ch = min(64, dvp)
    chunks = (time + CHUNK_WIDTH - 1) // CHUNK_WIDTH
    groups = (chunks + PREP_CHUNKS_PER_TASK - 1) // PREP_CHUNKS_PER_TASK
    owner_limit = min(CONSUMER_OWNER_CAP, batch * heads * groups)
    for value_width in (32, 16):
        if (MIN_VALUE_TILE <= value_width < ch
                and batch * heads * ((dvp + value_width - 1) // value_width) <= owner_limit):
            ch = value_width
    scale = float(case.get("scale", -1.0))
    if scale <= 0:
        scale = dk ** -0.5
    namespace = {"pl": pl, "B": batch, "T": time, "H": heads, "DK": dk, "DV": dv, "DKP": dkp,
                 "DVP": dvp, "CH": ch, "C": PHYSICAL_WIDTH, "NC": (time + CHUNK_WIDTH - 1) // CHUNK_WIDTH, "CW": CHUNK_WIDTH, "VALUE_CHUNKS": (dvp + ch - 1) // ch, "G": PREP_CHUNKS_PER_TASK, "NG": ((time + CHUNK_WIDTH - 1) // CHUNK_WIDTH + PREP_CHUNKS_PER_TASK - 1) // PREP_CHUNKS_PER_TASK,
                 "PL_DTYPE": PL_DTYPES[case["dtype"]][1], "SCALE": scale}
    source = _chunk_source(case)
    __import__("ast").parse(source)
    filename = "kda_lane2_r016_readiness_measured_pack_integration_cw%d_c%d_g%d_%s.py" % (CHUNK_WIDTH, PHYSICAL_WIDTH, PREP_CHUNKS_PER_TASK, hashlib.sha256(source.encode()).hexdigest())
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return (namespace["kda_recurrent"],) + parent_result[1:]

import re


def _simd_prepare(case):
    body = '''
head = group_task // NG
group_index = group_task % NG
batch_index = head // H
head_index = head % H
norm_ones = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)
norm_scratch = pl.create_tile([8, DKP], dtype=pl.FP32)
pair_scratch = pl.create_tile([8, DKP], dtype=pl.FP32)
qrows = pl.tile.full([C * G, DKP], dtype=pl.FP32, value=0.0)
krows = pl.tile.full([C * G, DKP], dtype=pl.FP32, value=0.0)
srows = pl.tile.full([C * G, DKP], dtype=pl.FP32, value=0.0)
vrows = pl.tile.full([C * G, DVP], dtype=pl.FP32, value=0.0)
brows = pl.tile.full([C, 8], dtype=pl.FP32, value=0.0)
for group_lane, (qrows, krows, srows, vrows) in pl.range(G, init_values=(qrows, krows, srows, vrows)):
    chunk = group_index * G + group_lane
    cumulative = pl.tile.full([1, DKP], dtype=pl.FP32, value=0.0)
    for token, (qrows, krows, srows, vrows, cumulative) in pl.range(CW, init_values=(qrows, krows, srows, vrows, cumulative)):
        token_index = chunk * CW + token
        row = (batch_index * T + pl.min(token_index, T - 1)) * H + head_index
        qt = __Q_LOAD__
        kt = __K_LOAD__
        vt = __V_LOAD__
        gt = __G_LOAD__
__NORMALIZATION__
        token_beta = pl.read(b1, [row, 0])
        if token_index >= T:
            qt = pl.mul(qt, 0.0)
            kt = pl.mul(kt, 0.0)
            vt = pl.mul(vt, 0.0)
            gt = pl.mul(gt, 0.0)
            token_beta = 0.0
        cumulative = pl.add(cumulative, gt)
        qrows = pl.assemble(qrows, pl.tile.slice(qt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token * G + group_lane, 0])
        krows = pl.assemble(krows, pl.tile.slice(kt, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token * G + group_lane, 0])
        srows = pl.assemble(srows, pl.tile.slice(cumulative, [1, DKP], [0, 0], pad_value=pl.PadValue.null), [token * G + group_lane, 0])
        vrows = pl.assemble(vrows, pl.tile.slice(vt, [1, DVP], [0, 0], pad_value=pl.PadValue.null), [token * G + group_lane, 0])
        pl.write(brows, [token, group_lane], token_beta)
        qdone, kdone, sdone, vdone, cumulative_done = pl.yield_(qrows, krows, srows, vrows, cumulative)
    qrows = qdone
    krows = kdone
    srows = sdone
    vrows = vdone
    q_all, k_all, s_all, v_all = pl.yield_(qrows, krows, srows, vrows)
qrows = q_all
krows = k_all
srows = s_all
vrows = v_all
last_log = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
last_log = pl.assemble(last_log, pl.tile.slice(srows, [G, DKP], [(CW - 1) * G, 0], pad_value=pl.PadValue.null), [0, 0])
last_decay = pl.exp(last_log)
panel_scratch = pl.create_tile([16, DKP], dtype=pl.FP32)
panel_target_offsets = pl.tile.full([16, DKP], dtype=pl.FP32, value=0.0)
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=0.0), [0 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=1.0), [1 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=2.0), [2 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=3.0), [3 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=4.0), [4 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=5.0), [5 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=6.0), [6 * G, 0])
panel_target_offsets = pl.assemble(panel_target_offsets, pl.tile.full([G, DKP], dtype=pl.FP32, value=7.0), [7 * G, 0])
beta_lane_pattern = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
target_row_offsets = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
pl.write(beta_lane_pattern, [0, 0], 1.0)
pl.write(target_row_offsets, [0, 0], 0.0)
pl.write(target_row_offsets, [0, 1], 0.0)
pl.write(beta_lane_pattern, [0, 2], 1.0)
pl.write(target_row_offsets, [0, 2], 1.0)
pl.write(target_row_offsets, [0, 3], 1.0)
pl.write(beta_lane_pattern, [0, 4], 1.0)
pl.write(target_row_offsets, [0, 4], 2.0)
pl.write(target_row_offsets, [0, 5], 2.0)
pl.write(beta_lane_pattern, [0, 6], 1.0)
pl.write(target_row_offsets, [0, 6], 3.0)
pl.write(target_row_offsets, [0, 7], 3.0)
pl.write(beta_lane_pattern, [0, 8], 1.0)
pl.write(target_row_offsets, [0, 8], 4.0)
pl.write(target_row_offsets, [0, 9], 4.0)
pl.write(beta_lane_pattern, [0, 10], 1.0)
pl.write(target_row_offsets, [0, 10], 5.0)
pl.write(target_row_offsets, [0, 11], 5.0)
pl.write(beta_lane_pattern, [0, 12], 1.0)
pl.write(target_row_offsets, [0, 12], 6.0)
pl.write(target_row_offsets, [0, 13], 6.0)
pl.write(beta_lane_pattern, [0, 14], 1.0)
pl.write(target_row_offsets, [0, 14], 7.0)
pl.write(target_row_offsets, [0, 15], 7.0)
beta_other_pattern = pl.sub(pl.tile.full([1, 16], dtype=pl.FP32, value=1.0), beta_lane_pattern)
low_source = pl.tile.full([C, C * G], dtype=pl.FP32, value=0.0)
query_source = pl.tile.full([C, C * G], dtype=pl.FP32, value=0.0)
for source_token, (low_source, query_source) in pl.range(CW, init_values=(low_source, query_source)):
    source_logs = pl.reshape(pl.tile.slice(srows, [G, DKP], [source_token * G, 0], pad_value=pl.PadValue.null), [1, G * DKP])
    source_keys = pl.reshape(pl.tile.slice(krows, [G, DKP], [source_token * G, 0], pad_value=pl.PadValue.null), [1, G * DKP])
    repeated_logs = pl.reshape(pl.col_expand(pl.create_tile([8, G * DKP], dtype=pl.FP32), source_logs), [16, DKP])
    repeated_keys = pl.reshape(pl.col_expand(pl.create_tile([8, G * DKP], dtype=pl.FP32), source_keys), [16, DKP])
    repeated_beta = pl.add(pl.mul(beta_lane_pattern, pl.read(brows, [source_token, 0])), pl.mul(beta_other_pattern, pl.read(brows, [source_token, 1])))
    for panel_start, (low_source, query_source) in pl.range(0, CW, 8, init_values=(low_source, query_source)):
        target_logs = pl.tile.slice(srows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)
        target_keys = pl.tile.slice(krows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)
        target_queries = pl.tile.slice(qrows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)
        target_indices = pl.add(panel_target_offsets, pl.cast(pl.cast(panel_start, pl.INT32), pl.FP32))
        active_mask = pl.tile.cmps(target_indices, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=5)
        raw_delta = pl.sub(target_logs, repeated_logs)
        safe_delta = pl.tile.select(active_mask, raw_delta, 0.0)
        causal_decay = pl.exp(safe_delta)
        decayed_key = pl.mul(repeated_keys, causal_decay)
        key_product = pl.mul(target_keys, decayed_key)
        query_product = pl.mul(target_queries, decayed_key)
        key_dot = pl.reshape(pl.row_sum(key_product, panel_scratch), [1, 16])
        query_dot = pl.reshape(pl.row_sum(query_product, panel_scratch), [1, 16])
        key_coeff = pl.mul(key_dot, repeated_beta)
        query_coeff = pl.mul(query_dot, repeated_beta)
        strict_row = pl.add(target_row_offsets, pl.cast(pl.cast(panel_start, pl.INT32), pl.FP32))
        key_coeff = pl.tile.select(pl.tile.cmps(strict_row, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=4), key_coeff, 0.0)
        query_coeff = pl.tile.select(pl.tile.cmps(strict_row, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=5), query_coeff, 0.0)
        low_source = pl.assemble(low_source, key_coeff, [source_token, panel_start * G])
        query_source = pl.assemble(query_source, query_coeff, [source_token, panel_start * G])
        low_panel_done, query_panel_done = pl.yield_(low_source, query_source)
    low_source = low_panel_done
    query_source = query_panel_done
    low_source_done, query_source_done = pl.yield_(low_source, query_source)
low_targets = pl.transpose(low_source_done, 0, 1)
output_coeff = pl.transpose(query_source_done, 0, 1)
low = pl.tile.full([C * C, 8], dtype=pl.FP32, value=0.0)
for target, (low,) in pl.range(CW, init_values=(low,)):
    low_target_rows = pl.tile.full([8, C], dtype=pl.FP32, value=0.0)
    low_target_rows = pl.assemble(low_target_rows, pl.tile.slice(low_targets, [G, C], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    low_target_padded = pl.transpose(low_target_rows, 0, 1)
    low = pl.assemble(low, low_target_padded, [target * C, 0])
    low_done = pl.yield_(low)
low = low_done
for target, (key_panel, value_panel_carried, query_panel, log_panel) in pl.range(CW, init_values=(krows, vrows, qrows, srows)):
    si = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
    ki = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
    qi = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
    value_seed = pl.tile.full([8, DVP], dtype=pl.FP32, value=0.0)
    si = pl.assemble(si, pl.tile.slice(log_panel, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    ki = pl.assemble(ki, pl.tile.slice(key_panel, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    qi = pl.assemble(qi, pl.tile.slice(query_panel, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    value_panel = pl.assemble(value_seed, pl.tile.slice(value_panel_carried, [G, DVP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    ur = pl.mul(value_panel, 1.0)
    wr = pl.mul(ki, pl.exp(si))
    for previous, (wr, ur) in pl.range(target, init_values=(wr, ur)):
        coefficient = pl.reshape(pl.tile.slice(low, [1, 8], [target * C + previous, 0], pad_value=pl.PadValue.null), [8, 1])
        previous_w = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
        previous_u = pl.tile.full([8, DVP], dtype=pl.FP32, value=0.0)
        previous_w = pl.assemble(previous_w, pl.tile.slice(key_panel, [G, DKP], [previous * G, 0], pad_value=pl.PadValue.null), [0, 0])
        previous_u = pl.assemble(previous_u, pl.tile.slice(value_panel_carried, [G, DVP], [previous * G, 0], pad_value=pl.PadValue.null), [0, 0])
        wr = pl.sub(wr, pl.row_expand_mul(previous_w, coefficient))
        ur = pl.sub(ur, pl.row_expand_mul(previous_u, coefficient))
        wr_done, ur_done = pl.yield_(wr, ur)
    key_panel = pl.assemble(key_panel, pl.tile.slice(wr_done, [G, DKP], [0, 0], pad_value=pl.PadValue.null), [target * G, 0])
    value_panel_carried = pl.assemble(value_panel_carried, pl.tile.slice(ur_done, [G, DVP], [0, 0], pad_value=pl.PadValue.null), [target * G, 0])
    query_panel = pl.assemble(query_panel, pl.tile.slice(pl.mul(qi, pl.exp(si)), [G, DKP], [0, 0], pad_value=pl.PadValue.null), [target * G, 0])
    token_beta = pl.reshape(pl.tile.slice(brows, [1, 8], [target, 0], pad_value=pl.PadValue.null), [8, 1])
    end_key = pl.row_expand_mul(pl.mul(ki, pl.exp(pl.sub(last_log, si))), token_beta)
    log_panel = pl.assemble(log_panel, pl.tile.slice(end_key, [G, DKP], [0, 0], pad_value=pl.PadValue.null), [target * G, 0])
    wdone, udone, qcoeff_done, bcoeff_done = pl.yield_(key_panel, value_panel_carried, query_panel, log_panel)
for group_lane in pl.range(G):
    chunk = group_index * G + group_lane
    if chunk < NC:
        chunk_task = head * NC + chunk
        projection_base = chunk_task * 2 * C
        action_base = chunk_task * (C + DKP)
        wc = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)
        qc = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)
        uc = pl.tile.full([C, DVP], dtype=pl.FP32, value=0.0)
        bc = pl.tile.full([C, DKP], dtype=pl.FP32, value=0.0)
        mc = pl.tile.full([C, C], dtype=pl.FP32, value=0.0)
        for token, (wc, qc, uc, bc, mc) in pl.range(CW, init_values=(wc, qc, uc, bc, mc)):
            wc = pl.assemble(wc, pl.tile.slice(wdone, [1, DKP], [token * G + group_lane, 0], pad_value=pl.PadValue.null), [token, 0])
            qc = pl.assemble(qc, pl.tile.slice(qcoeff_done, [1, DKP], [token * G + group_lane, 0], pad_value=pl.PadValue.null), [token, 0])
            uc = pl.assemble(uc, pl.tile.slice(udone, [1, DVP], [token * G + group_lane, 0], pad_value=pl.PadValue.null), [token, 0])
            bc = pl.assemble(bc, pl.tile.slice(bcoeff_done, [1, DKP], [token * G + group_lane, 0], pad_value=pl.PadValue.null), [token, 0])
            mc = pl.assemble(mc, pl.tile.slice(output_coeff, [1, C], [token * G + group_lane, 0], pad_value=pl.PadValue.null), [token, 0])
            wc_done, qc_done, uc_done, bc_done, mc_done = pl.yield_(wc, qc, uc, bc, mc)
        pl.store(wc_done, [projection_base, 0], prepared_p)
        pl.store(qc_done, [projection_base + C, 0], prepared_p)
        pl.store(uc_done, [chunk_task * C, 0], prepared_u)
        pl.store(mc_done, [action_base, 0], prepared_j)
        pl.store(pl.transpose(bc_done, 0, 1), [action_base + C, 0], prepared_j)
        dc = pl.tile.slice(last_decay, [1, DKP], [group_lane, 0], pad_value=pl.PadValue.null)
        pl.store(dc, [chunk_task, 0], prepared_d)
'''
    body = _block_input_body(body, case)
    return re.sub(r'\b(wc|qc|uc|bc|mc|dc)\b', lambda match: 'lane_' + match.group(0), body)


BLOCK_NORMALIZATION = False

def _panel_normalization(name):
    return "\n".join([
        "%s_square = pl.mul(%s, %s)" % (name, name, name),
        "%s_sum = pl.reshape(pl.row_sum(%s_square, panel_norm_scratch), [1, C])" % (name, name),
        "%s_denominator = pl.sqrt(pl.add(%s_sum, 1e-6))" % (name, name),
        "%s_inverse = pl.reshape(pl.div(panel_norm_ones, %s_denominator), [C, 1])" % (name, name),
        "%s = pl.row_expand_mul(%s, %s_inverse)" % (name, name, name),
    ])


def _block_input_body(body, case):
    prefix = """
    panel_valid_tokens = pl.min(CW, T - chunk * CW)
    panel_q = __PANEL_Q_LOAD__
    panel_k = __PANEL_K_LOAD__
    panel_v = __PANEL_V_LOAD__
    panel_g = __PANEL_G_LOAD__
    panel_beta = pl.add(pl.tile.full([C, 8], dtype=pl.FP32, value=0.0), pl.set_validshape(pl.fillpad(pl.load(beta_block, [batch_index * T + chunk * CW, head_index], [C, 8], valid_shape=[panel_valid_tokens, 1]), 0), C, 8))
__PANEL_NORMALIZATION__
"""
    for marker, tensor, width, logical in (("__PANEL_Q_LOAD__", "q_block", "DKP", "DK"), ("__PANEL_K_LOAD__", "k_block", "DKP", "DK"), ("__PANEL_V_LOAD__", "v_block", "DVP", "DV"), ("__PANEL_G_LOAD__", "g_block", "DKP", "DK")):
        expression = _padded_load(tensor, "[batch_index * T + chunk * CW, head_index * %s]" % logical, "C", width, "panel_valid_tokens", logical)
        if tensor != "g_block" and case["dtype"] != "fp32":
            expression = "pl.cast(%s, target_type=pl.FP32)" % expression
        expression = "pl.add(pl.tile.full([C, %s], dtype=pl.FP32, value=0.0), %s)" % (width, expression)
        prefix = prefix.replace(marker, expression)
    normalization = _panel_normalization("panel_q") + "\n" + _panel_normalization("panel_k")
    prefix = prefix.replace("__PANEL_NORMALIZATION__", _indented(normalization, 4) if BLOCK_NORMALIZATION and case.get("use_qk_l2norm", True) else "    pass")
    body = body.replace("pl.range(G, init_values=(qrows, krows, srows, vrows))", "pl.range(pl.min(G, NC - group_index * G), init_values=(qrows, krows, srows, vrows))", 1)
    body = body.replace("    cumulative = pl.tile.full([1, DKP], dtype=pl.FP32, value=0.0)", prefix + "    cumulative = pl.tile.full([1, DKP], dtype=pl.FP32, value=0.0)", 1)
    for marker, name, width in (("__Q_LOAD__", "panel_q", "DKP"), ("__K_LOAD__", "panel_k", "DKP"), ("__V_LOAD__", "panel_v", "DVP"), ("__G_LOAD__", "panel_g", "DKP")):
        body = body.replace(marker, "pl.tile.slice(%s, [1, %s], [token, 0], pad_value=pl.PadValue.null)" % (name, width))
    body = body.replace("token_beta = pl.read(b1, [row, 0])", "token_beta = pl.read(panel_beta, [token, 0])", 1)
    return "panel_norm_scratch = pl.create_tile([C, DKP], dtype=pl.FP32)\npanel_norm_ones = pl.tile.full([1, C], dtype=pl.FP32, value=1.0)\n" + body


_row_substitution_prepare = _simd_prepare

_COLUMN_UPDATE_SOLVE = '''output_coeff = pl.transpose(query_source_done, 0, 1)
initial_w = pl.mul(krows, pl.exp(srows))
packed_rhs = pl.tile.full([C * G, DKP + DVP], dtype=pl.FP32, value=0.0)
packed_rhs = pl.assemble(packed_rhs, initial_w, [0, 0])
packed_rhs = pl.assemble(packed_rhs, vrows, [0, DKP])
for source_token, (rhs_panel,) in pl.range(CW, init_values=(packed_rhs,)):
    source_rhs = pl.mul(pl.tile.slice(rhs_panel, [G, DKP + DVP], [source_token * G, 0], pad_value=pl.PadValue.null), 1.0)
    source_rhs_row = pl.reshape(source_rhs, [1, G * (DKP + DVP)])
    repeated_rhs = pl.reshape(pl.col_expand(pl.create_tile([C, G * (DKP + DVP)], dtype=pl.FP32), source_rhs_row), [C * G, DKP + DVP])
    coefficients = pl.reshape(pl.tile.slice(low_source_done, [1, C * G], [source_token, 0], pad_value=pl.PadValue.null), [C * G, 1])
    rhs_panel = pl.sub(rhs_panel, pl.row_expand_mul(repeated_rhs, coefficients))
    rhs_done = pl.yield_(rhs_panel)
wdone = pl.mul(pl.tile.slice(rhs_done, [C * G, DKP], [0, 0], pad_value=pl.PadValue.null), 1.0)
udone = pl.mul(pl.tile.slice(rhs_done, [C * G, DVP], [0, DKP], pad_value=pl.PadValue.null), 1.0)
qcoeff_done = pl.mul(qrows, pl.exp(srows))
bcoeff_seed = pl.tile.full([C * G, DKP], dtype=pl.FP32, value=0.0)
for target, (bcoeff_panel,) in pl.range(CW, init_values=(bcoeff_seed,)):
    si = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
    ki = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
    si = pl.assemble(si, pl.tile.slice(srows, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    ki = pl.assemble(ki, pl.tile.slice(krows, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])
    token_beta = pl.reshape(pl.tile.slice(brows, [1, 8], [target, 0], pad_value=pl.PadValue.null), [8, 1])
    end_key = pl.row_expand_mul(pl.mul(ki, pl.exp(pl.sub(last_log, si))), token_beta)
    bcoeff_panel = pl.assemble(bcoeff_panel, pl.tile.slice(end_key, [G, DKP], [0, 0], pad_value=pl.PadValue.null), [target * G, 0])
    bcoeff_done = pl.yield_(bcoeff_panel)
'''


def _simd_prepare(case):
    body = _row_substitution_prepare(case)
    start_marker = "low_targets = pl.transpose(low_source_done, 0, 1)\n"
    end_marker = "for group_lane in pl.range(G):\n"
    if body.count(start_marker) != 1 or body.count(end_marker) != 1:
        raise ValueError("Unexpected pinned preparation solve")
    start = body.index(start_marker)
    end = body.index(end_marker, start)
    body = body[:start] + _COLUMN_UPDATE_SOLVE + body[end:]
    if case["H"] == 1:
        before = "    panel_beta = pl.add(pl.tile.full([C, 8], dtype=pl.FP32, value=0.0), pl.set_validshape(pl.fillpad(pl.load(beta_block, [batch_index * T + chunk * CW, head_index], [C, 8], valid_shape=[panel_valid_tokens, 1]), 0), C, 8))"
        after = "\n".join([
            "    beta_row = pl.set_validshape(pl.fillpad(pl.load(beta_block, [0, batch_index * T + chunk * CW], [1, C], valid_shape=[1, panel_valid_tokens]), 0), 1, C)",
            "    beta_column = pl.transpose(beta_row, 0, 1)",
            "    panel_beta = pl.tile.full([C, 8], dtype=pl.FP32, value=0.0)",
            "    panel_beta = pl.assemble(panel_beta, beta_column, [0, 0])",
        ])
        if body.count(before) != 1:
            raise ValueError("Unexpected singleton-head beta load")
        body = body.replace(before, after, 1)
    return body


_ungrouped_chunk_source = _chunk_source


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected one grouped-staging rewrite site: " + before)
    return source.replace(before, after, 1)


def _isolate_task_locals(source, prefix):
    import ast
    import textwrap

    region = ast.parse(textwrap.dedent(source))
    statement = region.body[0]
    assigned = {
        node.id
        for child in statement.body
        for node in ast.walk(child)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    class RenameLocals(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in assigned:
                return ast.copy_location(ast.Name(id=prefix + node.id, ctx=node.ctx), node)
            return node

    for index, child in enumerate(statement.body):
        statement.body[index] = RenameLocals().visit(child)
    return textwrap.indent(ast.unparse(ast.fix_missing_locations(region)), "    ") + "\n"


def _chunk_source(case):
    source = _ungrouped_chunk_source(case)
    head_count = case["B"] * case["H"]
    if head_count == 1:
        return source
    buffer_start = source.index("    prepared_p = pl.create_tensor")
    prepare_start = source.index("    with pl.spmd(B * H * NG,")
    consumer_start = source.index("    for owner in pl.spmd(B * H * VALUE_CHUNKS,")
    return_start = source.rindex("    return y, final_state")
    header = source[:buffer_start]
    buffers = source[buffer_start:prepare_start]
    prepare = source[prepare_start:consumer_start]
    consumer = source[consumer_start:return_start]
    grouped_buffers = []
    grouped_preparations = []
    grouped_consumers = []
    first_heads = (head_count + 1) // 2
    dependency_groups = 4 if head_count >= 8 else head_count
    for group in range(dependency_groups):
        head_offset = (group * head_count + dependency_groups - 1) // dependency_groups
        group_end = ((group + 1) * head_count + dependency_groups - 1) // dependency_groups
        group_heads = group_end - head_offset
        local_buffers = buffers.replace("B * H * NC", "%d * NC" % group_heads)
        local_prepare = _replace_once(prepare, "pl.spmd(B * H * NG,", "pl.spmd(%d * NG," % group_heads)
        local_prepare = _replace_once(local_prepare, "head = group_task // NG", "head = group_task // NG + %d" % head_offset)
        local_prepare = _replace_once(local_prepare, "chunk_task = head * NC + chunk", "chunk_task = (head - %d) * NC + chunk" % head_offset)
        local_consumer = _replace_once(
            consumer,
            "    for owner in pl.spmd(B * H * VALUE_CHUNKS, name_hint='kda_four_product_owner', deps=[input_tid], optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=2)]):",
            "    with pl.spmd(%d * VALUE_CHUNKS, name_hint='kda_four_product_owner', deps=[input_tid], optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=2)]) as owner_tid:\n        owner = pl.tile.get_block_idx()" % group_heads,
        )
        local_consumer = _replace_once(local_consumer, "value_chunk = owner // (B * H)", "value_chunk = owner // %d" % group_heads)
        local_consumer = _replace_once(local_consumer, "head = owner % (B * H)", "head = owner %% %d + %d" % (group_heads, head_offset))
        local_consumer = _replace_once(local_consumer, "chunk_task = head * NC + chunk", "chunk_task = (head - %d) * NC + chunk" % head_offset)
        for name in ("prepared_p", "prepared_j", "prepared_u", "prepared_d", "input_tid", "owner_tid"):
            replacement = "%s_group%d" % (name, group)
            local_buffers = local_buffers.replace(name, replacement)
            local_prepare = local_prepare.replace(name, replacement)
            local_consumer = local_consumer.replace(name, replacement)
        local_prepare = local_prepare.replace("kda_independent_chunk_prepare", "kda_head_group%d_prepare" % group)
        local_consumer = local_consumer.replace("kda_four_product_owner", "kda_head_group%d_owner" % group)
        grouped_buffers.append(local_buffers)
        grouped_preparations.append(_isolate_task_locals(local_prepare, "group%d_prepare_" % group))
        grouped_consumers.append(_isolate_task_locals(local_consumer, "group%d_consumer_" % group))
    if head_count >= 2:
        grouped_tasks = [task for pair in zip(grouped_preparations, grouped_consumers) for task in pair]
    else:
        grouped_tasks = grouped_preparations + grouped_consumers
    return header + "".join(grouped_buffers + grouped_tasks) + "\n    return y, final_state\n"


_automatic_dependency_chunk_source = _chunk_source


def _chunk_source(case):
    source = _automatic_dependency_chunk_source(case)
    if case["B"] * case["H"] == 1:
        return source
    start = source.index("    with pl.spmd(")
    end = source.rindex("    return y, final_state")
    tasks = source[start:end]
    return source[:start] + "    with pl.manual_scope():\n" + "\n".join("    " + line if line else line for line in tasks.splitlines()) + "\n" + source[end:]


_column_update_chunk_source = _chunk_source


def _chunk_source(case):
    source = _column_update_chunk_source(case)
    if case["H"] == 1:
        before = "beta_block = pl.reshape(beta, [B * T, H])"
        if source.count(before) != 1:
            raise ValueError("Unexpected orchestration beta view")
        source = source.replace(before, "beta_block = pl.reshape(beta, [1, B * T * H])", 1)
    return source


_direct_coefficient_prepare = _simd_prepare


def _shared_replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Pinned shared-exponent rewrite site changed: " + before)
    return source.replace(before, after, 1)


def _simd_prepare(case):
    body = _direct_coefficient_prepare(case)
    factors = '''center_logs = pl.mul(pl.tile.slice(last_log, [G, DKP], [0, 0], pad_value=pl.PadValue.null), 0.5)
center_row = pl.reshape(center_logs, [1, G * DKP])
center_panel = pl.reshape(pl.col_expand(pl.create_tile([C, G * DKP], dtype=pl.FP32), center_row), [C * G, DKP])
centered_logs = pl.sub(srows, center_panel)
forward_factor = pl.exp(centered_logs)
key_reverse = pl.mul(krows, pl.exp(pl.neg(centered_logs)))
'''
    body = _shared_replace_once(body, "panel_scratch = pl.create_tile([16, DKP], dtype=pl.FP32)", factors + "panel_scratch = pl.create_tile([16, DKP], dtype=pl.FP32)")
    body = _shared_replace_once(body, "    source_logs = pl.reshape(pl.tile.slice(srows, [G, DKP], [source_token * G, 0], pad_value=pl.PadValue.null), [1, G * DKP])\n", "")
    body = _shared_replace_once(body, "source_keys = pl.reshape(pl.tile.slice(krows,", "source_keys = pl.reshape(pl.tile.slice(key_reverse,")
    body = _shared_replace_once(body, "    repeated_logs = pl.reshape(pl.col_expand(pl.create_tile([8, G * DKP], dtype=pl.FP32), source_logs), [16, DKP])\n", "")
    body = _shared_replace_once(body, "        target_logs = pl.tile.slice(srows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)\n", "")
    target_keys = "        target_keys = pl.tile.slice(krows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)"
    target_queries = "        target_queries = pl.tile.slice(qrows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)"
    forward_slice = "pl.tile.slice(forward_factor, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)"
    body = _shared_replace_once(body, target_keys, "        target_forward = " + forward_slice + "\n" + target_keys.replace("pl.tile.slice(krows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)", "pl.mul(pl.tile.slice(krows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null), target_forward)"))
    body = _shared_replace_once(body, target_queries, target_queries.replace("pl.tile.slice(qrows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)", "pl.mul(pl.tile.slice(qrows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null), target_forward)"))
    direct = '''        target_indices = pl.add(panel_target_offsets, pl.cast(pl.cast(panel_start, pl.INT32), pl.FP32))
        active_mask = pl.tile.cmps(target_indices, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=5)
        raw_delta = pl.sub(target_logs, repeated_logs)
        safe_delta = pl.tile.select(active_mask, raw_delta, 0.0)
        causal_decay = pl.exp(safe_delta)
        decayed_key = pl.mul(repeated_keys, causal_decay)
        key_product = pl.mul(target_keys, decayed_key)
        query_product = pl.mul(target_queries, decayed_key)'''
    shared = '''        key_product = pl.mul(target_keys, repeated_keys)
        query_product = pl.mul(target_queries, repeated_keys)'''
    return _shared_replace_once(body, direct, shared)


_source_major_shared_prepare = _simd_prepare


def _simd_prepare(case):
    body = _source_major_shared_prepare(case)
    if case['H'] == 1:
        body = _shared_replace_once(body,
            '    beta_column = pl.transpose(beta_row, 0, 1)\n'
            '    panel_beta = pl.tile.full([C, 8], dtype=pl.FP32, value=0.0)\n'
            '    panel_beta = pl.assemble(panel_beta, beta_column, [0, 0])',
            '    panel_beta = pl.tile.full([C, 8], dtype=pl.FP32, value=0.0)\n'
            '    for beta_token in pl.range(CW):\n'
            '        pl.write(panel_beta, [beta_token, 0], pl.read(beta_row, [0, beta_token]))')
    start = body.index('for source_token, (low_source, query_source) in pl.range(CW, init_values=(low_source, query_source)):')
    end = body.index('output_coeff = pl.transpose(query_source_done, 0, 1)', start)
    replacement = '''for panel_start, (low_source, query_source) in pl.range(0, CW, 8, init_values=(low_source, query_source)):
    target_forward = pl.tile.slice(forward_factor, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null)
    target_keys = pl.mul(pl.tile.slice(krows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null), target_forward)
    target_queries = pl.mul(pl.tile.slice(qrows, [16, DKP], [panel_start * G, 0], pad_value=pl.PadValue.null), target_forward)
    strict_row = pl.add(target_row_offsets, pl.cast(pl.cast(panel_start, pl.INT32), pl.FP32))
    for source_token, (low_source, query_source) in pl.range(CW, init_values=(low_source, query_source)):
        source_keys = pl.reshape(pl.tile.slice(key_reverse, [G, DKP], [source_token * G, 0], pad_value=pl.PadValue.null), [1, G * DKP])
        repeated_keys = pl.reshape(pl.col_expand(pl.create_tile([8, G * DKP], dtype=pl.FP32), source_keys), [16, DKP])
        repeated_beta = pl.add(pl.mul(beta_lane_pattern, pl.read(brows, [source_token, 0])), pl.mul(beta_other_pattern, pl.read(brows, [source_token, 1])))
        key_product = pl.mul(target_keys, repeated_keys)
        query_product = pl.mul(target_queries, repeated_keys)
        key_dot = pl.reshape(pl.row_sum(key_product, panel_scratch), [1, 16])
        query_dot = pl.reshape(pl.row_sum(query_product, panel_scratch), [1, 16])
        key_coeff = pl.mul(key_dot, repeated_beta)
        query_coeff = pl.mul(query_dot, repeated_beta)
        key_coeff = pl.tile.select(pl.tile.cmps(strict_row, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=4), key_coeff, 0.0)
        query_coeff = pl.tile.select(pl.tile.cmps(strict_row, pl.cast(pl.cast(source_token, pl.INT32), pl.FP32), cmp_type=5), query_coeff, 0.0)
        low_source = pl.assemble(low_source, key_coeff, [source_token, panel_start * G])
        query_source = pl.assemble(query_source, query_coeff, [source_token, panel_start * G])
        low_column_done, query_column_done = pl.yield_(low_source, query_source)
    low_source = low_column_done
    query_source = query_column_done
    low_source_done, query_source_done = pl.yield_(low_source, query_source)
'''
    return body[:start] + replacement + body[end:]


_full_support_prepare = _simd_prepare


def _simd_prepare(case):
    body = _full_support_prepare(case)
    body = _shared_replace_once(body, 'head_index = head % H', 'head_index = head % H\ngroup_active_tokens = pl.min(CW, T - group_index * G * CW)')
    body = body.replace('pl.range(0, CW, 8, init_values=(low_source, query_source))',
                        'pl.range(0, group_active_tokens, 8, init_values=(low_source, query_source))')
    body = body.replace('pl.range(CW, init_values=(low_source, query_source))',
                        'pl.range(group_active_tokens, init_values=(low_source, query_source))')
    body = body.replace('pl.range(CW, init_values=(packed_rhs,))',
                        'pl.range(group_active_tokens, init_values=(packed_rhs,))')
    body = body.replace('pl.range(CW, init_values=(bcoeff_seed,))',
                        'pl.range(group_active_tokens, init_values=(bcoeff_seed,))')
    body = _shared_replace_once(body, 'pl.range(CW, init_values=(lane_wc, lane_qc, lane_uc, lane_bc, lane_mc))',
                                'pl.range(pl.min(CW, T - chunk * CW), init_values=(lane_wc, lane_qc, lane_uc, lane_bc, lane_mc))')
    return body


ENDKEY_FACTOR_REUSE = 1
SEED_FACTOR_REUSE = 0
_active_support_decay_prepare = _simd_prepare


def _simd_prepare(case):
    body = _active_support_decay_prepare(case)
    scale = """lane_end_logs = pl.sub(pl.tile.slice(last_log, [G, DKP], [0, 0], pad_value=pl.PadValue.null), center_logs)
lane_decay_scale = pl.exp(lane_end_logs)
end_factor = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)
end_factor = pl.assemble(end_factor, lane_decay_scale, [0, 0])
"""
    if SEED_FACTOR_REUSE:
        scale += """lane_scale_row = pl.reshape(lane_decay_scale, [1, G * DKP])
seed_scale_panel = pl.reshape(pl.col_expand(pl.create_tile([C, G * DKP], dtype=pl.FP32), lane_scale_row), [C * G, DKP])
"""
    body = _shared_replace_once(body, 'output_coeff = pl.transpose(query_source_done, 0, 1)\n', scale + 'output_coeff = pl.transpose(query_source_done, 0, 1)\n')
    body = _shared_replace_once(body,
        '    si = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)\n'
        '    ki = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)\n'
        '    si = pl.assemble(si, pl.tile.slice(srows, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])\n'
        '    ki = pl.assemble(ki, pl.tile.slice(krows, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])',
        '    ki = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)\n'
        '    ki = pl.assemble(ki, pl.tile.slice(key_reverse, [G, DKP], [target * G, 0], pad_value=pl.PadValue.null), [0, 0])')
    body = _shared_replace_once(body,
        'end_key = pl.row_expand_mul(pl.mul(ki, pl.exp(pl.sub(last_log, si))), token_beta)',
        'end_key = pl.row_expand_mul(pl.mul(ki, end_factor), token_beta)')
    if SEED_FACTOR_REUSE:
        body = _shared_replace_once(body, 'initial_w = pl.mul(krows, pl.exp(srows))',
            'initial_w = pl.mul(pl.mul(krows, forward_factor), seed_scale_panel)')
        body = _shared_replace_once(body, 'qcoeff_done = pl.mul(qrows, pl.exp(srows))',
            'qcoeff_done = pl.mul(pl.mul(qrows, forward_factor), seed_scale_panel)')
    return body


OMIT_ZERO_LAST_SOLVE_COLUMN = 1
_endkey_full_column_prepare = _simd_prepare


def _simd_prepare(case):
    body = _endkey_full_column_prepare(case)
    return _shared_replace_once(
        body,
        "for source_token, (rhs_panel,) in pl.range(group_active_tokens, init_values=(packed_rhs,)):",
        "rhs_done = packed_rhs\n"
        "for source_token, (rhs_panel,) in pl.range(group_active_tokens - 1, init_values=(rhs_done,)):",
    )


MIN_VALUE_TILE = 32
CONSUMER_OWNER_CAP = 32


_readiness_unrepaired_prepare = _simd_prepare


def _simd_prepare(case):
    body = _readiness_unrepaired_prepare(case)
    if ((case["Dv"] + 15) // 16) * 16 > ((case["Dk"] + 15) // 16) * 16:
        before = "wdone = pl.mul(pl.tile.slice(rhs_done, [C * G, DKP], [0, 0], pad_value=pl.PadValue.null), 1.0)\nudone = pl.mul(pl.tile.slice(rhs_done, [C * G, DVP], [0, DKP], pad_value=pl.PadValue.null), 1.0)"
        first, second = before.split("\n")
        body = _shared_replace_once(body, before, second + "\n" + first)
    return body


_unpacked_preparation_chunk_source = _chunk_source
PREP_WORK_UNITS_PER_BLOCK = 2
MIN_PREP_WORK_UNITS = 16
MIN_PACKED_CHUNKS = 16


def _chunk_source(case):
    import ast

    source = _unpacked_preparation_chunk_source(case)
    chunk_count = (case["T"] + CHUNK_WIDTH - 1) // CHUNK_WIDTH
    if chunk_count < MIN_PACKED_CHUNKS:
        return source
    group_count = (chunk_count + PREP_CHUNKS_PER_TASK - 1) // PREP_CHUNKS_PER_TASK
    region = ast.parse(source)
    changed = False
    for statement in ast.walk(region):
        if not isinstance(statement, ast.With):
            continue
        context = statement.items[0].context_expr
        if not isinstance(context, ast.Call):
            continue
        hint = next((keyword.value.value for keyword in context.keywords
                     if keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)), "")
        if not hint.endswith("_prepare"):
            continue
        physical_count = eval(compile(ast.Expression(context.args[0]), "<prep-count>", "eval"),
                              {"B": case["B"], "H": case["H"], "NG": group_count})
        if physical_count < MIN_PREP_WORK_UNITS:
            continue
        first = statement.body[0]
        if not isinstance(first, ast.Assign) or not ast.unparse(first.value).endswith("get_block_idx()"):
            raise ValueError("Pinned preparation index changed")
        logical_name = first.targets[0].id
        block_name = logical_name + "_packed_block"
        local_name = logical_name + "_packed_local"
        block_count = (physical_count + PREP_WORK_UNITS_PER_BLOCK - 1) // PREP_WORK_UNITS_PER_BLOCK
        context.args[0] = ast.Constant(block_count)
        first.targets[0].id = block_name
        loop = ast.parse(
            "for " + local_name + " in pl.range(pl.min(2, " + str(physical_count) + " - " + block_name + " * 2)):\n"
            "    " + logical_name + " = " + block_name + " * 2 + " + local_name + "\n"
        ).body[0]
        loop.body.extend(statement.body[1:])
        statement.body = [first, loop]
        changed = True
    if not changed:
        return source
    return ast.unparse(ast.fix_missing_locations(region)) + "\n"
