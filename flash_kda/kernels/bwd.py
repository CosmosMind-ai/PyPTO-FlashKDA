"""KDA backward with value-parallel resident adjoints and token-parallel epilogues."""
from golden import TensorSpec

import math

import torch

import pypto.language as pl

PL_DTYPES = {"fp32": (torch.float32, pl.FP32), "fp16": (torch.float16, pl.FP16),
             "bf16": (torch.bfloat16, pl.BF16)}

# tolerances are per case dtype (returned by build): fp32 tight, bf16 wide -- the single
# pair covers every output (y/dq/dk/dv in the case dtype; dg/dbeta/dinitial_state/hs/dh
# fp32). Gradients through a length-T recurrence in fp32 sit far inside these bars.
TOLS = {"fp32": (3e-4, 3e-5), "fp16": (5e-3, 5e-4), "bf16": (3e-2, 3e-3)}

# The kernel body is rendered per case and exec'd (the @pl.jit parser rejects same-dtype
# casts and parses BOTH branches of `if`): {CO}/{CC} = input upcast, {YO}/{YC} = output
# downcast (both empty for fp32), {NRMQ}/{NRMK} = l2norm blocks (or "pass"),
# {CHUNKS} = the DV-chunk blocks (unrolled at build time, NC = DVP/CH in {1, 2}),
# {DHINIT}/{DHFIN} = dh GM copy in/out, {EPIQ}/{EPIK}/{EPIG} = the dq/dk/dg epilogues.
# The v-reductions leave row_sum as col-major [DKP,1] tiles and there is NO tile
# transpose on this stack (ttrans needs a row-major source; a same-scope GM scratch
# store->load transpose is an unfenced RAW hazard -- silently wrong values on device),
# so the l2norm adjoint is applied in COLUMN form: <qn,dqn> = col_sum(qcol * dqn_c) and
# <kn,dkn> = col_sum(kcol * dknv), each a single-read [1,1] compute-result dot taken AT
# THE POINT OF USE (long-lived multi-read dot scratch was observed torn on device).
# Each output row is then written by a pl.unroll(DK) scalar tile-write into a [1,DKP]
# row tile + one clipped store (the op-causal-conv1d pl.tile.write idiom).
_P1 = """
    for b in pl.parallel(0, B, 1):
        for h in pl.parallel(0, H, 1):
            for cb in pl.parallel(0, DVP, CH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_replay"):
                    bh = b * H + h
                    state = {LDS}
                    w8 = pl.create_tile([1, 8], dtype=pl.FP32)
                    {ST0}
                    for t in pl.range(T):
                        row = (b * T + t) * H + h
                        qt = {CO}{LDQ}{CC}
                        kt = {CO}{LDK}{CC}
                        {NRMQ1}
                        {NRMK1}
                        gt = pl.reshape({LDG}, [DKP, 1])
                        decay = pl.exp(gt)
                        state = pl.row_expand_mul(state, decay)
                        kcol = pl.reshape(kn1, [DKP, 1])
                        kh = pl.col_sum(pl.row_expand_mul(state, kcol))
                        r = pl.sub({CO}{LDV}{CC}, kh)
                        bt = pl.read(b1, [row, 0])
                        kb = pl.mul(kcol, bt)
                        upd = pl.row_expand_mul(pl.col_expand(state, r), kb)
                        state = pl.add(state, upd)
                        qcol = pl.reshape(qn1, [DKP, 1])
                        yt = pl.mul(pl.col_sum(pl.row_expand_mul(state, qcol)), SCALE)
                        yv = {YO}yt{YC}
                        {YV}
                        pl.store(yv, [row, cb], y2)
                        {STT}
                    {FS}
                    pl.store(state, [bh * DK, cb], f2)
"""

# per-DV-chunk adjoint block, rendered for c in range(NC); placeholders: {c} suffix,
# {cc} column offset, plus the shared load/cast fragments
_P2CH = """
                    H1c{c} = {LDH1}
                    dyc{c} = {CO}{LDDY}{CC}
                    pdq{c} = pl.row_sum(pl.mul(H1c{c}, pl.col_expand(H1c{c}, dyc{c})), tmpA)
                    dhgc{c} = dhcarry{c}
                    upd{c} = pl.row_expand_mul(pl.col_expand(dhgc{c}, dyc{c}), qcol)
                    dhgc{c} = pl.add(dhgc{c}, pl.mul(upd{c}, SCALE))
                    H0c{c} = {LDH0}
                    decayed_key{c} = pl.mul(kcol, egcol)
                    vc{c} = {CO}{LDVC}{CC}
                    kh{c} = pl.col_sum(pl.row_expand_mul(H0c{c}, decayed_key{c}))
                    rc{c} = pl.sub(vc{c}, kh{c})
                    ccv{c} = pl.col_sum(pl.row_expand_mul(dhgc{c}, kcol))
                    drc{c} = pl.mul(ccv{c}, bt)
                    dvo{c} = {YO}drc{c}{YC}
                    {DVV}
                    pl.store(dvo{c}, [row, {CCOFF}], dv2)
                    {DOTB}
                    dk1{c} = pl.row_sum(pl.mul(dhgc{c}, pl.col_expand(dhgc{c}, rc{c})), tmpA)
                    dk2{c} = pl.mul(pl.row_sum(pl.mul(H0c{c}, pl.col_expand(H0c{c}, drc{c})), tmpA), egcol)
                    dhp{c} = pl.sub(dhgc{c},
                                    pl.row_expand_mul(pl.col_expand(dhgc{c}, drc{c}), kcol))
                    dgp{c} = pl.mul(pl.row_sum(pl.mul(dhp{c}, H0c{c}), tmpA), egcol)
                    dhn{c} = pl.row_expand_mul(dhp{c}, egcol)
                    dhcarry{c} = dhn{c}
"""

_P2 = """
    for b in pl.parallel(0, B, 1):
        for h in pl.parallel(0, H, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="bwd_adjoint"):
                bh = b * H + h
                w8b = pl.create_tile([1, 8], dtype=pl.FP32)
                tmpA = pl.create_tile([DKP, CH], dtype=pl.FP32)
                {DHINIT}
                for tt in pl.range(T):
                    t = T - 1 - tt
                    row = (b * T + t) * H + h
                    qt = {CO}{LDQ}{CC}
                    kt = {CO}{LDK}{CC}
                    {NRMQ2}
                    {NRMK2}
                    gt = pl.reshape({LDG}, [DKP, 1])
                    egcol = pl.exp(gt)
                    qcol = pl.reshape(qn2, [DKP, 1])
                    kcol = pl.reshape(kn2, [DKP, 1])
                    bt = pl.read(b1, [row, 0])
{CHUNKS}
                    {ACCW}
                    {EPIQ}
                    {EPIK}
                    {EPIG}
                    pl.write(dbeta_scratch, [row, 0], dbeta_acc)
                {DHFIN}
"""

_KERNEL = """
@pl.jit
def kda_backward_decomposed(
    q: pl.Tensor[[B, T, H, DK], PL_DTYPE],
    k: pl.Tensor[[B, T, H, DK], PL_DTYPE],
    v: pl.Tensor[[B, T, H, DV], PL_DTYPE],
    g: pl.Tensor[[B, T, H, DK], pl.FP32],
    beta: pl.Tensor[[B, T, H], pl.FP32],
    initial_state: pl.Tensor[[B, H, DK, DV], pl.FP32],
    dy: pl.Tensor[[B, T, H, DV], PL_DTYPE],
    dfinal_state: pl.Tensor[[B, H, DK, DV], pl.FP32],
    hs: pl.InOut[pl.Tensor[[B, TP1, H, DK, DV], pl.FP32]],
    dh: pl.InOut[pl.Tensor[[B, H, DK, DV], pl.FP32]],
    y: pl.Out[pl.Tensor[[B, T, H, DV], PL_DTYPE]],
    final_state: pl.Out[pl.Tensor[[B, H, DK, DV], pl.FP32]],
    dq: pl.Out[pl.Tensor[[B, T, H, DK], PL_DTYPE]],
    dk: pl.Out[pl.Tensor[[B, T, H, DK], PL_DTYPE]],
    dv: pl.Out[pl.Tensor[[B, T, H, DV], PL_DTYPE]],
    dg: pl.Out[pl.Tensor[[B, T, H, DK], pl.FP32]],
    dbeta: pl.Out[pl.Tensor[[B, T, H], pl.FP32]],
    dinitial_state: pl.Out[pl.Tensor[[B, H, DK, DV], pl.FP32]],
):
    q2 = pl.reshape(q, [B * T * H, DK])
    k2 = pl.reshape(k, [B * T * H, DK])
    v2 = pl.reshape(v, [B * T * H, DV])
    g2 = pl.reshape(g, [B * T * H, DK])
    b1 = pl.reshape(beta, [B * T * H, 1])
    s2 = pl.reshape(initial_state, [B * H * DK, DV])
    dy2 = pl.reshape(dy, [B * T * H, DV])
    df2 = pl.reshape(dfinal_state, [B * H * DK, DV])
    hs2 = pl.reshape(hs, [B * TP1 * H * DKP, DVP])
    dh2 = pl.reshape(dh, [B * H * DK, DV])
    y2 = pl.reshape(y, [B * T * H, DV])
    f2 = pl.reshape(final_state, [B * H * DK, DV])
    dq2 = pl.reshape(dq, [B * T * H, DK])
    dk2 = pl.reshape(dk, [B * T * H, DK])
    dv2 = pl.reshape(dv, [B * T * H, DV])
    dg2 = pl.reshape(dg, [B * T * H, DK])
    dbe_flat = pl.reshape(dbeta, [1, B * T * H])
    dbeta_scratch = pl.create_tensor([B * T * H, 16], dtype=pl.FP32)
    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])
    # pass 1 (recompute forward): replay the contract-4.1 recurrence per (b, h,
    # dv-chunk), materialising H_0..H_T into hs and emitting y / final_state
{P1}
    # pass 2 (adjoint sweep): dH lives in the GM workspace dh (seeded from
    # dfinal_state); per token (reverse order): add the y_t contribution
    # scale*qn dy^T, then invert the rank-1 update / residual / decay chain.
    # dq/dk/dg leave the DV-chunk reductions as col-major [DKP,1] tiles; the
    # l2norm adjoint is applied in column form (the two norm dots use v-space
    # identities, no transpose) and each output row goes out via a
    # pl.unroll(DK) scalar tile-write into a [1,DKP] row tile + clipped store
{P2}
    for row_base in pl.parallel(0, B * T * H, 16):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbeta_compact"):
            dbeta_row = pl.create_tile([1, 16], dtype=pl.FP32)
            for lane in pl.unroll(16):
                if row_base + lane < B * T * H:
                    pl.tile.write(dbeta_row, [0, lane], pl.read(dbeta_scratch, [row_base + lane, 0]))
            valid_count = pl.min(16, B * T * H - row_base)
            dbeta_row = pl.set_validshape(dbeta_row, 1, valid_count)
            pl.store(dbeta_row, [0, row_base], dbe_flat)
    return y, final_state, dq, dk, dv, dg, dbeta, dinitial_state
"""


def _norm8(src, dst, invn, w8, ind):
    """l2norm of the [1, DKP] row `src` into `dst`, keeping 1/n in scalar `invn`
    (forward domain idiom: group-sum to [1,8], scalar combine, sqrt on a scratch
    tile -- [1,1] tiles are unallocatable on this stack). `ind` = continuation indent."""
    reads = "pl.read(ps, [0, 0])" + "".join(" + pl.read(ps, [0, %d])" % j for j in range(1, 8))
    lines = [
        "sq = pl.mul({0}, {0})",
        "ps = pl.col_sum(pl.reshape(sq, [DKP // 8, 8]))",
        "pl.write({3}, [0, 0], {1} + 1e-6)",
        "{2} = 1.0 / pl.read(pl.sqrt({3}), [0, 0])",
        "{4} = pl.mul({0}, {2})",
    ]
    return ("\n" + ind).join(l.format(src, reads, invn, w8, dst) for l in lines)


def _dot_row(a, b, out, ind):
    """total sum <a, b> of two [1, W] rows (W % 8 == 0): group to [W//8, 8], col_sum,
    8 scalar reads (the forward domain's proven reduction shape)."""
    first = "psd = pl.col_sum(pl.reshape(pl.mul(%s, %s), [%s // 8, 8]))" % (a, b, "CH")
    second = "%s = " % out + " + ".join("pl.read(psd, [0, %d])" % j for j in range(8))
    return ("\n" + ind).join([first, second])


def _dot_col(a, b, out):
    """total sum <a, b> of two [DKP, 1] columns, computed AT THE POINT OF USE in the
    epilogue: fresh product -> reshape [DKP,1] -> [DKP//8, 8] (order-preserving, 32B
    rows) -> col_sum -> 8 scalar reads. Point-of-use placement matters: with several
    long-lived multi-read dot scratches in flight, one was observed torn on device
    (per-head dk scalar off by a constant while sibling dots feeding dbeta/dq were
    fine); [1,1] compute-result reductions are REJECTED by codegen here, so the
    [1,8]-partials form stays."""
    first = "psd = pl.col_sum(pl.reshape(pl.mul(%s, %s), [DKP // 8, 8]))" % (a, b)
    second = "%s = " % out + " + ".join("pl.read(psd, [0, %d])" % j for j in range(8))
    return first + "\n" + " " * 20 + second


def build(shape):
    B, T, H = shape["B"], shape["T"], shape["H"]
    DK, DV = shape["Dk"], shape["Dv"]
    L2NORM = bool(shape.get("use_qk_l2norm", True))
    SCALE = float(shape.get("scale", -1.0))
    if SCALE <= 0:
        SCALE = 1.0 / math.sqrt(DK)
    TORCH_DTYPE, PL_DTYPE = PL_DTYPES[shape["dtype"]]
    RTOL, ATOL = TOLS[shape["dtype"]]
    need_cast = PL_DTYPE != pl.FP32
    align = 8 if PL_DTYPE == pl.FP32 else 16  # 32-byte tile rows
    DKP = -(-DK // align) * align
    DVP = -(-DV // align) * align
    TP1 = T + 1
    # DV chunking: full-width when the padded width fits one 64-col tile, else 64-col
    # chunks (all 128-wide cases divide exactly). NC in {1, 2} for every contract case.
    wide_geometry = DKP >= 128 and T > 1 and 1
    requested_width = 2 ** (5 if wide_geometry else 5)
    CH = min(requested_width, DVP)
    while (requested_width > 32 and 0 > 0
           and B * H * ((DVP + CH - 1) // CH) < 0):
        requested_width //= 2
        CH = min(requested_width, DVP)
    FCH = math.gcd(2 ** 5, DVP)
    CV = min(CH, DV)
    NC = (DVP + CH - 1) // CH

    def cast_pair(target):
        return ("pl.cast(", ", target_type=pl.%s)" % target) if need_cast else ("", "")

    down = {pl.FP32: "FP32", pl.FP16: "FP16", pl.BF16: "BF16"}[PL_DTYPE]
    co, cc = cast_pair("FP32")
    yo, yc = cast_pair(down)

    def load_expr(tensor, offsets, shp, valid):
        """GM load, padding rows/width when the case is unaligned (fillpad zeroes the
        pad; set_validshape re-marks the tile so no pad tag leaks downstream)."""
        rows, cols = shp.strip("[]").split(", ")
        base = "pl.load(%s, %s, %s" % (tensor, offsets, shp)
        if valid is None:
            return base + ")"
        return ("pl.set_validshape(pl.fillpad(" + base + ", valid_shape=%s), 0), %s, %s)"
                % (valid, rows, cols))

    dk_pad = DKP != DK
    cv_pad = CV != CH
    st_pad = dk_pad or cv_pad

    def pad_valid(rows_valid):
        v = []
        if rows_valid:
            v.append(str(DK))
        if cv_pad:
            v.append(str(CV))
        return ("[%s]" % ", ".join(v)) if v else None

    # NOTE on valid_shape: it clips rows then cols; render [DK, CV] / [1, DK] / [1, CV]
    v_rc = ("[%s, %s]" % (DK, CV)) if st_pad else None
    v_dk = ("[1, %s]" % DK) if dk_pad else None
    v_cv = ("[1, %s]" % CV) if cv_pad else None

    def st_store(tile):
        return ("%s = pl.set_validshape(%s, %s, %s)" % (tile, tile, DK, CV)) if st_pad else "pass"

    frag = {
        "LDS": load_expr("s2", "[bh * DK, cb]", "[DKP, FCH]", "[DK, pl.min(FCH, DV - cb)]"),
        "LDQ": load_expr("q2", "[row, 0]", "[1, DKP]", v_dk),
        "LDK": load_expr("k2", "[row, 0]", "[1, DKP]", v_dk),
        "LDV": load_expr("v2", "[row, cb]", "[1, FCH]", "[1, pl.min(FCH, DV - cb)]"),
        "LDG": load_expr("g2", "[row, 0]", "[1, DKP]", v_dk),
    }

    # ---- pass 1 (recompute forward, mirror of the op-kda reference) ----
    p1 = _P1.replace("CH", "FCH")
    p1 = p1.replace("{ST0}", "pl.store(pl.set_validshape(state, DKP, pl.min(FCH, DVP - cb)), [((b * TP1) * H + h) * DKP, cb], hs2)")
    p1 = p1.replace("{STT}", "pl.store(pl.set_validshape(state, DKP, pl.min(FCH, DVP - cb)), [((b * TP1 + t + 1) * H + h) * DKP, cb], hs2)")
    p1 = p1.replace("{FS}", "state = pl.set_validshape(state, DK, pl.min(FCH, DV - cb))")
    p1 = p1.replace("{YV}", "yv = pl.set_validshape(yv, 1, pl.min(FCH, DV - cb))")
    p1 = p1.replace("{NRMQ1}", _norm8("qt", "qn1", "qinv1", "w8", " " * 24) if L2NORM else
                    "qn1 = qt")
    p1 = p1.replace("{NRMK1}", _norm8("kt", "kn1", "kinv1", "w8", " " * 24) if L2NORM else
                    "kn1 = kt")

    # ---- pass 2 (adjoint sweep) ----
    p2chunks = ""
    for c in range(NC):
        ccofs = c * CH
        CV = min(CH, DV - ccofs)
        cv_pad = CV != CH
        st_pad = dk_pad or cv_pad
        v_rc = "[%s, %s]" % (DK, CV) if st_pad else None
        v_cv = "[1, %s]" % CV if cv_pad else None
        hs_valid = "[DKP, %s]" % min(CH, DVP - ccofs) if ccofs + CH > DVP else None
        blk = _P2CH
        blk = blk.replace("{LDH1}", load_expr(
            "hs2", "[((b * TP1 + t + 1) * H + h) * DKP, %d]" % ccofs, "[DKP, CH]", hs_valid))
        blk = blk.replace("{LDH0}", load_expr(
            "hs2", "[((b * TP1 + t) * H + h) * DKP, %d]" % ccofs, "[DKP, CH]", hs_valid))
        blk = blk.replace("{LDDH}", load_expr(
            "dh2", "[(b * H + h) * DK, %d]" % ccofs, "[DKP, CH]", v_rc))
        blk = blk.replace("{LDDY}", load_expr(
            "dy2", "[row, %d]" % ccofs, "[1, CH]", v_cv))
        blk = blk.replace("{LDVC}", load_expr(
            "v2", "[row, %d]" % ccofs, "[1, CH]", v_cv))
        blk = blk.replace("{DVV}", "dvo{c} = pl.set_validshape(dvo{c}, 1, %s)" % CV
                          if cv_pad else "pass")
        blk = blk.replace("{DHS}", "pl.store(%s, [(b * H + h) * DK, %d], dh2)"
                          % (("pl.set_validshape(dhn{c}, %s, %s)" % (DK, CV))
                             if st_pad else "dhn{c}", ccofs))
        blk = blk.replace("{DOTB}", _dot_row("ccv{c}", "rc{c}", "dbe{c}", " " * 20))
        blk = blk.replace("{CCOFF}", str(ccofs))
        blk = blk.replace("{c}", str(c))
        p2chunks += blk
    # accumulator wiring across the (<= 2) chunk blocks
    acc = "dqv = pdq0\n                    dbeta_acc = dbe0\n"
    acc += "                    dknv = pl.sub(pl.mul(dk10, bt), dk20)\n"
    acc += "                    dgv = dgp0"
    for c in range(1, NC):
        acc += ("\n                    dqv = pl.add(dqv, pdq{c})"
                "\n                    dbeta_acc = dbeta_acc + dbe{c}"
                "\n                    dknv = pl.add(dknv, pl.sub(pl.mul(dk1{c}, bt), dk2{c}))"
                "\n                    dgv = pl.add(dgv, dgp{c})").replace("{c}", str(c))

    # scalar tail: col-major [DKP,1] result -> [1,DKP] row tile via pl.tile.write
    # (op-causal-conv1d idiom) -> one clipped row store; no GM scratch, no transpose
    tail = ("for i in pl.unroll(DK):\n                        "
            "pl.tile.write(%s, [0, i], pl.read(%s, [i, 0]))")

    epiq = "dqn_c = pl.mul(dqv, SCALE)\n                    "
    if L2NORM:
        # <qn, dqn> straight from the accumulated column, at the point of use
        epiq += _dot_col("qcol", "dqn_c", "dqa") + "\n                    "
        epiq += "dqo_c = pl.mul(pl.sub(dqn_c, pl.mul(qcol, dqa)), qinv2)\n                    "
    else:
        epiq += "dqo_c = dqn_c\n                    "
    epiq += "dqf = {YO}dqo_c{YC}\n                    "
    epiq += "dqr = pl.create_tile([1, DKP], dtype=pl.%s)\n                    " % down
    epiq += tail % ("dqr", "dqf") + "\n                    "
    if dk_pad:
        epiq += "dqr = pl.set_validshape(dqr, 1, %s)\n                    " % DK
    epiq += "pl.store(dqr, [row, 0], dq2)"
    epik = ""
    if L2NORM:
        epik += _dot_col("kcol", "dknv", "dkd") + "\n                    "
        epik += "dko_c = pl.mul(pl.sub(dknv, pl.mul(kcol, dkd)), kinv2)\n                    "
    else:
        epik += "dko_c = dknv\n                    "
    epik += "dkf = {YO}dko_c{YC}\n                    "
    epik += "dkr = pl.create_tile([1, DKP], dtype=pl.%s)\n                    " % down
    epik += tail % ("dkr", "dkf") + "\n                    "
    if dk_pad:
        epik += "dkr = pl.set_validshape(dkr, 1, %s)\n                    " % DK
    epik += "pl.store(dkr, [row, 0], dk2)"
    epig = "dgr = pl.create_tile([1, DKP], dtype=pl.FP32)\n                    "
    epig += tail % ("dgr", "dgv") + "\n                    "
    if dk_pad:
        epig += "dgr = pl.set_validshape(dgr, 1, %s)\n                    " % DK
    epig += "pl.store(dgr, [row, 0], dg2)"

    dhinit_lines = []
    dhfin_lines = []
    for chunk_index in range(NC):
        chunk_offset = chunk_index * CH
        CV = min(CH, DV - chunk_offset)
        st_pad = dk_pad or CV != CH
        v_rc = "[%s, %s]" % (DK, CV) if st_pad else None
        adjoint_load = load_expr("df2", "[(b * H + h) * DK, %d]" % chunk_offset,
                                 "[DKP, CH]", v_rc)
        dhinit_lines.append("dhcarry%d = %s" % (chunk_index, adjoint_load))
        adjoint_final = "dhcarry%d" % chunk_index
        if st_pad:
            adjoint_final = "pl.set_validshape(%s, %s, %s)" % (adjoint_final, DK, CV)
        for destination in ("dh2", "di2"):
            dhfin_lines.append("pl.store(%s, [(b * H + h) * DK, %d], %s)" %
                               (adjoint_final, chunk_offset, destination))
    dhinit = "\n                ".join(dhinit_lines)
    dhfin = "\n                ".join(dhfin_lines)

    p2 = _P2
    p2 = p2.replace("{NRMQ2}", _norm8("qt", "qn2", "qinv2", "w8b", " " * 20) if L2NORM else "qn2 = qt")
    p2 = p2.replace("{NRMK2}", _norm8("kt", "kn2", "kinv2", "w8b", " " * 20) if L2NORM else "kn2 = kt")
    p2 = p2.replace("{CHUNKS}", p2chunks.strip("\n"))
    p2 = p2.replace("{ACCW}", acc)
    p2 = p2.replace("{EPIQ}", epiq)
    p2 = p2.replace("{EPIK}", epik)
    p2 = p2.replace("{EPIG}", epig)
    p2 = p2.replace("{DHINIT}", dhinit)
    p2 = p2.replace("{DHFIN}", dhfin)

    p2, allocations = _split_render(load_expr, DK, DV, DKP, DVP, CH, L2NORM, epiq, epik, epig)
    p1, p2, allocations = _apply_preparation(
        p1, p2, allocations, load_expr, DK, DKP, L2NORM, 0)
    tape_value_load = load_expr("v2", "[row, cb]", "[1, CH]", "[1, pl.min(CH, DV - cb)]")
    p1, p2, allocations = _repair_reductions(p1, p2, allocations, tape_value_load)
    GCH = min(2 ** (5 if wide_geometry else 5), DVP)
    GNC = (DVP + GCH - 1) // GCH
    ADVP = NC * CH
    AHEADS = math.gcd(B * H, 2 ** 3)
    while AHEADS > 1 and 4 * AHEADS * TP1 * DKP * ADVP > 128 * 1024 * 1024:
        AHEADS //= 2
    EPT = 2 ** 3
    guarded = ((DKP >= 128 and not wide_geometry)
               or (not L2NORM and not 1))
    effective_history_mode = 0 if guarded else 1
    if T > 2 ** 0:
        p2, allocations = _relocate_gradients(
            p2, allocations, load_expr, DK, DKP, L2NORM, effective_history_mode)
        if effective_history_mode > 0:
            p2, allocations = _fuse_token_gradients(
                p2, allocations, load_expr, DK, DKP, L2NORM, effective_history_mode)
            p2, allocations = _phase_isolate_gradients(
                p2, allocations, load_expr, DK, DKP, L2NORM,
                2 if wide_geometry else 0,
                (1 and DVP <= 64 and DKP <= 64 and DKP % 16 == 0
                 and (not 1
                      or B * T * H >= 2 ** 11
                      or not L2NORM)))
            if (wide_geometry and 1
                    and 4 * B * H * TP1 * DKP * ADVP > 128 * 1024 * 1024):
                p2, allocations = _bound_adjoint_tape(p2, allocations)
    kernel_template = _KERNEL.replace("    di2 =", allocations + "    di2 =")
    src = kernel_template.replace("{P1}", p1.strip("\n")).replace("{P2}", p2.strip("\n"))
    short_tape_bytes = 4 * TP1 * DKP * DVP
    short_live_bytes = 2 * short_tape_bytes + 48 * 4 * DKP * DVP + 64 * 4 * (DKP + DVP)
    short_eligible = (1 and 1 < T <= 2 ** 3
                      and DK == DKP and DV == DVP and min(DKP, DVP) >= 16
                      and max(DKP, DVP) <= 2 ** 4
                      and short_tape_bytes <= 2 ** 14
                      and short_live_bytes <= 160 * 1024)
    if short_eligible:
        src = _short_resident_source(load_expr, DK, DV, DKP, DVP, L2NORM, T)
    elif T == 1 and B * H == 1 and DKP * DVP <= 64:
        src = _resident_source(load_expr, DK, DV, DKP, DVP, L2NORM)
    elif T == 1 and 1:
        SCH = math.gcd(min(2 ** 5, DVP), DVP)
        SNC = (DVP + SCH - 1) // SCH
        src = _sharded_single_source(load_expr, DK, DV, DKP, DVP, L2NORM)
    if T > 1 and DKP <= 64 and DVP <= 64 and not short_eligible:
        src = _redispatch_narrow_gradients(src, B, T, H, EPT, 2)
    for a, b in (("{CO}", co), ("{CC}", cc), ("{YO}", yo), ("{YC}", yc)):
        src = src.replace(a, b)
    for a, b in frag.items():
        src = src.replace("{%s}" % a, b)

    if L2NORM and T > 1 and DKP <= 64 and DVP <= 64 and not short_eligible:
        src = _batched_token_prepare(src, B * T * H, DK, DKP, 16, shape['dtype'] != 'fp32')
    if L2NORM and T > 1 and DKP >= 128 and not short_eligible:
        src = _wide_token_prepare(
            src, B * T * H, DK, DKP, 8, need_cast, False)
    if L2NORM and T > 1 and DKP >= 128 and not short_eligible:
        src = _wide_gradient_panels(
            src, DVP, 32, 32 if DKP > 128 else 64, True)
    layout_wide = wide_geometry
    layout_narrow = DKP <= 64 and DVP <= 64 and not short_eligible
    layout_selected = (layout_wide and 0 in (0, 2)) or (layout_narrow and 0 in (1, 2))
    if T > 1 and layout_selected and DKP % 16 == 0 and CH % 16 == 0:
        src = _key_contiguous_layout(src, 0, DK, DV, DKP, DVP, CH, NC, ADVP)
    dispatch_selected = (wide_geometry and 2 in (0, 2)) or (layout_narrow and 2 in (1, 2))
    if T > 1 and dispatch_selected and 5:
        src = _recurrent_indexed_launches(src, 5)
    checkpoint_selected = (L2NORM and T >= 32 and DK == DKP and DV == DVP and ((DKP == 128 and DVP == 128) or (0 and DKP == 64 and DVP <= 64)))
    if 1 and checkpoint_selected:
        src = _checkpoint_continue(src, T, DVP, 8, need_cast, PL_DTYPE)
    if 1 and T > 1:
        src = _checkpoint_compact(src)
    if L2NORM and T > 1 and DK == DKP and DV == DVP and DKP <= 64 and DVP <= 64 and not short_eligible:
        src = _delete_scalar_recurrence(src, 1)
    pipeline_eligible = (T >= 17 and T > 1
                         and not short_eligible and ((DKP <= 64 and DVP <= 64) or checkpoint_selected))
    if pipeline_eligible:
        src = _pipeline_forward_replay(src, 2)
    if pipeline_eligible:
        src = _pipeline_adjoint_consumers(src, 1, 1, 0)
    if pipeline_eligible and DKP <= 64 and DVP <= 64:
        src, packing_hoisted = _hoist_immutable_packing(src)
        if packing_hoisted:
            src = _overlap_repaired_consumer(src, 'fused_token_gradients', 'fused_t')
    if T == 1 and not (B * H == 1 and DKP * DVP <= 64):
        src = _single_downstream_indexed_launches(src, 3)
    if 1 < T <= 32 and DKP <= 64 and DVP <= 64 and B * T * H <= 128 and not short_eligible:
        src = _light_vector_beta(src, B * T * H, 16, 1)
    elif T > 1 and DKP <= 64 and DVP <= 64 and not short_eligible:
        src = _light_vector_beta(src, B * T * H, 16, 1)
    if L2NORM and T > 1 and DKP <= 64 and DVP <= 64 and not short_eligible and 3:
        src = _narrow_tile_adjoints(src, 3, 1)
    if checkpoint_selected and (1 or 1):
        src = _checkpoint_tile_reductions(src, 1, 1, 16)
    endpoint_live_bytes = 4 * (16 * DKP * DVP + 96 * (DKP + DVP) + 1024)
    endpoint_selected = (1 and shape['dtype'] == 'fp32'
                         and 1 < T <= 16 and DK == DKP and DV == DVP
                         and DKP <= 32 and DVP <= 64 and B * H <= 8
                         and endpoint_live_bytes <= 160 * 1024)
    cast_endpoint_live_bytes = (endpoint_live_bytes
                                + _owner_slab_extra_bytes(T, DKP, DVP, 1)
                                + 4 * (8 * (DKP + DVP) + 4 * 32))
    cast_endpoint_selected = (need_cast and 1 < T <= 8
                              and DK == DKP == 16 and DV == DVP == 16
                              and B * H <= 8
                              and cast_endpoint_live_bytes <= 160 * 1024)
    endpoint_selected = endpoint_selected or cast_endpoint_selected
    if cast_endpoint_selected:
        src = _tiny_cast_endpoint_source(DK, DV, DKP, DVP, L2NORM, 1)
    elif endpoint_selected:
        src = _endpoint_source(DK, DV, DKP, DVP, L2NORM, need_cast, 0, 1)
    elif T > 1 and DKP <= 64 and DVP <= 64 and not short_eligible:
        src = _coalesce_narrow_phases(src, B, T, H, EPT, 1, T <= 32)
        if 1 and L2NORM and T > 32:
            src = _stage_owner_layout(src, DKP, DVP, EPT, 0)
    if 1 and endpoint_selected and max(16, DKP) % 16 == 0 and max(16, DVP) % 16 == 0:
        src = _light_endpoint_grid(src)
    if 1 and '    for row_base in pl.parallel(0, B * T * H, 16):\n' in src:
        src = _checkpoint_compact(src)
    if 1 and not short_eligible and not endpoint_selected and 1 < T <= 32 and DKP <= 32 and DVP <= 32 and DVP % 16 == 0 and (DK * DV) % 16 == 0:
        src = _light_reverse_grid(src)
    if checkpoint_selected and DK == DKP == 128 and DV == DVP == 128 and B * H * NC > 1:
        src = _wide_reverse_owner_grid(src, 2)
    src = _owner_local_dispatch(src, B, T, H, DK, DV, DKP, DVP, L2NORM, need_cast,
                                1, 1, 0, short_live_bytes)
    src = _single_tile_completion(src, T, DK, DV, DKP, DVP, L2NORM,
                                  3, 0, 0)
    src = _producer_beta_vectors(src, B, T, H, DK, DV, DKP, DVP, L2NORM,
                                 1)
    src = _checked_state_dual_projection(src, B, T, H, DK, DV, DKP, DVP,
                                         FCH, GCH, L2NORM, checkpoint_selected,
                                         3, 0)
    owned_geometry = _owned_geometry(B, T, H, DK, DV, DKP, DVP, ADVP, FCH, CH, GCH, GNC, EPT, L2NORM, need_cast, down, short_eligible, endpoint_selected)
    src = _owned_program(src, owned_geometry, 1)
    src = _wide_dq_completion(src, T, DK, DV, DKP, DVP, L2NORM, checkpoint_selected, 3)
    ns = {"pl": pl, "B": B, "T": "T", "H": H, "DK": DK, "DV": DV, "DKP": DKP, "DVP": DVP,
          "TP1": TP1, "CH": CH, "FCH": FCH, "NC": NC, "ROWS": B * T * H,
          "EPT": EPT, "GCH": GCH, "GNC": GNC, "ADVP": ADVP, "AHEADS": AHEADS,
          "PL_DTYPE": PL_DTYPE, "SCALE": SCALE}
    if T == 1 and 1:
        ns.update(SCH=math.gcd(min(2 ** 5, DVP), DVP))
        ns["SNC"] = (DVP + ns["SCH"] - 1) // ns["SCH"]
    ns["T"] = T
    # pl.jit parses via inspect.getsource: give the exec'd code a linecache-backed filename
    import linecache
    fname = "kda_backward_prepared_value_parallel_%s.py" % shape["dtype"]
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec"), ns)
    kda_backward_decomposed = ns["kda_backward_decomposed"]

    def gen_inputs():
        """contract 3.1 ranges (+ 4.3 specials), mirrored from the sealed forward harness"""
        l2n = L2NORM
        lo, hi = (-1.0, 1.0) if l2n else (-0.05, 0.05)
        q = torch.rand(B, T, H, DK) * (hi - lo) + lo
        k = torch.rand(B, T, H, DK) * (hi - lo) + lo
        v = torch.rand(B, T, H, DV) - 0.5
        g = -5.0 * torch.rand(B, T, H, DK)
        beta = torch.rand(B, T, H)
        s0 = torch.rand(B, H, DK, DV) - 0.5
        special = shape.get("special")
        if special == "near_zero_qk":
            q = torch.rand(B, T, H, DK) * 6e-4 - 3e-4
            k = torch.rand(B, T, H, DK) * 6e-4 - 3e-4
        elif special == "strong_decay":
            g = -1.0 - 4.0 * torch.rand(B, T, H, DK)
        elif special == "no_decay_explicit_scale":
            g = torch.zeros(B, T, H, DK)
        elif special == "zero_qk":
            q = torch.zeros(B, T, H, DK)
            k = torch.zeros(B, T, H, DK)
            g = torch.zeros(B, T, H, DK)
            beta = torch.ones(B, T, H)
        dy = torch.randn(B, T, H, DV)
        dfin = torch.randn(B, H, DK, DV) * 0.5
        return {"q": q.to(TORCH_DTYPE), "k": k.to(TORCH_DTYPE), "v": v.to(TORCH_DTYPE),
                "g": g, "beta": beta, "initial_state": s0,
                "dy": dy.to(TORCH_DTYPE), "dfinal_state": dfin}

    _init = gen_inputs()

    def specs():
        return [
            TensorSpec("q", [B, T, H, DK], TORCH_DTYPE, init_value=_init["q"]),
            TensorSpec("k", [B, T, H, DK], TORCH_DTYPE, init_value=_init["k"]),
            TensorSpec("v", [B, T, H, DV], TORCH_DTYPE, init_value=_init["v"]),
            TensorSpec("g", [B, T, H, DK], torch.float32, init_value=_init["g"]),
            TensorSpec("beta", [B, T, H], torch.float32, init_value=_init["beta"]),
            TensorSpec("initial_state", [B, H, DK, DV], torch.float32,
                       init_value=_init["initial_state"]),
            TensorSpec("dy", [B, T, H, DV], TORCH_DTYPE, init_value=_init["dy"]),
            TensorSpec("dfinal_state", [B, H, DK, DV], torch.float32,
                       init_value=_init["dfinal_state"]),
            TensorSpec("hs", [B, TP1, H, DKP, DVP], torch.float32),
            TensorSpec("dh", [B, H, DK, DV], torch.float32),
            TensorSpec("y", [B, T, H, DV], TORCH_DTYPE),
            TensorSpec("final_state", [B, H, DK, DV], torch.float32),
            TensorSpec("dq", [B, T, H, DK], TORCH_DTYPE),
            TensorSpec("dk", [B, T, H, DK], TORCH_DTYPE),
            TensorSpec("dv", [B, T, H, DV], TORCH_DTYPE),
            TensorSpec("dg", [B, T, H, DK], torch.float32),
            TensorSpec("dbeta", [B, T, H], torch.float32),
            TensorSpec("dinitial_state", [B, H, DK, DV], torch.float32),
        ]

    def golden(t):
        # fp64 differentiable forward + autograd reverse seeding (y, final_state) with
        # (dy, dfinal_state); validates itself under torch.autograd.gradcheck (see
        # runs/goldencheck_kdabwd.py) and against an independent naive adjoint loop.
        q = t["q"].detach().double().requires_grad_()
        k = t["k"].detach().double().requires_grad_()
        v = t["v"].detach().double().requires_grad_()
        g = t["g"].detach().double().requires_grad_()
        beta = t["beta"].detach().double().requires_grad_()
        s0 = t["initial_state"].detach().double().requires_grad_()
        state = s0
        hs_l = [state]
        ys = []
        for ti in range(T):
            qt, kt = q[:, ti], k[:, ti]
            if L2NORM:
                qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            hb = torch.exp(g[:, ti]).unsqueeze(-1) * state
            r = v[:, ti] - (kt.unsqueeze(-1) * hb).sum(dim=2)
            state = hb + beta[:, ti].reshape(B, H, 1, 1) * kt.unsqueeze(-1) * r.unsqueeze(2)
            ys.append(SCALE * (qt.unsqueeze(-1) * state).sum(dim=2))
            hs_l.append(state)
        y = torch.stack(ys, dim=1)
        dq, dk, dvv, dg, dbeta, di = torch.autograd.grad(
            [y, state], [q, k, v, g, beta, s0],
            [t["dy"].detach().double(), t["dfinal_state"].detach().double()])
        t["y"][:] = y.to(TORCH_DTYPE)
        t["final_state"][:] = state.float()
        hs_pad = torch.zeros(B, TP1, H, DKP, DVP)   # hs workspace is padded-layout:
        hs_pad[:, :, :, :DK, :DV] = torch.stack(hs_l, dim=1).float()  # pads stay zero
        t["hs"][:] = hs_pad
        t["dh"][:] = di.float()          # dh ends holding dL/dH_0 == dinitial_state
        t["dq"][:] = dq.to(TORCH_DTYPE)
        t["dk"][:] = dk.to(TORCH_DTYPE)
        t["dv"][:] = dvv.to(TORCH_DTYPE)
        t["dg"][:] = dg.float()
        t["dbeta"][:] = dbeta.float()
        t["dinitial_state"][:] = di.float()

    return kda_backward_decomposed, specs(), golden, RTOL, ATOL


def _split_render(load_expr, DK, DV, DKP, DVP, CH, L2NORM, epiq, epik, epig):
    row_load = "[1, %s]" % DK if DK != DKP else None
    state_load = "[DK, pl.min(CH, DV - cb)]"
    value_load = "[1, pl.min(CH, DV - cb)]"
    hs_load = "[DKP, pl.min(CH, DVP - cb)]"
    chunk = _P2CH
    replacements = {
        "LDH1": load_expr("hs2", "[((b * TP1 + t + 1) * H + h) * DKP, cb]", "[DKP, CH]", hs_load),
        "LDH0": load_expr("hs2", "[((b * TP1 + t) * H + h) * DKP, cb]", "[DKP, CH]", hs_load),
        "LDDY": load_expr("dy2", "[row, cb]", "[1, CH]", value_load),
        "LDVC": load_expr("v2", "[row, cb]", "[1, CH]", value_load),
        "DVV": "dvo{c} = pl.set_validshape(dvo{c}, 1, pl.min(CH, DV - cb))",
        "DOTB": _dot_row("ccv{c}", "rc{c}", "dbe{c}", " " * 20),
        "CCOFF": "cb",
    }
    for name, replacement in replacements.items():
        chunk = chunk.replace("{" + name + "}", replacement)
    chunk = chunk.replace("{c}", "0")
    chunk = chunk.replace(
        "pdq0 = pl.row_sum(pl.mul(H1c0, pl.col_expand(H1c0, dyc0)), tmpA)",
        "pdq0 = pl.row_sum(pl.mul(H1c0, pl.col_expand(H1c0, dyc0)), tmpA)\n"
        "                    partial_row = (part * ROWS + row) * DKP\n"
        "                    pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)")
    chunk = chunk.replace(
        "dk20 = pl.mul(pl.row_sum(pl.mul(H0c0, pl.col_expand(H0c0, drc0)), tmpA), egcol)",
        "dk20 = pl.mul(pl.row_sum(pl.mul(H0c0, pl.col_expand(H0c0, drc0)), tmpA), egcol)\n"
        "                    dknv = pl.sub(pl.mul(dk10, bt), dk20)\n"
        "                    pl.store(pl.row_expand_mul(carrier, dknv), [partial_row, 0], partial_k)")
    chunk = "\n".join("    " + line if line else line for line in chunk.strip("\n").splitlines())
    adjoint_load = load_expr("df2", "[bh * DK, cb]", "[DKP, CH]", state_load)
    qnorm = _norm8("qt", "qn2", "qinv2", "w8b", " " * 24) if L2NORM else "qn2 = qt"
    knorm = _norm8("kt", "kn2", "kinv2", "w8b", " " * 24) if L2NORM else "kn2 = kt"
    reverse = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for part in pl.parallel(NC):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="value_reverse"):
                    bh = b * H + h
                    cb = part * CH
                    w8b = pl.create_tile([1, 8], dtype=pl.FP32)
                    tmpA = pl.create_tile([DKP, CH], dtype=pl.FP32)
                    carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)
                    dhcarry0 = {ADJOINT}
                    for tt in pl.range(T):
                        t = T - 1 - tt
                        row = (b * T + t) * H + h
                        qt = {CO}{LDQ}{CC}
                        kt = {CO}{LDK}{CC}
                        {QNORM}
                        {KNORM}
                        gt = pl.reshape({LDG}, [DKP, 1])
                        egcol = pl.exp(gt)
                        qcol = pl.reshape(qn2, [DKP, 1])
                        kcol = pl.reshape(kn2, [DKP, 1])
                        bt = pl.read(b1, [row, 0])
{CHUNK}
                        pl.store(pl.row_expand_mul(carrier, dgp0), [partial_row, 0], partial_g)
                        pl.write(partial_beta, [part * ROWS + row, 0], dbe0)
                    dhlast = pl.set_validshape(dhcarry0, DK, pl.min(CH, DV - cb))
                    pl.store(dhlast, [bh * DK, cb], dh2)
                    pl.store(dhlast, [bh * DK, cb], di2)
"""
    for name, replacement in {"ADJOINT": adjoint_load, "QNORM": qnorm,
                               "KNORM": knorm, "CHUNK": chunk}.items():
        reverse = reverse.replace("{" + name + "}", replacement)

    def row_output(source, destination, scale, norm_row, inverse, name):
        lines = [f"{name} = pl.reshape({source}, [1, DKP])"]
        if scale != 1.0:
            lines.append(f"{name} = pl.mul({name}, SCALE)")
        if L2NORM and norm_row:
            lines.append(f"dot_parts = pl.col_sum(pl.reshape(pl.mul({name}, {norm_row}), [DKP // 8, 8]))")
            dot = " + ".join(f"pl.read(dot_parts, [0, {lane}])" for lane in range(8))
            lines.append(f"norm_dot = {dot}")
            lines.append(f"{name} = pl.mul(pl.sub({name}, pl.mul({norm_row}, norm_dot)), {inverse})")
        if destination != "dg2":
            lines.append(f"{name} = {{YO}}{name}{{YC}}")
        if DK != DKP:
            lines.append(f"{name} = pl.set_validshape({name}, 1, DK)")
        lines.append(f"pl.store({name}, [row, 0], {destination})")
        return ("\n" + " " * 24).join(lines)

    epiq = row_output("dqv", "dq2", 0.0, "qn2", "qinv2", "dq_row")
    epik = row_output("dknv", "dk2", 1.0, "kn2", "kinv2", "dk_row")
    epig = row_output("dgv", "dg2", 1.0, None, None, "dg_row")

    epilogue = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for token_base in pl.parallel(0, T, EPT):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_epilogue"):
                    w8b = pl.create_tile([1, 8], dtype=pl.FP32)
                    reduce_tmp = pl.create_tile([DKP, 8], dtype=pl.FP32)
                    for t in pl.range(token_base, pl.min(token_base + EPT, T)):
                        row = (b * T + t) * H + h
                        qt = {CO}{LDQ}{CC}
                        kt = {CO}{LDK}{CC}
                        {QNORM}
                        {KNORM}
                        qcol = pl.reshape(qn2, [DKP, 1])
                        kcol = pl.reshape(kn2, [DKP, 1])
                        pq = pl.load(partial_q, [row * DKP, 0], [DKP, 8])
                        pk = pl.load(partial_k, [row * DKP, 0], [DKP, 8])
                        pg = pl.load(partial_g, [row * DKP, 0], [DKP, 8])
                        dbeta_acc = pl.read(partial_beta, [row, 0])
                        for part in pl.range(1, NC):
                            partial_row = (part * ROWS + row) * DKP
                            pq = pl.add(pq, pl.load(partial_q, [partial_row, 0], [DKP, 8]))
                            pk = pl.add(pk, pl.load(partial_k, [partial_row, 0], [DKP, 8]))
                            pg = pl.add(pg, pl.load(partial_g, [partial_row, 0], [DKP, 8]))
                            dbeta_acc = dbeta_acc + pl.read(partial_beta, [part * ROWS + row, 0])
                        dqv = pl.mul(pl.row_sum(pq, reduce_tmp), 0.125)
                        dknv = pl.mul(pl.row_sum(pk, reduce_tmp), 0.125)
                        dgv = pl.mul(pl.row_sum(pg, reduce_tmp), 0.125)
                        {EPIQ}
                        {EPIK}
                        {EPIG}
                        pl.write(dbeta_scratch, [row, 0], dbeta_acc)
"""
    if not L2NORM:
        epilogue = epilogue.replace("                        qt = {CO}{LDQ}{CC}\n", "")
        epilogue = epilogue.replace("                        kt = {CO}{LDK}{CC}\n", "")
        epilogue = epilogue.replace("                        {QNORM}\n", "")
        epilogue = epilogue.replace("                        {KNORM}\n", "")
        epilogue = epilogue.replace("                        qcol = pl.reshape(qn2, [DKP, 1])\n", "")
        epilogue = epilogue.replace("                        kcol = pl.reshape(kn2, [DKP, 1])\n", "")
    for name, replacement in {"QNORM": qnorm, "KNORM": knorm,
                               "EPIQ": epiq,
                               "EPIK": epik,
                               "EPIG": epig}.items():
        epilogue = epilogue.replace("{" + name + "}", replacement)
    allocations = """
    partial_q = pl.create_tensor([NC * ROWS * DKP, 8], dtype=pl.FP32)
    partial_k = pl.create_tensor([NC * ROWS * DKP, 8], dtype=pl.FP32)
    partial_g = pl.create_tensor([NC * ROWS * DKP, 8], dtype=pl.FP32)
    partial_beta = pl.create_tensor([NC * ROWS, 16], dtype=pl.FP32)
"""
    return reverse + epilogue, allocations


def _apply_preparation(p1, p2, allocations, load_expr, DK, DKP, L2NORM, mode):
    if mode == 0 or (mode == 1 and not L2NORM):
        return p1, p2, allocations
    producer = [
        "    for row in pl.parallel(ROWS):",
        "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"token_prepare\"):",
    ]
    if L2NORM:
        allocations += """
    prepared_q = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_k = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    norm_meta = pl.create_tensor([ROWS, 16], dtype=pl.FP32)
"""
        producer += [
            "            w8p = pl.create_tile([1, 8], dtype=pl.FP32)",
            "            qt = {CO}{LDQ}{CC}",
            "            kt = {CO}{LDK}{CC}",
            "            " + _norm8("qt", "qnp", "qinvp", "w8p", " " * 12),
            "            pl.store(qnp, [row, 0], prepared_q)",
            "            pl.write(norm_meta, [row, 0], qinvp)",
            "            " + _norm8("kt", "knp", "kinvp", "w8p", " " * 12),
            "            pl.store(knp, [row, 0], prepared_k)",
            "            pl.write(norm_meta, [row, 1], kinvp)",
        ]
        for stage, scratch in ((1, "w8"), (2, "w8b")):
            for prefix, prepared in (("q", "prepared_q"), ("k", "prepared_k")):
                old = _norm8(prefix + "t", prefix + "n" + str(stage),
                             prefix + "inv" + str(stage), scratch, " " * 24)
                new = f"{prefix}n{stage} = pl.load({prepared}, [row, 0], [1, DKP])"
                if stage == 1:
                    p1 = p1.replace(old, new)
                else:
                    p2 = p2.replace(old, new)
        epilogue_offset = p2.index('name_hint="token_epilogue"')
        epilogue = p2[epilogue_offset:]
        epilogue = epilogue.replace(
            "qn2 = pl.load(prepared_q, [row, 0], [1, DKP])",
            "norm_values = pl.load(norm_meta, [row, 0], [1, 16])\n"
            "                        qinv2 = pl.read(norm_values, [0, 0])\n"
            "                        kinv2 = pl.read(norm_values, [0, 1])\n"
            "                        qn2 = pl.load(prepared_q, [row, 0], [1, DKP])")
        p2 = p2[:epilogue_offset] + epilogue
        p1 = p1.replace("                    w8 = pl.create_tile([1, 8], dtype=pl.FP32)\n", "")
        p2 = p2.replace("                    w8b = pl.create_tile([1, 8], dtype=pl.FP32)\n", "")
        for fragment in ("qt = {CO}{LDQ}{CC}", "kt = {CO}{LDK}{CC}"):
            p1 = p1.replace("                        " + fragment + "\n", "")
            p2 = p2.replace("                        " + fragment + "\n", "")
    if mode == 2:
        allocations += "\n    prepared_decay = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)\n"
        producer += [
            "            decay_row = pl.exp({LDG})",
            "            pl.store(decay_row, [row, 0], prepared_decay)",
        ]
        decay_load = "pl.reshape(pl.load(prepared_decay, [row, 0], [1, DKP]), [DKP, 1])"
        p1 = p1.replace("gt = pl.reshape({LDG}, [DKP, 1])\n                        decay = pl.exp(gt)",
                        "decay = " + decay_load)
        p2 = p2.replace("gt = pl.reshape({LDG}, [DKP, 1])\n                        egcol = pl.exp(gt)",
                        "egcol = " + decay_load)
    return "\n".join(producer) + "\n" + p1, p2, allocations


def _replace_once(source, before, after):
    count = source.count(before)
    if count != 1:
        raise ValueError(f"Selected-render replacement matched {count} times: {before}")
    return source.replace(before, after, 1)


def _repair_reductions(p1, p2, allocations, tape_value_load):
    if 1:
        p1 = _replace_once(
            p1,
            "                    w8 = pl.create_tile([1, 8], dtype=pl.FP32)",
            "                    w8 = pl.create_tile([1, 8], dtype=pl.FP32)\n"
            "                    key_reduction_scratch = pl.create_tile([DKP, FCH], dtype=pl.FP32)")
        p1 = _replace_once(
            p1,
            "kh = pl.col_sum(pl.row_expand_mul(state, kcol))",
            "kh = pl.col_sum(pl.row_expand_mul(state, kcol), key_reduction_scratch)")
        p1 = _replace_once(
            p1,
            "yt = pl.mul(pl.col_sum(pl.row_expand_mul(state, qcol)), SCALE)",
            "yt = pl.mul(pl.col_sum(pl.row_expand_mul(state, qcol), key_reduction_scratch), SCALE)")
    if 1:
        allocations += "\n    residual_tape = pl.create_tensor([B * T * H, DVP], dtype=pl.FP32)\n"
        p1 = _replace_once(
            p1,
            "                        r = pl.sub({CO}{LDV}{CC}, kh)",
            "                        r = pl.sub({CO}{LDV}{CC}, kh)\n"
            "                        pl.store(pl.set_validshape(r, 1, pl.min(FCH, DVP - cb)), [row, cb], residual_tape)")
        for fragment in (
            "                        decayed_key0 = pl.mul(kcol, egcol)\n",
            "                        vc0 = {CO}" + tape_value_load + "{CC}\n",
            "                        kh0 = pl.col_sum(pl.row_expand_mul(H0c0, decayed_key0))\n",
        ):
            p2 = _replace_once(p2, fragment, "")
        p2 = _replace_once(
            p2,
            "rc0 = pl.sub(vc0, kh0)",
            "rc0 = pl.set_validshape(pl.fillpad(pl.load(residual_tape, [row, cb], [1, CH], "
            "valid_shape=[1, pl.min(CH, DVP - cb)]), 0), 1, CH)")
    if 1:
        p2 = _replace_once(
            p2,
            "ccv0 = pl.col_sum(pl.row_expand_mul(dhgc0, kcol))",
            "ccv0 = pl.col_sum(pl.row_expand_mul(dhgc0, kcol), tmpA)")
        if not 1:
            p2 = _replace_once(
                p2,
                "kh0 = pl.col_sum(pl.row_expand_mul(H0c0, decayed_key0))",
                "kh0 = pl.col_sum(pl.row_expand_mul(H0c0, decayed_key0), tmpA)")
    if 1:
        for matrix, row in (("H1c0", "dyc0"), ("dhgc0", "rc0"), ("H0c0", "drc0")):
            p2 = _replace_once(
                p2,
                f"pl.mul({matrix}, pl.col_expand({matrix}, {row}))",
                f"pl.col_expand_mul({matrix}, {row})")
    return _single_state_reverse(p1, p2, allocations)


def _replace_line(source, prefix, replacement):
    lines = source.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.lstrip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Selected line replacement matched {len(matches)} times: {prefix}")
    lines[matches[0]] = replacement
    return "".join(lines)


def _single_state_reverse(p1, p2, allocations):
    indent = " " * 24
    if 0:
        p2 = _replace_line(p2, "H1c0 =", "")
        p2 = _replace_line(p2, "pdq0 =", "")
        p2 = _replace_once(
            p2,
            indent + "partial_row = (part * ROWS + row) * DKP\n"
            + indent + "pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)\n",
            "")
        residual_line = next(line for line in p2.splitlines(keepends=True)
                             if line.lstrip().startswith("rc0 ="))
        dot = " + ".join(f"pl.read(residual_dy_parts0, [0, {lane}])" for lane in range(8))
        identity = [
            "rc0 = pl.set_validshape(pl.fillpad(rc0, 0), 1, CH)",
            "residual_dy_parts0 = pl.col_sum(pl.reshape(pl.mul(rc0, dyc0), [CH // 8, 8]))",
            "residual_dy_dot0 = " + dot,
            "previous_dq0 = pl.row_sum(pl.col_expand_mul(H0c0, dyc0), tmpA)",
            "pdq0 = pl.add(pl.mul(previous_dq0, egcol), pl.mul(kcol, bt * residual_dy_dot0))",
            "partial_row = (part * ROWS + row) * DKP",
            "pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)",
        ]
        p2 = _replace_once(p2, residual_line,
                           residual_line + "".join(indent + line + "\n" for line in identity))
    if 1:
        p2 = _replace_once(
            p2,
            "upd0 = pl.row_expand_mul(pl.col_expand(dhgc0, dyc0), qcol)",
            "scaled_qcol = pl.mul(qcol, SCALE)\n" + indent
            + "upd0 = pl.row_expand_mul(pl.col_expand(dhgc0, dyc0), scaled_qcol)")
        p2 = _replace_once(p2, "dhgc0 = pl.add(dhgc0, pl.mul(upd0, SCALE))",
                           "dhgc0 = pl.add(dhgc0, upd0)")
        p2 = _replace_once(
            p2,
            indent + "dgp0 = pl.mul(pl.row_sum(pl.mul(dhp0, H0c0), tmpA), egcol)\n",
            "")
        p2 = _replace_once(
            p2,
            "dhn0 = pl.row_expand_mul(dhp0, egcol)",
            "dhn0 = pl.row_expand_mul(dhp0, egcol)\n" + indent
            + "dgp0 = pl.row_sum(pl.mul(dhn0, H0c0), tmpA)")
    return p1, p2, allocations


def _isolate_region_variables(source, prefix):
    names = set()
    for line in source.splitlines():
        stripped = line.strip()
        if " = " in stripped:
            name = stripped.split(" = ", 1)[0]
            if name.isidentifier():
                names.add(name)
        if stripped.startswith("for "):
            names.add(stripped.split()[1])
    result = []
    token = ""
    for character in source + "\n":
        if character.isalnum() or character == "_":
            token += character
        else:
            result.append(prefix + token if token in names else token)
            result.append(character)
            token = ""
    return "".join(result)


def _relocate_gradients(p2, allocations, load_expr, DK, DKP, L2NORM, mode):
    split = p2.index("    for b in pl.parallel(B):", p2.index('name_hint="value_reverse"'))
    reverse, epilogue = p2[:split], p2[split:]
    indent = " " * 24
    row_valid = "[1, %s]" % DK if DK != DKP else None
    qload = load_expr("q2", "[row, 0]", "[1, DKP]", row_valid)
    kload = load_expr("k2", "[row, 0]", "[1, DKP]", row_valid)
    gload = load_expr("g2", "[row, 0]", "[1, DKP]", row_valid)
    qnorm = _norm8("qt", "qn2", "qinv2", "w8b", indent) if L2NORM else "qn2 = qt"
    knorm = _norm8("kt", "kn2", "kinv2", "w8b", indent) if L2NORM else "kn2 = kt"
    if mode == 0:
        reverse = _replace_line(reverse, "H1c0 =", "")
        reverse = _replace_line(reverse, "pdq0 =", "")
        reverse = _replace_once(reverse, indent + "pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)\n", "")
        hload = load_expr("hs2", "[((b * TP1 + t + 1) * H + h) * DKP, cb]",
                          "[DKP, CH]", "[DKP, pl.min(CH, DVP - cb)]")
        dyload = load_expr("dy2", "[row, cb]", "[1, CH]", "[1, pl.min(CH, DV - cb)]")
        dq_stage = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for token_base in pl.parallel(0, T, EPT):
                for part in pl.parallel(NC):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="parallel_dq"):
                        cb = part * CH
                        tmpA = pl.create_tile([DKP, CH], dtype=pl.FP32)
                        carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)
                        for t in pl.range(token_base, pl.min(token_base + EPT, T)):
                            row = (b * T + t) * H + h
                            H1c0 = {HLOAD}
                            dyc0 = {CO}{DYLOAD}{CC}
                            pdq0 = pl.row_sum(pl.col_expand_mul(H1c0, dyc0), tmpA)
                            partial_row = (part * ROWS + row) * DKP
                            pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)
"""
        dq_stage = dq_stage.replace("{HLOAD}", hload).replace("{DYLOAD}", dyload)
        return reverse + _isolate_region_variables(dq_stage, "dqtask_") + epilogue, allocations
    adjoint = load_expr("df2", "[bh * DK, cb]", "[DKP, CH]",
                        "[DK, pl.min(CH, DV - cb)]")
    dyload = load_expr("dy2", "[row, cb]", "[1, CH]", "[1, pl.min(CH, DV - cb)]")
    thin = """
    for owner in pl.parallel(B * H * NC):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="thin_adjoint_reverse"):
            bh = owner // NC
            b = bh // H
            h = bh % H
            part = owner % NC
            cb = part * CH
            w8b = pl.create_tile([1, 8], dtype=pl.FP32)
            tmpA = pl.create_tile([DKP, CH], dtype=pl.FP32)
            dhcarry0 = {ADJOINT}
            {SEED}
            for tt in pl.range(T):
                t = T - 1 - tt
                row = (b * T + t) * H + h
                qt = {CO}{QLOAD}{CC}
                kt = {CO}{KLOAD}{CC}
                {QNORM}
                {KNORM}
                egcol = pl.exp(pl.reshape({GLOAD}, [DKP, 1]))
                qcol = pl.reshape(qn2, [DKP, 1])
                kcol = pl.reshape(kn2, [DKP, 1])
                bt = pl.read(b1, [row, 0])
                dyc0 = {CO}{DYLOAD}{CC}
                scaled_qcol = pl.mul(qcol, SCALE)
                upd0 = pl.row_expand_mul(pl.col_expand(dhcarry0, dyc0), scaled_qcol)
                dhgc0 = pl.add(dhcarry0, upd0)
                ccv0 = pl.col_sum(pl.row_expand_mul(dhgc0, kcol), tmpA)
                drc0 = pl.mul(ccv0, bt)
                dhp0 = pl.sub(dhgc0, pl.row_expand_mul(pl.col_expand(dhgc0, drc0), kcol))
                dhcarry0 = pl.row_expand_mul(dhp0, egcol)
                {HSTORE}
                pl.store(ccv0, [row, cb], contraction_tape)
            dhlast = pl.set_validshape(dhcarry0, DK, pl.min(CH, DV - cb))
            pl.store(dhlast, [bh * DK, cb], dh2)
            pl.store(dhlast, [bh * DK, cb], di2)
"""
    seed = "pl.store(dhcarry0, [(bh * TP1 + T) * DKP, cb], adjoint_tape)" if mode == 2 else "pass"
    hstore = "pl.store(%s, [(bh * TP1 + t) * DKP, cb], adjoint_tape)" % ("dhgc0" if mode == 1 else "dhcarry0")
    for name, replacement in {
        "ADJOINT": adjoint, "SEED": seed, "HSTORE": hstore,
        "QLOAD": qload, "KLOAD": kload, "GLOAD": gload,
        "QNORM": qnorm.replace("\n" + indent, "\n" + " " * 16),
        "KNORM": knorm.replace("\n" + indent, "\n" + " " * 16),
        "DYLOAD": dyload,
    }.items():
        thin = thin.replace("{" + name + "}", replacement)
    gradient = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for token_base in pl.parallel(0, T, EPT):
                for part in pl.parallel(GNC):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_parallel_gradients"):
                        bh = b * H + h
                        cb = part * GCH
                        w8b = pl.create_tile([1, 8], dtype=pl.FP32)
                        tmpA = pl.create_tile([DKP, GCH], dtype=pl.FP32)
                        carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)
                        for t in pl.range(token_base, pl.min(token_base + EPT, T)):
                            row = (b * T + t) * H + h
                            qt = {CO}{QLOAD}{CC}
                            kt = {CO}{KLOAD}{CC}
                            {QNORM}
                            {KNORM}
                            egcol = pl.exp(pl.reshape({GLOAD}, [DKP, 1]))
                            qcol = pl.reshape(qn2, [DKP, 1])
                            kcol = pl.reshape(kn2, [DKP, 1])
                            bt = pl.read(b1, [row, 0])
                            dyc0 = {CO}{DYLOAD}{CC}
                            H1c0 = {H1LOAD}
                            H0c0 = {H0LOAD}
                            rc0 = {RLOAD}
                            ccv0 = {CLOAD}
                            drc0 = pl.mul(ccv0, bt)
                            {GREAD}
                            pdq0 = pl.row_sum(pl.col_expand_mul(H1c0, dyc0), tmpA)
                            dk10 = pl.row_sum(pl.col_expand_mul(dhgc0, rc0), tmpA)
                            dk20 = pl.mul(pl.row_sum(pl.col_expand_mul(H0c0, drc0), tmpA), egcol)
                            dknv = pl.sub(pl.mul(dk10, bt), dk20)
                            {AREAD}
                            dgp0 = pl.row_sum(pl.mul(dhn0, H0c0), tmpA)
                            dvo0 = {YO}drc0{YC}
                            dvo0 = pl.set_validshape(dvo0, 1, pl.min(GCH, DV - cb))
                            pl.store(dvo0, [row, cb], dv2)
                            {DOTB}
                            partial_row = (part * ROWS + row) * DKP
                            pl.store(pl.row_expand_mul(carrier, pdq0), [partial_row, 0], partial_q)
                            pl.store(pl.row_expand_mul(carrier, dknv), [partial_row, 0], partial_k)
                            pl.store(pl.row_expand_mul(carrier, dgp0), [partial_row, 0], partial_g)
                            pl.write(partial_beta, [part * ROWS + row, 0], dbe0)
"""
    history_valid = "[DKP, pl.min(GCH, ADVP - cb)]"
    aread = load_expr("adjoint_tape", "[(bh * TP1 + t) * DKP, cb]", "[DKP, GCH]", history_valid)
    if mode == 1:
        gradient = gradient.replace("                            qt = {CO}{QLOAD}{CC}\n", "")
        gradient = gradient.replace("                            {QNORM}\n", "")
        gradient = gradient.replace("                            qcol = pl.reshape(qn2, [DKP, 1])\n", "")
        gread = "dhgc0 = " + aread
        aread = "dhp0 = pl.sub(dhgc0, pl.row_expand_mul(pl.col_expand(dhgc0, drc0), kcol))\n" + " " * 28 + "dhn0 = pl.row_expand_mul(dhp0, egcol)"
    else:
        next_adjoint = load_expr("adjoint_tape", "[(bh * TP1 + t + 1) * DKP, cb]", "[DKP, GCH]", history_valid)
        gread = "next_carry = " + next_adjoint + "\n" + " " * 28
        gread += "scaled_qcol = pl.mul(qcol, SCALE)\n" + " " * 28
        gread += "upd0 = pl.row_expand_mul(pl.col_expand(next_carry, dyc0), scaled_qcol)\n" + " " * 28
        gread += "dhgc0 = pl.add(next_carry, upd0)"
        aread = "dhn0 = " + aread
    for name, replacement in {
        "QLOAD": qload, "KLOAD": kload, "GLOAD": gload,
        "QNORM": qnorm.replace("\n" + indent, "\n" + " " * 28),
        "KNORM": knorm.replace("\n" + indent, "\n" + " " * 28),
        "DYLOAD": load_expr("dy2", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, DV - cb)]"),
        "H1LOAD": load_expr("hs2", "[((b * TP1 + t + 1) * H + h) * DKP, cb]", "[DKP, GCH]", "[DKP, pl.min(GCH, DVP - cb)]"),
        "H0LOAD": load_expr("hs2", "[((b * TP1 + t) * H + h) * DKP, cb]", "[DKP, GCH]", "[DKP, pl.min(GCH, DVP - cb)]"),
        "RLOAD": load_expr("residual_tape", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, DVP - cb)]"),
        "CLOAD": load_expr("contraction_tape", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, ADVP - cb)]"),
        "GREAD": gread, "AREAD": aread,
        "DOTB": _dot_row("ccv0", "rc0", "dbe0", " " * 28).replace("CH", "GCH"),
    }.items():
        gradient = gradient.replace("{" + name + "}", replacement)
    epilogue = epilogue.replace("NC", "GNC")
    allocations = allocations.replace("[NC * ROWS", "[GNC * ROWS")
    allocations += """
    adjoint_tape = pl.create_tensor([B * H * TP1 * DKP, ADVP], dtype=pl.FP32)
    contraction_tape = pl.create_tensor([ROWS, ADVP], dtype=pl.FP32)
"""
    return thin + _isolate_region_variables(gradient, "grad_") + epilogue, allocations


def _fuse_token_gradients(source, allocations, load_expr, DK, DKP, L2NORM, mode):
    boundary = source.index("    for grad_b in pl.parallel(B):", source.index('name_hint="thin_adjoint_reverse"'))
    thin = source[:boundary]
    indent = " " * 24
    row_valid = "[1, %s]" % DK if DK != DKP else None
    qnorm = _norm8("qt", "qn2", "qinv2", "w8q", indent) if L2NORM else "qn2 = qt"
    knorm = _norm8("kt", "kn2", "kinv2", "w8k", indent) if L2NORM else "kn2 = kt"
    gradient = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for token_base in pl.parallel(0, T, EPT):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_token_gradients"):
                    bh = b * H + h
                    w8q = pl.create_tile([1, 8], dtype=pl.FP32)
                    w8k = pl.create_tile([1, 8], dtype=pl.FP32)
                    tmpA = pl.create_tile([DKP, GCH], dtype=pl.FP32)
                    zero_columns = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)
                    for t in pl.range(token_base, pl.min(token_base + EPT, T)):
                        row = (b * T + t) * H + h
                        qt = {CO}{QLOAD}{CC}
                        kt = {CO}{KLOAD}{CC}
                        {QNORM}
                        {KNORM}
                        qcol = pl.reshape(qn2, [DKP, 1])
                        kcol = pl.reshape(kn2, [DKP, 1])
                        bt = pl.read(b1, [row, 0])
                        dqacc = pl.row_sum(zero_columns, tmpA)
                        dkacc = pl.row_sum(zero_columns, tmpA)
                        dgacc = pl.row_sum(zero_columns, tmpA)
                        dbeta_acc = 0.0
                        for part in pl.range(GNC):
                            cb = part * GCH
                            dyc0 = {CO}{DYLOAD}{CC}
                            H1c0 = {H1LOAD}
                            H0c0 = {H0LOAD}
                            rc0 = {RLOAD}
                            ccv0 = {CLOAD}
                            drc0 = pl.mul(ccv0, bt)
                            {GREAD}
                            pdq0 = pl.row_sum(pl.col_expand_mul(H1c0, dyc0), tmpA)
                            dk10 = pl.row_sum(pl.col_expand_mul(dhgc0, rc0), tmpA)
                            gt = pl.reshape({GLOAD}, [DKP, 1])
                            egcol = pl.exp(gt)
                            dk20 = pl.mul(pl.row_sum(pl.col_expand_mul(H0c0, drc0), tmpA), egcol)
                            dknv = pl.sub(pl.mul(dk10, bt), dk20)
                            {AREAD}
                            dgp0 = pl.row_sum(pl.mul(dhn0, H0c0), tmpA)
                            dvo0 = {YO}drc0{YC}
                            dvo0 = pl.set_validshape(dvo0, 1, pl.min(GCH, DV - cb))
                            pl.store(dvo0, [row, cb], dv2)
                            {DOTB}
                            dqacc = pl.add(dqacc, pdq0)
                            dkacc = pl.add(dkacc, dknv)
                            dgacc = pl.add(dgacc, dgp0)
                            dbeta_acc = dbeta_acc + dbe0
                        {DQOUT}
                        {DKOUT}
                        {DGOUT}
                        pl.write(dbeta_scratch, [row, 0], dbeta_acc)
"""
    valid = "[DKP, pl.min(GCH, ADVP - cb)]"
    current = load_expr("adjoint_tape", "[(bh * TP1 + t) * DKP, cb]", "[DKP, GCH]", valid)
    if mode == 1:
        gread = "dhgc0 = " + current
        aread = "dhp0 = pl.sub(dhgc0, pl.row_expand_mul(pl.col_expand(dhgc0, drc0), kcol))\n" + " " * 28
        aread += "dhn0 = pl.row_expand_mul(dhp0, egcol)"
    else:
        following = load_expr("adjoint_tape", "[(bh * TP1 + t + 1) * DKP, cb]", "[DKP, GCH]", valid)
        gread = "next_carry = " + following + "\n" + " " * 28
        gread += "scaled_qcol = pl.mul(qcol, SCALE)\n" + " " * 28
        gread += "upd0 = pl.row_expand_mul(pl.col_expand(next_carry, dyc0), scaled_qcol)\n" + " " * 28
        gread += "dhgc0 = pl.add(next_carry, upd0)"
        aread = "dhn0 = " + current

    repair = DKP >= 128 or not L2NORM
    if repair:
        gradient = _replace_once(
            gradient,
            "                    tmpA = pl.create_tile([DKP, GCH], dtype=pl.FP32)",
            "                    tmpA = pl.create_tile([DKP, GCH], dtype=pl.FP32)\n"
            "                    dq_scratch = pl.create_tile([DKP, GCH], dtype=pl.FP32)")
        gradient = _replace_once(
            gradient,
            "                        dqacc = pl.row_sum(zero_columns, tmpA)",
            "                        dq_zero_columns = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)\n"
            "                        dqacc = pl.row_sum(dq_zero_columns, dq_scratch)")
        gradient = _replace_once(
            gradient,
            "                            pdq0 = pl.row_sum(pl.col_expand_mul(H1c0, dyc0), tmpA)",
            "                            pdq0 = pl.row_sum(pl.col_expand_mul(H1c0, dyc0), dq_scratch)\n"
            "                            dqacc = pl.add(dqacc, pdq0)")
        gradient = _replace_once(gradient, "                            dqacc = pl.add(dqacc, pdq0)\n                            dkacc", "                            dkacc")

    def row_output(accumulator, destination, normalized, inverse, name, scaled):
        if repair and destination == "dq2":
            lines = [
                f"{name} = pl.create_tile([1, DKP], dtype=pl.FP32)",
                "for dq_lane in pl.unroll(DKP):",
                f"    pl.tile.write({name}, [0, dq_lane], pl.read({accumulator}, [dq_lane, 0]))",
            ]
        else:
            lines = [f"{name} = pl.reshape({accumulator}, [1, DKP])"]
        if scaled:
            lines.append(f"{name} = pl.mul({name}, SCALE)")
        if L2NORM and normalized:
            parts = name + "_parts"
            dot = name + "_dot"
            lines.append(f"{parts} = pl.col_sum(pl.reshape(pl.mul({name}, {normalized}), [DKP // 8, 8]))")
            lines.append(f"{dot} = " + " + ".join(f"pl.read({parts}, [0, {lane}])" for lane in range(8)))
            lines.append(f"{name} = pl.mul(pl.sub({name}, pl.mul({normalized}, {dot})), {inverse})")
        if destination != "dg2":
            lines.append(f"{name} = {{YO}}{name}{{YC}}")
        if DK != DKP:
            lines.append(f"{name} = pl.set_validshape({name}, 1, DK)")
        lines.append(f"pl.store({name}, [row, 0], {destination})")
        return ("\n" + indent).join(lines)

    replacements = {
        "QLOAD": load_expr("q2", "[row, 0]", "[1, DKP]", row_valid),
        "KLOAD": load_expr("k2", "[row, 0]", "[1, DKP]", row_valid),
        "GLOAD": load_expr("g2", "[row, 0]", "[1, DKP]", row_valid),
        "QNORM": qnorm, "KNORM": knorm,
        "DYLOAD": load_expr("dy2", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, DV - cb)]"),
        "H1LOAD": load_expr("hs2", "[((b * TP1 + t + 1) * H + h) * DKP, cb]", "[DKP, GCH]", "[DKP, pl.min(GCH, DVP - cb)]"),
        "H0LOAD": load_expr("hs2", "[((b * TP1 + t) * H + h) * DKP, cb]", "[DKP, GCH]", "[DKP, pl.min(GCH, DVP - cb)]"),
        "RLOAD": load_expr("residual_tape", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, DVP - cb)]"),
        "CLOAD": load_expr("contraction_tape", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, ADVP - cb)]"),
        "GREAD": gread, "AREAD": aread,
        "DOTB": _dot_row("ccv0", "rc0", "dbe0", " " * 28).replace("CH", "GCH"),
        "DQOUT": row_output("dqacc", "dq2", "qn2", "qinv2", "dq_row", True),
        "DKOUT": row_output("dkacc", "dk2", "kn2", "kinv2", "dk_row", False),
        "DGOUT": row_output("dgacc", "dg2", None, None, "dg_row", False),
    }
    for name, replacement in replacements.items():
        gradient = gradient.replace("{" + name + "}", replacement)
    allocations = "\n".join(line for line in allocations.splitlines() if not line.strip().startswith("partial_")) + "\n"
    return thin + _isolate_region_variables(gradient, "fused_"), allocations


def _resident_source(load_expr, DK, DV, DKP, DVP, L2NORM):
    row_valid = "[1, %s]" % DK if DK != DKP else None
    row_load = lambda tensor: load_expr(tensor, "[0, 0]", "[1, DKP]", row_valid)
    state_load = lambda tensor: load_expr(tensor, "[0, 0]", "[DKP, DVP]", "[DK, DV]")
    value_load = lambda tensor: load_expr(tensor, "[0, 0]", "[1, DVP]", "[1, DV]")
    indent = " " * 12
    lines = [
        "qraw = {CO}" + row_load("q2") + "{CC}",
        "kraw = {CO}" + row_load("k2") + "{CC}",
        "qnorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
        "knorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
    ]
    if L2NORM:
        lines.extend(_norm8("qraw", "qn", "qinv", "qnorm_scratch", "").splitlines())
        lines.extend(_norm8("kraw", "kn", "kinv", "knorm_scratch", "").splitlines())
    else:
        lines.extend(["qn = qraw", "kn = kraw"])
    lines.extend([
        "qcol = pl.reshape(qn, [DKP, 1])",
        "kcol = pl.reshape(kn, [DKP, 1])",
        "decay = pl.exp(pl.reshape(" + row_load("g2") + ", [DKP, 1]))",
        "previous_state = " + state_load("s2"),
        "pl.store(previous_state, [0, 0], hs2)",
        "decayed_state = pl.row_expand_mul(previous_state, decay)",
        "key_state = pl.col_sum(pl.row_expand_mul(decayed_state, kcol))",
        "value = {CO}" + value_load("v2") + "{CC}",
        "residual = pl.sub(value, key_state)",
        "weight = pl.read(b1, [0, 0])",
        "update = pl.row_expand_mul(pl.col_expand(decayed_state, residual), pl.mul(kcol, weight))",
        "updated_state = pl.add(decayed_state, update)",
        "out_y = pl.mul(pl.col_sum(pl.row_expand_mul(updated_state, qcol)), SCALE)",
        "out_y = {YO}out_y{YC}",
        "pl.store(pl.set_validshape(out_y, 1, DV), [0, 0], y2)",
        "output_grad = {CO}" + value_load("dy2") + "{CC}",
        "dq_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "dquery = pl.mul(pl.row_sum(pl.col_expand_mul(updated_state, output_grad), dq_scratch), SCALE)",
    ])
    lines.extend(_resident_row_output("dquery", "dq2", "qn", "qinv", "dq_row", DK, DKP, L2NORM))
    lines.extend([
        "pl.store(updated_state, [DKP, 0], hs2)",
        "pl.store(pl.set_validshape(updated_state, DK, DV), [0, 0], f2)",
        "incoming = " + state_load("df2"),
        "injection = pl.row_expand_mul(pl.col_expand(incoming, output_grad), pl.mul(qcol, SCALE))",
        "injected = pl.add(incoming, injection)",
        "contraction = pl.col_sum(pl.row_expand_mul(injected, kcol))",
        "dresidual = pl.mul(contraction, weight)",
        "out_dv = {YO}dresidual{YC}",
        "pl.store(pl.set_validshape(out_dv, 1, DV), [0, 0], dv2)",
        "dk_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "dg_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "shared_z = pl.row_sum(pl.col_expand_mul(previous_state, dresidual), dk_scratch)",
        "key_first = pl.row_sum(pl.col_expand_mul(injected, residual), dk_scratch)",
        "dkey = pl.sub(pl.mul(key_first, weight), pl.mul(shared_z, decay))",
    ])
    lines.extend(_resident_row_output("dkey", "dk2", "kn", "kinv", "dk_row", DK, DKP, L2NORM))
    lines.extend([
        "injected_state = pl.row_sum(pl.mul(injected, previous_state), dg_scratch)",
        "dgate = pl.mul(pl.sub(injected_state, pl.mul(kcol, shared_z)), decay)",
    ])
    lines.extend(_resident_row_output("dgate", "dg2", None, None, "dg_row", DK, DKP, L2NORM))
    lines.extend([
        "corrected = pl.sub(injected, pl.row_expand_mul(pl.col_expand(injected, dresidual), kcol))",
        "initial_grad = pl.row_expand_mul(corrected, decay)",
        "pl.store(pl.set_validshape(initial_grad, DK, DV), [0, 0], dh2)",
        "pl.store(pl.set_validshape(initial_grad, DK, DV), [0, 0], di2)",
    ])
    lines.extend(_dot_row("contraction", "residual", "weight_grad", "").replace("CH", "DVP").splitlines())
    lines.extend([
        "weight_row = pl.create_tile([1, 8], dtype=pl.FP32)",
        "pl.tile.write(weight_row, [0, 0], weight_grad)",
        "pl.store(pl.set_validshape(weight_row, 1, 1), [0, 0], dbe_flat)",
    ])
    header = _KERNEL[:_KERNEL.index("    dbeta_scratch =")]
    header += "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    header += "    for resident_task in pl.parallel(1):\n"
    header += "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"resident_single_token\"):\n"
    return header + "".join(indent + line + "\n" for line in lines) + "    return y, final_state, dq, dk, dv, dg, dbeta, dinitial_state\n"


def _resident_row_output(accumulator, destination, normalized, inverse, name, DK, DKP, L2NORM):
    lines = [
        f"{name} = pl.create_tile([1, DKP], dtype=pl.FP32)",
        f"for {name}_lane in pl.unroll(DKP):",
        f"    pl.tile.write({name}, [0, {name}_lane], pl.read({accumulator}, [{name}_lane, 0]))",
    ]
    if L2NORM and normalized:
        lines.extend(_dot_row(name, normalized, name + "_dot", "").replace("CH", "DKP").splitlines())
        lines.append(f"{name} = pl.mul(pl.sub({name}, pl.mul({normalized}, {name}_dot)), {inverse})")
    if destination != "dg2":
        lines.append(f"{name} = {{YO}}{name}{{YC}}")
    if DK != DKP:
        lines.append(f"{name} = pl.set_validshape({name}, 1, DK)")
    lines.append(f"pl.store({name}, [0, 0], {destination})")
    return lines


def _dependent_vector_layout(load_expr, DK, DKP, layout_keys, L2NORM, packed):
    indent = " " * 24
    lines = [
        "",
        "    for b in pl.parallel(B):",
        "        for h in pl.parallel(H):",
        "            for token_base in pl.parallel(0, T, EPT):",
        '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dependent_vector_gradient_layout"):',
        "                    w8q = pl.create_tile([1, 8], dtype=pl.FP32)",
        "                    w8k = pl.create_tile([1, 8], dtype=pl.FP32)",
        "                    for t in pl.range(token_base, pl.min(token_base + EPT, T)):",
        "                        row = (b * T + t) * H + h",
    ]
    if L2NORM:
        valid = f"[1, {DK}]" if DK != DKP else None
        for source, stem in (("q2", "q"), ("k2", "k")):
            value = load_expr(source, "[row, 0]", "[1, DKP]", valid)
            lines.append(indent + f"{stem}t = {{CO}}{value}{{CC}}")
            lines.append(indent + _norm8(stem + "t", stem + "n", stem + "inv", "w8" + stem, indent))
    if packed:
        value = load_expr("gradient_columns", f"[row * {layout_keys}, 0]",
                          f"[{layout_keys}, 8]", "[DKP, 8]" if DKP != layout_keys else None)
        lines.extend([
            indent + f"gradient_matrix = {value}",
            indent + "gradient_transposed = pl.transpose(gradient_matrix, 0, 1)",
        ])
    for destination, stem, normalized, inverse, scaled in (
        ("dq2", "dq", "qn", "qinv", True),
        ("dk2", "dk", "kn", "kinv", False),
        ("dg2", "dg", None, None, False),
    ):
        if packed:
            lane = {"dq": 0, "dk": 1, "dg": 2}[stem]
            lines.append(indent + f"{stem}_row = pl.tile.slice(gradient_transposed, [1, DKP], [{lane}, 0])")
        else:
            value = load_expr(destination + "_columns", f"[row * {layout_keys}, 0]",
                              f"[{layout_keys}, 8]", f"[DKP, 8]" if DKP != layout_keys else None)
            lines.extend([
                indent + f"{stem}_matrix = {value}",
                indent + f"{stem}_transposed = pl.transpose({stem}_matrix, 0, 1)",
                indent + f"{stem}_row = pl.tile.slice({stem}_transposed, [1, DKP], [0, 0])",
            ])
        if scaled:
            lines.append(indent + f"{stem}_row = pl.mul({stem}_row, SCALE)")
        if L2NORM and normalized:
            lines.extend([
                indent + f"{stem}_parts = pl.col_sum(pl.reshape(pl.mul({stem}_row, {normalized}), [DKP // 8, 8]))",
                indent + f"{stem}_dot = " + " + ".join(f"pl.read({stem}_parts, [0, {lane}])" for lane in range(8)),
                indent + f"{stem}_row = pl.mul(pl.sub({stem}_row, pl.mul({normalized}, {stem}_dot)), {inverse})",
            ])
        if destination != "dg2":
            lines.append(indent + f"{stem}_row = {{YO}}{stem}_row{{YC}}")
        if DK != DKP:
            lines.append(indent + f"{stem}_row = pl.tile.slice({stem}_row, [1, DKP], [0, 0], valid_shape=[1, {DK}])")
        lines.append(indent + f"pl.store({stem}_row, [row, 0], {destination})")
    return _isolate_region_variables("\n".join(lines) + "\n", "layout_")


def _phase_isolate_gradients(source, allocations, load_expr, DK, DKP, L2NORM, schedule, packed):
    gradient_start = source.index("    for fused_b in pl.parallel(B):")
    thin, gradient = source[:gradient_start], source[gradient_start:]
    if schedule:
        inner = " " * 28
        early = (inner + "fused_pdq0 = pl.row_sum(pl.col_expand_mul(fused_H1c0, fused_dyc0), fused_dq_scratch)\n"
                 + inner + "fused_dqacc = pl.add(fused_dqacc, fused_pdq0)\n")
        gradient = _replace_once(gradient, early, "")
        if schedule == 1:
            previous_line = next(line for line in gradient.splitlines(True)
                                 if line.lstrip().startswith("fused_H1c0 ="))
            gradient = _replace_once(gradient, previous_line, previous_line + early)
        else:
            for prefix in ("fused_dq_scratch =", "fused_dq_zero_columns =", "fused_dqacc =",
                           "fused_dyc0 =", "fused_H1c0 ="):
                gradient = _replace_line(gradient, prefix, "")
            start = gradient.index("                        fused_dq_row =")
            end = gradient.index("                        fused_dk_row =", start)
            gradient = gradient[:start] + gradient[end:]
            if L2NORM:
                qnorm_start = gradient.index("                        fused_sq = pl.mul(fused_qt, fused_qt)")
                knorm_start = gradient.index("                        fused_sq = pl.mul(fused_kt, fused_kt)", qnorm_start)
                gradient = gradient[:qnorm_start] + gradient[knorm_start:]
            else:
                gradient = _replace_line(gradient, "fused_qn2 =", "")
            for prefix in ("fused_qt =", "fused_w8q =", "fused_qcol ="):
                gradient = _replace_line(gradient, prefix, "")
            gradient += _independent_complete_dq(load_expr, DK, DKP, L2NORM)
    layout = ""
    if packed:
        start = gradient.index("                        fused_dq_row =")
        end = gradient.index("                        pl.write(dbeta_scratch", start)
        layout_keys = ((DKP + 15) // 16) * 16
        lines = ["layout_carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)"]
        for lane, stem in enumerate(("dq", "dk", "dg")):
            lines.extend([
                f"layout_{stem}_mask = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)",
                f"pl.tile.write(layout_{stem}_mask, [0, {lane}], 1.0)",
                f"layout_{stem}_carrier = pl.col_expand_mul(layout_carrier, layout_{stem}_mask)",
                f"packed_{stem} = pl.row_expand_mul(layout_{stem}_carrier, fused_{stem}acc)",
            ])
        lines.extend([
            "packed_gradients = pl.add(pl.add(packed_dq, packed_dk), packed_dg)",
            f"pl.store(packed_gradients, [fused_row * {layout_keys}, 0], gradient_columns)",
        ])
        gradient = gradient[:start] + "".join(" " * 24 + line + "\n" for line in lines) + gradient[end:]
        allocations += f"    gradient_columns = pl.create_tensor([ROWS * {layout_keys}, 8], dtype=pl.FP32)\n"
        layout = _dependent_vector_layout(load_expr, DK, DKP, layout_keys, L2NORM, True)
    return thin + gradient + layout, allocations


def _independent_complete_dq(load_expr, DK, DKP, L2NORM):
    indent = " " * 24
    valid = f"[1, {DK}]" if DK != DKP else None
    qload = load_expr("q2", "[row, 0]", "[1, DKP]", valid)
    hload = load_expr("hs2", "[((b * TP1 + t + 1) * H + h) * DKP, cb]",
                      "[DKP, GCH]", "[DKP, pl.min(GCH, DVP - cb)]")
    dyload = load_expr("dy2", "[row, cb]", "[1, GCH]", "[1, pl.min(GCH, DV - cb)]")
    norm = _norm8("qt", "qn2", "qinv2", "w8q", indent) if L2NORM else "qn2 = qt"
    output = _resident_row_output("dquery", "dq2", "qn2", "qinv2", "dq_row", DK, DKP, L2NORM)
    output = ("\n" + indent).join(output).replace("[0, 0], dq2)", "[row, 0], dq2)")
    stage = """
    for b in pl.parallel(B):
        for h in pl.parallel(H):
            for token_base in pl.parallel(0, T, EPT):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="independent_complete_dq"):
                    w8q = pl.create_tile([1, 8], dtype=pl.FP32)
                    dq_scratch = pl.create_tile([DKP, GCH], dtype=pl.FP32)
                    for t in pl.range(token_base, pl.min(token_base + EPT, T)):
                        row = (b * T + t) * H + h
                        qt = {CO}{QLOAD}{CC}
                        {QNORM}
                        dq_zero_columns = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)
                        dqacc = pl.row_sum(dq_zero_columns, dq_scratch)
                        for part in pl.range(GNC):
                            cb = part * GCH
                            next_state = {HLOAD}
                            output_grad = {CO}{DYLOAD}{CC}
                            dq_partial = pl.row_sum(pl.col_expand_mul(next_state, output_grad), dq_scratch)
                            dqacc = pl.add(dqacc, dq_partial)
                        dquery = pl.mul(dqacc, SCALE)
                        {OUTPUT}
"""
    for name, value in {"QLOAD": qload, "QNORM": norm, "HLOAD": hload,
                        "DYLOAD": dyload, "OUTPUT": output}.items():
        stage = stage.replace("{" + name + "}", value)
    return _isolate_region_variables(stage, "dqphase_")


def _sharded_single_source(load_expr, DK, DV, DKP, DVP, L2NORM):
    layout_keys = ((DKP + 15) // 16) * 16
    valid = f"[1, {DK}]" if DK != DKP else None
    row_load = lambda tensor: load_expr(tensor, "[bh, 0]", "[1, DKP]", valid)
    state_load = lambda tensor: load_expr(tensor, "[bh * DK, cb]", "[DKP, SCH]",
                                          "[DK, pl.min(SCH, DV - cb)]")
    value_load = lambda tensor: load_expr(tensor, "[bh, cb]", "[1, SCH]",
                                          "[1, pl.min(SCH, DV - cb)]")
    lines = [
        "bh = owner // SNC", "part = owner % SNC", "b = bh // H", "h = bh % H", "cb = part * SCH",
        f"partial_row = owner * {layout_keys}",
        "qraw = {CO}" + row_load("q2") + "{CC}",
        "kraw = {CO}" + row_load("k2") + "{CC}",
        "qnorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
        "knorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
    ]
    if L2NORM:
        lines.extend(_norm8("qraw", "qn", "qinv", "qnorm_scratch", "").splitlines())
        lines.extend(_norm8("kraw", "kn", "kinv", "knorm_scratch", "").splitlines())
    else:
        lines.extend(["qn = qraw", "kn = kraw"])
    lines.extend([
        "qcol = pl.reshape(qn, [DKP, 1])", "kcol = pl.reshape(kn, [DKP, 1])",
        "decay = pl.exp(pl.reshape(" + row_load("g2") + ", [DKP, 1]))",
        "previous_state = " + state_load("s2"),
        "pl.store(pl.set_validshape(previous_state, DKP, pl.min(SCH, DVP - cb)), [(b * TP1 * H + h) * DKP, cb], hs2)",
        "decayed_state = pl.row_expand_mul(previous_state, decay)",
        "key_scratch = pl.create_tile([DKP, SCH], dtype=pl.FP32)",
        "key_state = pl.col_sum(pl.row_expand_mul(decayed_state, kcol), key_scratch)",
        "value = {CO}" + value_load("v2") + "{CC}",
        "residual = pl.sub(value, key_state)", "weight = pl.read(b1, [bh, 0])",
        "update = pl.row_expand_mul(pl.col_expand(decayed_state, residual), pl.mul(kcol, weight))",
        "updated_state = pl.add(decayed_state, update)",
        "out_y = pl.mul(pl.col_sum(pl.row_expand_mul(updated_state, qcol), key_scratch), SCALE)",
        "out_y = {YO}out_y{YC}",
        "pl.store(pl.set_validshape(out_y, 1, pl.min(SCH, DV - cb)), [bh, cb], y2)",
        "pl.store(pl.set_validshape(updated_state, DKP, pl.min(SCH, DVP - cb)), [((b * TP1 + 1) * H + h) * DKP, cb], hs2)",
        "pl.store(pl.set_validshape(updated_state, DK, pl.min(SCH, DV - cb)), [bh * DK, cb], f2)",
        "output_grad = {CO}" + value_load("dy2") + "{CC}",
        "carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)",
        "incoming = " + state_load("df2"),
        "injection = pl.row_expand_mul(pl.col_expand(incoming, output_grad), pl.mul(qcol, SCALE))",
        "injected = pl.add(incoming, injection)",
        "contraction = pl.col_sum(pl.row_expand_mul(injected, kcol), key_scratch)",
        "dresidual = pl.mul(contraction, weight)", "out_dv = {YO}dresidual{YC}",
        "pl.store(pl.set_validshape(out_dv, 1, pl.min(SCH, DV - cb)), [bh, cb], dv2)",
        "dk_scratch = pl.create_tile([DKP, SCH], dtype=pl.FP32)",
        "key_first = pl.row_sum(pl.col_expand_mul(injected, residual), dk_scratch)",
        "key_second = pl.mul(pl.row_sum(pl.col_expand_mul(previous_state, dresidual), dk_scratch), decay)",
        "dkey = pl.sub(pl.mul(key_first, weight), key_second)",
        "pl.store(pl.row_expand_mul(carrier, dkey), [partial_row, 0], single_partial_k)",
        "corrected = pl.sub(injected, pl.row_expand_mul(pl.col_expand(injected, dresidual), kcol))",
        "initial_grad = pl.row_expand_mul(corrected, decay)",
        "pl.store(pl.set_validshape(initial_grad, DK, pl.min(SCH, DV - cb)), [bh * DK, cb], dh2)",
        "pl.store(pl.set_validshape(initial_grad, DK, pl.min(SCH, DV - cb)), [bh * DK, cb], di2)",
        "dg_scratch = pl.create_tile([DKP, SCH], dtype=pl.FP32)",
        "dgate = pl.row_sum(pl.mul(initial_grad, previous_state), dg_scratch)",
        "pl.store(pl.row_expand_mul(carrier, dgate), [partial_row, 0], single_partial_g)",
    ])
    lines.extend(_dot_row("contraction", "residual", "weight_grad", "").replace("CH", "SCH").splitlines())
    lines.append("pl.write(single_partial_beta, [owner, 0], weight_grad)")
    header = _KERNEL[:_KERNEL.index("    dbeta_scratch =")]
    header += "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    header += "    dbeta_scratch = pl.create_tensor([B * H, 16], dtype=pl.FP32)\n"
    for stem in ("k", "g"):
        header += f"    single_partial_{stem} = pl.create_tensor([B * H * SNC * {layout_keys}, 8], dtype=pl.FP32)\n"
    header += "    single_partial_beta = pl.create_tensor([B * H * SNC, 16], dtype=pl.FP32)\n"
    header += '    for owner in pl.spmd(B * H * SNC, name_hint="sharded_single_token"):\n'
    producer = header + "".join(" " * 8 + line + "\n" for line in lines)
    consumer = """
    for bh in pl.parallel(B * H):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="complete_single_gradients"):
            w8q = pl.create_tile([1, 8], dtype=pl.FP32)
            w8k = pl.create_tile([1, 8], dtype=pl.FP32)
"""
    lines = []
    if L2NORM:
        for tensor, stem in (("k2", "k"),):
            lines.append(f"{stem}raw = {{CO}}" + row_load(tensor) + "{CC}")
            lines.extend(_norm8(stem + "raw", stem + "n", stem + "inv", "w8" + stem, "").splitlines())
    lines.extend([
        f"ksum = pl.tile.full([{layout_keys}, 8], dtype=pl.FP32, value=0.0)",
        f"gsum = pl.tile.full([{layout_keys}, 8], dtype=pl.FP32, value=0.0)",
        "beta_sum = 0.0", "for part in pl.range(SNC):",
        f"    partial_row = (bh * SNC + part) * {layout_keys}",
        "    ksum = pl.add(ksum, " + load_expr("single_partial_k", "[partial_row, 0]", f"[{layout_keys}, 8]", "[DKP, 8]" if DKP != layout_keys else None) + ")",
        "    gsum = pl.add(gsum, " + load_expr("single_partial_g", "[partial_row, 0]", f"[{layout_keys}, 8]", "[DKP, 8]" if DKP != layout_keys else None) + ")",
        "    beta_sum = beta_sum + pl.read(single_partial_beta, [bh * SNC + part, 0])",
    ])
    for stem, dest, norm, inv, scaled in (("k", "dk2", "kn", "kinv", False),
                                         ("g", "dg2", None, None, False)):
        if DKP % 16 == 0:
            lines.extend([
                f"{stem}_transposed = pl.transpose({stem}sum, 0, 1)",
                f"{stem}_row = pl.tile.slice({stem}_transposed, [1, DKP], [0, 0])",
            ])
        else:
            lines.extend([
                f"{stem}_row = pl.create_tile([1, DKP], dtype=pl.FP32)",
                f"for {stem}_lane in pl.unroll(DKP):",
                f"    pl.tile.write({stem}_row, [0, {stem}_lane], pl.read({stem}sum, [{stem}_lane, 0]))",
            ])
        if scaled:
            lines.append(f"{stem}_row = pl.mul({stem}_row, SCALE)")
        if L2NORM and norm:
            lines.extend(_dot_row(stem + "_row", norm, stem + "_dot", "").replace("CH", "DKP").splitlines())
            lines.append(f"{stem}_row = pl.mul(pl.sub({stem}_row, pl.mul({norm}, {stem}_dot)), {inv})")
        if dest != "dg2":
            lines.append(f"{stem}_row = {{YO}}{stem}_row{{YC}}")
        if DK != DKP:
            lines.append(f"{stem}_row = pl.tile.slice({stem}_row, [1, DKP], [0, 0], valid_shape=[1, DK])")
        lines.append(f"pl.store({stem}_row, [bh, 0], {dest})")
    lines.append("pl.write(dbeta_scratch, [bh, 0], beta_sum)")
    consumer += "".join(" " * 12 + line + "\n" for line in lines)
    compact = _KERNEL[_KERNEL.index("    for row_base in pl.parallel"):]
    dependent = _independent_complete_dq(load_expr, DK, DKP, L2NORM)
    return producer + _isolate_region_variables(consumer, "single_reduce_") + dependent + compact


def _bound_adjoint_tape(source, allocations):
    dq_start = source.find("    for dqphase_b in pl.parallel(B):")
    independent = source[dq_start:] if dq_start >= 0 else ""
    grouped = source[:dq_start] if dq_start >= 0 else source
    tape_allocations = []
    remaining = []
    for line in allocations.splitlines():
        if line.strip().startswith("adjoint_tape ="):
            tape_allocations.append(line.replace("B * H * TP1", "AHEADS * TP1"))
        elif line.strip().startswith("contraction_tape ="):
            tape_allocations.append(line.replace("[ROWS,", "[AHEADS * T,"))
        else:
            remaining.append(line)
    assert len(tape_allocations) == 2
    grouped = _replace_once(grouped, "for owner in pl.parallel(B * H * NC):", "for owner in pl.parallel(AHEADS * NC):")
    grouped = _replace_once(grouped, "bh = owner // NC", "bh = head_base + owner // NC")
    grouped = grouped.replace("(bh * TP1 +", "((bh - head_base) * TP1 +")
    grouped = _replace_once(grouped, "pl.store(ccv0, [row, cb], contraction_tape)", "pl.store(ccv0, [(bh - head_base) * T + t, cb], contraction_tape)")
    header = "    for fused_b in pl.parallel(B):\n        for fused_h in pl.parallel(H):\n"
    assert grouped.count(header) == 1
    start = grouped.index(header)
    gradient = grouped[start + len(header):]
    gradient = "\n".join(line[4:] if line.startswith("    ") else line for line in gradient.splitlines()) + "\n"
    marker = "                fused_bh = fused_b * H + fused_h\n"
    replacement = ("                fused_bh = head_base + fused_head\n"
                   "                fused_b = fused_bh // H\n"
                   "                fused_h = fused_bh % H\n")
    gradient = _replace_once(gradient, marker, replacement)
    gradient = gradient.replace("(fused_bh * TP1 +", "((fused_bh - head_base) * TP1 +")
    gradient = _replace_once(gradient, "pl.load(contraction_tape, [fused_row, fused_cb]", "pl.load(contraction_tape, [(fused_bh - head_base) * T + fused_t, fused_cb]")
    grouped = grouped[:start] + "    for fused_head in pl.parallel(AHEADS):\n" + gradient
    body = "\n".join(tape_allocations) + "\n" + grouped
    body = "\n".join("    " + line if line else "" for line in body.splitlines()) + "\n"
    return "    for head_base in pl.parallel(0, B * H, AHEADS):\n" + body + independent, "\n".join(remaining) + "\n"


def _short_resident_source(load_expr, DK, DV, DKP, DVP, L2NORM, tokens):
    def row_output(accumulator, destination, normalized, inverse, name, DK, DKP, L2NORM):
        output = [f"{name} = {accumulator}"]
        if L2NORM and normalized:
            output.extend(_dot_row(name, normalized, name + "_dot", "").replace("CH", "DKP").splitlines())
            output.append(f"{name} = pl.mul(pl.sub({name}, pl.mul({normalized}, {name}_dot)), {inverse})")
        if destination != "dg2":
            output.append(f"{name} = {{YO}}{name}{{YC}}")
        if DK != DKP:
            output.append(f"{name} = pl.set_validshape({name}, 1, DK)")
        output.append(f"pl.store({name}, [0, 0], {destination})")
        return output
    key_valid = f"[1, {DK}]" if DK != DKP else None
    key_load = lambda tensor: load_expr(tensor, "[row, 0]", "[1, DKP]", key_valid)
    value_load = lambda tensor: load_expr(tensor, "[row, 0]", "[1, DVP]", None if DV == DVP else f"[1, {DV}]")
    state_load = lambda tensor: load_expr(tensor, "[head * DK, 0]", "[DKP, DVP]", None if (DK, DV) == (DKP, DVP) else f"[{DK}, {DV}]")
    lines = [
        "batch = head // H",
        "head_index = head % H",
        "initial = " + state_load("s2"),
        "tape_seed = pl.tile.full([TP1 * DKP, DVP], dtype=pl.FP32, value=0.0)",
        "tape_initial = pl.tile.assemble(tape_seed, initial, [0, 0])",
        "pl.store(initial, [(batch * TP1 * H + head_index) * DKP, 0], hs2)",
        "qnorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
        "knorm_scratch = pl.create_tile([1, 8], dtype=pl.FP32)",
        "for token, (state_carry, tape_carry) in pl.range(T, init_values=(initial, tape_initial)):",
        "    row = (batch * T + token) * H + head_index",
        "    qraw = {CO}" + key_load("q2") + "{CC}",
        "    kraw = {CO}" + key_load("k2") + "{CC}",
    ]
    if L2NORM:
        lines.extend("    " + line for line in _norm8("qraw", "qn", "qinv", "qnorm_scratch", "").splitlines())
        lines.extend("    " + line for line in _norm8("kraw", "kn", "kinv", "knorm_scratch", "").splitlines())
    else:
        lines.extend(["    qn = qraw", "    kn = kraw"])
    lines.extend([
        "    qcol = pl.reshape(qn, [DKP, 1])",
        "    kcol = pl.reshape(kn, [DKP, 1])",
        "    decay_row = pl.exp(" + key_load("g2") + ")",
        "    decay = pl.reshape(decay_row, [DKP, 1])",
        "    decayed = pl.row_expand_mul(state_carry, decay)",
        "    value = {CO}" + value_load("v2") + "{CC}",
        "    residual = pl.sub(value, pl.col_sum(pl.row_expand_mul(decayed, kcol)))",
        "    weight = pl.read(b1, [row, 0])",
        "    updated_state = pl.add(decayed, pl.row_expand_mul(pl.col_expand(decayed, residual), pl.mul(kcol, weight)))",
        "    tape_next = pl.tile.assemble(tape_carry, updated_state, [(token + 1) * DKP, 0])",
        "    pl.store(updated_state, [((batch * TP1 + token + 1) * H + head_index) * DKP, 0], hs2)",
        "    out_y = pl.mul(pl.col_sum(pl.row_expand_mul(updated_state, qcol)), SCALE)",
        "    out_y = {YO}out_y{YC}",
        "    pl.store(pl.set_validshape(out_y, 1, DV), [row, 0], y2)",
        "    final_replayed, complete_tape = pl.yield_(updated_state, tape_next)",
        "pl.store(pl.set_validshape(final_replayed, DK, DV), [head * DK, 0], f2)",
        "incoming = " + state_load("df2"),
        "dq_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "dk_first_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "dk_second_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "dg_scratch = pl.create_tile([DKP, DVP], dtype=pl.FP32)",
        "for reverse_token, (adjoint,) in pl.range(T, init_values=(incoming,)):",
        "    token = T - 1 - reverse_token",
        "    row = (batch * T + token) * H + head_index",
        "    qraw = {CO}" + key_load("q2") + "{CC}",
        "    kraw = {CO}" + key_load("k2") + "{CC}",
    ])
    if L2NORM:
        lines.extend("    " + line for line in _norm8("qraw", "qn", "qinv", "qnorm_scratch", "").splitlines())
        lines.extend("    " + line for line in _norm8("kraw", "kn", "kinv", "knorm_scratch", "").splitlines())
    else:
        lines.extend(["    qn = qraw", "    kn = kraw"])
    lines.extend([
        "    qcol = pl.reshape(qn, [DKP, 1])",
        "    kcol = pl.reshape(kn, [DKP, 1])",
        "    decay = pl.exp(pl.reshape(" + key_load("g2") + ", [DKP, 1]))",
        "    previous = pl.tile.slice(complete_tape, [DKP, DVP], [token * DKP, 0])",
        "    updated = pl.tile.slice(complete_tape, [DKP, DVP], [(token + 1) * DKP, 0])",
        "    output_grad = {CO}" + value_load("dy2") + "{CC}",
        "    dquery = pl.mul(pl.col_sum(pl.transpose(pl.col_expand_mul(updated, output_grad), 0, 1)), SCALE)",
    ])
    lines.extend("    " + line.replace("[0, 0], dq2)", "[row, 0], dq2)") for line in
                 row_output("dquery", "dq2", "qn", "qinv", "dq_row", DK, DKP, L2NORM))
    lines.extend([
        "    injected = pl.add(adjoint, pl.row_expand_mul(pl.col_expand(adjoint, output_grad), pl.mul(qcol, SCALE)))",
        "    contraction = pl.col_sum(pl.row_expand_mul(injected, kcol))",
        "    weight = pl.read(b1, [row, 0])",
        "    dresidual = pl.mul(contraction, weight)",
        "    out_dv = {YO}dresidual{YC}",
        "    pl.store(pl.set_validshape(out_dv, 1, DV), [row, 0], dv2)",
        "    decayed = pl.row_expand_mul(previous, decay)",
        "    value = {CO}" + value_load("v2") + "{CC}",
        "    residual = pl.sub(value, pl.col_sum(pl.row_expand_mul(decayed, kcol)))",
    ])
    lines.extend("    " + line for line in _dot_row("contraction", "residual", "weight_grad", "").replace("CH", "DVP").splitlines())
    lines.extend([
        "    pl.write(dbeta_scratch, [row, 0], weight_grad)",
        "    key_first = pl.col_sum(pl.transpose(pl.col_expand_mul(injected, residual), 0, 1))",
        "    key_second = pl.col_sum(pl.transpose(pl.col_expand_mul(decayed, dresidual), 0, 1))",
        "    dkey = pl.sub(pl.mul(key_first, weight), key_second)",
    ])
    lines.extend("    " + line.replace("[0, 0], dk2)", "[row, 0], dk2)") for line in
                 row_output("dkey", "dk2", "kn", "kinv", "dk_row", DK, DKP, L2NORM))
    lines.extend([
        "    corrected = pl.sub(injected, pl.row_expand_mul(pl.col_expand(injected, dresidual), kcol))",
        "    dgate = pl.mul(pl.col_sum(pl.transpose(pl.mul(corrected, previous), 0, 1)), decay_row)",
    ])
    lines.extend("    " + line.replace("[0, 0], dg2)", "[row, 0], dg2)") for line in
                 row_output("dgate", "dg2", None, None, "dg_row", DK, DKP, L2NORM))
    lines.extend([
        "    next_adjoint = pl.row_expand_mul(corrected, decay)",
        "    final_grad = pl.yield_(next_adjoint)",
        "pl.store(pl.set_validshape(final_grad, DK, DV), [head * DK, 0], dh2)",
        "pl.store(pl.set_validshape(final_grad, DK, DV), [head * DK, 0], di2)",
    ])
    import re

    forward_start = next(index for index, line in enumerate(lines) if line.startswith("for token,"))
    forward_end = next(index for index, line in enumerate(lines) if "final_replayed, complete_tape =" in line)
    reverse_start = next(index for index, line in enumerate(lines) if line.startswith("for reverse_token,"))
    reverse_end = next(index for index, line in enumerate(lines) if "final_grad = pl.yield_" in line)
    body = [line for line in lines[:forward_start] if not line.startswith(("tape_seed =", "tape_initial ="))]
    body.append("state_0 = initial")

    def specialize(block, replacements, prefix):
        assigned = set()
        for line in block:
            match = re.match(r"\s*(\w+) =", line)
            if match:
                assigned.add(match.group(1))
            match = re.match(r"\s*for (\w+) in", line)
            if match:
                assigned.add(match.group(1))
        names = {name: prefix + name for name in assigned}
        names.update(replacements)
        return [re.sub(r"\b\w+\b", lambda match: names.get(match.group(), match.group()), line) for line in block]

    forward = [line[4:] for line in lines[forward_start + 1:forward_end] if "tape_next =" not in line]
    for token in range(tokens):
        body.extend(specialize(forward, {"token": str(token), "state_carry": f"state_{token}",
                                        "updated_state": f"state_{token + 1}"}, f"forward_{token}_"))
    body.append(f"pl.store(pl.set_validshape(state_{tokens}, DK, DV), [head * DK, 0], f2)")
    body.extend(line for line in lines[forward_end + 2:reverse_start] if "_scratch =" not in line)
    body.append(f"adjoint_{tokens} = incoming")
    reverse = [line[4:] for line in lines[reverse_start + 2:reverse_end]]
    output_load = next(index for index, line in enumerate(reverse) if line.startswith("output_grad ="))
    reverse = reverse[:1] + reverse[output_load:]
    reverse = [line for line in reverse if not line.startswith(("previous =", "updated =", "weight =", "value =", "residual ="))]
    for token in range(tokens - 1, -1, -1):
        retained = {name: f"forward_{token}_{name}" for name in
                    ("qn", "kn", "qinv", "kinv", "qcol", "kcol", "decay", "decay_row", "weight", "residual")}
        body.extend(specialize(reverse, retained | {"token": str(token), "previous": f"state_{token}",
                                                   "updated": f"state_{token + 1}", "adjoint": f"adjoint_{token + 1}",
                                                   "next_adjoint": f"adjoint_{token}"}, f"reverse_{token}_"))
    body.extend(line.replace("final_grad", "adjoint_0") for line in lines[reverse_end + 1:])
    header = _KERNEL[:_KERNEL.index("    di2 =")]
    header += "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    header += "    for head in pl.parallel(B * H):\n"
    header += '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="short_resident_backward"):\n'
    compact = _KERNEL[_KERNEL.index("    for row_base in pl.parallel"):]
    return header + "".join(" " * 12 + line + "\n" for line in body) + compact


def _redispatch_narrow_gradients(source, batches, tokens, heads, group_size, mode):
    header = (
        "    for fused_b in pl.parallel(B):\n"
        "        for fused_h in pl.parallel(H):\n"
        "            for fused_token_base in pl.parallel(0, T, EPT):\n"
        '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_token_gradients"):\n'
    )
    if source.count(header) != 1:
        raise ValueError("Expected exactly one grouped narrow gradient producer")
    start = source.index(header)
    body_start = start + len(header)
    lines = source[body_start:].splitlines(True)
    body_end = 0
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip()) < 20:
            break
        body_end += len(line)
    body = source[body_start:body_start + body_end]
    groups_per_head = (tokens + group_size - 1) // group_size
    total_groups = batches * heads * groups_per_head
    if mode == 1:
        launch = (
            f"    for gradient_block in pl.parallel({total_groups}):\n"
            '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_token_gradients"):\n'
        )
        indentation = " " * 12
        remove_spaces = 8
    elif mode == 2:
        launch = f'    for gradient_block in pl.spmd({total_groups}, name_hint="fused_token_gradients"):\n'
        indentation = " " * 8
        remove_spaces = 12
    else:
        raise ValueError(f"Unsupported gradient dispatch mode: {mode}")
    decode = [
        f"gradient_head_owner = gradient_block // {groups_per_head}",
        "fused_b = gradient_head_owner // H",
        "fused_h = gradient_head_owner % H",
        f"fused_token_base = (gradient_block % {groups_per_head}) * EPT",
    ]
    replacement = launch + "".join(indentation + line + "\n" for line in decode)
    replacement += "".join(line[remove_spaces:] if line.strip() else line for line in body.splitlines(True))
    return source[:start] + replacement + source[body_start + body_end:]


def _batched_token_prepare(source, rows, keys, padded_keys, group_rows, cast_inputs):
    import re

    fragments = source.splitlines(keepends=True)
    output = []
    replaced = {"qn1": 0, "kn1": 0, "qn2": 0, "kn2": 0,
                "fused_qn2": 0, "fused_kn2": 0}
    if 'name_hint="dependent_vector_gradient_layout"' in source:
        replaced.update(layout_qn=0, layout_kn=0)
    position = 0
    while position < len(fragments):
        line = fragments[position]
        stripped = line.strip()
        if stripped.startswith(("qt = ", "kt = ", "fused_qt = ", "fused_kt = ", "layout_qt = ", "layout_kt = ")):
            position += 1
            continue
        if stripped.startswith(("w8 = pl.create_tile", "w8b = pl.create_tile",
                                "fused_w8q = pl.create_tile", "fused_w8k = pl.create_tile",
                                "layout_w8q = pl.create_tile", "layout_w8k = pl.create_tile")):
            position += 1
            continue
        if stripped.startswith(("sq = pl.mul(", "fused_sq = pl.mul(", "layout_sq = pl.mul(")):
            block = fragments[position:position + 5]
            if len(block) != 5:
                raise ValueError("Incomplete normalization block")
            match = re.fullmatch(r"(\s*)(\w+) = pl.mul\((\w+), (\w+)\)\n", block[-1])
            if match is None or match.group(2) not in replaced:
                raise ValueError("Normalization block changed: " + "".join(block))
            indent, normalized, raw, inverse = match.groups()
            query = normalized in ("qn1", "qn2", "fused_qn2", "layout_qn")
            tensor = "prepared_query" if query else "prepared_key"
            row = "fused_row" if normalized.startswith("fused_") else "layout_row" if normalized.startswith("layout_") else "row"
            output.append(f"{indent}{normalized} = pl.load({tensor}, [{row}, 0], [1, DKP])\n")
            if normalized.startswith(("fused_", "layout_")):
                inverse_tensor = "prepared_query_inverse" if query else "prepared_key_inverse"
                output.append(f"{indent}{inverse} = pl.read({inverse_tensor}, [{row}, 0])\n")
            replaced[normalized] += 1
            position += 5
            continue
        if stripped.startswith("gt = pl.reshape(") and "pl.load(g2," in stripped:
            if fragments[position + 1].strip() != "decay = pl.exp(gt)":
                raise ValueError("Replay decay changed")
            indent = line[:len(line) - len(line.lstrip())]
            output.append(indent + "decay = pl.reshape(pl.load(prepared_decay, [row, 0], [1, DKP]), [DKP, 1])\n")
            position += 2
            continue
        if stripped.startswith("egcol = pl.exp(pl.reshape(") and "pl.load(g2," in stripped:
            indent = line[:len(line) - len(line.lstrip())]
            output.append(indent + "egcol = pl.reshape(pl.load(prepared_decay, [row, 0], [1, DKP]), [DKP, 1])\n")
            position += 1
            continue
        if stripped.startswith("fused_gt = pl.reshape(") and "pl.load(g2," in stripped:
            if fragments[position + 1].strip() != "fused_egcol = pl.exp(fused_gt)":
                raise ValueError("Local gradient decay changed")
            indent = line[:len(line) - len(line.lstrip())]
            output.append(indent + "fused_egcol = pl.reshape(pl.load(prepared_decay, [fused_row, 0], [1, DKP]), [DKP, 1])\n")
            position += 2
            continue
        output.append(line)
        position += 1
    if any(count != 1 for count in replaced.values()):
        raise ValueError(f"Expected exactly one replacement per normalization block: {replaced}")
    source = "".join(output)
    initialization = "        fused_zero_columns = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)\n"
    token_loop = "        for fused_t in pl.range(fused_token_base, pl.min(fused_token_base + EPT, T)):\n"
    if source.count(initialization) != 1 or source.count(token_loop) != 1:
        raise ValueError("Narrow producer initialization changed")
    source = source.replace(initialization, "").replace(token_loop, token_loop + "    " + initialization)
    for raw in ("qt", "kt", "fused_qt", "fused_kt", "layout_qt", "layout_kt"):
        if re.search(r"\b" + raw + r"\b", source):
            raise ValueError(f"Unreplaced raw input use: {raw}")
    query_load = f"pl.set_validshape(pl.fillpad(pl.load(q2, [preparation_base, 0], [{group_rows}, DKP], valid_shape=[preparation_valid, DK]), 0), {group_rows}, DKP)"
    key_load = query_load.replace("pl.load(q2,", "pl.load(k2,")
    if cast_inputs:
        query_load = f"pl.cast({query_load}, target_type=pl.FP32)"
        key_load = f"pl.cast({key_load}, target_type=pl.FP32)"
    producer = f'''
    prepared_query = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_key = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_decay = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_query_inverse = pl.create_tensor([ROWS, 8], dtype=pl.FP32)
    prepared_key_inverse = pl.create_tensor([ROWS, 8], dtype=pl.FP32)
    for preparation_block in pl.spmd({(rows + group_rows - 1) // group_rows}, name_hint="batched_token_prepare"):
        preparation_base = preparation_block * {group_rows}
        preparation_valid = pl.min({group_rows}, ROWS - preparation_base)
        preparation_scratch = pl.create_tile([{group_rows}, DKP], dtype=pl.FP32)
        preparation_ones = pl.tile.full([1, {group_rows}], dtype=pl.FP32, value=1.0)
        preparation_meta = pl.tile.full([{group_rows}, 8], dtype=pl.FP32, value=1.0)
        preparation_query = {query_load}
        preparation_query_sum = pl.reshape(pl.row_sum(pl.mul(preparation_query, preparation_query), preparation_scratch), [1, {group_rows}])
        preparation_query_norm = pl.sqrt(pl.add(preparation_query_sum, 1e-6))
        preparation_query_inverse = pl.reshape(pl.div(preparation_ones, preparation_query_norm), [{group_rows}, 1])
        preparation_query = pl.row_expand_mul(preparation_query, preparation_query_inverse)
        pl.store(pl.set_validshape(preparation_query, preparation_valid, DKP), [preparation_base, 0], prepared_query)
        preparation_query_meta = pl.row_expand_mul(preparation_meta, preparation_query_inverse)
        pl.store(pl.set_validshape(preparation_query_meta, preparation_valid, 8), [preparation_base, 0], prepared_query_inverse)
        preparation_key = {key_load}
        preparation_key_sum = pl.reshape(pl.row_sum(pl.mul(preparation_key, preparation_key), preparation_scratch), [1, {group_rows}])
        preparation_key_norm = pl.sqrt(pl.add(preparation_key_sum, 1e-6))
        preparation_key_inverse = pl.reshape(pl.div(preparation_ones, preparation_key_norm), [{group_rows}, 1])
        preparation_key = pl.row_expand_mul(preparation_key, preparation_key_inverse)
        pl.store(pl.set_validshape(preparation_key, preparation_valid, DKP), [preparation_base, 0], prepared_key)
        preparation_key_meta = pl.row_expand_mul(preparation_meta, preparation_key_inverse)
        pl.store(pl.set_validshape(preparation_key_meta, preparation_valid, 8), [preparation_base, 0], prepared_key_inverse)
        preparation_gate = pl.set_validshape(pl.fillpad(pl.load(g2, [preparation_base, 0], [{group_rows}, DKP], valid_shape=[preparation_valid, DK]), 0), {group_rows}, DKP)
        preparation_decay = pl.exp(preparation_gate)
        pl.store(pl.set_validshape(preparation_decay, preparation_valid, DKP), [preparation_base, 0], prepared_decay)
'''
    marker = "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    if source.count(marker) != 1:
        raise ValueError("Selected allocation boundary changed")
    return source.replace(marker, producer + marker)


import ast
import re


def _wide_replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one wide selected-source match: {old[:100]!r}")
    return source.replace(old, new, 1)


def _wide_region(source, name):
    matches = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and any(
                keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == name for keyword in call.keywords
            ):
                matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"Expected one wide region {name}: {len(matches)}")
    node = matches[0]
    lines = source.splitlines(True)
    return sum(map(len, lines[:node.lineno - 1])), sum(map(len, lines[:node.end_lineno]))


def _wide_consumer(region, expected, row, decay_kind):
    fragments = region.splitlines(True)
    result = []
    replaced = {name: 0 for name in expected}
    raw_names = {item[0] for item in expected.values()}
    scratch_names = {item[2] for item in expected.values()}
    position = 0
    decay_replacements = 0
    while position < len(fragments):
        line = fragments[position]
        stripped = line.strip()
        target = stripped.split(" = ", 1)[0]
        if target in raw_names or target in scratch_names:
            position += 1
            continue
        if re.fullmatch(r"\w+_?sq = pl.mul\(.*", stripped) or stripped.startswith("sq = pl.mul("):
            block = fragments[position:position + 5]
            match = re.fullmatch(r"(\s*)(\w+) = pl.mul\((\w+), (\w+)\)\n", block[-1])
            if match is None or match.group(2) not in expected:
                raise ValueError("Unexpected wide normalization: " + "".join(block))
            indent, normalized, raw, inverse = match.groups()
            expected_raw, expected_inverse, scratch, query, adjoint = expected[normalized]
            if (raw, inverse) != (expected_raw, expected_inverse):
                raise ValueError(f"Wide normalization operands changed: {normalized}")
            tensor = "prepared_query" if query else "prepared_key"
            result.append(f"{indent}{normalized} = pl.load({tensor}, [{row}, 0], [1, DKP])\n")
            if adjoint:
                metadata = "prepared_query_inverse" if query else "prepared_key_inverse"
                result.append(f"{indent}{inverse} = pl.read({metadata}, [{row}, 0])\n")
            replaced[normalized] += 1
            position += 5
            continue
        if decay_kind == "replay" and stripped.startswith("gt = pl.reshape("):
            if fragments[position + 1].strip() != "decay = pl.exp(gt)":
                raise ValueError("Wide replay decay changed")
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + "decay = pl.reshape(pl.load(prepared_decay, [row, 0], [1, DKP]), [DKP, 1])\n")
            decay_replacements += 1
            position += 2
            continue
        if decay_kind == "reverse" and stripped.startswith("egcol = pl.exp(pl.reshape("):
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + "egcol = pl.reshape(pl.load(prepared_decay, [row, 0], [1, DKP]), [DKP, 1])\n")
            decay_replacements += 1
            position += 1
            continue
        if decay_kind == "fused" and stripped.startswith("fused_gt = pl.reshape("):
            if fragments[position + 1].strip() != "fused_egcol = pl.exp(fused_gt)":
                raise ValueError("Wide contraction decay changed")
            decay_replacements += 1
            position += 2
            continue
        result.append(line)
        position += 1
    if any(count != 1 for count in replaced.values()):
        raise ValueError(f"Wide normalization census failed: {replaced}")
    if decay_replacements != (0 if decay_kind is None else 1):
        raise ValueError(f"Wide decay census failed: {decay_kind}={decay_replacements}")
    region = "".join(result)
    for raw in raw_names | scratch_names:
        if re.search(r"\b" + raw + r"\b", region):
            raise ValueError(f"Wide consumer still uses {raw}")
    if decay_kind == "fused":
        loop = next(line for line in region.splitlines(True) if line.strip().startswith("for fused_part in"))
        indent = loop[:len(loop) - len(loop.lstrip())]
        region = _wide_replace_once(
            region, loop,
            indent + "fused_egcol = pl.reshape(pl.load(prepared_decay, [fused_row, 0], [1, DKP]), [DKP, 1])\n" + loop,
        )
        zero = next(line for line in region.splitlines(True) if line.strip().startswith("fused_zero_columns ="))
        token_loop = next(line for line in region.splitlines(True) if line.strip().startswith("for fused_t in"))
        region = _wide_replace_once(region, zero, "")
        region = _wide_replace_once(region, token_loop, token_loop + "    " + zero)
    if re.search(r"pl\.(?:load|exp)\((?:q2|k2|g2)\b", region):
        raise ValueError("Wide consumer retains raw token work")
    return region


def _wide_token_prepare(source, rows, keys, padded_keys, group_rows, cast_inputs, query_free):
    replay = {"kn1": ("kt", "kinv1", "w8", False, False)}
    if not query_free:
        replay["qn1"] = ("qt", "qinv1", "w8", True, False)
    consumers = [
        ("fwd_replay", replay, "row", "replay"),
        ("thin_adjoint_reverse", {
            "qn2": ("qt", "qinv2", "w8b", True, False),
            "kn2": ("kt", "kinv2", "w8b", False, False),
        }, "row", "reverse"),
        ("fused_token_gradients", {
            "fused_kn2": ("fused_kt", "fused_kinv2", "fused_w8k", False, True),
        }, "fused_row", "fused"),
        ("independent_complete_dq", {
            "dqphase_qn2": ("dqphase_qt", "dqphase_qinv2", "dqphase_w8q", True, True),
        }, "dqphase_row", None),
    ]
    for name, expected, row, decay_kind in consumers:
        start, end = _wide_region(source, name)
        region = _wide_consumer(source[start:end], expected, row, decay_kind)
        source = source[:start] + region + source[end:]
    query_load = f"pl.set_validshape(pl.fillpad(pl.load(q2, [preparation_base, 0], [{group_rows}, DKP], valid_shape=[preparation_valid, DK]), 0), {group_rows}, DKP)"
    key_load = query_load.replace("pl.load(q2,", "pl.load(k2,")
    if cast_inputs:
        query_load = f"pl.cast({query_load}, target_type=pl.FP32)"
        key_load = f"pl.cast({key_load}, target_type=pl.FP32)"
    producer = f'''
    prepared_query = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_key = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_decay = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)
    prepared_query_inverse = pl.create_tensor([ROWS, 8], dtype=pl.FP32)
    prepared_key_inverse = pl.create_tensor([ROWS, 8], dtype=pl.FP32)
    for preparation_block in pl.spmd({(rows + group_rows - 1) // group_rows}, name_hint="batched_token_prepare"):
        preparation_base = preparation_block * {group_rows}
        preparation_valid = pl.min({group_rows}, ROWS - preparation_base)
        preparation_scratch = pl.create_tile([{group_rows}, DKP], dtype=pl.FP32)
        preparation_ones = pl.tile.full([1, {group_rows}], dtype=pl.FP32, value=1.0)
        preparation_meta = pl.tile.full([{group_rows}, 8], dtype=pl.FP32, value=1.0)
        preparation_query = {query_load}
        preparation_query_sum = pl.reshape(pl.row_sum(pl.mul(preparation_query, preparation_query), preparation_scratch), [1, {group_rows}])
        preparation_query_norm = pl.sqrt(pl.add(preparation_query_sum, 1e-6))
        preparation_query_inverse = pl.reshape(pl.div(preparation_ones, preparation_query_norm), [{group_rows}, 1])
        preparation_query = pl.row_expand_mul(preparation_query, preparation_query_inverse)
        pl.store(pl.set_validshape(preparation_query, preparation_valid, DKP), [preparation_base, 0], prepared_query)
        preparation_query_meta = pl.row_expand_mul(preparation_meta, preparation_query_inverse)
        pl.store(pl.set_validshape(preparation_query_meta, preparation_valid, 8), [preparation_base, 0], prepared_query_inverse)
        preparation_key = {key_load}
        preparation_key_sum = pl.reshape(pl.row_sum(pl.mul(preparation_key, preparation_key), preparation_scratch), [1, {group_rows}])
        preparation_key_norm = pl.sqrt(pl.add(preparation_key_sum, 1e-6))
        preparation_key_inverse = pl.reshape(pl.div(preparation_ones, preparation_key_norm), [{group_rows}, 1])
        preparation_key = pl.row_expand_mul(preparation_key, preparation_key_inverse)
        pl.store(pl.set_validshape(preparation_key, preparation_valid, DKP), [preparation_base, 0], prepared_key)
        preparation_key_meta = pl.row_expand_mul(preparation_meta, preparation_key_inverse)
        pl.store(pl.set_validshape(preparation_key_meta, preparation_valid, 8), [preparation_base, 0], prepared_key_inverse)
        preparation_gate = pl.set_validshape(pl.fillpad(pl.load(g2, [preparation_base, 0], [{group_rows}, DKP], valid_shape=[preparation_valid, DK]), 0), {group_rows}, DKP)
        preparation_decay = pl.exp(preparation_gate)
        pl.store(pl.set_validshape(preparation_decay, preparation_valid, DKP), [preparation_base, 0], prepared_decay)
'''
    marker = "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    return _wide_replace_once(source, marker, producer + marker)


import re


def _wide_panel_region(source, region_name, padded_values, panel_width, retire_adjoint):
    start, end = _wide_region(source, region_name)
    region = source[start:end]
    if retire_adjoint:
        lines = region.splitlines(True)
        targets = ("fused_H0c0", "fused_dk20", "fused_dknv")
        moved = {}
        retained = []
        for line in lines:
            target = line.strip().split(" = ", 1)[0]
            if target in targets:
                if target in moved:
                    raise ValueError(f"Repeated panel statement: {target}")
                moved[target] = line
            else:
                retained.append(line)
        if set(moved) != set(targets):
            raise ValueError("Wide key/gate contraction statements changed")
        region = "".join(retained)
        anchor = next(line for line in retained if line.strip().startswith("fused_dhn0 ="))
        region = _wide_replace_once(region, anchor, anchor + "".join(moved[target] for target in targets))
    width = min(panel_width, padded_values)
    panels = (padded_values + width - 1) // width
    if not re.search(r"\bGCH\b", region) or not re.search(r"\bGNC\b", region):
        raise ValueError(f"Missing phase-local panel geometry: {region_name}")
    if width > 32:
        region = re.sub(r"\bGCH\b", str(width), region)
        region = re.sub(r"\bGNC\b", str(panels), region)
    return source[:start] + region + source[end:]


def _wide_gradient_panels(source, padded_values, dq_width, dkg_width, retire_adjoint):
    source = _wide_panel_region(
        source, "independent_complete_dq", padded_values, dq_width, False)
    return _wide_panel_region(
        source, "fused_token_gradients", padded_values, dkg_width, retire_adjoint)


def _layout_assignment(region, name, replacement):
    lines = region.splitlines(True)
    found = [line for line in lines if line.strip().startswith(name + " =")]
    if name in ("fused_dkacc", "fused_dgacc") and len(found) == 2:
        found = found[:1]
    if len(found) != 1:
        raise ValueError(f"Expected one layout assignment {name}: {len(found)}")
    line = found[0]
    indent = line[:len(line) - len(line.lstrip())]
    rendered = "".join(indent + item + "\n" for item in replacement.splitlines())
    return _wide_replace_once(region, line, rendered)


def _layout_row_gate(expression):
    class RemoveColumnView(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.func) == "pl.reshape":
                return node.args[0]
            return node
    return ast.unparse(RemoveColumnView().visit(ast.parse(expression, mode="eval")).body)


def _layout_region(source, name):
    matches = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.With):
            calls = [item.context_expr for item in node.items]
        elif isinstance(node, ast.For):
            calls = [node.iter]
        else:
            continue
        if any(isinstance(call, ast.Call) and any(
            keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == name for keyword in call.keywords
        ) for call in calls):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"Expected one layout region {name}: {len(matches)}")
    node = matches[0]
    lines = source.splitlines(True)
    return sum(map(len, lines[:node.lineno - 1])), sum(map(len, lines[:node.end_lineno]))


def _key_contiguous_layout(source, mode, DK, DV, DKP, DVP, CH, NC, ADVP):
    if mode == 0 or (mode == 2 and "gradient_columns =" in source):
        return source
    start, end = _layout_region(source, "thin_adjoint_reverse")
    region = source[start:end]
    bounded = "bh - head_base" in region
    local_head = "(bh - head_base)" if bounded else "bh"
    allocation_heads = "AHEADS" if bounded else "B * H"
    adjoint_offset = f"({local_head} * TP1 + t) * ADVP + cb"
    contraction_offset = f"({local_head} * T + t) * ADVP + cb"
    seed_line = next(line.strip() for line in region.splitlines() if line.strip().startswith("dhcarry0 ="))
    region = _layout_assignment(region, "tmpA", "tmpA = pl.create_tile([CH, DKP], dtype=pl.FP32)\nreverse_carrier = pl.tile.full([CH, 8], dtype=pl.FP32, value=1.0)")
    seed_source = next(line for line in region.splitlines(True) if line.strip() == seed_line)
    seed_indent = seed_source[:len(seed_source) - len(seed_source.lstrip())]
    region = _wide_replace_once(region, seed_source, seed_indent + "reverse_seed = " + seed_line.split(" = ", 1)[1] + "\n" + seed_indent + "reverse_seed_materialized = pl.add(pl.tile.full([DKP, CH], dtype=pl.FP32, value=0.0), reverse_seed)\n" + seed_indent + "dhcarry0 = pl.transpose(reverse_seed_materialized, 0, 1)\n")
    region = _layout_assignment(region, "qcol", "qrow = qn2")
    region = _layout_assignment(region, "kcol", "krow = kn2")
    gate_line = next(line.strip() for line in region.splitlines() if line.strip().startswith("egcol ="))
    region = _layout_assignment(region, "egcol", "egrow = " + _layout_row_gate(gate_line.split(" = ", 1)[1]))
    region = _layout_assignment(region, "scaled_qcol", "scaled_qrow = pl.mul(qrow, SCALE)\ndycol = pl.reshape(dyc0, [CH, 1])")
    region = _layout_assignment(region, "upd0", "upd0 = pl.col_expand_mul(pl.row_expand(dhcarry0, dycol), scaled_qrow)")
    region = _layout_assignment(region, "ccv0", "ccv0 = pl.row_sum(pl.col_expand_mul(dhgc0, krow), tmpA)")
    region = _layout_assignment(region, "dhp0", "dhp0 = pl.sub(dhgc0, pl.col_expand_mul(pl.row_expand(dhgc0, drc0), krow))")
    region = _wide_replace_once(region, "dhcarry0 = pl.row_expand_mul(dhp0, egcol)", "dhcarry0 = pl.col_expand_mul(dhp0, egrow)")
    store_lines = [line for line in region.splitlines(True) if "adjoint_tape)" in line or "contraction_tape)" in line]
    if len(store_lines) != 2:
        raise ValueError("Unexpected history publication")
    for line in store_lines:
        indent = line[:len(line) - len(line.lstrip())]
        if "adjoint_tape)" in line:
            replacement = f"pl.store(dhgc0, [{adjoint_offset}, 0], native_adjoint_tape)"
        else:
            replacement = f"pl.store(pl.row_expand_mul(reverse_carrier, ccv0), [{contraction_offset}, 0], native_contraction_tape)"
        region = _wide_replace_once(region, line, indent + replacement + "\n")
    region = _layout_assignment(region, "dhlast", "reverse_final = pl.transpose(dhcarry0, 0, 1)\ndhlast = pl.set_validshape(reverse_final, DK, pl.min(CH, DV - cb))")
    source = source[:start] + region + source[end:]
    allocation_line = next(line for line in source.splitlines(True) if line.strip().startswith("adjoint_tape ="))
    indent = allocation_line[:len(allocation_line) - len(allocation_line.lstrip())]
    native_allocations = (
        indent + f"native_adjoint_tape = pl.create_tensor([{allocation_heads} * TP1 * ADVP, DKP], dtype=pl.FP32)\n"
        + indent + f"native_contraction_tape = pl.create_tensor([{allocation_heads} * T * ADVP, 8], dtype=pl.FP32)\n")
    source = _wide_replace_once(source, allocation_line, allocation_line + native_allocations)
    if mode == 1:
        end = _layout_region(source, "thin_adjoint_reverse")[1]
        owner_line = next(line for line in source[:end].splitlines(True)[::-1] if line.strip().startswith("for owner in pl.parallel("))
        base_indent = owner_line[:len(owner_line) - len(owner_line.lstrip())]
        restore = [
            f"for restore_owner in pl.parallel({allocation_heads} * NC):",
            '    with pl.at(level=pl.Level.CORE_GROUP, name_hint="restore_reverse_history"):',
            "        restore_head = restore_owner // NC",
            "        restore_part = restore_owner % NC",
            "        restore_cb = restore_part * CH",
            "        for restore_t in pl.range(T):",
            "            restore_adjoint = pl.load(native_adjoint_tape, [(restore_head * TP1 + restore_t) * ADVP + restore_cb, 0], [CH, DKP])",
            "            restore_matrix = pl.transpose(restore_adjoint, 0, 1)",
            "            pl.store(pl.set_validshape(restore_matrix, DKP, pl.min(CH, ADVP - restore_cb)), [(restore_head * TP1 + restore_t) * DKP, restore_cb], adjoint_tape)",
            "            restore_carrier = pl.load(native_contraction_tape, [(restore_head * T + restore_t) * ADVP + restore_cb, 0], [CH, 8])",
            "            restore_transposed = pl.transpose(restore_carrier, 0, 1)",
            "            restore_row = pl.tile.slice(restore_transposed, [1, CH], [0, 0], valid_shape=[1, pl.min(CH, ADVP - restore_cb)])",
        ]
        row = "restore_head * T + restore_t" if bounded else "((restore_head // H) * T + restore_t) * H + restore_head % H"
        restore.append(f"            pl.store(restore_row, [{row}, restore_cb], contraction_tape)")
        source = source[:end] + "".join(base_indent + line + "\n" for line in restore) + source[end:]
        return source
    for prefix in ("adjoint_tape =", "contraction_tape ="):
        line = next(line for line in source.splitlines(True) if line.strip().startswith(prefix))
        source = _wide_replace_once(source, line, "")
    start, end = _layout_region(source, "fused_token_gradients")
    region = source[start:end]
    scratch_line = next(line.strip() for line in region.splitlines() if line.strip().startswith("fused_tmpA ="))
    width = re.search(r"\[DKP, (\w+)\]", scratch_line).group(1)
    local_head = "(fused_bh - head_base)" if bounded else "fused_bh"
    aoffset = f"({local_head} * TP1 + fused_t) * ADVP + fused_cb"
    coffset = f"({local_head} * T + fused_t) * ADVP + fused_cb"
    region = _layout_assignment(region, "fused_tmpA", f"fused_tmpA = pl.create_tile([{width}, DKP], dtype=pl.FP32)\nfused_carrier_scratch = pl.create_tile([{width}, 8], dtype=pl.FP32)")
    region = _layout_assignment(region, "fused_zero_columns", "fused_zero_rows = pl.tile.full([8, DKP], dtype=pl.FP32, value=0.0)")
    for stem in ("dk", "dg"):
        region = _layout_assignment(region, f"fused_{stem}acc", f"fused_{stem}acc = pl.col_sum(fused_zero_rows, fused_tmpA)")
    region = _layout_assignment(region, "fused_kcol", "fused_krow = fused_kn2")
    gate_line = next(line.strip() for line in region.splitlines() if line.strip().startswith("fused_egcol ="))
    region = _layout_assignment(region, "fused_egcol", "fused_egrow = " + _layout_row_gate(gate_line.split(" = ", 1)[1]))
    region = _layout_assignment(region, "fused_ccv0", f"fused_ccarrier = pl.set_validshape(pl.fillpad(pl.load(native_contraction_tape, [{coffset}, 0], [{width}, 8], valid_shape=[pl.min({width}, ADVP - fused_cb), 8]), 0), {width}, 8)\nfused_ccol = pl.mul(pl.row_sum(fused_ccarrier, fused_carrier_scratch), 0.125)\nfused_carrier_materialized = pl.add(pl.tile.full([{width}, 8], dtype=pl.FP32, value=0.0), fused_ccarrier)\nfused_ctransposed = pl.transpose(fused_carrier_materialized, 0, 1)\nfused_ccv0 = pl.tile.slice(fused_ctransposed, [1, {width}], [0, 0])")
    region = _layout_assignment(region, "fused_drc0", "fused_drc0 = pl.mul(fused_ccv0, fused_bt)\nfused_drcol = pl.mul(fused_ccol, fused_bt)\nfused_rcol = pl.reshape(fused_rc0, [" + width + ", 1])")
    region = _layout_assignment(region, "fused_dhgc0", f"fused_dhgc0 = pl.set_validshape(pl.fillpad(pl.load(native_adjoint_tape, [{aoffset}, 0], [{width}, DKP], valid_shape=[pl.min({width}, ADVP - fused_cb), DKP]), 0), {width}, DKP)")
    region = _layout_assignment(region, "fused_dk10", "fused_dk10 = pl.col_sum(pl.row_expand_mul(fused_dhgc0, fused_rcol), fused_tmpA)")
    region = _layout_assignment(region, "fused_dhp0", "fused_dhp0 = pl.sub(fused_dhgc0, pl.col_expand_mul(pl.row_expand(fused_dhgc0, fused_drcol), fused_krow))")
    region = _layout_assignment(region, "fused_dhn0", "fused_dhn0 = pl.col_expand_mul(fused_dhp0, fused_egrow)")
    previous = next(line.strip() for line in region.splitlines() if line.strip().startswith("fused_H0c0 ="))
    region = _layout_assignment(region, "fused_H0c0", "fused_previous_key_value = " + previous.split(" = ", 1)[1] + "\nfused_previous_materialized = pl.add(pl.tile.full([DKP, " + width + "], dtype=pl.FP32, value=0.0), fused_previous_key_value)\nfused_H0c0 = pl.transpose(fused_previous_materialized, 0, 1)")
    region = _layout_assignment(region, "fused_dk20", "fused_dk20 = pl.mul(pl.col_sum(pl.row_expand_mul(fused_H0c0, fused_drcol), fused_tmpA), fused_egrow)")
    region = _layout_assignment(region, "fused_dgp0", "fused_dgp0 = pl.col_sum(pl.mul(fused_dhn0, fused_H0c0), fused_tmpA)")
    region = _wide_replace_once(region, "fused_dk_row = pl.reshape(fused_dkacc, [1, DKP])", "fused_dk_row = fused_dkacc")
    region = _wide_replace_once(region, "fused_dg_row = pl.reshape(fused_dgacc, [1, DKP])", "fused_dg_row = fused_dgacc")
    return source[:start] + region + source[end:]


import ast
import re


def _launch_kinds(source):
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.For):
            expression = node.iter
        elif isinstance(node, ast.With):
            expression = node.items[0].context_expr
        else:
            continue
        if not isinstance(expression, ast.Call):
            continue
        for keyword in expression.keywords:
            if keyword.arg == "name_hint":
                name = keyword.value.value
                if name in result:
                    raise ValueError(f"Duplicated selected launch: {name}")
                result[name] = ast.unparse(expression.func)
    return result


def _indexed_region(source, header, launch, body_indent, launch_indent, isolate_prefix=""):
    if source.count(header) != 1:
        raise ValueError(f"Missing or duplicated selected launch: {header!r}")
    start = source.index(header)
    body_start = start + len(header)
    body = []
    for line in source[body_start:].splitlines(True):
        if line.strip() and len(line) - len(line.lstrip()) < body_indent:
            break
        body.append(line)
    if not body or not any(line.strip() for line in body):
        raise ValueError("Empty selected launch body")
    body_end = body_start + sum(len(line) for line in body)
    delta = body_indent - launch_indent - 4
    rewritten = "".join(line[delta:] if line.strip() else line for line in body)
    region = launch + rewritten
    if isolate_prefix:
        region = _isolate_region_variables(region, isolate_prefix)
    return source[:start] + region + source[body_end:]


def _recurrent_indexed_launches(source, mask):
    original = _launch_kinds(source)
    for name in ("fwd_replay", "thin_adjoint_reverse", "fused_token_gradients"):
        if name not in original:
            raise ValueError(f"Unsupported selected recurrence: {name}")
    expected = set()
    if mask & 1:
        header = (
            "    for b in pl.parallel(0, B, 1):\n"
            "        for h in pl.parallel(0, H, 1):\n"
            "            for cb in pl.parallel(0, DVP, FCH):\n"
            '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_replay"):\n'
        )
        launch = (
            '    for replay_block in pl.spmd(B * H * (DVP // FCH), name_hint="fwd_replay"):\n'
            "        replay_head = replay_block // (DVP // FCH)\n"
            "        b = replay_head // H\n"
            "        h = replay_head % H\n"
            "        cb = (replay_block % (DVP // FCH)) * FCH\n"
        )
        source = _indexed_region(source, header, launch, 20, 4, "replay_local_")
        expected.add("fwd_replay")
    if mask & 2:
        bounded = "    for head_base in pl.parallel(0, B * H, AHEADS):" in source
        indent = " " * (8 if bounded else 4)
        owners = "AHEADS * NC" if bounded else "B * H * NC"
        header = (indent + f"for owner in pl.parallel({owners}):\n"
                  + indent + '    with pl.at(level=pl.Level.CORE_GROUP, name_hint="thin_adjoint_reverse"):\n')
        launch = indent + f'for owner in pl.spmd({owners}, name_hint="thin_adjoint_reverse"):\n'
        source = _indexed_region(source, header, launch, len(indent) + 8, len(indent))
        expected.add("thin_adjoint_reverse")
    if mask & 4:
        for stem, region in (("fused_", "fused_token_gradients"),
                             ("dqphase_", "independent_complete_dq"),
                             ("layout_", "dependent_vector_gradient_layout")):
            marker = f'name_hint="{region}"'
            count = source.count(marker)
            if count > 1:
                raise ValueError(f"Duplicated selected consumer: {region}")
            if count == 0:
                if region == "fused_token_gradients":
                    raise ValueError("Missing selected fused consumer")
                continue
            region_line = next(line for line in source.splitlines() if marker in line)
            if "pl.spmd(" in region_line:
                if region != "fused_token_gradients":
                    raise ValueError(f"Unexpected pre-indexed consumer: {region}")
                continue
            bounded = region == "fused_token_gradients" and "for fused_head in pl.parallel(AHEADS):" in source
            if bounded:
                header = (
                    "        for fused_head in pl.parallel(AHEADS):\n"
                    "            for fused_token_base in pl.parallel(0, T, EPT):\n"
                    '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_token_gradients"):\n'
                )
                launch = (
                    '        for fused_dispatch_block in pl.spmd(AHEADS * ((T + EPT - 1) // EPT), name_hint="fused_token_gradients"):\n'
                    "            fused_head = fused_dispatch_block // ((T + EPT - 1) // EPT)\n"
                    "            fused_token_base = (fused_dispatch_block % ((T + EPT - 1) // EPT)) * EPT\n"
                )
                source = _indexed_region(source, header, launch, 20, 8)
            else:
                header = (
                    f"    for {stem}b in pl.parallel(B):\n"
                    f"        for {stem}h in pl.parallel(H):\n"
                    f"            for {stem}token_base in pl.parallel(0, T, EPT):\n"
                    f'                with pl.at(level=pl.Level.CORE_GROUP, name_hint="{region}"):\n'
                )
                launch = (
                    f'    for {stem}dispatch_block in pl.spmd(B * H * ((T + EPT - 1) // EPT), name_hint="{region}"):\n'
                    f"        {stem}dispatch_head = {stem}dispatch_block // ((T + EPT - 1) // EPT)\n"
                    f"        {stem}b = {stem}dispatch_head // H\n"
                    f"        {stem}h = {stem}dispatch_head % H\n"
                    f"        {stem}token_base = ({stem}dispatch_block % ((T + EPT - 1) // EPT)) * EPT\n"
                )
                source = _indexed_region(source, header, launch, 20, 4)
            expected.add(region)
    actual = _launch_kinds(source)
    changed = {name for name in original if original[name] != actual.get(name)}
    if actual.keys() != original.keys() or changed != expected:
        raise ValueError(f"Selected launch rewrite mismatch: expected={expected}, actual={changed}")
    if any(actual[name] != "pl.spmd" for name in expected):
        raise ValueError("Selected launch was not indexed")
    compile(source, "selected_recurrent_launches.py", "exec")
    return source


import re
import textwrap


def _checkpoint_compact(source):
    anchor = "    for row_base in pl.parallel(0, B * T * H, 16):\n"
    start = source.index(anchor)
    region_start, end = _layout_region(source, "dbeta_compact")
    body = source[region_start:end].splitlines(True)[1:]
    replacement = '    for compact_block in pl.spmd((B * T * H + 15) // 16, name_hint="dbeta_compact"):\n'
    replacement += "        row_base = compact_block * 16\n"
    replacement += "".join(line[4:] if line.strip() else line for line in body)
    return source[:start] + replacement + source[end:]


def _checkpoint_continue(source, tokens, padded_values, window, need_cast, input_dtype):
    bounded = "    for head_base in pl.parallel(0, B * H, AHEADS):" in source
    heads = "AHEADS" if bounded else "B * H"
    head_base = "head_base" if bounded else "0"
    indent = "        " if bounded else "    "
    slots = (tokens + window - 1) // window + 1
    windows = slots - 1
    start, end = _layout_region(source, "fused_token_gradients")
    original = source[start:end]
    width_match = re.search(r"fused_tmpA = pl.create_tile\(\[DKP, ([^\]]+)\]", original)
    width = width_match.group(1)
    panels = (padded_values + int(width) - 1) // int(width) if width.isdigit() else 1
    if not width.isdigit():
        if padded_values > 64:
            raise ValueError("Unsupported symbolic panel width")
        width = "GCH"
        panels = "GNC"
    old_store = next(line for line in source.splitlines(True) if line.strip().startswith("pl.store(dhgc0,") and "adjoint_tape)" in line)
    local_head = "(bh - head_base)" if bounded else "bh"
    store_indent = old_store[:len(old_store) - len(old_store.lstrip())]
    new_store = store_indent + f"if t % {window} == 0:\n"
    new_store += store_indent + f"    pl.store(dhcarry0, [({local_head} * {slots} + t // {window}) * DKP, cb], adjoint_checkpoints)\n"
    source = _replace_once(source, old_store, new_store)
    old_allocation = next(line for line in source.splitlines(True) if line.strip().startswith("adjoint_tape ="))
    allocation = indent + f"adjoint_checkpoints = pl.create_tensor([{heads} * {slots} * DKP, ADVP], dtype=pl.FP32)\n"
    allocation += indent + f"checkpoint_partial_k = pl.create_tensor([{panels} * {heads} * T, DKP], dtype=pl.FP32)\n"
    allocation += indent + f"checkpoint_partial_g = pl.create_tensor([{panels} * {heads} * T, DKP], dtype=pl.FP32)\n"
    allocation += indent + f"checkpoint_partial_beta = pl.create_tensor([{panels} * {heads} * T, 16], dtype=pl.FP32)\n"
    source = _replace_once(source, old_allocation, allocation)
    assignments = {}
    for line in original.splitlines():
        stripped = line.strip()
        if " = " in stripped:
            name, expression = stripped.split(" = ", 1)
            assignments[name] = expression
    required = ("fused_rc0", "fused_ccv0", "fused_drc0", "fused_dk10", "fused_dhp0", "fused_dhn0", "fused_H0c0", "fused_dk20", "fused_dknv", "fused_dgp0", "fused_psd", "fused_dbe0")
    for name in required:
        if name not in assignments:
            raise ValueError(f"Missing original consumer assignment {name}")
    arithmetic = "\n".join(f"{name} = {assignments[name]}" for name in required)
    dvo = next(line.strip() for line in original.splitlines() if line.strip().startswith("fused_dvo0 ="))
    finish_start = original.index("fused_dk_row =")
    finish = textwrap.dedent(original[finish_start:].split("\n", 1)[0] + "\n" + textwrap.dedent(original[finish_start:].split("\n", 1)[1]))
    finish_lines = original[finish_start:].splitlines()
    finish = "\n".join([finish_lines[0]] + [line.lstrip() for line in finish_lines[1:]])
    template = '''
for checkpoint_block in pl.spmd({HEADS} * {WINDOWS} * {PANELS}, name_hint="checkpoint_gradient_panels"):
    fused_part = checkpoint_block % {PANELS}
    checkpoint_window = (checkpoint_block // {PANELS}) % {WINDOWS}
    checkpoint_head = checkpoint_block // ({PANELS} * {WINDOWS})
    fused_bh = {BASE} + checkpoint_head
    fused_b = fused_bh // H
    fused_h = fused_bh % H
    fused_cb = fused_part * {WIDTH}
    checkpoint_start = checkpoint_window * {WINDOW}
    checkpoint_end = pl.min(checkpoint_start + {WINDOW}, T)
    fused_tmpA = pl.create_tile([DKP, {WIDTH}], dtype=pl.FP32)
    if checkpoint_end == T:
        checkpoint_carry = pl.load(df2, [fused_bh * DK, fused_cb], [DKP, {WIDTH}])
    else:
        checkpoint_carry = pl.load(adjoint_checkpoints, [(checkpoint_head * {SLOTS} + checkpoint_end // {WINDOW}) * DKP, fused_cb], [DKP, {WIDTH}])
    for checkpoint_offset in pl.range(checkpoint_end - checkpoint_start):
        fused_t = checkpoint_end - 1 - checkpoint_offset
        fused_row = (fused_b * T + fused_t) * H + fused_h
        fused_kn2 = pl.load(prepared_key, [fused_row, 0], [1, DKP])
        fused_qn2 = pl.load(prepared_query, [fused_row, 0], [1, DKP])
        fused_kcol = pl.reshape(fused_kn2, [DKP, 1])
        checkpoint_qcol = pl.reshape(fused_qn2, [DKP, 1])
        fused_bt = pl.read(b1, [fused_row, 0])
        fused_egcol = pl.reshape(pl.load(prepared_decay, [fused_row, 0], [1, DKP]), [DKP, 1])
        checkpoint_dy = {DYLOAD}
        checkpoint_scaled_query = pl.mul(checkpoint_qcol, SCALE)
        checkpoint_injection = pl.row_expand_mul(pl.col_expand(checkpoint_carry, checkpoint_dy), checkpoint_scaled_query)
        fused_dhgc0 = pl.add(checkpoint_carry, checkpoint_injection)
{ARITHMETIC}
        checkpoint_carry = fused_dhn0
        {DVCAST}
        pl.store(fused_dvo0, [fused_row, fused_cb], dv2)
        checkpoint_partial_row = (fused_part * {HEADS} + checkpoint_head) * T + fused_t
        pl.store(pl.reshape(fused_dknv, [1, DKP]), [checkpoint_partial_row, 0], checkpoint_partial_k)
        pl.store(pl.reshape(fused_dgp0, [1, DKP]), [checkpoint_partial_row, 0], checkpoint_partial_g)
        pl.write(checkpoint_partial_beta, [checkpoint_partial_row, 0], fused_dbe0)
for checkpoint_finish_block in pl.spmd({HEADS} * ((T + EPT - 1) // EPT), name_hint="checkpoint_gradient_finish"):
    checkpoint_head = checkpoint_finish_block // ((T + EPT - 1) // EPT)
    checkpoint_token_base = (checkpoint_finish_block % ((T + EPT - 1) // EPT)) * EPT
    fused_bh = {BASE} + checkpoint_head
    fused_b = fused_bh // H
    fused_h = fused_bh % H
    fused_tmpA = pl.create_tile([DKP, {WIDTH}], dtype=pl.FP32)
    for fused_t in pl.range(checkpoint_token_base, pl.min(checkpoint_token_base + EPT, T)):
        fused_row = (fused_b * T + fused_t) * H + fused_h
        fused_zero_columns = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)
        fused_dkacc = pl.row_sum(fused_zero_columns, fused_tmpA)
        fused_dgacc = pl.row_sum(fused_zero_columns, fused_tmpA)
        fused_dbeta_acc = 0.0
        fused_kn2 = pl.load(prepared_key, [fused_row, 0], [1, DKP])
        fused_kinv2 = pl.read(prepared_key_inverse, [fused_row, 0])
        for fused_part in pl.range({PANELS}):
            checkpoint_partial_row = (fused_part * {HEADS} + checkpoint_head) * T + fused_t
            checkpoint_dk = pl.reshape(pl.load(checkpoint_partial_k, [checkpoint_partial_row, 0], [1, DKP]), [DKP, 1])
            checkpoint_dg = pl.reshape(pl.load(checkpoint_partial_g, [checkpoint_partial_row, 0], [1, DKP]), [DKP, 1])
            fused_dkacc = pl.add(fused_dkacc, checkpoint_dk)
            fused_dgacc = pl.add(fused_dgacc, checkpoint_dg)
            checkpoint_beta = pl.read(checkpoint_partial_beta, [checkpoint_partial_row, 0])
            fused_dbeta_acc = fused_dbeta_acc + checkpoint_beta
{FINISH}
'''
    dyload = f"pl.load(dy2, [fused_row, fused_cb], [1, {width}])"
    if need_cast:
        dyload = f"pl.cast({dyload}, target_type=pl.FP32)"
    replacements = dict(HEADS=heads, BASE=head_base, WINDOWS=windows, PANELS=panels, WIDTH=width,
                        WINDOW=window, SLOTS=slots, DYLOAD=dyload, DVCAST=dvo,
                        ARITHMETIC=textwrap.indent(arithmetic, "        "), FINISH=textwrap.indent(finish, "        "))
    for name, value in replacements.items():
        template = template.replace("{" + name + "}", str(value))
    replacement = textwrap.indent(template.strip("\n") + "\n", indent)
    start, end = _layout_region(source, "fused_token_gradients")
    source = source[:start] + replacement + source[end:]
    if "adjoint_tape" in source:
        raise ValueError("Full adjoint history survived continuation rewrite")
    return source


def _deletion_line(source, old, replacements):
    matches = [line for line in source.splitlines(True) if line.strip() == old]
    if len(matches) != 1:
        raise ValueError(f"Expected one deletion line {old!r}: {len(matches)}")
    line = matches[0]
    indent = line[:len(line) - len(line.lstrip())]
    return source.replace(line, "".join(indent + replacement + "\n" for replacement in replacements), 1)


def _delete_scalar_recurrence(source, mask):
    allocations = []
    preparation = []
    if mask & 1:
        allocations.append("    prepared_scaled_query = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)\n")
        source = _deletion_line(
            source,
            "pl.store(pl.set_validshape(preparation_query, preparation_valid, DKP), [preparation_base, 0], prepared_query)",
            ["pl.store(pl.set_validshape(preparation_query, preparation_valid, DKP), [preparation_base, 0], prepared_query)",
             "preparation_scaled_query = pl.mul(preparation_query, SCALE)",
             "pl.store(pl.set_validshape(preparation_scaled_query, preparation_valid, DKP), [preparation_base, 0], prepared_scaled_query)"])
        source = _deletion_line(source, "qn2 = pl.load(prepared_query, [row, 0], [1, DKP])", [])
        source = _deletion_line(source, "qcol = pl.reshape(qn2, [DKP, 1])", [])
        source = _deletion_line(source, "scaled_qcol = pl.mul(qcol, SCALE)", [
            "scaled_qcol = pl.reshape(pl.load(prepared_scaled_query, [row, 0], [1, DKP]), [DKP, 1])"])
    if mask & 6:
        allocations.append("    preparation_beta_flat = pl.reshape(beta, [1, ROWS])\n")
        preparation += [
            "preparation_beta_row = pl.set_validshape(pl.fillpad(pl.load(preparation_beta_flat, [0, preparation_base], [1, 16], valid_shape=[1, preparation_valid]), 0), 1, 16)",
            "preparation_beta_column = pl.reshape(preparation_beta_row, [16, 1])",
        ]
    if mask & 2:
        allocations.append("    prepared_beta_key = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)\n")
        preparation += [
            "preparation_beta_key = pl.row_expand_mul(preparation_key, preparation_beta_column)",
            "pl.store(pl.set_validshape(preparation_beta_key, preparation_valid, DKP), [preparation_base, 0], prepared_beta_key)",
        ]
        source = _deletion_line(source, "replay_local_bt = pl.read(b1, [replay_local_row, 0])", [])
        source = _deletion_line(source, "replay_local_kb = pl.mul(replay_local_kcol, replay_local_bt)", [
            "replay_local_kb = pl.reshape(pl.load(prepared_beta_key, [replay_local_row, 0], [1, DKP]), [DKP, 1])"])
    if mask & 4:
        allocations.append("    prepared_beta_vector = pl.create_tensor([ROWS, CH], dtype=pl.FP32)\n")
        preparation += [
            "preparation_beta_ones = pl.tile.full([16, CH], dtype=pl.FP32, value=1.0)",
            "preparation_beta_vector = pl.row_expand_mul(preparation_beta_ones, preparation_beta_column)",
            "pl.store(pl.set_validshape(preparation_beta_vector, preparation_valid, CH), [preparation_base, 0], prepared_beta_vector)",
        ]
        source = _deletion_line(source, "bt = pl.read(b1, [row, 0])", [
            "beta_vector_row = pl.load(prepared_beta_vector, [row, 0], [1, CH])"])
        source = _deletion_line(source, "drc0 = pl.mul(ccv0, bt)", [
            "drc0 = pl.mul(ccv0, beta_vector_row)"])
    marker = "    prepared_query = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)\n"
    if source.count(marker) != 1:
        raise ValueError("Batched preparation allocation changed")
    source = source.replace(marker, "".join(allocations) + marker, 1)
    if preparation:
        source = _deletion_line(
            source,
            "pl.store(pl.set_validshape(preparation_decay, preparation_valid, DKP), [preparation_base, 0], prepared_decay)",
            ["pl.store(pl.set_validshape(preparation_decay, preparation_valid, DKP), [preparation_base, 0], prepared_decay)"] + preparation)
    return source


def _pipeline_forward_replay(source, stages):
    import ast

    tree = ast.parse(source)
    regions = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.With)):
            continue
        calls = [node.iter] if isinstance(node, ast.For) else [item.context_expr for item in node.items]
        if any(isinstance(call, ast.Call) and any(keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "fwd_replay" for keyword in call.keywords) for call in calls):
            regions.append(node)
    if len(regions) != 1:
        raise ValueError("Expected exactly one selected fwd_replay region")
    loops = [node for node in ast.walk(regions[0]) if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == "pl.range" and len(node.iter.args) == 1 and ast.unparse(node.iter.args[0]) == "T"]
    if len(loops) != 1:
        raise ValueError("Expected exactly one selected replay token loop")
    loop = loops[0]
    lines = source.splitlines(keepends=True)
    original = lines[loop.lineno - 1]
    if original.count("pl.range(T)") != 1:
        raise ValueError("Unexpected selected replay loop header")
    lines[loop.lineno - 1] = original.replace("pl.range(T)", f"pl.pipeline(T, stage={stages})", 1)
    result = "".join(lines)
    modified = ast.parse(result)
    pipelines = [node for node in ast.walk(modified) if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.pipeline"]
    if len(pipelines) != 1:
        raise ValueError("Expected one pipeline in final selected source")
    pipelines[0].func.attr = "range"
    pipelines[0].keywords = []
    if ast.dump(tree, include_attributes=False) != ast.dump(modified, include_attributes=False):
        raise ValueError("Pipeline changed more than the replay loop scheduling construct")
    return result


def _pipeline_adjoint_consumers(source, reverse_pipeline, consumer_pipeline, finish_pipeline):
    import ast

    targets = {
        "thin_adjoint_reverse": ("tt", reverse_pipeline),
        "fused_token_gradients": ("fused_t", consumer_pipeline),
        "checkpoint_gradient_panels": ("checkpoint_offset", consumer_pipeline),
        "dependent_vector_gradient_layout": ("layout_t", finish_pipeline),
        "checkpoint_gradient_finish": ("fused_t", finish_pipeline),
        "independent_complete_dq": ("dqphase_t", finish_pipeline),
    }
    tree = ast.parse(source)
    regions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            calls = [node.iter]
        elif isinstance(node, ast.With):
            calls = [item.context_expr for item in node.items]
        else:
            continue
        for call in calls:
            if not isinstance(call, ast.Call):
                continue
            for keyword in call.keywords:
                if keyword.arg != "name_hint" or not isinstance(keyword.value, ast.Constant):
                    continue
                name = keyword.value.value
                if name in targets:
                    if name in regions:
                        raise ValueError(f"Duplicate selected region {name}")
                    regions[name] = node
    if "thin_adjoint_reverse" not in regions:
        raise ValueError("Missing selected thin reverse")
    if sum(name in regions for name in ("fused_token_gradients", "checkpoint_gradient_panels")) != 1:
        raise ValueError("Expected exactly one selected main consumer")
    if "checkpoint_gradient_panels" in regions and "checkpoint_gradient_finish" not in regions:
        raise ValueError("Missing checkpoint finish")
    replacements = []
    changed_lines = set()
    lines = source.splitlines(keepends=True)
    for name, region in regions.items():
        variable, enabled = targets[name]
        loops = [node for node in ast.walk(region)
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                 and node.target.id == variable and isinstance(node.iter, ast.Call)
                 and ast.unparse(node.iter.func) == "pl.range"]
        if len(loops) != 1:
            raise ValueError(f"Expected one {variable} token loop in {name}")
        if not enabled:
            continue
        loop = loops[0]
        if name == "checkpoint_gradient_panels":
            continue
        if any(isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.write"
               for node in ast.walk(loop)):
            continue
        call = loop.iter
        if call.lineno != call.end_lineno or call.keywords:
            raise ValueError(f"Unexpected token constructor in {name}")
        original = ast.get_source_segment(source, call)
        replacement = original.replace("pl.range(", "pl.pipeline(", 1)[:-1] + ", stage=2)"
        line_index = call.lineno - 1
        header = lines[line_index]
        if header.count(original) != 1:
            raise ValueError(f"Ambiguous loop header in {name}")
        replacements.append((line_index, original, replacement))
        changed_lines.add(call.lineno)
    for line_index, original, replacement in replacements:
        lines[line_index] = lines[line_index].replace(original, replacement, 1)
    result = "".join(lines)
    modified = ast.parse(result)
    for node in ast.walk(modified):
        if isinstance(node, ast.Call) and node.lineno in changed_lines and ast.unparse(node.func) == "pl.pipeline":
            node.func.attr = "range"
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "stage"]
    if ast.dump(tree, include_attributes=False) != ast.dump(modified, include_attributes=False):
        raise ValueError("Changed more than selected token-loop scheduling")
    return result


def _checkpoint_numeric_panels(source, padded_values, width):
    import re

    start, end = _layout_region(source, "fused_token_gradients")
    region = source[start:end]
    region = re.sub(r"\bGCH\b", str(width), region)
    region = re.sub(r"\bGNC\b", str((padded_values + width - 1) // width), region)
    return source[:start] + region + source[end:]


def _consumer_token_loop(source, region_name, variable):
    import ast

    tree = ast.parse(source)
    regions = []
    for node in ast.walk(tree):
        calls = [node.iter] if isinstance(node, ast.For) else []
        if isinstance(node, ast.With):
            calls = [item.context_expr for item in node.items]
        if any(isinstance(call, ast.Call) and any(
                keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == region_name for keyword in call.keywords)
                for call in calls):
            regions.append(node)
    if len(regions) != 1:
        raise ValueError(f"Expected one repaired consumer {region_name}")
    loops = [node for node in ast.walk(regions[0])
             if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
             and node.target.id == variable]
    if len(loops) != 1:
        raise ValueError(f"Expected one repaired token loop {variable}")
    return tree, loops[0]


def _hoist_immutable_packing(source):
    import ast

    tree, loop = _consumer_token_loop(source, "fused_token_gradients", "fused_t")
    names = {"layout_carrier"}
    names.update(f"layout_{stem}_{suffix}" for stem in ("dq", "dk", "dg")
                 for suffix in ("mask", "carrier"))
    assignments = [node for node in loop.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in names]
    if not assignments:
        return source, False
    if {node.targets[0].id for node in assignments} != names or len(assignments) != len(names):
        raise ValueError("Incomplete invariant packing selector set")
    expected = ["layout_carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)"]
    for lane, stem in enumerate(("dq", "dk", "dg")):
        expected.extend([
            f"layout_{stem}_mask = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)",
            f"pl.tile.write(layout_{stem}_mask, [0, {lane}], 1.0)",
            f"layout_{stem}_carrier = pl.col_expand_mul(layout_carrier, layout_{stem}_mask)",
        ])
    expected_nodes = [ast.parse(statement).body[0] for statement in expected]
    expected_dumps = [ast.dump(node, include_attributes=False) for node in expected_nodes]
    selected = [node for node in loop.body
                if ast.dump(node, include_attributes=False) in expected_dumps]
    if [ast.dump(node, include_attributes=False) for node in selected] != expected_dumps:
        raise ValueError("Invariant packing initialization changed")
    selected_ids = {id(node) for node in selected}
    if any(isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.write"
           for statement in loop.body if id(statement) not in selected_ids
           for node in ast.walk(statement)):
        return source, False
    for statement in loop.body:
        if id(statement) in selected_ids:
            continue
        if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in names
               for node in ast.walk(statement)):
            raise ValueError("Packing selector mutated by token body")
    lines = source.splitlines(keepends=True)
    moved_lines = {line for node in selected for line in range(node.lineno - 1, node.end_lineno)}
    preheader = [" " * loop.col_offset + statement + "\n" for statement in expected]
    result = "".join(("".join(preheader) if index == loop.lineno - 1 else "") + line
                     for index, line in enumerate(lines) if index not in moved_lines)
    _, modified_loop = _consumer_token_loop(result, "fused_token_gradients", "fused_t")
    loop.body = [statement for statement in loop.body if id(statement) not in selected_ids]
    if ast.dump(loop, include_attributes=False) != ast.dump(modified_loop, include_attributes=False):
        raise ValueError("Hoist changed the per-token packing arithmetic")
    return result, True


def _overlap_repaired_consumer(source, region_name, variable):
    import ast

    tree, loop = _consumer_token_loop(source, region_name, variable)
    if any(isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.write"
           for node in ast.walk(loop)):
        raise ValueError("Mutable tile write survived repaired consumer")
    call = loop.iter
    if not isinstance(call, ast.Call) or ast.unparse(call.func) != "pl.range" or call.keywords:
        raise ValueError("Unexpected repaired consumer token constructor")
    if call.lineno != call.end_lineno:
        raise ValueError("Multiline repaired consumer constructor")
    original = ast.get_source_segment(source, call)
    replacement = original.replace("pl.range(", "pl.pipeline(", 1)[:-1] + ", stage=2)"
    lines = source.splitlines(keepends=True)
    if lines[call.lineno - 1].count(original) != 1:
        raise ValueError("Ambiguous repaired consumer token header")
    lines[call.lineno - 1] = lines[call.lineno - 1].replace(original, replacement, 1)
    result = "".join(lines)
    modified, modified_loop = _consumer_token_loop(result, region_name, variable)
    modified_loop.iter.func.attr = "range"
    modified_loop.iter.keywords = []
    if ast.dump(tree, include_attributes=False) != ast.dump(modified, include_attributes=False):
        raise ValueError("Consumer overlap changed more than its schedule")
    return result


def _single_downstream_indexed_launches(source, mask):
    original = _launch_kinds(source)
    expected = set()
    if mask & 1:
        header = (
            "    for single_reduce_bh in pl.parallel(B * H):\n"
            '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="complete_single_gradients"):\n'
        )
        launch = '    for single_reduce_bh in pl.spmd(B * H, name_hint="complete_single_gradients"):\n'
        source = _indexed_region(source, header, launch, 12, 4)
        expected.add("complete_single_gradients")
    if mask & 2:
        header = (
            "    for dqphase_b in pl.parallel(B):\n"
            "        for dqphase_h in pl.parallel(H):\n"
            "            for dqphase_token_base in pl.parallel(0, T, EPT):\n"
            '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="independent_complete_dq"):\n'
        )
        launch = (
            '    for dqphase_dispatch_head in pl.spmd(B * H, name_hint="independent_complete_dq"):\n'
            "        dqphase_b = dqphase_dispatch_head // H\n"
            "        dqphase_h = dqphase_dispatch_head % H\n"
            "        dqphase_token_base = 0\n"
        )
        source = _indexed_region(source, header, launch, 20, 4)
        expected.add("independent_complete_dq")
    actual = _launch_kinds(source)
    changed = {name for name in original if original[name] != actual.get(name)}
    if actual.keys() != original.keys() or changed != expected:
        raise ValueError(f"Single-token launch rewrite mismatch: expected={expected}, actual={changed}")
    if any(actual[name] != "pl.spmd" for name in expected):
        raise ValueError("Single-token consumer was not indexed")
    compile(source, "selected_single_launches.py", "exec")
    return source


def _vector_beta_lines(rows, block_rows, indent, base="publication_base", stop="publication_stop"):
    lines = [
        f"publication_valid = {stop} - {base}",
        f"publication_scratch = pl.create_tile([{block_rows}, DVP], dtype=pl.FP32)",
        f"publication_contractions = pl.set_validshape(pl.fillpad(pl.load(contraction_tape, [{base}, 0], [{block_rows}, DVP], valid_shape=[publication_valid, DV]), 0), {block_rows}, DVP)",
        f"publication_residuals = pl.set_validshape(pl.fillpad(pl.load(residual_tape, [{base}, 0], [{block_rows}, DVP], valid_shape=[publication_valid, DV]), 0), {block_rows}, DVP)",
        "publication_products = pl.mul(publication_contractions, publication_residuals)",
        f"publication_values = pl.reshape(pl.row_sum(publication_products, publication_scratch), [1, {block_rows}])",
        f"pl.store(pl.set_validshape(publication_values, 1, publication_valid), [0, {base}], dbe_flat)",
    ]
    return "".join(" " * indent + line + "\n" for line in lines)


def _line_gradient_owner(region, rows, block_rows):
    dedented = textwrap.dedent(region)
    syntax = ast.parse(dedented)
    launch = syntax.body[0]
    if not isinstance(launch, ast.For) or ast.unparse(launch.iter.func) != "pl.spmd":
        raise ValueError("Expected indexed gradient owner")
    loops = [node for node in launch.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "fused_t"]
    if len(loops) != 1:
        raise ValueError("Expected one gradient token loop")
    token_loop = loops[0]
    loop_kind = ast.unparse(token_loop.iter.func)
    if loop_kind not in ("pl.range", "pl.pipeline"):
        raise ValueError("Unsupported gradient iteration kind")
    lines = dedented.splitlines(True)
    excluded = {"gradient_head_owner", "fused_b", "fused_h", "fused_token_base", "fused_bh"}
    prologue = []
    for statement in launch.body:
        if statement is token_loop:
            break
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id in excluded for target in statement.targets):
            continue
        prologue.extend(lines[statement.lineno - 1:statement.end_lineno])
    body = []
    row_assignments = 0
    for line in lines[token_loop.lineno:token_loop.end_lineno]:
        if line.strip() == "fused_row = (fused_b * T + fused_t) * H + fused_h":
            row_assignments += 1
        else:
            body.append(line)
    if row_assignments != 1:
        raise ValueError("Gradient row hook changed")
    arguments = "publication_base, publication_stop"
    if loop_kind == "pl.pipeline":
        stages = [keyword.value for keyword in token_loop.iter.keywords if keyword.arg == "stage"]
        if len(stages) != 1:
            raise ValueError("Missing consumer pipeline depth")
        arguments += ", stage=" + ast.unparse(stages[0])
    replacement = f'    for publication_block in pl.spmd({(rows + block_rows - 1) // block_rows}, name_hint="fused_token_gradients"):\n'
    replacement += f"        publication_base = publication_block * {block_rows}\n"
    replacement += f"        publication_stop = pl.min(publication_base + {block_rows}, ROWS)\n"
    replacement += "".join("    " + line if line.strip() else line for line in prologue)
    replacement += f"        for fused_row in {loop_kind}({arguments}):\n"
    replacement += "            fused_b = fused_row // (T * H)\n"
    replacement += "            fused_t = (fused_row // H) % T\n"
    replacement += "            fused_h = fused_row % H\n"
    replacement += "            fused_bh = fused_b * H + fused_h\n"
    replacement += "".join("    " + line if line.strip() else line for line in body)
    replacement += _vector_beta_lines(rows, block_rows, 8)
    return replacement


def _line_layout_owner(region, rows, block_rows):
    dedented = textwrap.dedent(region)
    syntax = ast.parse(dedented)
    launch = syntax.body[0]
    loops = [node for node in launch.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "layout_t"]
    if len(loops) != 1:
        raise ValueError("Expected one gradient layout token loop")
    token_loop = loops[0]
    lines = dedented.splitlines(True)
    excluded = {"layout_dispatch_head", "layout_b", "layout_h", "layout_token_base"}
    prologue = []
    for statement in launch.body:
        if statement is token_loop:
            break
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id in excluded for target in statement.targets):
            continue
        prologue.extend(lines[statement.lineno - 1:statement.end_lineno])
    body = []
    row_assignments = 0
    for line in lines[token_loop.lineno:token_loop.end_lineno]:
        if line.strip() == "layout_row = (layout_b * T + layout_t) * H + layout_h":
            row_assignments += 1
        else:
            body.append(line)
    if row_assignments != 1:
        raise ValueError("Layout row hook changed")
    replacement = f'    for publication_block in pl.spmd({(rows + block_rows - 1) // block_rows}, name_hint="dependent_vector_gradient_layout"):\n'
    replacement += f"        publication_base = publication_block * {block_rows}\n"
    replacement += f"        publication_stop = pl.min(publication_base + {block_rows}, ROWS)\n"
    replacement += "".join("    " + line if line.strip() else line for line in prologue)
    replacement += "        for layout_row in pl.range(publication_base, publication_stop):\n"
    replacement += "            layout_b = layout_row // (T * H)\n"
    replacement += "            layout_t = (layout_row // H) % T\n"
    replacement += "            layout_h = layout_row % H\n"
    replacement += "".join("    " + line if line.strip() else line for line in body)
    replacement += _vector_beta_lines(rows, block_rows, 8)
    return replacement


def _light_vector_beta(source, rows, block_rows, mode):
    start, end = _layout_region(source, "fused_token_gradients")
    region = source[start:end]
    removable = ("fused_dbeta_acc =", "fused_psd =", "fused_dbe0 =", "pl.write(dbeta_scratch,")
    removed = [line for line in region.splitlines(True) if line.strip().startswith(removable)]
    if len(removed) != 5:
        raise ValueError(f"Expected five scalar dbeta statements, found {len(removed)}")
    region = "".join(line for line in region.splitlines(True) if not line.strip().startswith(removable))
    has_layout = "dependent_vector_gradient_layout" in _launch_kinds(source)
    fuse_gradient = mode == 2
    if fuse_gradient:
        region = _line_gradient_owner(region, rows, block_rows)
    source = source[:start] + region + source[end:]
    compact_start, compact_end = _layout_region(source, "dbeta_compact")
    if mode == 1 or (mode == 3 and not has_layout):
        publication = f'    for publication_block in pl.spmd({(rows + block_rows - 1) // block_rows}, name_hint="dbeta_vector_publish"):\n'
        publication += f"        publication_base = publication_block * {block_rows}\n"
        publication += f"        publication_stop = pl.min(publication_base + {block_rows}, ROWS)\n"
        publication += _vector_beta_lines(rows, block_rows, 8)
    else:
        publication = ""
    source = source[:compact_start] + publication + source[compact_end:]
    if mode == 3 and has_layout:
        layout_start, layout_end = _layout_region(source, "dependent_vector_gradient_layout")
        layout = _line_layout_owner(source[layout_start:layout_end], rows, block_rows)
        source = source[:layout_start] + layout + source[layout_end:]
    allocation = "    dbeta_scratch = pl.create_tensor([B * T * H, 16], dtype=pl.FP32)\n"
    if source.count(allocation) != 1:
        raise ValueError("Expected one scalar scratch allocation")
    source = source.replace(allocation, "", 1)
    if "dbeta_scratch" in source or "fused_dbeta_acc" in source:
        raise ValueError("A remaining producer needs scalar scratch")
    compile(source, "selected_vector_beta.py", "exec")
    return source


def _short_vector_beta(source, tokens):
    start, end = _layout_region(source, "short_resident_backward")
    region = source[start:end]
    beta_rows = ((tokens + 7) // 8) * 8
    initializer = f"short_beta_products = pl.tile.full([{beta_rows}, DVP], dtype=pl.FP32, value=0.0)"
    lines = region.splitlines(True)
    prefix = None
    output = []
    products = 0
    publications = 0
    for line in lines:
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        if stripped.startswith("head_index ="):
            output.append(line)
            output.append(indent + initializer + "\n")
            prefix = indent
        elif re.match(r"reverse_\d+_psd =", stripped):
            token = int(re.match(r"reverse_(\d+)_psd =", stripped).group(1))
            output.append(indent + f"short_beta_products = pl.tile.assemble(short_beta_products, pl.mul(reverse_{token}_contraction, forward_{token}_residual), [{token}, 0])\n")
            products += 1
        elif re.match(r"reverse_\d+_weight_grad =", stripped):
            continue
        elif stripped.startswith("pl.write(dbeta_scratch,"):
            publications += 1
        else:
            output.append(line)
    if prefix is None or products != tokens or publications != tokens:
        raise ValueError("Short beta producer hooks changed")
    for statement in (
        f"short_beta_scratch = pl.create_tile([{beta_rows}, DVP], dtype=pl.FP32)",
        "short_beta_columns = pl.row_sum(short_beta_products, short_beta_scratch)",
        f"short_beta_carrier = pl.tile.full([{beta_rows}, 16], dtype=pl.FP32, value=1.0)",
        "short_beta_slots = pl.row_expand_mul(short_beta_carrier, short_beta_columns)",
        "short_beta_output = pl.reshape(dbeta_scratch, [B * T, H * 16])",
        "pl.store(pl.set_validshape(short_beta_slots, T, 16), [batch * T, head_index * 16], short_beta_output)",
    ):
        output.append(prefix + statement + "\n")
    replacement = "".join(output)
    if "_weight_grad =" in replacement or "_psd =" in replacement:
        raise ValueError("Short scalar beta reduction remains")
    source = source[:start] + replacement + source[end:]
    compile(source, "selected_short_vector_beta.py", "exec")
    return source


def _tile_total8(product, prefix, width):
    return [
        f"{prefix}_parts = pl.col_sum(pl.reshape({product}, [{width} // 8, 8]))",
        f"{prefix}_carrier = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)",
        f"{prefix}_scratch = pl.create_tile([8, 8], dtype=pl.FP32)",
        f"{prefix}_rows = pl.col_expand({prefix}_carrier, {prefix}_parts)",
        f"{prefix}_total = pl.reshape(pl.row_sum({prefix}_rows, {prefix}_scratch), [1, 8])",
    ]


def _tile_norm_lines(raw, normalized, inverse, prefix):
    lines = _tile_total8(f"pl.mul({raw}, {raw})", prefix, "DKP")
    lines += [
        f"{prefix}_ones = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)",
        f"{inverse} = pl.div({prefix}_ones, pl.sqrt(pl.add({prefix}_total, 1e-6)))",
        f"{normalized} = pl.reshape(pl.col_expand_mul(pl.reshape({raw}, [DKP // 8, 8]), {inverse}), [1, DKP])",
    ]
    return lines


def _tile_adjoint_lines(gradient, normalized, inverse, prefix):
    lines = _tile_total8(f"pl.mul({gradient}, {normalized})", prefix, "DKP")
    lines += [
        f"{prefix}_correction = pl.reshape(pl.col_expand_mul(pl.reshape({normalized}, [DKP // 8, 8]), {prefix}_total), [1, DKP])",
        f"{prefix}_difference = pl.sub({gradient}, {prefix}_correction)",
        f"{gradient} = pl.reshape(pl.col_expand_mul(pl.reshape({prefix}_difference, [DKP // 8, 8]), {inverse}), [1, DKP])",
        f"{gradient} = pl.add({gradient}, 0.0)",
    ]
    return lines


def _vector_slot_compact(source):
    start, end = _layout_region(source, "dbeta_compact")
    replacement = '''    for compact_block in pl.spmd((B * T * H + 15) // 16, name_hint="dbeta_compact"):
        compact_base = compact_block * 16
        compact_valid = pl.min(16, B * T * H - compact_base)
        compact_slots = pl.set_validshape(pl.fillpad(pl.load(dbeta_scratch, [compact_base, 0], [16, 16], valid_shape=[compact_valid, 16]), 0), 16, 16)
        compact_scratch = pl.create_tile([16, 16], dtype=pl.FP32)
        compact_values = pl.mul(pl.reshape(pl.row_sum(compact_slots, compact_scratch), [1, 16]), 0.0625)
        pl.store(pl.set_validshape(compact_values, 1, compact_valid), [0, compact_base], dbe_flat)
'''
    return source[:start] + replacement + source[end:]


def _short_tile_reductions(source, tokens, mask):
    start, end = _layout_region(source, "short_resident_backward")
    lines = source[start:end].splitlines(True)
    output = []
    position = 0
    norm_count = 0
    adjoint_count = 0
    beta_count = 0
    while position < len(lines):
        line = lines[position]
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        norm = re.fullmatch(r"(forward_\d+)_sq = pl.mul\((\w+), \2\)", stripped)
        adjoint = re.fullmatch(r"(reverse_\d+)_psd = pl.col_sum\(pl.reshape\(pl.mul\((\w+), (\w+)\), \[DKP // 8, 8\]\)\)", stripped)
        beta = re.fullmatch(r"(reverse_(\d+))_psd = pl.col_sum\(pl.reshape\(pl.mul\((\w+), (\w+)\), \[DVP // 8, 8\]\)\)", stripped)
        if norm and mask & 1:
            block = lines[position:position + 5]
            normalized = re.fullmatch(r"\s*(\w+) = pl.mul\((\w+), (\w+)\)\n", block[-1])
            if normalized is None:
                raise ValueError("Short norm block changed")
            target, raw, inverse = normalized.groups()
            output += [indent + item + "\n" for item in _tile_norm_lines(raw, target, inverse, target + "_tile")]
            norm_count += 1
            position += 5
            continue
        if adjoint and mask & 2:
            match = re.fullmatch(r"\s*(\w+) = pl.mul\(pl.sub\(\1, pl.mul\((\w+), (\w+)\)\), (\w+)\)\n", lines[position + 2])
            if match is None:
                raise ValueError("Short adjoint block changed")
            gradient, normalized, dot, inverse = match.groups()
            if mask & 1:
                inverse_tile = inverse
            else:
                inverse_tile = gradient + "_inverse_tile"
                output.append(indent + f"{inverse_tile} = pl.tile.full([1, 8], dtype=pl.FP32, value={inverse})\n")
            output += [indent + item + "\n" for item in _tile_adjoint_lines(gradient, normalized, inverse_tile, gradient + "_tile")]
            adjoint_count += 1
            position += 3
            continue
        if norm and not mask & 1 and mask & 2:
            pass
        if beta and mask & 4:
            prefix, token, contraction, residual = beta.groups()
            output.append(indent + f"short_beta_products = pl.tile.assemble(short_beta_products, pl.mul({contraction}, {residual}), [{token}, 0])\n")
            if not lines[position + 2].strip().startswith("pl.write(dbeta_scratch,"):
                raise ValueError("Short beta publication changed")
            beta_count += 1
            position += 3
            continue
        output.append(line)
        if stripped.startswith("head_index =") and mask & 4:
            padded = ((tokens + 7) // 8) * 8
            output.append(indent + f"short_beta_products = pl.tile.full([{padded}, DVP], dtype=pl.FP32, value=0.0)\n")
        position += 1
    normalized_path = any("_sq = pl.mul(" in line for line in lines)
    if normalized_path and mask & 1 and norm_count != 2 * tokens:
        raise ValueError(f"Expected forward q/k norms: {norm_count}")
    if normalized_path and mask & 2 and adjoint_count != 2 * tokens:
        raise ValueError(f"Expected dq/dk adjoints: {adjoint_count}")
    if normalized_path and mask & 1 and not mask & 2:
        for token in range(tokens):
            for stem in ("dq", "dk"):
                prefix = f"reverse_{token}_{stem}_row"
                inverse = f"forward_{token}_{'q' if stem == 'dq' else 'k'}inv"
                old = next(item for item in output if item.strip().startswith(prefix + " = pl.mul(pl.sub("))
                indent = old[:len(old) - len(old.lstrip())]
                target = old.strip().replace(f", {inverse})", ", 1.0)")
                result = [target, f"{prefix} = pl.reshape(pl.col_expand_mul(pl.reshape({prefix}, [DKP // 8, 8]), {inverse}), [1, DKP])"]
                output[output.index(old):output.index(old) + 1] = [indent + item + "\n" for item in result]
    if mask & 4:
        if beta_count != tokens:
            raise ValueError("Missing short beta token")
        indent = " " * 12
        padded = ((tokens + 7) // 8) * 8
        output += [indent + item + "\n" for item in (
            f"short_beta_scratch = pl.create_tile([{padded}, DVP], dtype=pl.FP32)",
            "short_beta_values = pl.row_sum(short_beta_products, short_beta_scratch)",
            f"short_beta_slots = pl.tile.full([{padded}, 16], dtype=pl.FP32, value=0.0)",
            "short_beta_slots = pl.row_expand(short_beta_slots, short_beta_values)",
        )]
        for token in range(tokens):
            output += [indent + f"short_beta_slot_{token} = pl.tile.slice(short_beta_slots, [1, 16], [{token}, 0])\n",
                       indent + f"pl.store(short_beta_slot_{token}, [(batch * T + {token}) * H + head_index, 0], dbeta_scratch)\n"]
    source = source[:start] + "".join(output) + source[end:]
    if mask & 4:
        source = _vector_slot_compact(source)
    return source


def _region_tile_adjoints(source, region_name, mask):
    start, end = _layout_region(source, region_name)
    lines = source[start:end].splitlines(True)
    output = []
    position = 0
    changed = 0
    while position < len(lines):
        line = lines[position]
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        inverse = re.fullmatch(r"(\w+) = pl.read\((prepared_(query|key)_inverse), \[(\w+), 0\]\)", stripped)
        if inverse and mask & (1 if inverse.group(3) == "query" else 2):
            target, tensor, stem, row = inverse.groups()
            output.append(indent + f"{target} = pl.load({tensor}, [{row}, 0], [1, 8])\n")
            position += 1
            continue
        product = re.fullmatch(r"(\w+) = pl.col_sum\(pl.reshape\(pl.mul\((\w+), (\w+)\), \[DKP // 8, 8\]\)\)", stripped)
        if product and position + 2 < len(lines):
            match = re.fullmatch(r"\s*(\w+) = pl.mul\(pl.sub\(\1, pl.mul\((\w+), (\w+)\)\), (\w+)\)\n", lines[position + 2])
            if match:
                gradient, normalized, dot, inverse = match.groups()
                bit = 1 if "dq" in gradient else 2
                if mask & bit:
                    output += [indent + item + "\n" for item in _tile_adjoint_lines(gradient, normalized, inverse, gradient + "_tile")]
                    changed += 1
                    position += 3
                    continue
        output.append(line)
        position += 1
    if not changed:
        return source
    return source[:start] + "".join(output) + source[end:]


def _narrow_tile_adjoints(source, mask, scope):
    names = _launch_kinds(source)
    source = _region_tile_adjoints(source, "fused_token_gradients", mask)
    if scope and "dependent_vector_gradient_layout" in names:
        source = _region_tile_adjoints(source, "dependent_vector_gradient_layout", mask)
    return source


def _checkpoint_tile_reductions(source, beta_mode, finish_norm, block_rows):
    if finish_norm:
        source = _region_tile_adjoints(source, "checkpoint_gradient_finish", 2)
    if not beta_mode:
        return source
    bounded = "contraction_tape = pl.create_tensor([AHEADS * T," in source
    remove = ("checkpoint_partial_beta =", "fused_psd =", "fused_dbe0 =", "pl.write(checkpoint_partial_beta,", "fused_dbeta_acc =", "checkpoint_beta =", "pl.write(dbeta_scratch,")
    source = "".join(line for line in source.splitlines(True) if not line.strip().startswith(remove))
    if bounded:
        start, end = _layout_region(source, "checkpoint_gradient_finish")
        indent = " " * 8
        publisher = f'''for beta_head_block in pl.spmd(AHEADS * ((T + {block_rows - 1}) // {block_rows}), name_hint="checkpoint_beta_vector_publish"):
    beta_local_head = beta_head_block // ((T + {block_rows - 1}) // {block_rows})
    beta_token_base = (beta_head_block % ((T + {block_rows - 1}) // {block_rows})) * {block_rows}
    beta_head = head_base + beta_local_head
    beta_batch = beta_head // H
    beta_head_index = beta_head % H
    beta_products = pl.tile.full([{block_rows}, DVP], dtype=pl.FP32, value=0.0)
    for beta_lane in pl.unroll({block_rows}):
        beta_token = beta_token_base + beta_lane
        if beta_token < T:
            beta_global_row = (beta_batch * T + beta_token) * H + beta_head_index
            beta_contraction = pl.load(contraction_tape, [beta_local_head * T + beta_token, 0], [1, DVP])
            beta_residual = pl.load(residual_tape, [beta_global_row, 0], [1, DVP])
            beta_products = pl.tile.assemble(beta_products, pl.mul(beta_contraction, beta_residual), [beta_lane, 0])
    beta_scratch = pl.create_tile([{block_rows}, DVP], dtype=pl.FP32)
    beta_values = pl.row_sum(beta_products, beta_scratch)
    beta_slots = pl.tile.full([{block_rows}, 16], dtype=pl.FP32, value=0.0)
    beta_slots = pl.row_expand(beta_slots, beta_values)
    for beta_lane in pl.unroll({block_rows}):
        beta_token = beta_token_base + beta_lane
        if beta_token < T:
            beta_global_row = (beta_batch * T + beta_token) * H + beta_head_index
            beta_slot = pl.tile.slice(beta_slots, [1, 16], [beta_lane, 0])
            pl.store(beta_slot, [beta_global_row, 0], dbeta_scratch)
'''
        source = source[:end] + "\n" + "".join(indent + line if line.strip() else line for line in publisher.splitlines(True)) + source[end:]
        source = _vector_slot_compact(source)
    else:
        start, end = _layout_region(source, "dbeta_compact")
        publisher = f'    for publication_block in pl.spmd((B * T * H + {block_rows - 1}) // {block_rows}, name_hint="checkpoint_beta_vector_publish"):\n'
        publisher += f"        publication_base = publication_block * {block_rows}\n"
        publisher += f"        publication_stop = pl.min(publication_base + {block_rows}, ROWS)\n"
        publisher += _vector_beta_lines(0, block_rows, 8)
        source = source[:start] + publisher + source[end:]
        source = "".join(line for line in source.splitlines(True) if not line.strip().startswith("dbeta_scratch ="))
    if "checkpoint_partial_beta" in source or "fused_dbe0" in source or "fused_dbeta_acc" in source:
        raise ValueError("Checkpoint scalar beta survived")
    ast.parse(source)
    return source


def _coalescing_launch(source, name):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Expected one selected JIT function")
    matches = [node for node in functions[0].body if isinstance(node, ast.For)
               and isinstance(node.iter, ast.Call)
               and ast.unparse(node.iter.func) == "pl.spmd"
               and any(keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value == name for keyword in node.iter.keywords)]
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level indexed launch: {name}")
    return matches[0]


def _attach_ready_beta(source, batches, tokens, heads, group_size):
    rows = batches * tokens * heads
    slabs = (rows + 15) // 16
    owners = batches * heads * ((tokens + group_size - 1) // group_size)
    if owners < slabs:
        return source
    global_tapes = (
        "    contraction_tape = pl.create_tensor([ROWS, ADVP], dtype=pl.FP32)",
        "    residual_tape = pl.create_tensor([B * T * H, DVP], dtype=pl.FP32)",
        "    adjoint_tape = pl.create_tensor([B * H * TP1 * DKP, ADVP], dtype=pl.FP32)",
    )
    if any(source.count(allocation) != 1 for allocation in global_tapes):
        return source
    producer = _coalescing_launch(source, "fused_token_gradients")
    publisher = _coalescing_launch(source, "dbeta_vector_publish")
    if not isinstance(producer.target, ast.Name) or producer.target.id != "gradient_block":
        raise ValueError("Gradient owner decoding changed")
    if not isinstance(producer.iter.args[0], ast.Constant) or producer.iter.args[0].value != owners:
        raise ValueError("Gradient task count changed")
    token_loops = [node for node in producer.body if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == "fused_t"]
    if len(token_loops) != 1 or producer.body[-1] is not token_loops[0]:
        raise ValueError("Expected the inherited terminal gradient token loop")
    lines = source.splitlines(True)
    publication = "".join(lines[publisher.lineno:publisher.end_lineno])
    expected = (
        "        publication_base = publication_block * 16\n"
        "        publication_stop = pl.min(publication_base + 16, ROWS)\n"
        + _vector_beta_lines(rows, 16, 8)
    )
    if publication != expected:
        raise ValueError("Earned standalone beta vector formula changed")
    sidecar = f"        if gradient_block < {slabs}:\n"
    sidecar += "".join("    " + line.replace("publication_block", "gradient_block")
                       for line in publication.splitlines(True))
    output = []
    for index, line in enumerate(lines, 1):
        if not publisher.lineno <= index <= publisher.end_lineno:
            output.append(line)
        if index == producer.end_lineno:
            output.append(sidecar)
    result = "".join(output)
    modified = _coalescing_launch(result, "fused_token_gradients")
    modified.body.pop()
    if ast.dump(producer, include_attributes=False) != ast.dump(modified, include_attributes=False):
        raise ValueError("Sidecar changed original gradient work or ownership")
    if "dbeta_vector_publish" in _launch_kinds(result):
        raise ValueError("Standalone beta launch survived")
    return result


def _complete_layout_locally(source):
    if "dependent_vector_gradient_layout" not in _launch_kinds(source):
        return source
    producer = _coalescing_launch(source, "fused_token_gradients")
    layout = _coalescing_launch(source, "dependent_vector_gradient_layout")
    token_loops = [node for node in layout.body if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == "layout_t"]
    if len(token_loops) != 1:
        raise ValueError("Packed layout token loop changed")
    lines = source.splitlines(True)
    epilogue = []
    matrix_loads = 0
    for statement in token_loops[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = ast.unparse(statement.targets[0])
            if name == "layout_row":
                continue
            if name == "layout_gradient_matrix":
                if "pl.load(gradient_columns," not in ast.unparse(statement):
                    raise ValueError("Expected the original packed-column reload")
                epilogue.append("            layout_gradient_matrix = pl.add(packed_gradients, 0.0)\n")
                matrix_loads += 1
                continue
        epilogue.extend(line.replace("layout_row", "fused_row")
                        for line in lines[statement.lineno - 1:statement.end_lineno])
    if matrix_loads != 1:
        raise ValueError("Expected one packed gradient reload")
    stores = [node for node in ast.walk(producer) if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.store"
              and isinstance(node.value.args[-1], ast.Name) and node.value.args[-1].id == "gradient_columns"]
    if len(stores) != 1:
        raise ValueError("Expected one original packed gradient store")
    store = stores[0]
    allocation_lines = [index for index, line in enumerate(lines, 1)
                        if line.strip().startswith("gradient_columns = pl.create_tensor(")]
    if len(allocation_lines) != 1:
        raise ValueError("Expected one private packed-column allocation")
    output = []
    for index, line in enumerate(lines, 1):
        if index == store.lineno:
            output.extend(epilogue)
        if (layout.lineno <= index <= layout.end_lineno or store.lineno <= index <= store.end_lineno
                or index in allocation_lines):
            continue
        output.append(line)
    result = "".join(output)
    if "gradient_columns" in result or "dependent_vector_gradient_layout" in _launch_kinds(result):
        raise ValueError("Private packed-layout handoff still has consumers")
    original_loop = next(node for node in producer.body if isinstance(node, ast.For)
                         and isinstance(node.target, ast.Name) and node.target.id == "fused_t")
    modified = _coalescing_launch(result, "fused_token_gradients")
    modified_loop = next(node for node in modified.body if isinstance(node, ast.For)
                         and isinstance(node.target, ast.Name) and node.target.id == "fused_t")
    if ast.dump(original_loop.iter, include_attributes=False) != ast.dump(modified_loop.iter, include_attributes=False):
        raise ValueError("Local layout changed producer scheduling")
    return result


def _coalesce_narrow_phases(source, batches, tokens, heads, group_size, beta_sidecar, local_layout):
    if beta_sidecar:
        source = _attach_ready_beta(source, batches, tokens, heads, group_size)
    if local_layout:
        source = _complete_layout_locally(source)
    compile(source, "selected_coalesced_narrow.py", "exec")
    return source


def _endpoint_source(DK, DV, DKP, DVP, normalized, cast_inputs, shared, endpoints):
    key_pitch = max(16, DKP)
    value_pitch = max(16, DVP)
    packet = 32 if cast_inputs else 16
    bytes_per_element = 2 if cast_inputs else 4
    staged = {
        "y": (DV * bytes_per_element) % 64 != 0,
        "dq": (DK * bytes_per_element) % 64 != 0,
        "dk": (DK * bytes_per_element) % 64 != 0,
        "dv": (DV * bytes_per_element) % 64 != 0,
        "dg": (DK * 4) % 64 != 0,
    }
    header = _KERNEL[:_KERNEL.index("    di2 =")]
    header += "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    allocations = []
    for name, required in staged.items():
        if required:
            pitch = value_pitch if name in ("y", "dv") else key_pitch
            allocations.append(f"    endpoint_{name} = pl.create_tensor([ROWS, {pitch}], dtype=pl.FP32)")
            width = DV if name in ("y", "dv") else DK
            allocations.append(f"    {name}_flat = pl.reshape({name}2, [1, ROWS * {width}])")
    if shared:
        for name in ("q", "k", "decay"):
            allocations.append(f"    endpoint_{name}prep = pl.create_tensor([ROWS, {key_pitch}], dtype=pl.FP32)")
        allocations.append("    endpoint_inverse = pl.create_tensor([ROWS, 16], dtype=pl.FP32)")
    source = header + "\n".join(allocations) + "\n"

    def load(tensor, width, upcast=False):
        expression = f"pl.load({tensor}, [row, 0], [1, {width}])"
        return f"pl.cast({expression}, pl.FP32)" if upcast and cast_inputs else expression

    def preparation(prefix, query, key, decay, inverses=False, use_shared=True):
        result = []
        if shared and use_shared:
            for name, wanted in (("q", query), ("k", key), ("decay", decay)):
                if wanted:
                    result.append(f"{prefix}_{name} = " + load(f"endpoint_{name}prep", DKP))
            if inverses and normalized:
                result.extend([f"{prefix}_qinv = pl.read(endpoint_inverse, [row, 0])",
                               f"{prefix}_kinv = pl.read(endpoint_inverse, [row, 1])"])
        else:
            for name, wanted in (("q", query), ("k", key)):
                if wanted:
                    result.append(f"{prefix}_{name}raw = " + load(name + "2", DKP, True))
                    if normalized:
                        result.extend(_norm8(f"{prefix}_{name}raw", f"{prefix}_{name}",
                                             f"{prefix}_{name}inv", f"{prefix}_{name}scratch", "").replace(
                                                 "ps", prefix + "_" + name + "partials").replace(
                                                 "sq =", prefix + "_" + name + "sq =").replace(
                                                 "pl.reshape(sq,", f"pl.reshape({prefix}_{name}sq,").splitlines())
                    else:
                        result.append(f"{prefix}_{name} = {prefix}_{name}raw")
            if decay:
                result.append(f"{prefix}_decay = pl.exp(" + load("g2", DKP) + ")")
        return result

    def scratch(prefix):
        return [f"{prefix}_{name}scratch = pl.create_tile([1, 8], dtype=pl.FP32)" for name in ("q", "k")] if normalized and not shared else []

    if shared:
        source += '    for row in pl.parallel(ROWS):\n        with pl.at(level=pl.Level.CORE_GROUP, name_hint="endpoint_prepare"):\n'
        prepared = []
        if normalized:
            prepared += [f"prep_{name}scratch = pl.create_tile([1, 8], dtype=pl.FP32)" for name in ("q", "k")]
        prepared += preparation("prep", True, True, True, use_shared=False)
        for name in ("q", "k", "decay"):
            prepared.append(f"pl.store(prep_{name}, [row, 0], endpoint_{name}prep)")
        if normalized:
            prepared.extend(["pl.write(endpoint_inverse, [row, 0], prep_qinv)", "pl.write(endpoint_inverse, [row, 1], prep_kinv)"])
        source += "".join(" " * 12 + line + "\n" for line in prepared)

    def output(name, value):
        if staged[name]:
            return [f"pl.store({value}, [row, 0], endpoint_{name})"]
        if name != "dg" and cast_inputs:
            return [f"{name}_published = pl.cast({value}, PL_DTYPE)", f"pl.store({name}_published, [row, 0], {name}2)"]
        return [f"pl.store({value}, [row, 0], {name}2)"]

    def normalize_gradient(value, vector, inverse, prefix):
        if not normalized:
            return []
        return _dot_row(value, vector, prefix + "_dot", "").replace("CH", "DKP").splitlines() + [
            f"{value} = pl.mul(pl.sub({value}, pl.mul({vector}, {prefix}_dot)), {inverse})"]

    body = [
        f"endpoint = endpoint_base + endpoint_slot",
        "head = endpoint // T",
        "target = endpoint % T",
        "batch = head // H",
        "head_index = head % H",
        "initial = pl.load(s2, [head * DK, 0], [DKP, DVP])",
        "if target == 0:",
        "    pl.store(initial, [(batch * TP1 * H + head_index) * DKP, 0], hs2)",
    ]
    body += scratch("prefix")
    body += ["for forward_token, (prefix_carry,) in pl.range(target, init_values=(initial,)):",
             "    row = (batch * T + forward_token) * H + head_index"]
    body += ["    " + line for line in preparation("prefix", False, True, True)]
    body += [
        "    prefix_key = pl.reshape(prefix_k, [DKP, 1])",
        "    prefix_decayed = pl.row_expand_mul(prefix_carry, pl.reshape(prefix_decay, [DKP, 1]))",
        "    prefix_value = " + load("v2", DVP, True),
        "    prefix_residual = pl.sub(prefix_value, pl.col_sum(pl.row_expand_mul(prefix_decayed, prefix_key)))",
        "    prefix_beta = pl.read(b1, [row, 0])",
        "    prefix_next = pl.add(prefix_decayed, pl.row_expand_mul(pl.col_expand(prefix_decayed, prefix_residual), pl.mul(prefix_key, prefix_beta)))",
        "    state_before = pl.yield_(prefix_next)",
        "row = (batch * T + target) * H + head_index",
    ]
    body += scratch("local") + preparation("local", True, True, True, True)
    body += [
        "local_key = pl.reshape(local_k, [DKP, 1])",
        "local_query = pl.reshape(local_q, [DKP, 1])",
        "local_decay = pl.reshape(local_decay, [DKP, 1])",
        "local_decayed = pl.row_expand_mul(state_before, local_decay)",
        "local_value = " + load("v2", DVP, True),
        "local_residual = pl.sub(local_value, pl.col_sum(pl.row_expand_mul(local_decayed, local_key)))",
        "local_beta = pl.read(b1, [row, 0])",
        "state_after = pl.add(local_decayed, pl.row_expand_mul(pl.col_expand(local_decayed, local_residual), pl.mul(local_key, local_beta)))",
        "pl.store(state_after, [((batch * TP1 + target + 1) * H + head_index) * DKP, 0], hs2)",
        "if target == T - 1:",
        "    pl.store(state_after, [head * DK, 0], f2)",
        "local_y = pl.mul(pl.col_sum(pl.row_expand_mul(state_after, local_query)), SCALE)",
    ]
    body += output("y", "local_y")
    body += ["incoming = pl.load(df2, [head * DK, 0], [DKP, DVP])"]
    body += scratch("suffix")
    body += ["for reverse_index, (suffix_carry,) in pl.range(T - target - 1, init_values=(incoming,)):",
             "    suffix_token = T - 1 - reverse_index",
             "    row = (batch * T + suffix_token) * H + head_index"]
    body += ["    " + line for line in preparation("suffix", True, True, True)]
    body += [
        "    suffix_key = pl.reshape(suffix_k, [DKP, 1])",
        "    suffix_query = pl.reshape(suffix_q, [DKP, 1])",
        "    suffix_dy = " + load("dy2", DVP, True),
        "    suffix_injected = pl.add(suffix_carry, pl.row_expand_mul(pl.col_expand(suffix_carry, suffix_dy), pl.mul(suffix_query, SCALE)))",
        "    suffix_contraction = pl.col_sum(pl.row_expand_mul(suffix_injected, suffix_key))",
        "    suffix_beta = pl.read(b1, [row, 0])",
        "    suffix_dresidual = pl.mul(suffix_contraction, suffix_beta)",
        "    suffix_corrected = pl.sub(suffix_injected, pl.row_expand_mul(pl.col_expand(suffix_injected, suffix_dresidual), suffix_key))",
        "    suffix_next = pl.row_expand_mul(suffix_corrected, pl.reshape(suffix_decay, [DKP, 1]))",
        "    adjoint_after = pl.yield_(suffix_next)",
        "row = (batch * T + target) * H + head_index",
        "local_dy = " + load("dy2", DVP, True),
        "local_dquery = pl.mul(pl.col_sum(pl.transpose(pl.col_expand_mul(state_after, local_dy), 0, 1)), SCALE)",
    ]
    body += normalize_gradient("local_dquery", "local_q", "local_qinv", "query")
    body += output("dq", "local_dquery")
    body += [
        "local_injected = pl.add(adjoint_after, pl.row_expand_mul(pl.col_expand(adjoint_after, local_dy), pl.mul(local_query, SCALE)))",
        "local_contraction = pl.col_sum(pl.row_expand_mul(local_injected, local_key))",
        "local_dresidual = pl.mul(local_contraction, local_beta)",
    ]
    body += output("dv", "local_dresidual")
    body += _dot_row("local_contraction", "local_residual", "local_dbeta", "").replace("CH", "DVP").splitlines()
    body += [
        "pl.write(dbeta_scratch, [row, 0], local_dbeta)",
        "local_key_first = pl.col_sum(pl.transpose(pl.col_expand_mul(local_injected, local_residual), 0, 1))",
        "local_key_second = pl.col_sum(pl.transpose(pl.col_expand_mul(local_decayed, local_dresidual), 0, 1))",
        "local_dkey = pl.sub(pl.mul(local_key_first, local_beta), local_key_second)",
    ]
    body += normalize_gradient("local_dkey", "local_k", "local_kinv", "key")
    body += output("dk", "local_dkey")
    body += [
        "local_corrected = pl.sub(local_injected, pl.row_expand_mul(pl.col_expand(local_injected, local_dresidual), local_key))",
        "local_dgate = pl.mul(pl.col_sum(pl.transpose(pl.mul(local_corrected, state_before), 0, 1)), pl.reshape(local_decay, [1, DKP]))",
    ]
    body += output("dg", "local_dgate")
    body += [
        "if target == 0:",
        "    local_initial_grad = pl.row_expand_mul(local_corrected, local_decay)",
        "    pl.store(local_initial_grad, [head * DK, 0], dh2)",
        "    pl.store(local_initial_grad, [head * DK, 0], di2)",
    ]
    source += f'    for endpoint_base in pl.parallel(0, ROWS, {endpoints}):\n'
    source += '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_endpoint_recompute"):\n'
    source += f'            for endpoint_slot in pl.range({endpoints}):\n                if endpoint_base + endpoint_slot < ROWS:\n'
    source += "".join(" " * 20 + line + "\n" for line in body)
    if any(staged.values()):
        source += f'    for packet_base in pl.parallel(0, ROWS * {max(DK, DV)}, {packet}):\n'
        source += '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="endpoint_publication"):\n'
        for name, required in staged.items():
            if not required:
                continue
            width = DV if name in ("y", "dv") else DK
            own_packet = 16 if name == "dg" else packet
            if own_packet != packet:
                pieces = packet // own_packet
            else:
                pieces = 1
            source += f"            for {name}_piece in pl.unroll({pieces}):\n"
            source += f"                {name}_base = packet_base + {name}_piece * {own_packet}\n"
            source += f"                if {name}_base < ROWS * {width}:\n"
            source += f"                    {name}_packet = pl.create_tile([1, {own_packet}], dtype=pl.FP32)\n"
            source += f"                    for {name}_lane in pl.unroll({own_packet}):\n"
            source += f"                        {name}_index = {name}_base + {name}_lane\n"
            source += f"                        if {name}_index < ROWS * {width}:\n"
            source += f"                            pl.tile.write({name}_packet, [0, {name}_lane], pl.read(endpoint_{name}, [{name}_index // {width}, {name}_index % {width}]))\n"
            if name != "dg" and cast_inputs:
                source += f"                    {name}_packet = pl.cast({name}_packet, PL_DTYPE)\n"
            source += f"                    {name}_packet = pl.set_validshape({name}_packet, 1, pl.min({own_packet}, ROWS * {width} - {name}_base))\n"

            source += f"                    pl.store({name}_packet, [0, {name}_base], {name}_flat)\n"
    source += _KERNEL[_KERNEL.index("    for row_base in pl.parallel"):]
    return source


def _stage_owner_layout(source, keys, values, group_size, collection_pipeline):
    if "dependent_vector_gradient_layout" not in _launch_kinds(source):
        return source
    replication = 2 if collection_pipeline else 1
    panel = min(32, values)
    live_bytes = 4 * (group_size * keys * 8 + replication * (
        12 * keys * panel + 12 * keys * 8 + 32 * (keys + values) + 1024))
    if live_bytes > 160 * 1024:
        return source
    producer = _coalescing_launch(source, "fused_token_gradients")
    layout = _coalescing_launch(source, "dependent_vector_gradient_layout")
    collect = [node for node in producer.body if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "fused_t"]
    finish = [node for node in layout.body if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == "layout_t"]
    if len(collect) != 1 or len(finish) != 1:
        raise ValueError("Inherited owner/token loops changed")
    collect = collect[0]
    expected_args = "fused_token_base, pl.min(fused_token_base + EPT, T)"
    if ", ".join(ast.unparse(argument) for argument in collect.iter.args) != expected_args:
        raise ValueError("Inherited gradient interval changed")
    stores = [node for node in ast.walk(collect) if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.store"
              and isinstance(node.value.args[-1], ast.Name) and node.value.args[-1].id == "gradient_columns"]
    if len(stores) != 1 or stores[0] is not collect.body[-1]:
        raise ValueError("Expected terminal packed-column publication")
    store = stores[0]
    if ast.unparse(store.value.args[0]) != "packed_gradients":
        raise ValueError("Packed-gradient arithmetic changed")
    lines = source.splitlines(True)
    epilogue = ["        for staged_t in pl.range(fused_token_base, pl.min(fused_token_base + EPT, T)):\n"]
    reloads = 0
    for statement in finish[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = ast.unparse(statement.targets[0])
            if name == "layout_row":
                epilogue.append("            staged_row = (fused_b * T + staged_t) * H + fused_h\n")
                continue
            if name == "layout_gradient_matrix":
                if "pl.load(gradient_columns," not in ast.unparse(statement):
                    raise ValueError("Expected original packed-layout reload")
                epilogue.append("            layout_gradient_view = pl.tile.slice(staged_packed, [DKP, 8], [(staged_t - fused_token_base) * DKP, 0])\n")
                epilogue.append("            layout_gradient_matrix = pl.add(layout_gradient_view, 0.0)\n")
                reloads += 1
                continue
        epilogue.extend(line.replace("layout_row", "staged_row")
                        for line in lines[statement.lineno - 1:statement.end_lineno])
    if reloads != 1:
        raise ValueError("Expected exactly one layout-matrix load")
    allocation = [index for index, line in enumerate(lines, 1)
                  if line.strip().startswith("gradient_columns = pl.create_tensor(")]
    if len(allocation) != 1:
        raise ValueError("Expected private GM packed allocation")
    loop = "pl.pipeline" if collection_pipeline else "pl.range"
    stage = ", stage=2" if collection_pipeline else ""
    header = (
        "        staged_initial = pl.tile.full([EPT * DKP, 8], dtype=pl.FP32, value=0.0)\n"
        f"        for fused_t, (staged_carry,) in {loop}({expected_args}{stage}, init_values=(staged_initial,)):\n"
    )
    output = []
    for index, line in enumerate(lines, 1):
        if index == collect.lineno:
            output.append(header)
        elif index == store.lineno:
            output.append("            staged_next = pl.tile.assemble(staged_carry, packed_gradients, [(fused_t - fused_token_base) * DKP, 0])\n")
            output.append("            staged_packed = pl.yield_(staged_next)\n")
        elif not (layout.lineno <= index <= layout.end_lineno or index in allocation
                  or store.lineno <= index <= store.end_lineno):
            output.append(line)
        if index == collect.end_lineno:
            output.extend(epilogue)
    result = "".join(output)
    if "gradient_columns" in result or "dependent_vector_gradient_layout" in _launch_kinds(result):
        raise ValueError("Staged completion retained a GM handoff")
    modified = _coalescing_launch(result, "fused_token_gradients")
    if ast.dump(producer.iter, include_attributes=False) != ast.dump(modified.iter, include_attributes=False):
        raise ValueError("Staging changed gradient ownership")
    compile(result, "selected_owner_staged_layout.py", "exec")
    return result


def _light_endpoint_grid(source):
    header = (
        "    for endpoint_base in pl.parallel(0, ROWS, 1):\n"
        '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_endpoint_recompute"):\n'
    )
    launch = '    for endpoint_base in pl.spmd(ROWS, name_hint="token_endpoint_recompute"):\n'
    return _indexed_region(source, header, launch, 12, 4)


def _light_reverse_grid(source):
    marker = 'name_hint="thin_adjoint_reverse"'
    if marker not in source:
        return source
    region_line = next(line for line in source.splitlines() if marker in line)
    if "pl.spmd(" in region_line:
        return source
    bounded = "    for head_base in pl.parallel(0, B * H, AHEADS):" in source
    indent = " " * (8 if bounded else 4)
    owners = "AHEADS * NC" if bounded else "B * H * NC"
    header = (
        indent + f"for owner in pl.parallel({owners}):\n"
        + indent + '    with pl.at(level=pl.Level.CORE_GROUP, name_hint="thin_adjoint_reverse"):\n'
    )
    launch = indent + f'for owner in pl.spmd({owners}, name_hint="thin_adjoint_reverse"):\n'
    return _indexed_region(source, header, launch, len(indent) + 8, len(indent))


def _wide_reverse_owner_grid(source, schedule):
    if schedule == 2:
        return _light_reverse_grid(source)
    if schedule != 1:
        raise ValueError("Unknown reverse schedule")
    bounded = "    for head_base in pl.parallel(0, B * H, AHEADS):" in source
    indent = " " * (8 if bounded else 4)
    heads = "AHEADS" if bounded else "B * H"
    header = (
        indent + f"for owner in pl.parallel({heads} * NC):\n"
        + indent + '    with pl.at(level=pl.Level.CORE_GROUP, name_hint="thin_adjoint_reverse"):\n'
    )
    launch = (
        indent + "for reverse_part in pl.range(NC):\n"
        + indent + f'    for reverse_head in pl.spmd({heads}, name_hint="thin_adjoint_reverse"):\n'
        + indent + "        owner = reverse_head * NC + reverse_part\n"
    )
    if source.count(header) != 1:
        raise ValueError("Expected one ordinary wide reverse owner loop")
    return source.replace(header, launch, 1)


def _owner_slab_extra_bytes(tokens, keys, values, mode):
    rows = ((tokens + 15) // 16) * 16
    return 4 * (10 * rows * keys + 6 * rows * 8 + (4 * rows * values if mode == 2 else 0) + 4096)


def _owner_slab_producer(tokens, cast_inputs, normalized, endpoint, mode):
    rows = ((tokens + 15) // 16) * 16
    lines = []
    for name, tensor, width in (("q", "q", "DK"), ("k", "k", "DK"), ("gate", "g", "DK")):
        start = "target" if endpoint and name == "q" else "0"
        valid = "T - target" if endpoint and name == "q" else "T"
        lines.append(f"slab_{name}_view = pl.reshape({tensor}, [B * T, H * {width}])")
        expression = (f"pl.set_validshape(pl.fillpad(pl.load(slab_{name}_view, "
                      f"[batch * T + {start}, head_index * {width}], [{rows}, DKP], "
                      f"valid_shape=[{valid}, DK]), 0), {rows}, DKP)")
        if cast_inputs and name != "gate":
            expression = f"pl.cast({expression}, pl.FP32)"
        lines.append(f"slab_{name}_raw = {expression}")
        if name == "gate":
            lines.append("slab_decay = pl.exp(slab_gate_raw)")
        elif normalized:
            lines += [
                f"slab_{name}_scratch = pl.create_tile([{rows}, DKP], dtype=pl.FP32)",
                f"slab_{name}_sum = pl.reshape(pl.row_sum(pl.mul(slab_{name}_raw, slab_{name}_raw), slab_{name}_scratch), [1, {rows}])",
                f"slab_{name}_ones = pl.tile.full([1, {rows}], dtype=pl.FP32, value=1.0)",
                f"slab_{name}_inverse = pl.reshape(pl.div(slab_{name}_ones, pl.sqrt(pl.add(slab_{name}_sum, 1e-6))), [{rows}, 1])",
                f"slab_{name} = pl.row_expand_mul(slab_{name}_raw, slab_{name}_inverse)",
                f"slab_{name}_meta_seed = pl.tile.full([{rows}, 8], dtype=pl.FP32, value=1.0)",
                f"slab_{name}_meta = pl.row_expand_mul(slab_{name}_meta_seed, slab_{name}_inverse)",
            ]
        else:
            lines.append(f"slab_{name} = slab_{name}_raw")
    if mode == 2:
        for name, tensor in (("value", "v"), ("dy", "dy")):
            start = "target" if endpoint and name == "dy" else "0"
            valid = "T - target" if endpoint and name == "dy" else "target + 1" if endpoint else "T"
            lines.append(f"slab_{name}_view = pl.reshape({tensor}, [B * T, H * DV])")
            expression = (f"pl.set_validshape(pl.fillpad(pl.load(slab_{name}_view, "
                          f"[batch * T + {start}, head_index * DV], [{rows}, DVP], "
                          f"valid_shape=[{valid}, DV]), 0), {rows}, DVP)")
            if cast_inputs:
                expression = f"pl.cast({expression}, pl.FP32)"
            lines.append(f"slab_{name} = {expression}")
    return lines


def _owner_inverse_multiply(line, inverse):
    import re
    match = re.fullmatch(r"(\w+) = pl.mul\((pl.sub\(.+\)), " + re.escape(inverse) + r"\)", line)
    if match is None:
        raise ValueError("Normalization-adjoint multiply changed: " + line)
    target, difference = match.groups()
    return [f"{target} = {difference}",
            f"{target} = pl.reshape(pl.col_expand_mul(pl.reshape({target}, [DKP // 8, 8]), {inverse}), [1, DKP])"]


def _owner_local_slabs(source, tokens, normalized, cast_inputs, endpoint, mode):
    import re
    lines = source.splitlines(True)
    output = []
    position = 0
    inserted = 0
    preparations = 0
    while position < len(lines):
        line = lines[position]
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        if stripped == "head_index = head % H":
            output.append(line)
            output.extend(indent + item + "\n" for item in _owner_slab_producer(tokens, cast_inputs, normalized, endpoint, mode))
            inserted += 1
            position += 1
            continue
        if re.fullmatch(r"(?:qnorm_scratch|knorm_scratch|(?:prefix|local|suffix)_[qk]scratch) = pl.create_tile\(.+", stripped):
            position += 1
            continue
        resident_raw = re.fullmatch(r"forward_\d+_[qk]raw = .+", stripped) if not endpoint else None
        if resident_raw:
            position += 1
            continue
        resident_norm = re.fullmatch(r"(forward_(\d+)_)[s]q = pl.mul\((forward_\d+_)([qk])raw, .+", stripped) if not endpoint and normalized else None
        resident_plain = re.fullmatch(r"(forward_(\d+)_)([qk])n = forward_\d+_[qk]raw", stripped) if not endpoint and not normalized else None
        if resident_norm or resident_plain:
            match = resident_norm or resident_plain
            prefix, index = match.group(1), match.group(2)
            stem = match.group(4) if resident_norm else match.group(3)
            target, inverse = prefix + stem + "n", prefix + stem + "inv"
            count = 5 if normalized else 1
            if not lines[position + count - 1].strip().startswith(target + " ="):
                raise ValueError("Resident preparation changed")
            output.append(indent + f"{target} = pl.tile.slice(slab_{stem}, [1, DKP], [{index}, 0])\n")
            if normalized:
                output.append(indent + f"{inverse} = pl.tile.slice(slab_{stem}_meta, [1, 8], [{index}, 0])\n")
            preparations += 1
            position += count
            continue
        raw = re.fullmatch(r"((?:prefix|local|suffix)_)([qk])raw = .+", stripped) if endpoint else None
        if raw:
            prefix, stem = raw.groups()
            if endpoint:
                index = "forward_token" if prefix == "prefix_" else "target" if prefix == "local_" else "suffix_token"
                relative = f"{index} - target" if stem == "q" else index
                target = prefix + stem
                inverse = prefix + stem + "inv"
            else:
                index = re.fullmatch(r"forward_(\d+)_", prefix).group(1)
                relative = index
                target = prefix + stem + "n"
                inverse = prefix + stem + "inv"
            count = 6 if normalized else 2
            block = lines[position:position + count]
            if not block[-1].strip().startswith(target + " ="):
                raise ValueError("Preparation block changed: " + "".join(block))
            output.append(indent + f"{target} = pl.tile.slice(slab_{stem}, [1, DKP], [{relative}, 0])\n")
            if normalized:
                output.append(indent + f"{inverse} = pl.tile.slice(slab_{stem}_meta, [1, 8], [{relative}, 0])\n")
            preparations += 1
            position += count
            continue
        decay = re.fullmatch(r"(forward_(\d+)_decay_row) = pl.exp\(.+", stripped) if not endpoint else re.fullmatch(r"((prefix|local|suffix)_decay) = pl.exp\(.+", stripped)
        if decay:
            target, prefix = decay.groups()
            index = ("forward_token" if prefix == "prefix" else "target" if prefix == "local" else "suffix_token") if endpoint else prefix
            output.append(indent + f"{target} = pl.tile.slice(slab_decay, [1, DKP], [{index}, 0])\n")
            position += 1
            continue
        if normalized and " = pl.mul(pl.sub(" in stripped:
            inverse = re.search(r", ((?:forward_\d+_|local_)[qk]inv)\)$", stripped)
            if inverse:
                output.extend(indent + item + "\n" for item in _owner_inverse_multiply(stripped, inverse.group(1)))
                position += 1
                continue
        if mode == 2:
            value = re.fullmatch(r"((?:forward_(\d+)_value)|(?:reverse_(\d+)_output_grad)|(?:(prefix|local)_value)|(?:(local|suffix)_dy)) = .+", stripped)
            if value:
                target, forward, reverse, prefix, suffix = value.groups()
                name = "dy" if reverse is not None or suffix is not None else "value"
                index = forward if forward is not None else reverse if reverse is not None else "forward_token" if prefix == "prefix" else "suffix_token - target" if suffix == "suffix" else "0" if suffix == "local" else "target"
                output.append(indent + f"{target} = pl.tile.slice(slab_{name}, [1, DVP], [{index}, 0])\n")
                position += 1
                continue
        output.append(line)
        position += 1
    if inserted != 1 or preparations != (5 if endpoint else 2 * tokens):
        raise ValueError(f"Unexpected owner-local replacement counts: {inserted}, {preparations}")
    result = "".join(output)
    views = []
    retained = []
    for line in result.splitlines(True):
        if re.fullmatch(r"\s*slab_\w+_view = pl.reshape\(.+\)\n", line):
            views.append("    " + line.strip() + "\n")
        else:
            retained.append(line)
    result = "".join(retained)
    boundary = "    di2 = pl.reshape(dinitial_state, [B * H * DK, DV])\n"
    if result.count(boundary) != 1:
        raise ValueError("Owner slab tensor-view boundary changed")
    result = result.replace(boundary, boundary + "".join(views), 1)
    ast.parse(result)
    return result


def _owner_local_dispatch(source, batches, tokens, heads, keys, values, padded_keys, padded_values,
                          normalized, cast_inputs, mode, resident, neighbor, short_live_bytes):
    aligned = (keys == padded_keys and values == padded_values
               and min(padded_keys, padded_values) >= 16
               and padded_keys % 16 == 0 and padded_values % 16 == 0)
    endpoint_base = 4 * (16 * padded_keys * padded_values + 96 * (padded_keys + padded_values) + 1024)
    extra = _owner_slab_extra_bytes(tokens, padded_keys, padded_values, mode) if mode else 0
    neighbor_selected = (neighbor and cast_inputs and normalized and 8 < tokens <= 32
                         and aligned and max(padded_keys, padded_values) <= 32
                         and batches * heads <= 8 and endpoint_base + extra <= 160 * 1024)
    if neighbor_selected:
        source = _endpoint_source(keys, values, padded_keys, padded_values, normalized, cast_inputs, 0, 1)
        source = _light_endpoint_grid(source)
    if not mode or not aligned:
        return source
    if 'name_hint="token_endpoint_recompute"' in source and endpoint_base + extra <= 160 * 1024:
        return _owner_local_slabs(source, tokens, normalized, cast_inputs, True, mode)
    if resident and 'name_hint="short_resident_backward"' in source and short_live_bytes + extra <= 160 * 1024:
        return _owner_local_slabs(source, tokens, normalized, cast_inputs, False, mode)
    return source


def _single_materialized_row(column, name):
    return [
        f"{name}_carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)",
        f"{name}_replicated = pl.row_expand_mul({name}_carrier, {column})",
        f"{name}_materialized = pl.add(pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0), {name}_replicated)",
        f"{name}_transposed = pl.transpose({name}_materialized, 0, 1)",
        f"{name} = pl.tile.slice({name}_transposed, [1, DKP], [0, 0])",
    ]


def _single_normalization(source):
    import re
    lines = source.splitlines(True)
    output = []
    position = 0
    norms = 0
    adjoints = 0
    while position < len(lines):
        line = lines[position]
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        norm = re.fullmatch(r"\w+ = pl.mul\((\w+), \1\)", stripped)
        dot = re.fullmatch(r"\w+ = pl.col_sum\(pl.reshape\(pl.mul\((\w+), (\w+)\), \[DKP // 8, 8\]\)\)", stripped)
        if norm and position + 4 < len(lines) and "1e-6" in lines[position + 2]:
            final = re.fullmatch(r"\s*(\w+) = pl.mul\((\w+), (\w+)\)\n", lines[position + 4])
            if final is None or final.group(2) != norm.group(1):
                raise ValueError("Single-token normalization changed")
            target, raw, inverse = final.groups()
            output.extend(indent + item + "\n" for item in _tile_norm_lines(raw, target, inverse, target + "_single_tile"))
            norms += 1
            position += 5
            continue
        if dot:
            final = re.fullmatch(r"\s*(\w+) = pl.mul\(pl.sub\(\1, pl.mul\((\w+), \w+\)\), (\w+)\)\n", lines[position + 2])
            if final is None:
                raise ValueError("Single-token normalization adjoint changed")
            gradient, normalized, inverse = final.groups()
            if (gradient, normalized) != dot.groups():
                raise ValueError("Single-token normalization dot changed")
            output.extend(indent + item + "\n" for item in _tile_adjoint_lines(gradient, normalized, inverse, gradient + "_single_tile"))
            adjoints += 1
            position += 3
            continue
        output.append(line)
        position += 1
    if (norms, adjoints) != (4, 2):
        raise ValueError(f"Single-token normalization census: {norms}, {adjoints}")
    return "".join(output)


def _single_tile_completion(source, tokens, keys, values, padded_keys, padded_values, normalized, mask, tile_norm, scope):
    if tokens != 1 or "sharded_single_token" not in source:
        return source
    if min(padded_keys, padded_values) < 64 or padded_keys % 16:
        return source
    if not scope and (keys != padded_keys or values != padded_values):
        return source
    panel = math.gcd(min(32, padded_values), padded_values)
    live_bytes = 4 * (5 * padded_keys * panel + 12 * padded_keys * 8 + 1024)
    if live_bytes > 160 * 1024:
        return source
    layout_keys = ((padded_keys + 15) // 16) * 16
    if mask & 1:
        old = (
            "dqphase_dq_row = pl.create_tile([1, DKP], dtype=pl.FP32)\n"
            "            for dqphase_dq_row_lane in pl.unroll(DKP):\n"
            "                pl.tile.write(dqphase_dq_row, [0, dqphase_dq_row_lane], pl.read(dqphase_dquery, [dqphase_dq_row_lane, 0]))"
        )
        replacement = "\n            ".join(_single_materialized_row("dqphase_dquery", "dqphase_dq_row"))
        source = _replace_once(source, old, replacement)
    if mask & 2:
        for stem, column in (("k", "dkey"), ("g", "dgate")):
            source = _replace_once(
                source,
                f"single_partial_{stem} = pl.create_tensor([B * H * SNC * {layout_keys}, 8], dtype=pl.FP32)",
                f"single_partial_{stem} = pl.create_tensor([B * H * SNC, {layout_keys}], dtype=pl.FP32)",
            )
            old = f"pl.store(pl.row_expand_mul(carrier, {column}), [partial_row, 0], single_partial_{stem})"
            row = f"single_private_{stem}"
            replacement = _single_materialized_row(column, row)
            replacement.append(f"pl.store({row}, [owner, 0], single_partial_{stem})")
            source = _replace_once(source, old, "\n        ".join(replacement))
            source = _replace_once(source,
                f"single_reduce_{stem}sum = pl.tile.full([{layout_keys}, 8], dtype=pl.FP32, value=0.0)",
                f"single_reduce_{stem}sum = pl.tile.full([1, {layout_keys}], dtype=pl.FP32, value=0.0)")
            source = _replace_once(source,
                f"pl.load(single_partial_{stem}, [single_reduce_partial_row, 0], [{layout_keys}, 8])",
                f"pl.load(single_partial_{stem}, [single_reduce_bh * SNC + single_reduce_part, 0], [1, {layout_keys}])")
            source = _replace_once(source,
                f"single_reduce_{stem}_transposed = pl.transpose(single_reduce_{stem}sum, 0, 1)\n"
                f"        single_reduce_{stem}_row = pl.tile.slice(single_reduce_{stem}_transposed, [1, DKP], [0, 0])",
                f"single_reduce_{stem}_row = pl.tile.slice(single_reduce_{stem}sum, [1, DKP], [0, 0])")
    if tile_norm and normalized:
        source = _single_normalization(source)
    compile(source, "selected_single_tile_completion.py", "exec")
    return source


def _producer_beta_vectors(source, batches, tokens, heads, keys, values, padded_keys, padded_values, normalized, mask):
    eligible = (normalized and tokens >= 17 and keys == padded_keys and values == padded_values
                and padded_keys <= 64 and padded_values <= 64 and padded_values % 32 == 0
                and batches * heads * (padded_values // 32) <= 16)
    if not eligible:
        return source
    allocation = "    prepared_query = pl.create_tensor([ROWS, DKP], dtype=pl.FP32)\n"
    source = _replace_once(source, allocation,
        "    producer_beta_flat = pl.reshape(beta, [1, ROWS])\n"
        "    producer_beta_tape = pl.create_tensor([ROWS, 8], dtype=pl.FP32)\n" + allocation)
    begin, end = _layout_region(source, "batched_token_prepare")
    region = source[begin:end]
    loop = next(node for node in ast.walk(ast.parse(textwrap.dedent(region))) if isinstance(node, ast.For))
    indent = " " * (4 + loop.body[0].col_offset)
    region += "".join(indent + line + "\n" for line in (
        "producer_beta_input = pl.set_validshape(pl.fillpad(pl.load(producer_beta_flat, [0, preparation_base], [1, 16], valid_shape=[1, preparation_valid]), 0), 1, 16)",
        "producer_beta_column = pl.reshape(producer_beta_input, [16, 1])",
        "producer_beta_carrier = pl.tile.full([16, 8], dtype=pl.FP32, value=1.0)",
        "producer_beta_replicated = pl.row_expand_mul(producer_beta_carrier, producer_beta_column)",
        "pl.store(pl.set_validshape(producer_beta_replicated, preparation_valid, 8), [preparation_base, 0], producer_beta_tape)",
    ))
    source = source[:begin] + region + source[end:]
    if mask & 1:
        source = _deletion_line(source, "replay_local_bt = pl.read(b1, [replay_local_row, 0])", [
            "replay_local_beta_vector = pl.load(producer_beta_tape, [replay_local_row, 0], [1, 8])"])
        source = _deletion_line(source, "replay_local_kb = pl.mul(replay_local_kcol, replay_local_bt)", [
            "replay_local_key_groups = pl.reshape(replay_local_kn1, [DKP // 8, 8])",
            "replay_local_weighted_groups = pl.col_expand_mul(replay_local_key_groups, replay_local_beta_vector)",
            "replay_local_kb = pl.reshape(replay_local_weighted_groups, [DKP, 1])"])
    if mask & 2:
        source = _deletion_line(source, "bt = pl.read(b1, [row, 0])", [
            "reverse_beta_vector = pl.load(producer_beta_tape, [row, 0], [1, 8])"])
        source = _deletion_line(source, "drc0 = pl.mul(ccv0, bt)", [
            "reverse_contraction_groups = pl.reshape(ccv0, [CH // 8, 8])",
            "reverse_weighted_groups = pl.col_expand_mul(reverse_contraction_groups, reverse_beta_vector)",
            "drc0 = pl.reshape(reverse_weighted_groups, [1, CH])"])
    return source


def _dual_projection_line(source, original, replacement):
    lines = source.splitlines(True)
    matches = [index for index, line in enumerate(lines) if line.strip() == original]
    if len(matches) != 1:
        raise ValueError(f"Expected one projection hook: {original}")
    index = matches[0]
    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    lines[index:index + 1] = [indent + line + "\n" for line in replacement]
    return "".join(lines)


def _checked_state_dual_projection(source, B, T, H, DK, DV, DKP, DVP,
                                   FCH, GCH, normalized, checkpoint_selected,
                                   scope, workspace):
    if not (normalized and T >= 32 and DK == DKP and DV == DVP
            and FCH == GCH == 32 and DVP % 32 == 0):
        return source
    wide = bool(scope & 1) and checkpoint_selected and DKP == DVP == 128
    narrow = bool(scope & 2) and DKP == 64 and DVP <= 64
    if not (wide or narrow):
        return source
    consumer_name = "independent_complete_dq" if wide else "fused_token_gradients"
    if f'name_hint="{consumer_name}"' not in source or "prepared_query = pl.create_tensor" not in source:
        return source
    begin, end = _layout_region(source, "fwd_replay")
    replay = source[begin:end]
    replay = _dual_projection_line(replay, "replay_local_qn1 = pl.load(prepared_query, [replay_local_row, 0], [1, DKP])", [])
    replay = _dual_projection_line(replay, "replay_local_qcol = pl.reshape(replay_local_qn1, [DKP, 1])", [])
    replay = _dual_projection_line(replay, "replay_local_yt = pl.mul(pl.col_sum(pl.row_expand_mul(replay_local_state, replay_local_qcol), replay_local_key_reduction_scratch), SCALE)", [])
    cast_lines = [line.strip() for line in replay.splitlines() if line.strip().startswith("replay_local_yv = ")]
    if len(cast_lines) != 2:
        raise ValueError("Expected projection cast and clip")
    original_cast = cast_lines[0].split(" = ", 1)[1]
    for line in cast_lines:
        replay = _dual_projection_line(replay, line, [])
    replay = _dual_projection_line(replay, "pl.store(replay_local_yv, [replay_local_row, replay_local_cb], y2)", [])
    if "y2)" in replay or "replay_local_q" in replay:
        raise ValueError("Replay projection deletion incomplete")
    source = source[:begin] + replay + source[end:]
    begin, end = _layout_region(source, consumer_name)
    consumer = source[begin:end]
    if wide:
        scratch = "dual_projection_scratch" if workspace else "dqphase_dq_scratch"
        consumer = _dual_projection_line(consumer, "dqphase_qn2 = pl.load(prepared_query, [dqphase_row, 0], [1, DKP])", [
            "dqphase_qn2 = pl.load(prepared_query, [dqphase_row, 0], [1, DKP])",
            "dual_projection_qcol = pl.reshape(dqphase_qn2, [DKP, 1])",
        ])
        anchor = "dqphase_dqacc = pl.add(dqphase_dqacc, dqphase_dq_partial)"
        state, query, row, column, width = "dqphase_next_state", "dual_projection_qcol", "dqphase_row", "dqphase_cb", "GCH"
        scratch_anchor = "dqphase_dq_scratch = pl.create_tile([DKP, GCH], dtype=pl.FP32)"
    else:
        scratch = "dual_projection_scratch" if workspace else "fused_tmpA"
        anchor = "fused_dgacc = pl.add(fused_dgacc, fused_dgp0)"
        state, query, row, column, width = "fused_H1c0", "fused_qcol", "fused_row", "fused_cb", "GCH"
        scratch_anchor = "fused_tmpA = pl.create_tile([DKP, GCH], dtype=pl.FP32)"
    if workspace:
        consumer = _dual_projection_line(consumer, scratch_anchor, [scratch_anchor, f"dual_projection_scratch = pl.create_tile([DKP, {width}], dtype=pl.FP32)"])
    consumer = _dual_projection_line(consumer, anchor, [
        anchor,
        f"dual_projection_value = pl.mul(pl.col_sum(pl.row_expand_mul({state}, {query}), {scratch}), SCALE)",
        "dual_projection_output = " + original_cast.replace("replay_local_yt", "dual_projection_value"),
        f"dual_projection_output = pl.set_validshape(dual_projection_output, 1, pl.min({width}, DV - {column}))",
        f"pl.store(dual_projection_output, [{row}, {column}], y2)",
    ])
    return source[:begin] + consumer + source[end:]


def _owned_geometry(batches, tokens, heads, keys, values, key_pitch, state_pitch,
                    adjoint_pitch, forward_width, reverse_width, gradient_width,
                    panels, group_tokens, normalized, cast_inputs, output_type,
                    resident, endpoint):
    packed = (key_pitch % 16 == 0 and
              (not normalized or batches * tokens * heads >= 2048))
    inherited_projection = (normalized and tokens >= 32 and keys == key_pitch
                            and values == state_pitch and forward_width == gradient_width == 32
                            and state_pitch % 32 == 0 and key_pitch == 64)
    supported = (tokens > 1 and key_pitch <= 64 and state_pitch <= 64
                 and not resident and not endpoint and not inherited_projection
                 and not (packed and tokens > 32))
    return dict(B=batches, T=tokens, H=heads, DK=keys, DV=values, DKP=key_pitch,
                DVP=state_pitch, ADVP=adjoint_pitch, FCH=forward_width, CH=reverse_width,
                GCH=gradient_width, GNC=panels, EPT=group_tokens,
                normalized=normalized, cast=cast_inputs, output_type=output_type,
                packed=packed, supported=supported,
                isolated=not normalized or panels == 1,
                pipeline=tokens >= 17 and (normalized or packed),
                replay_pipeline=tokens >= 17,
                beta_vectors=normalized and tokens >= 17 and keys == key_pitch
                             and values == state_pitch and state_pitch % 32 == 0
                             and batches * heads * (state_pitch // 32) <= 16,
                groups=(tokens + group_tokens - 1) // group_tokens,
                query_reads=("prepared_query", "prepared_query_inverse") if normalized else ("q2",),
                pair_reads=("hs2", "dy2"), pair_writes=("y2", "dq2"),
                state_reads=("hs2", "residual_tape", "contraction_tape", "adjoint_tape"),
                state_writes=("dk2", "dv2", "dg2", "dbe_flat"),
                slots={"dq": 0, "dk": 1, "dg": 2}, adjoint_heads="global")


def _owned_load(tensor, offsets, shape, valid=None):
    expression = f"pl.load({tensor}, {offsets}, {shape}"
    if valid is None:
        return expression + ")"
    rows, columns = shape.strip("[]").split(", ")
    return f"pl.set_validshape(pl.fillpad({expression}, valid_shape={valid}), 0), {rows}, {columns})"


def _owned_cast(expression, geometry, output=False):
    if not geometry["cast"]:
        return expression
    target = geometry["output_type"] if output else "FP32"
    return f"pl.cast({expression}, target_type=pl.{target})"


def _owned_row_load(tensor, row, geometry):
    valid = f"[1, {geometry['DK']}]" if geometry["DK"] != geometry["DKP"] else None
    expression = _owned_load(tensor, f"[{row}, 0]", "[1, DKP]", valid)
    return _owned_cast(expression, geometry) if tensor in ("q2", "k2") else expression


def _owned_launch(geometry, name, prefix):
    groups = geometry["groups"]
    owners = geometry["B"] * geometry["H"] * groups
    lines = [f'for {prefix}block in pl.spmd({owners}, name_hint="{name}"):',
             f"    {prefix}head = {prefix}block // {groups}",
             f"    {prefix}b = {prefix}head // H",
             f"    {prefix}h = {prefix}head % H",
             f"    {prefix}base = ({prefix}block % {groups}) * EPT"]
    return lines


def _owned_loop(geometry, prefix):
    bounds = f"{prefix}base, pl.min({prefix}base + EPT, T)"
    return f"for {prefix}t in pl.pipeline({bounds}, stage=2):" if geometry["pipeline"] else f"for {prefix}t in pl.range({bounds}):"


def _owned_completion(geometry, prefix, stem, packed=False):
    accumulator = prefix + stem + "acc"
    row = prefix + stem + "_row"
    lines = []
    if packed:
        slot = geometry["slots"][stem]
        lines.append(f"{row} = pl.tile.slice({prefix}transposed, [1, DKP], [{slot}, 0])")
    elif stem == "dq" and not geometry["normalized"]:
        lines += [f"{row} = pl.create_tile([1, DKP], dtype=pl.FP32)",
                  f"for {prefix}lane in pl.unroll(DKP):",
                  f"    pl.tile.write({row}, [0, {prefix}lane], pl.read({accumulator}, [{prefix}lane, 0]))"]
    else:
        lines.append(f"{row} = pl.reshape({accumulator}, [1, DKP])")
    if stem == "dq":
        lines.append(f"{row} = pl.mul({row}, SCALE)")
    if stem in ("dq", "dk") and geometry["normalized"]:
        normalized = prefix + ("qn" if stem == "dq" else "kn")
        inverse = prefix + ("qinv" if stem == "dq" else "kinv")
        lines += _tile_adjoint_lines(row, normalized, inverse, row + "_tile")
        lines.append(f"{row} = pl.add({row}, 0.0)")
    if stem != "dg":
        lines.append(f"{row} = {_owned_cast(row, geometry, True)}")
    if geometry["DK"] != geometry["DKP"]:
        lines.append(f"{row} = pl.set_validshape({row}, 1, {geometry['DK']})")
    lines.append(f"pl.store({row}, [{prefix}row, 0], {stem}2)")
    return lines


def _owned_carriers(geometry, prefix, stems):
    lines = [f"{prefix}carrier = pl.tile.full([DKP, 8], dtype=pl.FP32, value=1.0)"]
    for stem in stems:
        slot = geometry["slots"][stem]
        lines += [f"{prefix}{stem}_mask = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)",
                  f"pl.tile.write({prefix}{stem}_mask, [0, {slot}], 1.0)",
                  f"{prefix}{stem}_carrier = pl.col_expand_mul({prefix}carrier, {prefix}{stem}_mask)"]
    return lines


def _owned_pack(geometry, prefix, stems):
    lines = [f"{prefix}packed_{stem} = pl.row_expand_mul({prefix}{stem}_carrier, {prefix}{stem}acc)" for stem in stems]
    expression = prefix + "packed_" + stems[0]
    for stem in stems[1:]:
        expression = f"pl.add({expression}, {prefix}packed_{stem})"
    lines += [f"{prefix}matrix = pl.add({expression}, 0.0)",
              f"{prefix}transposed = pl.transpose({prefix}matrix, 0, 1)"]
    for stem in stems:
        lines += _owned_completion(geometry, prefix, stem, True)
    return lines


def _owned_pair_panel(geometry, prefix):
    row, column = prefix + "row", prefix + "cb"
    dy = _owned_cast(_owned_load("dy2", f"[{row}, {column}]", "[1, GCH]",
                               f"[1, pl.min(GCH, DV - {column})]"), geometry)
    state = _owned_load("hs2", f"[(({prefix}b * TP1 + {prefix}t + 1) * H + {prefix}h) * DKP, {column}]",
                        "[DKP, GCH]", f"[DKP, pl.min(GCH, DVP - {column})]")
    return [f"{prefix}cb = {prefix}part * GCH", f"{prefix}dy = {dy}",
            f"{prefix}next_state = {state}",
            f"{prefix}pdq = pl.row_sum(pl.col_expand_mul({prefix}next_state, {prefix}dy), {prefix}scratch)",
            f"{prefix}dqacc = pl.add({prefix}dqacc, {prefix}pdq)",
            f"{prefix}y_value = pl.mul(pl.col_sum(pl.row_expand_mul({prefix}next_state, {prefix}qcol), {prefix}scratch), SCALE)",
            f"if {column} < DV:",
            f"    {prefix}y_output = {_owned_cast(prefix + 'y_value', geometry, True)}",
            f"    {prefix}y_output = pl.set_validshape({prefix}y_output, 1, pl.min(GCH, DV - {column}))",
            f"    pl.store({prefix}y_output, [{row}, {column}], y2)"]


def _owned_query(geometry, prefix):
    if geometry["normalized"]:
        return [f"{prefix}qn = pl.load(prepared_query, [{prefix}row, 0], [1, DKP])",
                f"{prefix}qinv = pl.load(prepared_query_inverse, [{prefix}row, 0], [1, 8])",
                f"{prefix}qcol = pl.reshape({prefix}qn, [DKP, 1])"]
    return [f"{prefix}qn = {_owned_row_load('q2', prefix + 'row', geometry)}",
            f"{prefix}qcol = pl.reshape({prefix}qn, [DKP, 1])"]


def _owned_indent(lines, spaces):
    return [" " * spaces + line for line in lines]


def _owned_pair(geometry):
    prefix = "pair_"
    lines = _owned_launch(geometry, "explicit_paired_projection", prefix)
    lines += _owned_indent([f"{prefix}scratch = pl.create_tile([DKP, GCH], dtype=pl.FP32)"], 4)
    if geometry["packed"]:
        lines += _owned_indent(_owned_carriers(geometry, prefix, ("dq",)), 4)
    lines += _owned_indent([_owned_loop(geometry, prefix)], 4)
    body = [f"{prefix}row = ({prefix}b * T + {prefix}t) * H + {prefix}h"]
    body += _owned_query(geometry, prefix)
    body += [f"{prefix}zeros = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)",
             f"{prefix}dqacc = pl.row_sum({prefix}zeros, {prefix}scratch)",
             f"for {prefix}part in pl.range(GNC):"]
    body += _owned_indent(_owned_pair_panel(geometry, prefix), 4)
    body += (_owned_pack(geometry, prefix, ("dq",)) if geometry["packed"]
             else _owned_completion(geometry, prefix, "dq"))
    lines += _owned_indent(body, 8)
    return "\n".join(_owned_indent(lines, 4)) + "\n"


def _owned_state_panel(geometry, prefix, mixed):
    row, column = prefix + "row", prefix + "cb"
    previous = _owned_load("hs2", f"[(({prefix}b * TP1 + {prefix}t) * H + {prefix}h) * DKP, {column}]",
                           "[DKP, GCH]", f"[DKP, pl.min(GCH, DVP - {column})]")
    residual = _owned_load("residual_tape", f"[{row}, {column}]", "[1, GCH]", f"[1, pl.min(GCH, DVP - {column})]")
    contraction = _owned_load("contraction_tape", f"[{row}, {column}]", "[1, GCH]", f"[1, pl.min(GCH, ADVP - {column})]")
    adjoint = _owned_load("adjoint_tape", f"[({prefix}bh * TP1 + {prefix}t) * DKP, {column}]",
                          "[DKP, GCH]", f"[DKP, pl.min(GCH, ADVP - {column})]")
    lines = _owned_pair_panel(geometry, prefix) if mixed else [f"{column} = {prefix}part * GCH"]
    lines += [f"{prefix}previous = {previous}", f"{prefix}residual = {residual}",
              f"{prefix}contraction = {contraction}",
              f"{prefix}dr = pl.mul({prefix}contraction, {prefix}beta)",
              f"{prefix}adjoint = {adjoint}",
              f"{prefix}dk1 = pl.row_sum(pl.col_expand_mul({prefix}adjoint, {prefix}residual), {prefix}scratch)"]
    if geometry["normalized"]:
        lines.append(f"{prefix}decay = pl.reshape(pl.load(prepared_decay, [{row}, 0], [1, DKP]), [DKP, 1])")
    else:
        lines += [f"{prefix}gate = pl.reshape({_owned_row_load('g2', row, geometry)}, [DKP, 1])",
                  f"{prefix}decay = pl.exp({prefix}gate)"]
    lines += [f"{prefix}dk2 = pl.mul(pl.row_sum(pl.col_expand_mul({prefix}previous, {prefix}dr), {prefix}scratch), {prefix}decay)",
              f"{prefix}dkn = pl.sub(pl.mul({prefix}dk1, {prefix}beta), {prefix}dk2)",
              f"{prefix}dhp = pl.sub({prefix}adjoint, pl.row_expand_mul(pl.col_expand({prefix}adjoint, {prefix}dr), {prefix}kcol))",
              f"{prefix}dhn = pl.row_expand_mul({prefix}dhp, {prefix}decay)",
              f"{prefix}dgp = pl.row_sum(pl.mul({prefix}dhn, {prefix}previous), {prefix}scratch)",
              f"if {column} < DV:",
              f"    {prefix}dv_output = {_owned_cast(prefix + 'dr', geometry, True)}",
              f"    {prefix}dv_output = pl.set_validshape({prefix}dv_output, 1, pl.min(GCH, DV - {column}))",
              f"    pl.store({prefix}dv_output, [{row}, {column}], dv2)",
              f"{prefix}dkacc = pl.add({prefix}dkacc, {prefix}dkn)",
              f"{prefix}dgacc = pl.add({prefix}dgacc, {prefix}dgp)"]
    return lines


def _owned_state(geometry):
    prefix = "state_"
    mixed = not geometry["isolated"]
    stems = ("dq", "dk", "dg") if mixed else ("dk", "dg")
    lines = _owned_launch(geometry, "fused_token_gradients", prefix)
    lines += _owned_indent([f"{prefix}bh = {prefix}b * H + {prefix}h",
                            f"{prefix}scratch = pl.create_tile([DKP, GCH], dtype=pl.FP32)"], 4)
    if geometry["packed"]:
        lines += _owned_indent(_owned_carriers(geometry, prefix, stems), 4)
    lines += _owned_indent([_owned_loop(geometry, prefix)], 4)
    body = [f"{prefix}zeros = pl.tile.full([DKP, 8], dtype=pl.FP32, value=0.0)",
            f"{prefix}row = ({prefix}b * T + {prefix}t) * H + {prefix}h"]
    if mixed:
        body += _owned_query(geometry, prefix)
    if geometry["normalized"]:
        body += [f"{prefix}kn = pl.load(prepared_key, [{prefix}row, 0], [1, DKP])",
                 f"{prefix}kinv = pl.load(prepared_key_inverse, [{prefix}row, 0], [1, 8])"]
    else:
        body += [f"{prefix}kn = {_owned_row_load('k2', prefix + 'row', geometry)}"]
    body += [f"{prefix}kcol = pl.reshape({prefix}kn, [DKP, 1])",
             f"{prefix}beta = pl.read(b1, [{prefix}row, 0])"]
    body += [f"{prefix}{stem}acc = pl.row_sum({prefix}zeros, {prefix}scratch)" for stem in stems]
    body += [f"for {prefix}part in pl.range(GNC):"]
    body += _owned_indent(_owned_state_panel(geometry, prefix, mixed), 4)
    if geometry["packed"]:
        body += _owned_pack(geometry, prefix, stems)
    else:
        for stem in stems:
            body += _owned_completion(geometry, prefix, stem)
    lines += _owned_indent(body, 8)
    owners = geometry["B"] * geometry["H"] * geometry["groups"]
    slabs = (geometry["B"] * geometry["T"] * geometry["H"] + 15) // 16
    if owners < slabs:
        raise ValueError("No complete dbeta sidecar ownership")
    lines += _owned_indent([f"if {prefix}block < {slabs}:",
                            f"    publication_base = {prefix}block * 16",
                            "    publication_stop = pl.min(publication_base + 16, ROWS)"], 4)
    lines += _vector_beta_lines(0, 16, 8).rstrip().splitlines()
    return "\n".join(_owned_indent(lines, 4)) + "\n"


def _owned_replay(geometry):
    prefix = "replay_local_"
    state_load = _owned_load("s2", "[replay_local_bh * DK, replay_local_cb]", "[DKP, FCH]",
                             "[DK, pl.min(FCH, DV - replay_local_cb)]")
    lines = ['for replay_local_replay_block in pl.spmd(B * H * (DVP // FCH), name_hint="fwd_replay"):',
             "    replay_local_replay_head = replay_local_replay_block // (DVP // FCH)",
             "    replay_local_b = replay_local_replay_head // H",
             "    replay_local_h = replay_local_replay_head % H",
             "    replay_local_cb = (replay_local_replay_block % (DVP // FCH)) * FCH",
             "    replay_local_bh = replay_local_b * H + replay_local_h",
             f"    replay_local_state = {state_load}"]
    if not geometry["normalized"]:
        lines.append("    replay_local_w8 = pl.create_tile([1, 8], dtype=pl.FP32)")
    lines += ["    replay_local_key_reduction_scratch = pl.create_tile([DKP, FCH], dtype=pl.FP32)",
              "    pl.store(pl.set_validshape(replay_local_state, DKP, pl.min(FCH, DVP - replay_local_cb)), [((replay_local_b * TP1) * H + replay_local_h) * DKP, replay_local_cb], hs2)"]
    loop = "pl.pipeline(T, stage=2)" if geometry["replay_pipeline"] else "pl.range(T)"
    lines.append(f"    for replay_local_t in {loop}:")
    body = ["replay_local_row = (replay_local_b * T + replay_local_t) * H + replay_local_h"]
    if geometry["normalized"]:
        body += ["replay_local_kn1 = pl.load(prepared_key, [replay_local_row, 0], [1, DKP])",
                 "replay_local_decay = pl.reshape(pl.load(prepared_decay, [replay_local_row, 0], [1, DKP]), [DKP, 1])"]
    else:
        body += [f"replay_local_kt = {_owned_row_load('k2', 'replay_local_row', geometry)}",
                 "replay_local_kn1 = replay_local_kt",
                 f"replay_local_gt = pl.reshape({_owned_row_load('g2', 'replay_local_row', geometry)}, [DKP, 1])",
                 "replay_local_decay = pl.exp(replay_local_gt)"]
    value_load = _owned_cast(_owned_load("v2", "[replay_local_row, replay_local_cb]", "[1, FCH]",
                                         "[1, pl.min(FCH, DV - replay_local_cb)]"), geometry)
    body += ["replay_local_state = pl.row_expand_mul(replay_local_state, replay_local_decay)",
             "replay_local_kcol = pl.reshape(replay_local_kn1, [DKP, 1])",
             "replay_local_kh = pl.col_sum(pl.row_expand_mul(replay_local_state, replay_local_kcol), replay_local_key_reduction_scratch)",
             f"replay_local_r = pl.sub({value_load}, replay_local_kh)",
             "pl.store(pl.set_validshape(replay_local_r, 1, pl.min(FCH, DVP - replay_local_cb)), [replay_local_row, replay_local_cb], residual_tape)"]
    if geometry["beta_vectors"]:
        body += ["replay_local_beta_vector = pl.load(producer_beta_tape, [replay_local_row, 0], [1, 8])",
                 "replay_local_key_groups = pl.reshape(replay_local_kn1, [DKP // 8, 8])",
                 "replay_local_weighted_groups = pl.col_expand_mul(replay_local_key_groups, replay_local_beta_vector)",
                 "replay_local_kb = pl.reshape(replay_local_weighted_groups, [DKP, 1])"]
    else:
        body += ["replay_local_bt = pl.read(b1, [replay_local_row, 0])",
                 "replay_local_kb = pl.mul(replay_local_kcol, replay_local_bt)"]
    body += ["replay_local_upd = pl.row_expand_mul(pl.col_expand(replay_local_state, replay_local_r), replay_local_kb)",
             "replay_local_state = pl.add(replay_local_state, replay_local_upd)",
             "pl.store(pl.set_validshape(replay_local_state, DKP, pl.min(FCH, DVP - replay_local_cb)), [((replay_local_b * TP1 + replay_local_t + 1) * H + replay_local_h) * DKP, replay_local_cb], hs2)"]
    lines += _owned_indent(body, 8)
    lines += ["    replay_local_state = pl.set_validshape(replay_local_state, DK, pl.min(FCH, DV - replay_local_cb))",
              "    pl.store(replay_local_state, [replay_local_bh * DK, replay_local_cb], f2)"]
    return "\n".join(_owned_indent(lines, 4)) + "\n"


def _owned_boundaries(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    lines = source.splitlines(True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    regions = {}
    for statement in function.body:
        if not isinstance(statement, ast.For):
            continue
        tags = [keyword.value.value for node in ast.walk(statement.iter) if isinstance(node, ast.Call)
                for keyword in node.keywords if keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)]
        if tags:
            regions[tags[0]] = (offsets[statement.lineno - 1], offsets[statement.end_lineno])
    return regions


def _owned_program(source, geometry, pair_order):
    if not geometry["supported"]:
        return source
    regions = _owned_boundaries(source)
    if "fwd_replay" not in regions or "fused_token_gradients" not in regions:
        return source
    replay_begin, replay_end = regions["fwd_replay"]
    consumer_begin, consumer_end = regions["fused_token_gradients"]
    reverse = source[replay_end:consumer_begin]
    import textwrap
    reverse_tree = ast.parse(textwrap.dedent(reverse))
    if len(reverse_tree.body) != 1 or not isinstance(reverse_tree.body[0], ast.For):
        return source
    reverse_tags = [keyword.value.value for node in ast.walk(reverse_tree) if isinstance(node, ast.Call)
                    for keyword in node.keywords if keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)]
    if reverse_tags != ["thin_adjoint_reverse"]:
        return source
    required = ["residual_tape", "adjoint_tape", "contraction_tape"]
    if geometry["normalized"]:
        required += ["prepared_query", "prepared_query_inverse", "prepared_key", "prepared_key_inverse", "prepared_decay"]
    if geometry["beta_vectors"]:
        required += ["producer_beta_tape"]
    for tensor in required:
        if f"    {tensor} = pl.create_tensor" not in source[:replay_begin]:
            return source
    pair = _owned_pair(geometry) if geometry["isolated"] else ""
    replay = _owned_replay(geometry)
    state = _owned_state(geometry)
    programs = replay + (pair + reverse if pair_order else reverse + pair) + state
    result = source[:replay_begin] + programs + source[consumer_end:]
    ast.parse(result)
    return result


def _wide_dq_completion(source, tokens, keys, values, key_pitch, value_pitch,
                        normalized, checkpoint_selected, form):
    if not (form and normalized and checkpoint_selected and tokens >= 32
            and keys == key_pitch == 128 and values == value_pitch == 128):
        return source
    regions = _owned_boundaries(source)
    if "independent_complete_dq" not in regions:
        return source
    begin, end = regions["independent_complete_dq"]
    region = source[begin:end]
    indent = " " * 12
    inverse = "dqphase_qinv2 = pl.read(prepared_query_inverse, [dqphase_row, 0])"
    packing = "\n".join(indent + statement for statement in (
        "dqphase_dq_row = pl.create_tile([1, DKP], dtype=pl.FP32)",
        "for dqphase_dq_row_lane in pl.unroll(DKP):",
        "    pl.tile.write(dqphase_dq_row, [0, dqphase_dq_row_lane], pl.read(dqphase_dquery, [dqphase_dq_row_lane, 0]))",
    )) + "\n"
    adjoint = "\n".join(indent + statement for statement in (
        "dqphase_psd = pl.col_sum(pl.reshape(pl.mul(dqphase_dq_row, dqphase_qn2), [DKP // 8, 8]))",
        "dqphase_dq_row_dot = " + " + ".join(f"pl.read(dqphase_psd, [0, {lane}])" for lane in range(8)),
        "dqphase_dq_row = pl.mul(pl.sub(dqphase_dq_row, pl.mul(dqphase_qn2, dqphase_dq_row_dot)), dqphase_qinv2)",
    )) + "\n"
    required = (
        indent + inverse + "\n", packing, adjoint,
        indent + "dqphase_dquery = pl.mul(dqphase_dqacc, SCALE)\n",
        indent + "pl.store(dqphase_dq_row, [dqphase_row, 0], dq2)\n",
        indent + "dqphase_qn2 = pl.load(prepared_query, [dqphase_row, 0], [1, DKP])\n",
        "dual_projection_value = pl.mul(pl.col_sum(pl.row_expand_mul(dqphase_next_state, dual_projection_qcol), dqphase_dq_scratch), SCALE)",
    )
    if any(region.count(statement) != 1 for statement in required):
        return source
    if form in (1, 2, 3):
        region = region.replace(indent + inverse + "\n", indent + "dqphase_qinv2 = pl.load(prepared_query_inverse, [dqphase_row, 0], [1, 8])\n")
        replacement = _tile_adjoint_lines("dqphase_dq_row", "dqphase_qn2", "dqphase_qinv2", "dqphase_dq_row_tile")
        region = region.replace(adjoint, "".join(indent + statement + "\n" for statement in replacement))
    if form in (2, 4):
        replacement = [
            "dqphase_dq_row = pl.reshape(dqphase_dquery, [1, DKP])",
            "dqphase_dq_row = pl.add(dqphase_dq_row, 0.0)",
        ]
        region = region.replace(packing, "".join(indent + statement + "\n" for statement in replacement))
    elif form == 3:
        replacement = _single_materialized_row("dqphase_dquery", "dqphase_dq_row")
        replacement.append("dqphase_dq_row = pl.add(dqphase_dq_row, 0.0)")
        region = region.replace(packing, "".join(indent + statement + "\n" for statement in replacement))
    result = source[:begin] + region + source[end:]
    ast.parse(result)
    return result


def _tiny_endpoint_vector_publication():
    lines = [
        '    for packet_base in pl.parallel(0, ROWS * 16, 32):',
        '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="endpoint_publication"):',
        '            packet_row = packet_base // 16',
    ]
    for name in ("y", "dq", "dk", "dv"):
        lines.extend([
            f"            {name}_packet_seed = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)",
            f"            {name}_first_row = pl.load(endpoint_{name}, [packet_row, 0], [1, 16])",
            f"            {name}_packet_first = pl.tile.assemble({name}_packet_seed, {name}_first_row, [0, 0])",
            f"            {name}_packet = pl.add({name}_packet_first, 0.0)",
            "            if packet_row + 1 < ROWS:",
            f"                {name}_second_row = pl.load(endpoint_{name}, [packet_row + 1, 0], [1, 16])",
            f"                {name}_packet = pl.tile.assemble({name}_packet, {name}_second_row, [0, 16])",
            f"            {name}_packet_materialized = pl.add({name}_packet, 0.0)",
            f"            {name}_packet_cast = pl.cast({name}_packet_materialized, PL_DTYPE)",
            f"            {name}_packet_valid = pl.set_validshape({name}_packet_cast, 1, pl.min(32, ROWS * 16 - packet_base))",
            f"            pl.store({name}_packet_valid, [0, packet_base], {name}_flat)",
        ])
    return "\n".join(lines) + "\n"


def _tiny_cast_endpoint_source(keys, values, padded_keys, padded_values, normalized, publication_form):
    if (keys, values, padded_keys, padded_values) != (16, 16, 16, 16):
        raise ValueError("Tiny packet publisher requires aligned 16-wide rows")
    source = _endpoint_source(keys, values, padded_keys, padded_values, normalized, True, 0, 1)
    if not publication_form:
        return source
    begin_marker = '    for packet_base in pl.parallel(0, ROWS * 16, 32):\n'
    end_marker = '    for row_base in pl.parallel'
    if source.count(begin_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("Endpoint publisher boundaries changed")
    begin = source.index(begin_marker)
    end = source.index(end_marker, begin)
    return source[:begin] + _tiny_endpoint_vector_publication() + source[end:]
