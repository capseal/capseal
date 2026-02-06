"""Constraint evaluation for geometry AIR rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .geom_air import GeomAIRParams, GeomEvalTable, pack_vm_value, MODULUS


@dataclass
class ConstraintResidual:
    pc_step: int
    gas_step: int
    cnt_step: int
    m11_step: int
    m12_step: int
    m22_step: int
    s_steps: List[int]
    pow_steps: List[int]


@dataclass
class RowValues:
    pc: int
    pc_next: int
    gas: int
    gas_next: int
    opcode: int
    acc: int
    x1: int
    x2: int
    cnt: int
    cnt_next: int
    m11: int
    m11_next: int
    m12: int
    m12_next: int
    m22: int
    m22_next: int
    sketches: List[int]
    sketches_next: List[int]
    pow: List[int]
    pow_next: List[int]


def _get(col: List[int], idx: int) -> int:
    if idx < len(col):
        return col[idx] % MODULUS
    return 0


def _get_with_mask(tbl: GeomEvalTable, name: str, values: List[int], idx: int) -> int:
    val = _get(values, idx)
    if tbl.column_masks:
        mask_vals = tbl.column_masks.get(name)
        if mask_vals is not None:
            val = (val - _get(mask_vals, idx)) % MODULUS
    return val


def _get_sketch(tbl: GeomEvalTable, j: int, idx: int) -> int:
    return _get_with_mask(tbl, f"sketches_{j}", tbl.sketches[j], idx)


def _get_pow(tbl: GeomEvalTable, j: int, idx: int) -> int:
    return _get_with_mask(tbl, f"powers_{j}", tbl.powers[j], idx)


def _row_values_from_table(
    params: GeomAIRParams,
    tbl: GeomEvalTable,
    t: int,
) -> RowValues:
    next_idx = t + 1
    sketches = [_get_sketch(tbl, j, t) for j in range(params.num_challenges)]
    sketches_next = [_get_sketch(tbl, j, next_idx) for j in range(params.num_challenges)]
    pow_vals = [_get_pow(tbl, j, t) for j in range(params.num_challenges)]
    pow_next = [_get_pow(tbl, j, next_idx) for j in range(params.num_challenges)]
    return RowValues(
        pc=_get_with_mask(tbl, "PC", tbl.PC, t),
        pc_next=_get_with_mask(tbl, "PC", tbl.PC, next_idx),
        gas=_get_with_mask(tbl, "GAS", tbl.GAS, t),
        gas_next=_get_with_mask(tbl, "GAS", tbl.GAS, next_idx),
        opcode=_get_with_mask(tbl, "OP", tbl.OP, t),
        acc=_get_with_mask(tbl, "ACC", tbl.ACC, t),
        x1=_get_with_mask(tbl, "X1", tbl.X1, t),
        x2=_get_with_mask(tbl, "X2", tbl.X2, t),
        cnt=_get_with_mask(tbl, "CNT", tbl.CNT, t),
        cnt_next=_get_with_mask(tbl, "CNT", tbl.CNT, next_idx),
        m11=_get_with_mask(tbl, "M11", tbl.M11, t),
        m11_next=_get_with_mask(tbl, "M11", tbl.M11, next_idx),
        m12=_get_with_mask(tbl, "M12", tbl.M12, t),
        m12_next=_get_with_mask(tbl, "M12", tbl.M12, next_idx),
        m22=_get_with_mask(tbl, "M22", tbl.M22, t),
        m22_next=_get_with_mask(tbl, "M22", tbl.M22, next_idx),
        sketches=sketches,
        sketches_next=sketches_next,
        pow=pow_vals,
        pow_next=pow_next,
    )


def eval_constraints_from_row(params: GeomAIRParams, row: RowValues) -> ConstraintResidual:
    pc_res = (row.pc_next - (row.pc + 1)) % MODULUS
    gas_res = (row.gas_next - (row.gas - (1 + (row.opcode % 3)))) % MODULUS
    cnt_res = (row.cnt_next - (row.cnt + 1)) % MODULUS
    m11_res = (row.m11_next - (row.m11 + row.x1 * row.x1)) % MODULUS
    m12_res = (row.m12_next - (row.m12 + row.x1 * row.x2)) % MODULUS
    m22_res = (row.m22_next - (row.m22 + row.x2 * row.x2)) % MODULUS

    val = pack_vm_value(row.pc, row.opcode, row.gas, row.acc)
    s_res: List[int] = []
    pow_res: List[int] = []
    for j, rj in enumerate(params.r_challenges):
        s_res.append((row.sketches_next[j] - (row.sketches[j] + val * row.pow[j])) % MODULUS)
        pow_res.append((row.pow_next[j] - (row.pow[j] * rj)) % MODULUS)

    return ConstraintResidual(
        pc_step=pc_res,
        gas_step=gas_res,
        cnt_step=cnt_res,
        m11_step=m11_res,
        m12_step=m12_res,
        m22_step=m22_res,
        s_steps=s_res,
        pow_steps=pow_res,
    )


def eval_constraints_at_row(params: GeomAIRParams, tbl: GeomEvalTable, t: int) -> ConstraintResidual:
    trace_len = tbl.trace_len
    if trace_len <= 1:
        zero_vec = [0] * params.num_challenges
        return ConstraintResidual(0, 0, 0, 0, 0, 0, zero_vec, zero_vec)
    if t >= trace_len - 1:
        zero_vec = [0] * params.num_challenges
        return ConstraintResidual(0, 0, 0, 0, 0, 0, zero_vec, zero_vec)

    row_vals = _row_values_from_table(params, tbl, t)
    return eval_constraints_from_row(params, row_vals)


def composition_value_from_row(
    params: GeomAIRParams,
    row_vals: RowValues,
    residual: ConstraintResidual,
    idx: int,
    trace_len: int,
    alphas: Dict[str, int],
    sigma_expected: Dict[str, int] | None,
) -> int:
    total = 0

    def add(name: str, value: int) -> None:
        nonlocal total
        alpha = alphas.get(name)
        if alpha is None:
            return
        total = (total + alpha * (value % MODULUS)) % MODULUS

    add("pc_step", residual.pc_step)
    add("gas_step", residual.gas_step)
    add("cnt_step", residual.cnt_step)
    add("m11_step", residual.m11_step)
    add("m12_step", residual.m12_step)
    add("m22_step", residual.m22_step)
    for j, val in enumerate(residual.s_steps):
        add(f"s_step_{j}", val)
    for j, val in enumerate(residual.pow_steps):
        add(f"pow_step_{j}", val)

    final_idx = max(trace_len - 1, 0)
    if sigma_expected and idx == final_idx:
        m11_target = sigma_expected.get("m11")
        if m11_target is not None:
            add("m11_final", (row_vals.m11 - m11_target) % MODULUS)
        m12_target = sigma_expected.get("m12")
        if m12_target is not None:
            add("m12_final", (row_vals.m12 - m12_target) % MODULUS)
        m22_target = sigma_expected.get("m22")
        if m22_target is not None:
            add("m22_final", (row_vals.m22 - m22_target) % MODULUS)
        cnt_target = sigma_expected.get("cnt")
        if cnt_target is not None:
            add("cnt_final", (row_vals.cnt - cnt_target) % MODULUS)

    return total


def row_values_from_columns(
    params: GeomAIRParams,
    current: Dict[str, int],
    nxt: Dict[str, int],
) -> RowValues:
    def g(src: Dict[str, int], name: str) -> int:
        return src.get(name, 0) % MODULUS

    sketches = [g(current, f"sketches_{j}") for j in range(params.num_challenges)]
    sketches_next = [g(nxt, f"sketches_{j}") for j in range(params.num_challenges)]
    pow_vals = [g(current, f"powers_{j}") for j in range(params.num_challenges)]
    pow_next = [g(nxt, f"powers_{j}") for j in range(params.num_challenges)]
    return RowValues(
        pc=g(current, "PC"),
        pc_next=g(nxt, "PC"),
        gas=g(current, "GAS"),
        gas_next=g(nxt, "GAS"),
        opcode=g(current, "OP"),
        acc=g(current, "ACC"),
        x1=g(current, "X1"),
        x2=g(current, "X2"),
        cnt=g(current, "CNT"),
        cnt_next=g(nxt, "CNT"),
        m11=g(current, "M11"),
        m11_next=g(nxt, "M11"),
        m12=g(current, "M12"),
        m12_next=g(nxt, "M12"),
        m22=g(current, "M22"),
        m22_next=g(nxt, "M22"),
        sketches=sketches,
        sketches_next=sketches_next,
        pow=pow_vals,
        pow_next=pow_next,
    )


def build_composition_vector(
    params: GeomAIRParams,
    table: GeomEvalTable,
    alphas: Dict[str, int],
    sigma_expected: Dict[str, int] | None = None,
) -> List[int]:
    """Combine constraint residuals with Fiat–Shamir alphas into C[t]."""

    domain_size = table.domain_size
    trace_len = table.trace_len
    comp = [0] * domain_size

    max_t = max(0, trace_len - 1)
    for t in range(max_t):
        row_vals = _row_values_from_table(params, table, t)
        residual = eval_constraints_from_row(params, row_vals)
        comp[t] = composition_value_from_row(
            params,
            row_vals,
            residual,
            t,
            trace_len,
            alphas,
            sigma_expected,
        )

    if sigma_expected:
        final_idx = max(trace_len - 1, 0)
        if final_idx < domain_size:
            row_vals = _row_values_from_table(params, table, final_idx)
            residual = ConstraintResidual(
                pc_step=0,
                gas_step=0,
                cnt_step=0,
                m11_step=0,
                m12_step=0,
                m22_step=0,
                s_steps=[0] * params.num_challenges,
                pow_steps=[0] * params.num_challenges,
            )
            comp[final_idx] = (
                comp[final_idx]
                + composition_value_from_row(
                    params,
                    row_vals,
                    residual,
                    final_idx,
                    trace_len,
                    alphas,
                    sigma_expected,
                )
            ) % MODULUS

    return comp
