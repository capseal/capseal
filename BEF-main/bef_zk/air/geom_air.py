"""Geometry AIR row/trace builder for the STC-backed zk demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from bef_zk.fri.config import MODULUS  # Goldilocks prime (2^61 - 1)

Field = int


def modp(value: int, modulus: int = MODULUS) -> int:
    return value % modulus


def pack_vm_value(pc: Field, opcode: Field, gas: Field, acc: Field) -> Field:
    pc_mask = pc & ((1 << 16) - 1)
    op_mask = opcode & ((1 << 8) - 1)
    gas_mask = gas & ((1 << 12) - 1)
    acc_mask = acc & ((1 << 25) - 1)
    return ((pc_mask << 45) | (op_mask << 37) | (gas_mask << 25) | acc_mask) % MODULUS


@dataclass
class GeomAIRParams:
    steps: int
    num_challenges: int
    r_challenges: List[Field]
    matrix: List[List[Field]]  # 2x2 dynamics matrix
    seed: bytes = b""
    modulus: int = MODULUS


@dataclass
class GeomInitialState:
    pc: Field = 0
    opcode: Field = 0
    gas: Field = 100
    acc: Field = 0
    x1: Field = 0
    x2: Field = 0
    cnt: Field = 0
    m11: Field = 0
    m12: Field = 0
    m22: Field = 0
    s: List[Field] = field(default_factory=list)
    pow: List[Field] = field(default_factory=list)

    def ensure_lengths(self, m: int) -> None:
        if not self.s:
            self.s = [0] * m
        if not self.pow:
            self.pow = [1] * m


@dataclass
class GeomAirRow:
    pc: Field
    opcode: Field
    gas: Field
    acc: Field
    x1: Field
    x2: Field
    cnt: Field
    m11: Field
    m12: Field
    m22: Field
    s: List[Field]
    pow: List[Field]


@dataclass
class GeomTrace:
    params: GeomAIRParams
    rows: List[GeomAirRow]


@dataclass
class GeomEvalTable:
    trace_len: int
    domain_size: int
    PC: List[Field]
    OP: List[Field]
    GAS: List[Field]
    ACC: List[Field]
    X1: List[Field]
    X2: List[Field]
    CNT: List[Field]
    M11: List[Field]
    M12: List[Field]
    M22: List[Field]
    sketches: List[List[Field]]
    powers: List[List[Field]]
    column_masks: Dict[str, List[Field]] | None = None


def next_power_of_two(n: int) -> int:
    out = 1
    while out < n:
        out <<= 1
    return out


def simulate_trace(program: Sequence[int], params: GeomAIRParams, init: GeomInitialState) -> GeomTrace:
    init.ensure_lengths(params.num_challenges)
    rows: List[GeomAirRow] = []
    pc = init.pc
    gas = init.gas
    acc = init.acc
    opcode = init.opcode
    x1 = init.x1
    x2 = init.x2
    cnt = init.cnt
    m11 = init.m11
    m12 = init.m12
    m22 = init.m22
    s = list(init.s)
    pow_vec = list(init.pow)

    a11, a12 = params.matrix[0]
    a21, a22 = params.matrix[1]

    for t in range(params.steps):
        opcode = program[t % len(program)]
        packed_val = pack_vm_value(pc, opcode, gas, acc)

        # VM semantics
        if opcode == 1:  # add
            acc = (acc + gas) % MODULUS
        elif opcode == 2:  # mul
            acc = (acc * gas) % MODULUS
        elif opcode == 3:  # xor-like
            acc = (acc ^ gas) % MODULUS
        gas = (gas - (1 + opcode % 3)) % MODULUS
        pc = (pc + 1) % MODULUS

        # Geometry state update (linear dynamics)
        nx1 = (a11 * x1 + a12 * x2) % MODULUS
        nx2 = (a21 * x1 + a22 * x2) % MODULUS

        # Accumulate SPD moments
        cnt = (cnt + 1) % MODULUS
        m11 = (m11 + x1 * x1) % MODULUS
        m12 = (m12 + x1 * x2) % MODULUS
        m22 = (m22 + x2 * x2) % MODULUS

        # STC accumulator
        new_s = []
        new_pow = []
        for j, rj in enumerate(params.r_challenges):
            new_s_val = (s[j] + packed_val * pow_vec[j]) % MODULUS
            new_pow_val = (pow_vec[j] * rj) % MODULUS
            new_s.append(new_s_val)
            new_pow.append(new_pow_val)
        row = GeomAirRow(
            pc=pc,
            opcode=opcode,
            gas=gas,
            acc=acc,
            x1=x1,
            x2=x2,
            cnt=cnt,
            m11=m11,
            m12=m12,
            m22=m22,
            s=list(new_s),
            pow=list(new_pow),
        )
        rows.append(row)
        x1, x2 = nx1, nx2
        s, pow_vec = new_s, new_pow
    return GeomTrace(params=params, rows=rows)


def trace_to_eval_table(
    trace: GeomTrace,
    domain_size: int | None = None,
    column_masks: Dict[str, List[Field]] | None = None,
) -> GeomEvalTable:
    n = domain_size or next_power_of_two(trace.params.steps)
    pad = n - trace.params.steps
    def pad_col(values: List[Field]) -> List[Field]:
        return values + [0] * pad

    def maybe_mask(name: str, values: List[Field]) -> List[Field]:
        padded = pad_col(values)
        if column_masks is None:
            return padded
        mask_vals = column_masks.get(name)
        if mask_vals is None:
            return padded
        if len(mask_vals) != len(padded):
            raise ValueError(f"mask length mismatch for column {name}")
        return [
            (padded[i] + mask_vals[i]) % MODULUS
            for i in range(len(padded))
        ]

    pc_col = maybe_mask("PC", [row.pc for row in trace.rows])
    op_col = maybe_mask("OP", [row.opcode for row in trace.rows])
    gas_col = maybe_mask("GAS", [row.gas for row in trace.rows])
    acc_col = maybe_mask("ACC", [row.acc for row in trace.rows])
    x1_col = maybe_mask("X1", [row.x1 for row in trace.rows])
    x2_col = maybe_mask("X2", [row.x2 for row in trace.rows])
    cnt_col = maybe_mask("CNT", [row.cnt for row in trace.rows])
    m11_col = maybe_mask("M11", [row.m11 for row in trace.rows])
    m12_col = maybe_mask("M12", [row.m12 for row in trace.rows])
    m22_col = maybe_mask("M22", [row.m22 for row in trace.rows])

    sketches_cols = []
    powers_cols = []
    m = trace.params.num_challenges
    for j in range(m):
        sketches_cols.append(
            maybe_mask(f"sketches_{j}", [row.s[j] for row in trace.rows])
        )
        powers_cols.append(
            maybe_mask(f"powers_{j}", [row.pow[j] for row in trace.rows])
        )

    return GeomEvalTable(
        trace_len=len(trace.rows),
        domain_size=n,
        PC=pc_col,
        OP=op_col,
        GAS=gas_col,
        ACC=acc_col,
        X1=x1_col,
        X2=x2_col,
        CNT=cnt_col,
        M11=m11_col,
        M12=m12_col,
        M22=m22_col,
        sketches=sketches_cols,
        powers=powers_cols,
        column_masks=column_masks,
    )


class GeomAIR:
    """AIR helper exposing eval-table builder and per-constraint columns."""

    def __init__(self, params: GeomAIRParams, domain_size: int | None = None):
        self.params = params
        self.domain_size = domain_size or next_power_of_two(params.steps)
        base = [
            "pc_step",
            "gas_step",
            "cnt_step",
            "m11_step",
            "m12_step",
            "m22_step",
        ]
        s_names = [f"s_step_{j}" for j in range(params.num_challenges)]
        pow_names = [f"pow_step_{j}" for j in range(params.num_challenges)]
        boundary = ["m11_final", "m12_final", "m22_final", "cnt_final"]
        self.constraint_names = base + s_names + pow_names + boundary

    def build_eval_table(self, trace: GeomTrace) -> GeomEvalTable:
        return trace_to_eval_table(trace, self.domain_size)

    def empty_constraint_dict(self) -> Dict[str, List[Field]]:
        return {name: [] for name in self.constraint_names}

    def build_composition_vector(
        self,
        trace: GeomTrace,
        alphas: Dict[str, Field],
        sigma_expected: Dict[str, Field] | None = None,
    ) -> List[Field]:
        table = trace_to_eval_table(trace, self.domain_size)
        from .geom_constraints import build_composition_vector

        return build_composition_vector(
            self.params,
            table,
            alphas,
            sigma_expected=sigma_expected,
        )

    def eval_constraints(self, table: GeomEvalTable) -> Dict[str, List[Field]]:
        from .geom_constraints import eval_constraints_at_row

        residuals = self.empty_constraint_dict()
        for t in range(self.domain_size):
            res = eval_constraints_at_row(self.params, table, t)
            residuals["pc_step"].append(res.pc_step)
            residuals["gas_step"].append(res.gas_step)
            residuals["cnt_step"].append(res.cnt_step)
            residuals["m11_step"].append(res.m11_step)
            residuals["m12_step"].append(res.m12_step)
            residuals["m22_step"].append(res.m22_step)
            for j, val in enumerate(res.s_steps):
                residuals[f"s_step_{j}"].append(val)
            for j, val in enumerate(res.pow_steps):
                residuals[f"pow_step_{j}"].append(val)
        return residuals
