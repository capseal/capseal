"""Helpers for working with masked AIR columns."""
from __future__ import annotations

from typing import Dict, List

from ..air.geom_air import GeomAIRParams, GeomEvalTable


BASE_COLUMNS = [
    "PC",
    "OP",
    "GAS",
    "ACC",
    "X1",
    "X2",
    "CNT",
    "M11",
    "M12",
    "M22",
]


def column_names(params: GeomAIRParams) -> List[str]:
    names = list(BASE_COLUMNS)
    for j in range(params.num_challenges):
        names.append(f"sketches_{j}")
    for j in range(params.num_challenges):
        names.append(f"powers_{j}")
    return names


def extract_masked_columns(table: GeomEvalTable, params: GeomAIRParams) -> Dict[str, List[int]]:
    columns: Dict[str, List[int]] = {
        "PC": table.PC,
        "OP": table.OP,
        "GAS": table.GAS,
        "ACC": table.ACC,
        "X1": table.X1,
        "X2": table.X2,
        "CNT": table.CNT,
        "M11": table.M11,
        "M12": table.M12,
        "M22": table.M22,
    }
    for j in range(params.num_challenges):
        columns[f"sketches_{j}"] = table.sketches[j]
    for j in range(params.num_challenges):
        columns[f"powers_{j}"] = table.powers[j]
    return columns


def build_row_matrix(masked_columns: Dict[str, List[int]], params: GeomAIRParams) -> List[List[int]]:
    names = column_names(params)
    if not names:
        return []
    domain_size = len(masked_columns[names[0]])
    matrix: List[List[int]] = []
    for idx in range(domain_size):
        row: List[int] = []
        for name in names:
            col = masked_columns.get(name)
            if col is None or idx >= len(col):
                raise ValueError(f"missing column data for {name}")
            row.append(int(col[idx]))
        matrix.append(row)
    return matrix
