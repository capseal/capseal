import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.prover import zk_prove_geom, MODULUS
from bef_zk.zk_geom.verifier import zk_verify_geom
from bef_zk.zk_geom.serialization import (
    proof_to_json,
    proof_from_json,
    proof_to_bytes,
    proof_from_bytes,
)

PROGRAM = [1, 2, 3, 1, 2, 0]


def _default_params(steps: int = 16) -> GeomAIRParams:
    return GeomAIRParams(
        steps=steps,
        num_challenges=2,
        r_challenges=[1234567, 89101112],
        matrix=[[2, 1], [1, 1]],
    )


def _fri_cfg(steps: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    return FRIConfig(
        field_modulus=(1 << 61) - 1,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=min(4, max_rounds),
        num_queries=4,
    )


def test_proof_round_trip_serialization() -> None:
    params = _default_params(steps=8)
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    proof = zk_prove_geom(PROGRAM, params, init, fri_cfg, vc)
    serialized = proof_to_json(proof)
    restored = proof_from_json(serialized)
    ok, _ = zk_verify_geom(PROGRAM, params, init, vc, restored)
    assert ok


def test_proof_round_trip_binary_serialization() -> None:
    params = _default_params(steps=8)
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    proof = zk_prove_geom(PROGRAM, params, init, fri_cfg, vc)
    blob = proof_to_bytes(proof)
    restored = proof_from_bytes(blob)
    ok, _ = zk_verify_geom(PROGRAM, params, init, vc, restored)
    assert ok


def test_proof_detects_row_value_corruption() -> None:
    params = _default_params(steps=8)
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    proof = zk_prove_geom(PROGRAM, params, init, fri_cfg, vc)
    tampered = proof_from_json(proof_to_json(proof))
    tampered.row_openings[0].row_values[0] = (
        tampered.row_openings[0].row_values[0] + 1
    ) % MODULUS
    ok, _ = zk_verify_geom(PROGRAM, params, init, vc, tampered)
    assert not ok


def test_proof_detects_mask_digest_corruption() -> None:
    params = _default_params(steps=8)
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    proof = zk_prove_geom(PROGRAM, params, init, fri_cfg, vc)
    tampered = proof_from_json(proof_to_json(proof))
    tampered.mask_digest = bytes(reversed(tampered.mask_digest))
    ok, _ = zk_verify_geom(PROGRAM, params, init, vc, tampered)
    assert not ok
