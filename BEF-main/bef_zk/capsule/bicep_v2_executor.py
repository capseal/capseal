#!/usr/bin/env python3
"""
Production bicep_v2 emitter (Level 1.5 contract).

Emits canonical run directory matching verifier-independent/test_bicep_v2.py:
  run_dir/
    trace.jsonl
    manifest.json
    commitments.json
    audit_openings/
      audit_step_0000.json
      ...

Deterministic and byte-for-byte identical to the test generator when given
the same seed + parameters.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from canonical.bicep_state import (
    AddressableRNG,
    canonical_json_bytes,
    quantize,
    build_merkle_tree,
    compute_challenge_seed,
    sample_audit_indices,
    generate_audit_opening,
    total_leaves,
    sde_step_em,
    sde_step_with_jump,
)


DEFAULT_SEED = "cafebabe" * 8


def _manifest_dict(
    rng: AddressableRNG,
    num_paths: int,
    num_channels: int,
    audit_k: int,
    precision_bits: int,
    sde_params: dict,
) -> dict:
    return {
        "transition_spec_id": "bicep_v2",
        "state_view_schema_id": "bicep_state_v2",
        "transition_fn_id": "bicep_sde_em_v1",
        "output_fn_id": "bicep_features_v1",
        "seed_commitment": rng.seed_commitment,
        "x_quant_scheme_id": "fixed_point_v1",
        "x_quant_precision_bits": precision_bits,
        "sampling_scheme_id": "standard_v1",  # matches generator
        "state_layout_id": "flat_path_channel_v1",
        "state_num_paths": num_paths,
        "state_num_channels": num_channels,
        "sde_params": sde_params,
        "audit_k": audit_k,
    }


def emit_bicep_v2_run(
    run_dir: Path,
    seed_hex: str = DEFAULT_SEED,
    num_steps: int = 10,
    num_paths: int = 4,
    num_channels: int = 4,
    audit_k: int = 8,
    precision_bits: int = 24,
    sde_params: Optional[dict] = None,
) -> dict:
    """Emit full bicep_v2 run directory; return manifest dict."""
    run_dir.mkdir(parents=True, exist_ok=True)

    if sde_params is None:
        sde_params = {
            "theta": 2.0,
            "mu0": 0.0,
            "sigma": 0.5,
            "dt": 0.01,
            "jump_rate": 0.0,
            "jump_scale": 0.0,
        }

    rng = AddressableRNG.from_hex(seed_hex)
    n_leaves = total_leaves(num_paths, num_channels)

    manifest = _manifest_dict(rng, num_paths, num_channels, audit_k, precision_bits, sde_params)
    manifest_hash = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()

    # Genesis state (quantized RNG)
    state_q = [quantize(rng.rand("state_init", 0, idx), precision_bits) for idx in range(n_leaves)]
    state_root, levels = build_merkle_tree(state_q)

    # Genesis output_chain (same formula as generator)
    output_chain = hashlib.sha256(
        f"bicep_v2:output_chain:genesis:{rng.seed_commitment}".encode()
    ).hexdigest()

    # Hash chain head_0
    head = hashlib.sha256(f"genesis:{manifest_hash}".encode()).hexdigest()

    # 1) Write manifest.json
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # 2) Prepare outputs
    trace_path = run_dir / "trace.jsonl"
    openings_dir = run_dir / "audit_openings"
    openings_dir.mkdir(exist_ok=True)

    # 3) Emit rows + audit openings
    with open(trace_path, "w") as tf:
        for t in range(num_steps):
            # Quantized outputs per step (num_channels features)
            x_t_q = [quantize(rng.rand("input", t, i), precision_bits) for i in range(num_channels)]

            # Range-based RNG records for SDE noise (as in generator)
            rand_addrs = [
                {
                    "tag": "sde_noise",
                    "t": t,
                    "i_start": 0,
                    "i_count": n_leaves,
                    "layout_id": "flat_path_channel_v1",
                }
            ]
            rng_use_hash = hashlib.sha256(canonical_json_bytes(rand_addrs)).hexdigest()

            view_pre = {"state_root": state_root, "output_chain": output_chain}

            # SDE step for entire state
            state_pre_q = state_q[:]
            levels_pre = levels

            state_post_q = []
            jump_rate = sde_params.get("jump_rate", 0.0)
            for idx in range(n_leaves):
                eps = rng.rand_normal("sde_noise", t, idx)
                if jump_rate > 0:
                    jump_flag = rng.rand("jump_mask", t, idx)
                    jump_mag = rng.rand_normal("jump_mag", t, idx)
                    next_q = sde_step_with_jump(
                        state_q[idx], sde_params, eps, jump_flag, jump_mag, precision_bits
                    )
                else:
                    next_q = sde_step_em(state_q[idx], sde_params, eps, precision_bits)
                state_post_q.append(next_q)

            state_q = state_post_q
            state_root_new, levels_new = build_merkle_tree(state_q)

            # Update output_chain
            output_chain_new = hashlib.sha256(
                output_chain.encode() + canonical_json_bytes(x_t_q)
            ).hexdigest()

            view_post = {"state_root": state_root_new, "output_chain": output_chain_new}

            row = {
                "schema": "bicep_trace_v2",
                "t": t,
                "x_t_q": x_t_q,
                "view_pre": view_pre,
                "view_post": view_post,
                "rand_addrs": rand_addrs,
                "rng_use_hash": rng_use_hash,
            }

            # Write row deterministically
            tf.write(json.dumps(row, sort_keys=True) + "\n")

            # Generate and save audit opening
            challenge_seed = compute_challenge_seed(manifest_hash, head, state_root, rng_use_hash)
            audit_indices = sample_audit_indices(challenge_seed, n_leaves, audit_k)
            audit = generate_audit_opening(
                t,
                state_pre_q,
                state_post_q,
                levels_pre,
                levels_new,
                state_root,
                state_root_new,
                audit_indices,
                challenge_seed,
            )
            with open(openings_dir / f"audit_step_{t:04d}.json", "w") as af:
                json.dump(audit, af, indent=2, sort_keys=True)

            # Advance hash head and state
            row_digest = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
            head = hashlib.sha256(f"{head}:{row_digest}".encode()).hexdigest()
            state_root = state_root_new
            levels = levels_new
            output_chain = output_chain_new

    # 4) commitments.json (minimal, matches generator)
    commitments = {
        "manifest_hash": manifest_hash,
        "head_T": head,
        "total_steps": num_steps,
        "spec": "bicep_v2",
    }
    with open(run_dir / "commitments.json", "w") as f:
        json.dump(commitments, f, indent=2, sort_keys=True)

    return manifest


def main():
    # Simple CLI for manual runs
    run_dir: Optional[Path] = None
    seed_hex = DEFAULT_SEED
    num_steps = 10
    num_paths = 4
    num_channels = 4
    audit_k = 8
    precision_bits = 24

    args = sys.argv[1:]
    if "--run-dir" in args:
        i = args.index("--run-dir")
        if i + 1 < len(args):
            run_dir = Path(args[i + 1])
    if "--seed" in args:
        i = args.index("--seed")
        if i + 1 < len(args):
            seed_hex = args[i + 1]

    if run_dir is None:
        print("Usage: python -m bef_zk.capsule.bicep_v2_executor --run-dir <path> [--seed <hex>]")
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    emit_bicep_v2_run(
        run_dir,
        seed_hex=seed_hex,
        num_steps=num_steps,
        num_paths=num_paths,
        num_channels=num_channels,
        audit_k=audit_k,
        precision_bits=precision_bits,
    )
    print(f"bicep_v2 run emitted at: {run_dir}")


if __name__ == "__main__":
    main()
