"""Helpers to compute and display security-assumption blocks for CLI output."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import click


def _estimate_fri_security(blowup: Optional[int], queries: Optional[int]) -> Optional[int]:
    if not blowup or not queries:
        return None
    if blowup <= 1 or queries <= 0:
        return None
    return int(round(queries * math.log2(blowup)))


def _proof_primitive_from_capsule(capsule: Dict[str, Any]) -> Dict[str, Any]:
    header = capsule.get("header") or capsule
    pc = header.get("proof_system") or {}
    fri_meta = (pc.get("fri_params") or {}) if isinstance(pc, dict) else {}
    blowup = fri_meta.get("blowup") or fri_meta.get("blowup_factor")
    queries = fri_meta.get("num_queries") or header.get("num_queries")
    rounds = fri_meta.get("num_rounds")
    primitive = {
        "vc": "stc_sha256",
        "pc": "fri",
        "hash": "sha256",
        "backend": header.get("backend_id") or header.get("proof_system_id"),
        "fri_params": {
            "blowup": blowup,
            "queries": queries,
            "rounds": rounds,
            "est_security_bits": _estimate_fri_security(blowup, queries),
        },
        "fs_model": "random_oracle",
        "instance_binding": "absorbed",
        "air_binding": {
            "trace_spec_hash": header.get("trace_spec_hash"),
            "statement_hash": header.get("statement_hash"),
        },
    }
    return primitive


def policy_info_from_capsule(capsule: Dict[str, Any]) -> Dict[str, Any] | None:
    header = capsule.get("header") or {}
    policy_ref = header.get("policy_ref") or capsule.get("policy")
    if not isinstance(policy_ref, dict):
        return None
    return {
        "policy_id": policy_ref.get("policy_id"),
        "policy_version": policy_ref.get("policy_version"),
        "policy_hash": policy_ref.get("policy_hash"),
        "track_id": policy_ref.get("track_id"),
    }


def dataset_info_from_capsule(capsule: Dict[str, Any]) -> Dict[str, Any] | None:
    datasets = []
    ds_ref = (capsule.get("dataset_ref") or {}).get("datasets") or []
    for entry in ds_ref:
        datasets.append(
            {
                "dataset_id": entry.get("dataset_id"),
                "root": entry.get("root"),
                "num_chunks": entry.get("num_chunks"),
            }
        )
    if not datasets:
        return None
    return {"datasets": datasets}


def build_security_assumptions(
    *,
    operation: str,
    capsule: Dict[str, Any] | None = None,
    verify_result: Dict[str, Any] | None = None,
    sandbox_isolation: Dict[str, Any] | None = None,
    policy_info: Dict[str, Any] | None = None,
    da_info: Dict[str, Any] | None = None,
    dataset_info: Dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> Dict[str, Any]:
    assumptions: Dict[str, Any] = {"operation": operation}
    if capsule:
        assumptions["proof_primitive"] = _proof_primitive_from_capsule(capsule)
        header = capsule.get("header") or {}
        if header.get("profile"):
            assumptions["profile"] = header.get("profile")
    elif verify_result and verify_result.get("backend_id"):
        primitive = {
            "vc": "stc_sha256",
            "pc": "fri",
            "hash": "sha256",
            "backend": verify_result.get("backend_id"),
            "fri_params": {
                "blowup": verify_result.get("fri_blowup"),
                "queries": verify_result.get("num_queries"),
                "est_security_bits": _estimate_fri_security(
                    verify_result.get("fri_blowup"), verify_result.get("num_queries")
                ),
            },
            "fs_model": "random_oracle",
            "instance_binding": "absorbed",
        }
        primitive["air_binding"] = {
            "trace_spec_hash": verify_result.get("trace_spec_hash"),
            "statement_hash": verify_result.get("statement_hash"),
        }
        assumptions["proof_primitive"] = primitive

    if sandbox_isolation:
        assumptions["sandbox"] = sandbox_isolation
    if policy_info:
        assumptions["policy"] = policy_info
    if da_info:
        assumptions["da"] = da_info
    if dataset_info:
        assumptions["dataset"] = dataset_info
    if verify_result:
        assumptions["proof_integrity"] = {
            "capsule_bound": verify_result.get("capsule_hash_ok"),
            "row_commitment_bound": verify_result.get("row_index_commitment_ok"),
            "proof_verified": verify_result.get("proof_verified"),
        }
    if warnings:
        assumptions["warnings"] = warnings
    return assumptions


def print_security_assumptions(assumptions: dict[str, Any] | None) -> None:
    if not assumptions:
        return
    click.echo("\nSecurity Assumptions")
    pp = assumptions.get("proof_primitive") or {}
    if pp:
        backend = (pp.get("backend") or pp.get("pc") or "").upper()
        click.echo(f"  Proof primitive: {backend} (VC={pp.get('vc')}, PC={pp.get('pc')})")
        fri = pp.get("fri_params") or {}
        if fri.get("blowup") or fri.get("queries"):
            bits = fri.get("est_security_bits")
            note = f" (~{bits} bits)" if bits else ""
            click.echo(
                f"  FRI params: blowup={fri.get('blowup')}, queries={fri.get('queries')}" + note
            )
        air = pp.get("air_binding") or {}
        if air.get("trace_spec_hash") or air.get("statement_hash"):
            click.echo(
                f"  AIR binding: trace_spec={str(air.get('trace_spec_hash',''))[:8]}…, "
                f"statement={str(air.get('statement_hash',''))[:8]}…"
            )
    policy = assumptions.get("policy") or {}
    if policy.get("policy_id"):
        version = policy.get("policy_version") or "unspecified"
        p_hash = policy.get("policy_hash")
        hash_note = f", hash={p_hash[:12]}…" if p_hash else ""
        click.echo(
            f"  Policy: {policy.get('policy_id')} (version {version}{hash_note})"
        )
    profile = assumptions.get("profile")
    if profile:
        click.echo(f"  Profile: {profile}")
    sandbox = assumptions.get("sandbox") or {}
    if sandbox:
        desc = []
        if sandbox.get("network_degraded"):
            desc.append("net=degraded")
        elif sandbox.get("network"):
            desc.append("net=isolated")
        else:
            desc.append("net=shared")
        fs = "pivot" if sandbox.get("pivot_root") else (
            "restricted" if sandbox.get("filesystem") else "full"
        )
        desc.append(f"fs={fs}")
        if sandbox.get("memory_limit"):
            desc.append("mem=limited")
        if sandbox.get("fallback_from"):
            desc.append(f"fallback={sandbox['fallback_from']}")
        click.echo("  Sandbox: " + ", ".join(desc))
    dataset = assumptions.get("dataset") or {}
    if dataset.get("datasets"):
        click.echo(f"  Datasets committed: {len(dataset['datasets'])}")
    da = assumptions.get("da") or {}
    if da.get("status"):
        click.echo("  DA: " + da["status"])
    warns = assumptions.get("warnings") or []
    if warns:
        click.echo("  Warnings: " + ", ".join(warns))
