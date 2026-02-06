from __future__ import annotations

import datetime
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from canonical.project_trace import canonical_json_bytes, sha256_bytes

from bef_zk.capsule.finding_utils import (
    FINDING_NORM_VERSION,
    compute_finding_fingerprint,
    policy_severity_order,
    severity_rank,
    snippet_hash,
)
from bef_zk.capsule.review_agent import run_llm_explain


@dataclass
class ExplainResult:
    summary_path: Path
    receipt_path: Path
    report_path: Optional[Path]
    explanations: list[dict[str, Any]]
    selected_findings: int
    input_hash: str
    cached: bool


def _collect_review_findings(run_path: Path) -> list[dict[str, Any]]:
    review_dir = run_path / "reviews"
    if not review_dir.exists():
        return []
    findings: list[dict[str, Any]] = []
    for review_file in sorted(review_dir.glob("review_shard_*.json")):
        try:
            review = json.loads(review_file.read_text())
        except Exception:
            continue
        backend_id = review.get("backend_id") or review.get("backend", "")
        for finding in review.get("findings", []):
            entry = dict(finding)
            entry.setdefault(
                "finding_fingerprint",
                compute_finding_fingerprint(
                    entry, backend_id, norm_version=entry.get("finding_norm_version")
                ),
            )
            entry["_source_review"] = review_file.name
            findings.append(entry)
    return findings


def _load_diff_findings(diff_path: Path) -> list[dict[str, Any]]:
    data = json.loads(diff_path.read_text())
    if isinstance(data, list):
        return data
    for key in ("new_findings", "new", "findings"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Unrecognized diff file format: {diff_path}")


def _dedupe_findings(
    findings: list[dict[str, Any]],
    backend_id: str,
    norm_fallback: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    deduped: list[dict[str, Any]] = []
    fingerprint_map: dict[str, dict[str, Any]] = {}
    for f in findings:
        entry = dict(f)
        fp = entry.get("finding_fingerprint")
        norm_version = entry.get("finding_norm_version") or norm_fallback
        if not fp:
            fp = compute_finding_fingerprint(entry, backend_id, norm_version=norm_version)
            entry["finding_fingerprint"] = fp
        if not fp:
            continue
        if fp in fingerprint_map:
            fingerprint_map[fp].setdefault("_occurrences", []).append(entry)
            continue
        entry.setdefault("_occurrences", []).append(entry)
        fingerprint_map[fp] = entry
        deduped.append(entry)
    return deduped, fingerprint_map


def _group_key_for_explain(meta: dict[str, Any], explanation: dict[str, Any]) -> tuple[str, str]:
    rule = meta.get("rule_id") or ""
    if rule:
        return ("rule", rule)
    recommendation = explanation.get("recommendation", "").strip().lower()
    if recommendation:
        return ("recommendation", recommendation)
    return ("fingerprint", explanation.get("fingerprint", ""))


def _render_explain_markdown(
    trace_root: str,
    rollup_hash: str,
    input_hash: str,
    provider: str,
    model_name: str,
    min_severity: str,
    max_findings: int,
    severity_rank_map: dict[str, int],
    explanations: list[dict[str, Any]],
    entry_metadata: dict[str, dict[str, Any]],
    report_top: int,
) -> str:
    order = sorted(severity_rank_map.keys(), key=lambda s: severity_rank_map[s])
    severity_counts = Counter(meta.get("severity", "info") for meta in entry_metadata.values())
    total_selected = len(entry_metadata)

    groups: list[dict[str, Any]] = []
    group_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for idx, explanation in enumerate(explanations):
        fp = explanation.get("fingerprint")
        if not fp:
            continue
        meta = entry_metadata.get(fp)
        if not meta:
            continue
        key = _group_key_for_explain(meta, explanation)
        group = group_lookup.get(key)
        severity = meta.get("severity", "info")
        severity_idx = severity_rank_map.get(severity, len(severity_rank_map))
        if group is None:
            title = meta.get("rule_id") or explanation.get("analysis", "") or fp
            title = title.strip() or fp
            group = {
                "key": key,
                "title": title,
                "rule_id": meta.get("rule_id", ""),
                "severity": severity,
                "severity_rank": severity_idx,
                "analysis": explanation.get("analysis", ""),
                "recommendation": explanation.get("recommendation", ""),
                "suggested_change": explanation.get("suggested_change", ""),
                "occurrences": [],
                "fingerprints": [],
                "order": idx,
                "total_occurrences": 0,
                "_seen_occ": set(),
            }
            group_lookup[key] = group
            groups.append(group)
        elif severity_idx < group["severity_rank"]:
            group["severity"] = severity
            group["severity_rank"] = severity_idx

        group["fingerprints"].append(fp)

        for occ in meta.get("_occurrences", [meta]):
            occ_fp = occ.get("finding_fingerprint") or fp
            file_path = occ.get("file_path") or meta.get("file_path", "")
            line_range = occ.get("line_range") or meta.get("line_range", [0, 0])
            lr_start = line_range[0] if isinstance(line_range, (list, tuple)) and line_range else 0
            lr_end = line_range[1] if isinstance(line_range, (list, tuple)) and len(line_range) > 1 else lr_start
            occ_key = (file_path, lr_start, lr_end, occ_fp)
            if occ_key in group["_seen_occ"]:
                continue
            group["_seen_occ"].add(occ_key)
            group["occurrences"].append(
                {
                    "file_path": file_path,
                    "line_range": [lr_start, lr_end],
                    "fingerprint": occ_fp,
                    "snippet_hash": occ.get("snippet_hash") or meta.get("snippet_hash", ""),
                    "chunk_hashes": occ.get("chunk_hashes") or meta.get("chunk_hashes", []),
                }
            )
            group["total_occurrences"] += 1

    groups.sort(
        key=lambda g: (
            g["severity_rank"],
            -g["total_occurrences"],
            g["title"].lower(),
            g["order"],
        )
    )

    top_n = max(report_top, 0)
    if top_n == 0 or top_n > len(groups):
        top_n = len(groups)

    lines = ["# CapSeal Explain Report", ""]
    lines.append(f"**Trace root:** `{trace_root}`")
    if rollup_hash:
        lines.append(f"**Rollup hash:** `{rollup_hash}`")
    lines.append(f"**Explain input hash:** `{input_hash}`")
    lines.append(f"**Backend:** `{provider}` model `{model_name}`")
    lines.append(f"**Selection:** min_severity={min_severity}, max_findings={max_findings}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Selected findings: **{total_selected}**")
    for sev in order:
        count = severity_counts.get(sev, 0)
        lines.append(f"- {sev.title()}: {count}")
    lines.append("")

    if not groups:
        lines.append("No explanations returned.")
        return "\n".join(lines)

    lines.append(f"## Findings (Top {top_n})")
    lines.append("")
    for idx, group in enumerate(groups[:top_n], 1):
        lines.append(f"### {idx}. [{group['severity'].upper()}] {group['title']}")
        lines.append("")
        if group.get("rule_id"):
            lines.append(f"**Rule:** `{group['rule_id']}`")
            lines.append("")
        lines.append("**Analysis**")
        lines.append(group["analysis"] or "(missing)")
        lines.append("")
        lines.append("**Recommendation**")
        lines.append(group["recommendation"] or "(missing)")
        lines.append("")
        if group.get("suggested_change"):
            lines.append("**Suggested change**")
            lines.append(group["suggested_change"])
            lines.append("")
        lines.append(f"**Occurrences ({group['total_occurrences']})**")
        for occ in group["occurrences"]:
            chunks = occ.get("chunk_hashes") or []
            chunk_str = ", ".join(f"`{c[:12]}...`" for c in chunks[:3])
            if len(chunks) > 3:
                chunk_str += f" (+{len(chunks)-3} more)"
            snippet_hash_value = occ.get("snippet_hash") or ""
            suffix = []
            if snippet_hash_value:
                suffix.append(f"snippet `{snippet_hash_value[:12]}...`")
            if chunk_str:
                suffix.append(f"chunks {chunk_str}")
            suffix_text = f" ({'; '.join(suffix)})" if suffix else ""
            lines.append(
                f"- `{occ['file_path']}` lines {occ['line_range'][0]}-{occ['line_range'][1]} "
                f"— fp `{occ['fingerprint'][:12]}...`{suffix_text}"
            )
        lines.append("")

    remaining = len(groups) - top_n
    if remaining > 0:
        lines.append("<details>")
        lines.append(f"<summary>Show {remaining} more finding groups</summary>")
        lines.append("")
        for idx, group in enumerate(groups[top_n:], top_n + 1):
            lines.append(f"#### {idx}. [{group['severity'].upper()}] {group['title']}")
            summary_line = group["analysis"] or group["recommendation"]
            lines.append(f"- {summary_line}")
            lines.append(
                f"- Occurrences: {group['total_occurrences']} across {len(group['occurrences'])} anchors"
            )
            lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    footer = [f"trace_root `{trace_root[:16]}...`"]
    footer.append(f"input `{input_hash[:16]}...`")
    if rollup_hash:
        footer.append(f"rollup `{rollup_hash[:16]}...`")
    lines.append(f"*Generated by capseal-0.3.0 ({', '.join(footer)})*")
    return "\n".join(lines)


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def verify_explain_receipt(run_path: Path, receipt_path: Path, quiet: bool = False) -> bool:
    if not receipt_path.exists():
        if not quiet:
            print(f"  [FAIL] receipt not found: {receipt_path}")
        return False

    receipt_data = json.loads(receipt_path.read_text())
    files_to_check = {
        "prompt_entries_hash": receipt_data.get("prompt_entries_path"),
        "prompt_text_hash": receipt_data.get("prompt_text_path"),
        "raw_hash": receipt_data.get("raw_path"),
        "summary_hash": receipt_data.get("summary_path"),
    }
    if receipt_data.get("report_path"):
        files_to_check["report_hash"] = receipt_data.get("report_path")

    def _hash_file(path: Path) -> str:
        if not path.exists():
            return ""
        return sha256_bytes(path.read_bytes())

    all_ok = True
    for hash_field, rel_path in files_to_check.items():
        if not rel_path:
            if not quiet:
                print(f"  [FAIL] {hash_field}: missing path")
            all_ok = False
            continue
        abs_path = (run_path / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path)
        if not abs_path.exists():
            if not quiet:
                print(f"  [FAIL] {hash_field}: file not found ({rel_path})")
            all_ok = False
            continue
        actual = _hash_file(abs_path)
        expected = receipt_data.get(hash_field, "")
        if actual != expected:
            if not quiet:
                print(f"  [FAIL] {hash_field}: hash mismatch")
            all_ok = False
        else:
            if not quiet:
                print(f"  [PASS] {hash_field}: {actual[:16]}...")

    recorded = receipt_data.get("receipt_hash", "")
    body = dict(receipt_data)
    body.pop("receipt_hash", None)
    expected_receipt_hash = sha256_bytes(canonical_json_bytes(body))
    if expected_receipt_hash == recorded:
        if not quiet:
            print(f"  [PASS] receipt_hash: {recorded[:16]}...")
    else:
        if not quiet:
            print("  [FAIL] receipt_hash mismatch")
        all_ok = False

    return all_ok


def run_explain_pipeline(
    run_path: Path,
    provider: str,
    model: str,
    temperature: float,
    llm_max_tokens: int,
    max_findings: int,
    min_severity: str,
    diff_path: Optional[Path],
    output_format: str,
    report_top: int,
    force: bool,
    out: Optional[Path],
) -> ExplainResult:
    review_dir = run_path / "reviews"
    if not review_dir.exists():
        raise RuntimeError("No reviews/ directory found. Run `capsule review` first.")

    commitments = json.loads((run_path / "commitments.json").read_text())
    manifest = json.loads((run_path / "manifest.json").read_text())
    trace_root = commitments.get("head_T", "")
    manifest_policy_version = manifest.get("policy_version", "unknown")
    review_rules_manifest = manifest.get("review_rules", {})

    agg_backend_id = ""
    agg_policy_id = ""
    agg_policy_version = manifest_policy_version
    agg_review_rules = review_rules_manifest
    agg_norm_version = FINDING_NORM_VERSION
    agg_path = review_dir / "aggregate.json"
    if agg_path.exists():
        agg_data = json.loads(agg_path.read_text())
        agg_backend_id = agg_data.get("backend_id", agg_backend_id)
        agg_policy_id = agg_data.get("policy_id", agg_policy_id)
        agg_norm_version = agg_data.get("finding_norm_version", agg_norm_version)
        agg_policy_version = agg_data.get("policy_version", agg_policy_version)
        agg_review_rules = agg_data.get("review_rules", agg_review_rules)
    if not agg_policy_id:
        agg_policy_id = manifest.get("policy_id", "")

    findings = _collect_review_findings(run_path)
    if not findings:
        raise RuntimeError("No review_shard_*.json files found.")

    if not agg_norm_version:
        for f in findings:
            if f.get("finding_norm_version"):
                agg_norm_version = f["finding_norm_version"]
                break

    findings, fingerprint_map = _dedupe_findings(findings, agg_backend_id, agg_norm_version)
    if not findings:
        raise RuntimeError("No findings available after deduplication.")

    severity_order = policy_severity_order(agg_review_rules)
    severity_rank_map = severity_rank(severity_order)

    if diff_path:
        diff_entries = _load_diff_findings(Path(diff_path))
        selected_findings = []
        for item in diff_entries:
            fp = item.get("finding_fingerprint") or item.get("fingerprint")
            if not fp:
                continue
            source = fingerprint_map.get(fp)
            if source:
                selected_findings.append(source)
    else:
        threshold = severity_rank_map.get(min_severity)
        if threshold is None:
            threshold = severity_rank_map.get("info", len(severity_rank_map))
        filtered = [
            f for f in findings
            if severity_rank_map.get(f.get("severity", "info"), len(severity_rank_map)) <= threshold
        ]
        if not filtered:
            raise RuntimeError(f"No findings at severity >= {min_severity}")
        filtered.sort(
            key=lambda f: (
                severity_rank_map.get(f.get("severity", "info"), len(severity_rank_map)),
                f.get("file_path", ""),
                f.get("rule_id", ""),
            )
        )
        if max_findings > 0:
            filtered = filtered[:max_findings]
        selected_findings = filtered

    entries = []
    for f in selected_findings:
        entries.append({
            "fingerprint": f.get("finding_fingerprint", ""),
            "severity": f.get("severity", "info"),
            "rule_id": f.get("rule_id", ""),
            "file_path": f.get("file_path", ""),
            "line_range": f.get("line_range", [0, 0]),
            "message": f.get("message", ""),
            "snippet": f.get("snippet", f.get("message", "")),
        })

    if not entries:
        raise RuntimeError("No findings selected for explanation.")

    selection_info = {
        "trace_root": trace_root,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": llm_max_tokens,
        "min_severity": min_severity,
        "max_findings": max_findings,
        "fingerprints": [e["fingerprint"] for e in entries],
        "diff_path": str(Path(diff_path).resolve()) if diff_path else "",
        "backend_id": agg_backend_id,
        "policy_id": agg_policy_id,
        "policy_version": agg_policy_version,
        "finding_norm_version": agg_norm_version,
    }
    input_hash = sha256_bytes(canonical_json_bytes(selection_info))

    explain_root = review_dir / "explain_llm"
    cache_dir = explain_root / input_hash
    summary_path = Path(out) if out else cache_dir / "summary.json"
    receipt_path = cache_dir / "receipt.json"
    if summary_path.exists() and not force:
        cached_ok = receipt_path.exists() and verify_explain_receipt(run_path, receipt_path, quiet=True)
        if cached_ok:
            summary_data = json.loads(summary_path.read_text())
            explanations = summary_data.get("explanations", [])
            report_rel = summary_data.get("report_path")
            report_path = (run_path / report_rel) if report_rel else None
            return ExplainResult(
                summary_path=summary_path,
                receipt_path=receipt_path,
                report_path=report_path,
                explanations=explanations,
                selected_findings=summary_data.get("selected_findings", len(explanations)),
                input_hash=input_hash,
                cached=True,
            )
        else:
            summary_path.unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)

    cache_dir.mkdir(parents=True, exist_ok=True)

    prompt_entries_path = cache_dir / "prompt_entries.json"
    prompt_entries_path.write_text(json.dumps(entries, indent=2, sort_keys=True))
    prompt_text_path = cache_dir / "prompt.txt"
    raw_path = cache_dir / "raw.txt"

    prompt, raw_output, explanations = run_llm_explain(
        entries,
        provider,
        model,
        temperature=temperature,
        max_tokens=llm_max_tokens,
    )
    prompt_text_path.write_text(prompt)
    raw_path.write_text(raw_output)

    summary_data = {
        "schema": "llm_explain_v1",
        "trace_root": trace_root,
        "backend_id": agg_backend_id,
        "policy_id": agg_policy_id,
        "policy_version": agg_policy_version,
        "finding_norm_version": agg_norm_version,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": llm_max_tokens,
        "min_severity": min_severity,
        "max_findings": max_findings,
        "selected_findings": len(entries),
        "input_hash": input_hash,
        "prompt_entries_path": _relpath(prompt_entries_path, run_path),
        "prompt_text_path": _relpath(prompt_text_path, run_path),
        "raw_path": _relpath(raw_path, run_path),
        "explanations": explanations,
    }

    entry_metadata: dict[str, dict[str, Any]] = {}
    for entry in entries:
        fp = entry["fingerprint"]
        source = fingerprint_map.get(fp, {})
        meta = {
            "fingerprint": fp,
            "severity": entry.get("severity", "info"),
            "rule_id": entry.get("rule_id", ""),
            "file_path": entry.get("file_path", ""),
            "line_range": entry.get("line_range", [0, 0]),
            "snippet_hash": source.get("snippet_hash", ""),
            "chunk_hashes": source.get("chunk_hashes", []),
            "message": entry.get("message", ""),
            "occurrences": source.get("_occurrences", [source]),
        }
        entry_metadata[fp] = meta

    rollup_hash = ""
    rollup_path = run_path / "workflow" / "rollup.json"
    if rollup_path.exists():
        with open(rollup_path) as f:
            rollup_data = json.load(f)
        rollup_hash = rollup_data.get("rollup_hash", "")

    report_path = cache_dir / "report.md"
    markdown = _render_explain_markdown(
        trace_root,
        rollup_hash,
        input_hash,
        provider,
        model,
        min_severity,
        max_findings,
        severity_rank_map,
        explanations,
        entry_metadata,
        report_top,
    )
    report_path.write_text(markdown)
    summary_data["report_path"] = _relpath(report_path, run_path)

    summary_path.write_text(json.dumps(summary_data, indent=2, sort_keys=True))

    receipt = {
        "schema": "llm_explain_receipt_v1",
        "trace_root": trace_root,
        "rollup_hash": rollup_hash,
        "backend_id": agg_backend_id,
        "policy_id": agg_policy_id,
        "policy_version": agg_policy_version,
        "finding_norm_version": agg_norm_version,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": llm_max_tokens,
        "min_severity": min_severity,
        "max_findings": max_findings,
        "input_hash": input_hash,
        "selected_fingerprints": [e["fingerprint"] for e in entries],
        "prompt_entries_path": summary_data["prompt_entries_path"],
        "prompt_text_path": summary_data["prompt_text_path"],
        "raw_path": summary_data["raw_path"],
        "summary_path": _relpath(summary_path, run_path),
        "report_path": summary_data["report_path"],
        "prompt_entries_hash": sha256_bytes(prompt_entries_path.read_bytes()),
        "prompt_text_hash": sha256_bytes(prompt.encode()),
        "raw_hash": sha256_bytes(raw_output.encode()),
        "summary_hash": sha256_bytes(summary_path.read_bytes()),
        "report_hash": sha256_bytes(report_path.read_bytes()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tool_version": "capseal-0.3.0",
        "diff_source": str(diff_path) if diff_path else "",
    }
    body = dict(receipt)
    receipt["receipt_hash"] = sha256_bytes(canonical_json_bytes(body))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))

    return ExplainResult(
        summary_path=summary_path,
        receipt_path=receipt_path,
        report_path=report_path,
        explanations=explanations,
        selected_findings=len(entries),
        input_hash=input_hash,
        cached=False,
    )
