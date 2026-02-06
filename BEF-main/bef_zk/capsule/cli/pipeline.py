"""Pipeline verification - multi-stage edge continuity checks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click


@dataclass
class StageSpec:
    """Specification for a pipeline stage."""
    stage_name: str
    capsule_path: str
    capsule_hash: str  # Expected SHA256 of capsule file
    inputs_commitment: str  # Hash of inputs (dataset roots, config, etc.)
    outputs_commitment: str  # Hash of outputs (results, artifacts)


@dataclass
class EdgeSpec:
    """Edge constraint: stage_from outputs == stage_to inputs."""
    from_stage: str
    to_stage: str
    # Specific binding: which output field connects to which input field
    bindings: list[dict[str, str]] = field(default_factory=list)


@dataclass
class PipelineIndex:
    """Pipeline index defining stages and edges."""
    pipeline_id: str
    pipeline_version: str
    stages: list[StageSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    final_outputs: dict[str, str] = field(default_factory=dict)  # name -> hash

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineIndex":
        stages = [
            StageSpec(
                stage_name=s["stage_name"],
                capsule_path=s["capsule_path"],
                capsule_hash=s["capsule_hash"],
                inputs_commitment=s["inputs_commitment"],
                outputs_commitment=s["outputs_commitment"],
            )
            for s in d.get("stages", [])
        ]
        edges = [
            EdgeSpec(
                from_stage=e["from_stage"],
                to_stage=e["to_stage"],
                bindings=e.get("bindings", []),
            )
            for e in d.get("edges", [])
        ]
        return cls(
            pipeline_id=d["pipeline_id"],
            pipeline_version=d.get("pipeline_version", "1.0"),
            stages=stages,
            edges=edges,
            final_outputs=d.get("final_outputs", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_version": self.pipeline_version,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "capsule_path": s.capsule_path,
                    "capsule_hash": s.capsule_hash,
                    "inputs_commitment": s.inputs_commitment,
                    "outputs_commitment": s.outputs_commitment,
                }
                for s in self.stages
            ],
            "edges": [
                {
                    "from_stage": e.from_stage,
                    "to_stage": e.to_stage,
                    "bindings": e.bindings,
                }
                for e in self.edges
            ],
            "final_outputs": self.final_outputs,
        }

    @classmethod
    def load(cls, path: Path) -> "PipelineIndex":
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class EdgeCheckResult:
    """Result of an edge continuity check."""
    from_stage: str
    to_stage: str
    status: str  # "pass", "fail"
    from_outputs: str
    to_inputs: str
    match: bool
    error: str | None = None


@dataclass
class StageCheckResult:
    """Result of a stage hash verification."""
    stage_name: str
    status: str  # "pass", "fail", "skip"
    expected_hash: str
    actual_hash: str | None
    match: bool
    error: str | None = None


@dataclass
class PipelineCheckResult:
    """Complete pipeline verification result."""
    pipeline_id: str
    overall_status: str  # "PASS", "FAIL"
    stage_checks: list[StageCheckResult] = field(default_factory=list)
    edge_checks: list[EdgeCheckResult] = field(default_factory=list)
    final_outputs_match: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "overall_status": self.overall_status,
            "stage_checks": [
                {
                    "stage_name": s.stage_name,
                    "status": s.status,
                    "expected_hash": s.expected_hash,
                    "actual_hash": s.actual_hash,
                    "match": s.match,
                    "error": s.error,
                }
                for s in self.stage_checks
            ],
            "edge_checks": [
                {
                    "from_stage": e.from_stage,
                    "to_stage": e.to_stage,
                    "status": e.status,
                    "from_outputs": e.from_outputs,
                    "to_inputs": e.to_inputs,
                    "match": e.match,
                    "error": e.error,
                }
                for e in self.edge_checks
            ],
            "final_outputs_match": self.final_outputs_match,
        }


def _hash_file(path: Path) -> str | None:
    """Compute SHA256 of file."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pipeline(pipeline_path: Path, base_dir: Path | None = None) -> PipelineCheckResult:
    """Verify a pipeline index.

    Checks:
    1. Each stage capsule hash matches the file
    2. Edge continuity: stage_from.outputs_commitment == stage_to.inputs_commitment
    """
    index = PipelineIndex.load(pipeline_path)
    base = base_dir or pipeline_path.parent

    result = PipelineCheckResult(
        pipeline_id=index.pipeline_id,
        overall_status="PASS",
    )

    # Build stage lookup
    stages_by_name: dict[str, StageSpec] = {s.stage_name: s for s in index.stages}

    # Check 1: Stage capsule hashes
    for stage in index.stages:
        capsule_path = Path(stage.capsule_path)
        if not capsule_path.is_absolute():
            capsule_path = base / capsule_path

        actual_hash = _hash_file(capsule_path)
        match = actual_hash == stage.capsule_hash

        if not match:
            result.overall_status = "FAIL"

        result.stage_checks.append(StageCheckResult(
            stage_name=stage.stage_name,
            status="pass" if match else "fail",
            expected_hash=stage.capsule_hash,
            actual_hash=actual_hash,
            match=match,
            error=None if match else ("File not found" if actual_hash is None else "Hash mismatch"),
        ))

    # Check 2: Edge continuity
    for edge in index.edges:
        from_stage = stages_by_name.get(edge.from_stage)
        to_stage = stages_by_name.get(edge.to_stage)

        if not from_stage or not to_stage:
            result.overall_status = "FAIL"
            result.edge_checks.append(EdgeCheckResult(
                from_stage=edge.from_stage,
                to_stage=edge.to_stage,
                status="fail",
                from_outputs="",
                to_inputs="",
                match=False,
                error=f"Stage not found: {edge.from_stage if not from_stage else edge.to_stage}",
            ))
            continue

        match = from_stage.outputs_commitment == to_stage.inputs_commitment

        if not match:
            result.overall_status = "FAIL"

        result.edge_checks.append(EdgeCheckResult(
            from_stage=edge.from_stage,
            to_stage=edge.to_stage,
            status="pass" if match else "fail",
            from_outputs=from_stage.outputs_commitment,
            to_inputs=to_stage.inputs_commitment,
            match=match,
            error=None if match else "Continuity mismatch: outputs != inputs",
        ))

    return result


@click.group("pipeline")
def pipeline_group():
    """Pipeline verification commands."""
    pass


@pipeline_group.command("verify")
@click.argument("pipeline_index", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--base-dir", type=click.Path(exists=True, path_type=Path), help="Base directory for relative paths")
def verify_pipeline_cmd(pipeline_index: Path, json_output: bool, base_dir: Path | None):
    """Verify pipeline stage hashes and edge continuity.

    Checks:
    - Each stage capsule hash matches the recorded hash
    - Edge continuity: stage outputs == next stage inputs

    Exit codes:
    - 0: All checks pass
    - 1: One or more checks failed
    """
    result = verify_pipeline(pipeline_index, base_dir)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        status_icon = "✓" if result.overall_status == "PASS" else "✗"
        click.echo(f"{status_icon} Pipeline: {result.pipeline_id} - {result.overall_status}")
        click.echo("")

        click.echo("Stage Checks:")
        for s in result.stage_checks:
            icon = "✓" if s.match else "✗"
            click.echo(f"  {icon} {s.stage_name}: {s.status}")
            if s.error:
                click.echo(f"      Error: {s.error}")

        if result.edge_checks:
            click.echo("")
            click.echo("Edge Checks:")
            for e in result.edge_checks:
                icon = "✓" if e.match else "✗"
                click.echo(f"  {icon} {e.from_stage} → {e.to_stage}: {e.status}")
                if e.error:
                    click.echo(f"      Error: {e.error}")

    import sys
    sys.exit(0 if result.overall_status == "PASS" else 1)


@pipeline_group.command("init")
@click.option("--output", "-o", type=click.Path(path_type=Path), default="pipeline_index.json",
              help="Output file path")
@click.option("--pipeline-id", type=str, required=True, help="Pipeline identifier")
@click.argument("capsules", nargs=-1, type=click.Path(exists=True, path_type=Path))
def init_pipeline(output: Path, pipeline_id: str, capsules: tuple[Path, ...]):
    """Initialize a pipeline index from capsule files.

    Creates a pipeline_index.json with stage entries for each capsule.
    You'll need to manually add edges and adjust commitments.

    Example:
        capseal pipeline init --pipeline-id my_pipeline stage1/capsule.json stage2/capsule.json
    """
    stages = []
    for i, capsule_path in enumerate(capsules):
        capsule_hash = _hash_file(capsule_path)
        if not capsule_hash:
            click.echo(f"Warning: Could not hash {capsule_path}", err=True)
            continue

        # Load capsule to get some metadata
        try:
            capsule_data = json.loads(capsule_path.read_text())
        except Exception:
            capsule_data = {}

        stage_name = capsule_data.get("trace_id", f"stage_{i}")

        # Placeholder commitments - user should fill these in
        stages.append({
            "stage_name": stage_name,
            "capsule_path": str(capsule_path),
            "capsule_hash": capsule_hash,
            "inputs_commitment": "TODO: fill in inputs commitment",
            "outputs_commitment": "TODO: fill in outputs commitment",
        })

    # Create edges between consecutive stages
    edges = []
    for i in range(len(stages) - 1):
        edges.append({
            "from_stage": stages[i]["stage_name"],
            "to_stage": stages[i + 1]["stage_name"],
            "bindings": [],
        })

    index = {
        "pipeline_id": pipeline_id,
        "pipeline_version": "1.0",
        "stages": stages,
        "edges": edges,
        "final_outputs": {},
    }

    output.write_text(json.dumps(index, indent=2))
    click.echo(f"Created: {output}")
    click.echo(f"  Stages: {len(stages)}")
    click.echo(f"  Edges: {len(edges)}")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Fill in inputs_commitment and outputs_commitment for each stage")
    click.echo("  2. Run: capseal pipeline verify pipeline_index.json")
