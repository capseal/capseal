"""Pydantic models that define the HTTP API contract for the CapSeal Flask service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class PolicyRef(BaseModel):
    """Reference to a policy artifact."""

    policyId: Optional[str] = None
    policyVersion: Optional[str] = None
    policyPath: Optional[str] = Field(default=None, description="Server-visible path to policy.json")
    policy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline policy JSON (primarily for testing); server may reject this.",
    )

    @field_validator("policy", mode="before")
    @classmethod
    def _require_path_or_inline(cls, value: Optional[Dict[str, Any]], info):  # type: ignore[override]
        data = getattr(info, "data", {})
        if not value and not data.get("policyPath"):
            raise ValueError("policy.policyPath or policy.policy must be provided")
        return value


class DatasetMapping(BaseModel):
    """Mapping between dataset identifier and on-disk path or remote URL."""

    id: str
    path: Optional[str] = Field(default=None, description="Server-visible filesystem path")
    url: Optional[HttpUrl] = Field(default=None, description="Remote source URL (requires fetch phase)")

    @field_validator("path", mode="before")
    @classmethod
    def _require_path_or_url(cls, path: Optional[str], info):  # type: ignore[override]
        data = getattr(info, "data", {})
        if not path and not data.get("url"):
            raise ValueError("dataset mapping requires either path or url")
        return path


class FetchRequest(BaseModel):
    url: HttpUrl
    datasetId: str
    outputDir: Optional[str] = None
    policy: PolicyRef
    sandbox: bool = True
    sandboxAllowNetwork: bool = True
    datasetTreeArity: int = Field(default=16, ge=2, le=64)


class RunRequest(BaseModel):
    traceId: str = "run"
    outputDir: Optional[str] = None
    policy: PolicyRef
    policyId: str
    backend: str = Field(default="geom", pattern="^(geom|risc0)$")
    steps: int = Field(default=64, gt=0)
    queries: int = Field(default=8, gt=0)
    challenges: int = Field(default=2, gt=0)
    datasets: List[DatasetMapping] = Field(default_factory=list)
    profile: str = Field(default="default", pattern="^(default|train|eval)$")
    sandbox: bool = False
    sandboxAllowNetwork: bool = False
    sandboxMemory: Optional[int] = Field(default=4096, gt=0)
    sandboxTimeout: Optional[int] = Field(default=600, gt=0)


class EmitRequest(BaseModel):
    source: Optional[str] = None
    capsulePath: Optional[str] = None
    outPath: str
    artifactsDir: Optional[str] = None
    archiveDir: Optional[str] = None
    policyPath: Optional[str] = None
    manifestsDir: Optional[str] = None
    profile: str = Field(default="proof-only", pattern="^(proof-only|da|replay)$")

    @field_validator("source", mode="before")
    @classmethod
    def _require_source_or_capsule(cls, source: Optional[str], info):  # type: ignore[override]
        data = getattr(info, "data", {})
        if not source and not data.get("capsulePath"):
            raise ValueError("source or capsulePath is required")
        return source


class VerifyRequest(BaseModel):
    capsulePath: str
    mode: str = Field(default="proof-only", pattern="^(proof-only|da|replay)$")
    policyPath: Optional[str] = None
    manifestsDir: Optional[str] = None
    datasets: List[DatasetMapping] = Field(default_factory=list)
    daProfileId: Optional[str] = Field(default=None, description="Name of da_profiles/<id>.json to load")
    daProfilePath: Optional[str] = Field(default=None, description="Explicit path to DA profile JSON")


class ReplayRequest(BaseModel):
    capsulePath: str
    datasets: List[DatasetMapping] = Field(default_factory=list)
    tolerance: float = 0.0
    range: Optional[str] = None
    sample: Optional[int] = Field(default=None, ge=1)
    sampleSeed: Optional[int] = None
    maxDivergences: Optional[int] = Field(default=100, ge=1)
    untilDiverge: bool = False
    verbose: bool = False


class AuditRequest(BaseModel):
    capsulePath: str
    format: str = Field(default="summary", pattern="^(summary|json|jsonl|csv)$")
    verifyChain: bool = True
    filterType: Optional[str] = None
    fromSeq: int = Field(default=0, ge=0)
    toSeq: Optional[int] = Field(default=None, ge=0)


class RowRequest(BaseModel):
    capsulePath: str
    row: int = Field(ge=0)
    schemaId: Optional[str] = None


class SandboxTestRequest(BaseModel):
    backend: Optional[str] = Field(default=None, description="backend name to force (optional)")


class JobResponse(BaseModel):
    jobId: str
    status: str
    location: Optional[str] = None


class CommandResponse(BaseModel):
    requestId: str
    status: str
    exitCode: int
    errorCode: Optional[str] = None
    command: List[str]
    stdout: str
    stderr: str
    result: Optional[Dict[str, Any]] = None
