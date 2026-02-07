"""Opening governance helpers (tickets, co-signing, budgets)."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, Iterable, List

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from bef_zk.codec import ENCODING_ID, canonical_encode
from bef_zk.policy import PolicyError, PolicyConfig

DEFAULT_OPENING_STATE_DIR = Path.home() / ".capsule_openings"


@dataclass
class OpeningState:
    rows_opened: int = 0
    bytes_opened: int = 0
    last_hash: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_opened": self.rows_opened,
            "bytes_opened": self.bytes_opened,
            "last_hash": self.last_hash,
        }

    @classmethod
    def from_head(cls, path: Path) -> "OpeningState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return cls()
        return cls(
            rows_opened=int(data.get("rows_opened") or 0),
            bytes_opened=int(data.get("bytes_opened") or 0),
            last_hash=data.get("last_hash"),
        )


class OpeningPolicyEnforcer:
    """Applies the openings_governance + co_signing rules."""

    def __init__(
        self,
        *,
        policy_config: PolicyConfig,
        capsule_data: Dict[str, Any],
        policy_path: Path,
        state_dir: Path | None = None,
        allow_reset: bool = False,
    ) -> None:
        self.policy_config = policy_config
        self.capsule = capsule_data
        self.policy_path = policy_path
        self.rules = policy_config.rules.get("openings_governance") or {}
        self.co_rules = policy_config.rules.get("co_signing") or {}
        if not self.rules.get("openings_enabled"):
            raise PolicyError("openings are disabled by policy")
        self.state_dir = state_dir or DEFAULT_OPENING_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.capsule_hash = str(capsule_data.get("capsule_hash") or "")
        policy_hash = str((capsule_data.get("policy") or {}).get("policy_hash") or "")
        self.registry_path = self.state_dir / "registry.json"
        self.registry = self._load_registry()
        self.registry_key = f"{self.capsule_hash}:{policy_hash}"
        self.ledger_path = self.state_dir / f"{self.registry_key}.ledger.jsonl"
        self.head_path = self.state_dir / f"{self.registry_key}.head.json"
        self.allow_reset = allow_reset
        self._ensure_state_files()
        self.state = OpeningState.from_head(self.head_path)

    def enforce(
        self,
        *,
        ticket_path: Path | None,
        dataset_id: str | None,
        row_indices: List[int],
        bytes_requested: int,
    ) -> None:
        if self.rules.get("require_opening_ticket") and not ticket_path:
            raise PolicyError("policy requires a signed opening ticket")
        ticket = self._load_ticket(ticket_path) if ticket_path else None
        if ticket:
            self._validate_ticket(ticket, dataset_id=dataset_id, row_indices=row_indices, bytes_requested=bytes_requested)
        self._enforce_budgets(len(row_indices), bytes_requested)
        self._persist_state(len(row_indices), bytes_requested)

    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self.registry, indent=2))

    def _ensure_state_files(self) -> None:
        ledger_exists = self.ledger_path.exists()
        head_exists = self.head_path.exists()
        registry_entry = self.registry.get(self.registry_key)
        if (ledger_exists and not head_exists) or (head_exists and not ledger_exists):
            ledger_exists = False
            head_exists = False
        if registry_entry and not (ledger_exists and head_exists):
            if not self.allow_reset:
                raise PolicyError(
                    "opening ledger missing; use --allow-opening-reset to reinitialize budgets"
                )
            self.registry.pop(self.registry_key, None)
            self._save_registry()
        if not self.ledger_path.exists():
            self.ledger_path.write_text("")
        if not self.head_path.exists():
            state = OpeningState()
            self.head_path.write_text(json.dumps(state.to_dict(), indent=2))
            self.registry[self.registry_key] = {
                "initialized_at": datetime.now(tz=timezone.utc).isoformat(),
                "last_hash": None,
            }
            self._save_registry()

    def _load_ticket(self, path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PolicyError(f"invalid ticket JSON: {path}") from exc
        ticket = data.get("ticket") or data
        signatures = data.get("signatures") or ticket.get("signatures")
        if signatures is None:
            raise PolicyError("opening ticket missing signatures block")
        ticket["signatures"] = signatures
        return ticket

    def _validate_ticket(
        self,
        ticket: Dict[str, Any],
        *,
        dataset_id: str | None,
        row_indices: List[int],
        bytes_requested: int,
    ) -> None:
        ticket_body = {k: v for k, v in ticket.items() if k != "signatures"}
        ticket_hash = hashlib.sha256(canonical_encode(ticket_body, encoding_id=ENCODING_ID)).digest()
        if self.rules.get("ticket_binding") == "CAPSULE_HASH":
            if ticket_body.get("capsule_hash") != self.capsule_hash:
                raise PolicyError("ticket does not reference this capsule")
        if dataset_id and ticket_body.get("dataset_id") and ticket_body.get("dataset_id") != dataset_id:
            raise PolicyError("ticket dataset_id mismatch")
        requested = sorted(row_indices)
        ticket_indices = sorted(ticket_body.get("row_indices") or [])
        if requested and ticket_indices:
            if requested != ticket_indices:
                raise PolicyError("ticket row indices do not match requested rows")
        now = datetime.now(tz=timezone.utc).timestamp()
        expires_at = int(ticket_body.get("expires_at_unix") or 0)
        if expires_at and now > expires_at:
            raise PolicyError("opening ticket has expired")
        issued_at = int(ticket_body.get("issued_at_unix") or 0)
        if issued_at and now < issued_at:
            raise PolicyError("opening ticket not yet valid")
        limit = int(ticket_body.get("max_bytes_to_disclose") or 0)
        if limit and bytes_requested > limit:
            raise PolicyError("requested data exceeds ticket disclosure budget")
        self._validate_signatures(ticket_body, ticket.get("signatures") or [], ticket_hash)
        self._enforce_index_policy(row_indices)
        self._enforce_redaction(ticket_body)

    def _validate_signatures(
        self,
        ticket_body: Dict[str, Any],
        signatures: Iterable[Dict[str, Any]],
        digest: bytes,
    ) -> None:
        if not signatures:
            raise PolicyError("opening ticket missing signatures")
        if not self.co_rules.get("required"):
            return
        min_sigs = int(self.co_rules.get("min_signatures") or 1)
        roles_required = set(self.co_rules.get("roles_required") or [])
        authorized = {
            signer["pubkey"]: signer
            for signer in self.co_rules.get("authorized_signers", [])
        }
        satisfied_roles: set[str] = set()
        for sig in signatures:
            pub = sig.get("pubkey")
            signature_b64 = sig.get("signature")
            if pub not in authorized:
                raise PolicyError("ticket signed by unauthorized key")
            try:
                pub_bytes = base64.b64decode(pub)
                signature_bytes = base64.b64decode(signature_b64)
            except Exception as exc:  # noqa: BLE001
                raise PolicyError("invalid ticket signature encoding") from exc
            try:
                ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes).verify(signature_bytes, digest)
            except InvalidSignature as exc:  # noqa: B904
                raise PolicyError("ticket signature verification failed") from exc
            signer_role = authorized[pub].get("role")
            if signer_role:
                satisfied_roles.add(signer_role)
        if len(signatures) < min_sigs:
            raise PolicyError("not enough signatures on ticket")
        missing_roles = roles_required - satisfied_roles
        if missing_roles:
            raise PolicyError(f"ticket missing signatures for roles: {sorted(missing_roles)}")

    def _enforce_index_policy(self, row_indices: List[int]) -> None:
        if not row_indices:
            return
        index_rules = self.rules.get("openings_index_policy")
        if not index_rules:
            return
        if index_rules.get("mode") != "DA_SEED_ONLY":
            return
        sampling_cfg = self.capsule.get("da_sampling") or {}
        seed_hex = sampling_cfg.get("seed_hex")
        population = int(sampling_cfg.get("population_size") or 0)
        sample_count = int(sampling_cfg.get("k") or 0)
        unique = bool(sampling_cfg.get("unique", True))
        if sampling_cfg.get("fn_id") != "sha256_mod_v1":
            sampling_cfg = {}
        if not sampling_cfg:
            seed_source = index_rules.get("da_seed_source") or ""
            if seed_source == "capsule.da_policy.deterministic_seed":
                seed_hex = ((self.capsule.get("da_policy") or {}).get("deterministic_seed"))
            da_policy = self.capsule.get("da_policy") or {}
            population = int((self.capsule.get("chunk_meta") or {}).get("num_chunks") or 0)
            sample_count = int(da_policy.get("k_samples") or len(row_indices))
            unique = True
        if not seed_hex:
            raise PolicyError("policy requires DA seed but capsule did not record one")
        allowed = _derive_da_indices(seed_hex, population, sample_count, unique=unique)
        for idx in row_indices:
            if idx not in allowed:
                raise PolicyError("row index not allowed by DA seed policy")

    def _enforce_redaction(self, ticket_body: Dict[str, Any]) -> None:
        redaction_rules = self.rules.get("redaction") or {}
        if not redaction_rules:
            return
        ticket_mode = ticket_body.get("redaction_mode")
        if ticket_mode and ticket_mode != redaction_rules.get("mode"):
            raise PolicyError("ticket redaction mode does not match policy")

    def _enforce_budgets(self, rows: int, bytes_requested: int) -> None:
        budget = self.rules.get("openings_budget") or {}
        max_per_request = int(budget.get("max_openings_per_request") or 0)
        if max_per_request and rows > max_per_request:
            raise PolicyError("request exceeds per-request opening budget")
        max_total_rows = int(budget.get("max_row_openings_total") or 0)
        if max_total_rows and self.state.rows_opened + rows > max_total_rows:
            raise PolicyError("policy opening budget exhausted (rows)")
        max_total_bytes = int(budget.get("max_bytes_opened_total") or 0)
        if max_total_bytes and self.state.bytes_opened + bytes_requested > max_total_bytes:
            raise PolicyError("policy opening budget exhausted (bytes)")

    def _persist_state(self, rows: int, bytes_requested: int) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "delta_rows": rows,
            "delta_bytes": bytes_requested,
            "totals": {
                "rows": self.state.rows_opened + rows,
                "bytes": self.state.bytes_opened + bytes_requested,
            },
            "prev_hash": self.state.last_hash or "0" * 64,
            "request_id": str(uuid4()),
        }
        encoded = canonical_encode(entry, encoding_id=ENCODING_ID)
        entry_hash = hashlib.sha256(encoded).hexdigest()
        entry["state_hash"] = entry_hash
        with self.ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry) + "\n")
        self.state.rows_opened += rows
        self.state.bytes_opened += bytes_requested
        self.state.last_hash = entry_hash
        self.head_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        self.registry[self.registry_key] = {
            "last_hash": entry_hash,
            "updated_at": entry["timestamp"],
        }
        self._save_registry()


def _derive_da_indices(seed_hex: str, total_rows: int, sample_count: int, *, unique: bool = True) -> set[int]:
    if total_rows <= 0:
        return set()
    seed_int = int(seed_hex, 16)
    limit = min(sample_count or total_rows, total_rows if unique else sample_count or total_rows)
    indices: set[int] = set()
    counter = 0
    while len(indices) < limit:
        material = seed_int.to_bytes(max(1, (seed_int.bit_length() + 7) // 8), "big") + counter.to_bytes(4, "big")
        digest = hashlib.sha256(material).digest()
        candidate = int.from_bytes(digest[:8], "big") % total_rows
        counter += 1
        if not unique or candidate not in indices:
            indices.add(candidate)
    return indices


def resolve_policy_path_for_capsule(
    capsule_data: Dict[str, Any],
    base_dir: Path,
    explicit_policy: Path | None,
) -> Path | None:
    if explicit_policy:
        return explicit_policy
    pol_section = capsule_data.get("policy") or {}
    recorded_path = pol_section.get("policy_path")
    if recorded_path:
        path = Path(recorded_path)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            return path
    fallback = base_dir / "policy.json"
    if fallback.exists():
        return fallback
    return None
