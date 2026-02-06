"""Policy-governed network access helper."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from bef_zk.codec import ENCODING_ID, canonical_encode

from .rules import PolicyError

HASH_PREFIX_NETWORK_LOG = b"NETWORK_LOG_V1::"


@dataclass
class NetworkFetchRecord:
    url: str
    timestamp: int
    bytes: int
    sha256: str
    domain: str

    def to_obj(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "timestamp": int(self.timestamp),
            "bytes": int(self.bytes),
            "sha256": self.sha256,
            "domain": self.domain,
        }

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "NetworkFetchRecord":
        return cls(
            url=str(obj.get("url", "")),
            timestamp=int(obj.get("timestamp", 0)),
            bytes=int(obj.get("bytes", 0)),
            sha256=str(obj.get("sha256", "")),
            domain=str(obj.get("domain", "")),
        )


def compute_network_log_root(entries: Iterable[NetworkFetchRecord]) -> str:
    acc = b"\x00" * 32
    for entry in entries:
        encoded = canonical_encode(entry.to_obj(), encoding_id=ENCODING_ID)
        acc = hashlib.sha256(HASH_PREFIX_NETWORK_LOG + acc + encoded).digest()
    return acc.hex()


class NetworkGateway:
    """Download helper that enforces policy-defined network budgets."""

    def __init__(self, rules: Dict[str, Any] | None, *, base_dir: Path) -> None:
        self.rules = rules or {}
        self.enabled = bool(self.rules.get("enabled"))
        self.allowed_domains = {d.lower() for d in self.rules.get("allowed_domains", [])}
        self.require_tls = bool(self.rules.get("require_tls"))
        self.max_bytes = int(self.rules.get("max_bytes_total") or 0) or None
        self.bytes_total = 0
        self.entries: List[NetworkFetchRecord] = []
        self.base_dir = base_dir

    def _check_policy(self, url: str) -> str:
        if not self.enabled:
            raise PolicyError("network access disabled by policy")
        parsed = url_parse.urlparse(url)
        if self.require_tls and parsed.scheme.lower() != "https":
            raise PolicyError("policy requires HTTPS network access")
        domain = (parsed.hostname or "").lower()
        if self.allowed_domains and domain not in self.allowed_domains:
            raise PolicyError(f"domain '{domain}' not permitted by policy")
        return domain

    def fetch(self, url: str, dest_path: Path, *, timeout: float = 15.0) -> NetworkFetchRecord:
        domain = self._check_policy(url)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        total = 0
        try:
            with url_request.urlopen(url, timeout=timeout) as resp, dest_path.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if self.max_bytes is not None and self.bytes_total + total > self.max_bytes:
                        raise PolicyError("network byte budget exceeded")
                    fh.write(chunk)
                    hasher.update(chunk)
        except url_error.URLError as exc:
            raise PolicyError(f"network fetch failed for {url}: {exc}") from exc
        self.bytes_total += total
        record = NetworkFetchRecord(
            url=url,
            timestamp=int(time.time()),
            bytes=total,
            sha256=hasher.hexdigest(),
            domain=domain,
        )
        self.entries.append(record)
        return record

    def iter_entries(self) -> List[NetworkFetchRecord]:
        return list(self.entries)


def enforce_network_log(rules: Dict[str, Any] | None, entries: Iterable[NetworkFetchRecord]) -> None:
    if not rules:
        return
    enabled = bool(rules.get("enabled"))
    if not enabled:
        if list(entries):
            raise PolicyError("network usage recorded but policy disables network access")
        return
    allowed = {d.lower() for d in rules.get("allowed_domains", [])}
    max_bytes = int(rules.get("max_bytes_total") or 0) or None
    require_tls = bool(rules.get("require_tls"))
    bytes_total = 0
    for entry in entries:
        bytes_total += int(entry.bytes)
        if max_bytes is not None and bytes_total > max_bytes:
            raise PolicyError("network log exceeds policy byte budget")
        if allowed and entry.domain.lower() not in allowed:
            raise PolicyError(f"network log domain '{entry.domain}' not allowed")
        if require_tls and not entry.url.lower().startswith("https://"):
            raise PolicyError("network log contains non-HTTPS fetch")


__all__ = [
    "NetworkGateway",
    "NetworkFetchRecord",
    "compute_network_log_root",
    "enforce_network_log",
]
