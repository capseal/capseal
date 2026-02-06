from __future__ import annotations

import time
from typing import Mapping

import pytest

from bef_zk.da.provider import (
    AvailabilityError,
    ChunkFetchResult,
    ChunkInclusionProof,
    DAProvider,
    PolicyAwareDAClient,
)


class _StubProvider(DAProvider):
    def __init__(self, *, failures: int = 0, sleep_ms: int = 0) -> None:
        self.failures = failures
        self.sleep_ms = sleep_ms
        self.calls = 0

    def fetch_batch(
        self,
        indices,
        *,
        timeout_ms: int | None = None,
    ) -> Mapping[int, ChunkFetchResult]:
        self.calls += 1
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000)
        if self.calls <= self.failures:
            raise AvailabilityError("simulated failure")
        proof = ChunkInclusionProof(siblings=[], arity=2, tree_size=1)
        return {idx: ChunkFetchResult(values=[idx], proof=proof) for idx in indices}


def test_policy_client_retries_and_succeeds() -> None:
    flaky = _StubProvider(failures=1)
    client = PolicyAwareDAClient(flaky, retries=1, timeout_ms=100)
    result = client.fetch_batch([0])
    assert 0 in result
    assert flaky.calls == 2


def test_policy_client_exhausts_retries() -> None:
    flaky = _StubProvider(failures=2)
    client = PolicyAwareDAClient(flaky, retries=1, timeout_ms=100)
    with pytest.raises(AvailabilityError):
        client.fetch_batch([0])
    assert flaky.calls == 2


def test_policy_client_timeout() -> None:
    slow = _StubProvider(sleep_ms=50)
    client = PolicyAwareDAClient(slow, retries=0, timeout_ms=10)
    with pytest.raises(AvailabilityError):
        client.fetch_batch([0])
