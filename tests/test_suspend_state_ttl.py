"""Restored suspend TTL uses exported suspended_at and policy_ttl_seconds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aetherdialect._contracts_base import SuspendedSessionExpiredError
from aetherdialect._contracts_core import PipelineSuspended
from aetherdialect._main_session import PipelineSession


@pytest.mark.fast
def test_expired_restored_session_refuses() -> None:
    owner = SimpleNamespace(
        limits=SimpleNamespace(suspended_session_ttl_seconds=3600),
        _sandbox_closed=False,
    )
    sess = PipelineSession(owner, mode="writer")
    sess._suspended = PipelineSuspended("sql_confirm", "confirm?", SimpleNamespace(suspended_at=None))
    sess._session_busy = True
    sess._restored_suspended_at = datetime.now(UTC) - timedelta(seconds=100)
    sess._restored_policy_ttl_seconds = 10
    with pytest.raises(SuspendedSessionExpiredError):
        sess._enforce_suspended_session_ttl()
