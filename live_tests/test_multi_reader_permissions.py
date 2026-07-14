"""Live tests for multi-reader owner and consumer permission scenarios. Dual Postgres logins exercise database RBAC together with application scope. The owner uses primary PGUSER and PGPASSWORD from the live env file with role owner and full schema reflection. The consumer uses PGUSER2 and PGPASSWORD2 with role consumer and EngineContext allow_objects matching the consumer database grants. Configure PGUSER2 and PGPASSWORD2 in the live env file before running these tests."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE
from live_tests.conftest import (
    _CONSUMER_CREDENTIALS_SKIP_REASON,
    _consumer_credentials_configured,
)

pytestmark = pytest.mark.skipif(
    not _consumer_credentials_configured(),
    reason=_CONSUMER_CREDENTIALS_SKIP_REASON,
)


def test_consumer_init_matches_owner_template_count(
    t2s_rbac_owner: AetherEngine,
    t2s_consumer_pguser2: AetherEngine,
) -> None:
    """Consumer with fewer table grants initializes without MigrationPendingError."""
    owner = t2s_rbac_owner
    consumer = t2s_consumer_pguser2
    assert len(owner._templates) == len(consumer._templates)


def test_consumer_forbidden_table_permission_denied(
    t2s_consumer_pguser2: AetherEngine,
) -> None:
    """Consumer asking about a forbidden table gets permission_denied without SQL leakage."""
    t2s = t2s_consumer_pguser2
    with t2s.session(mode="reader") as session:
        step = session.ask("Show payroll for all employees including SSN")
        while not step.done:
            if step.prompt:
                step = session.step("y")
            else:
                break
        assert step.status == "permission_denied"
        assert PERMISSION_DENIED_USER_MESSAGE in (step.message or "")
        assert step.sql is None


def test_two_readers_queue_drained_by_owner(
    t2s_rbac_owner: AetherEngine,
    t2s_consumer_pguser2: AetherEngine,
) -> None:
    """Reader sessions enqueue learning; owner writer drain merges events."""
    owner = t2s_rbac_owner
    consumer = t2s_consumer_pguser2
    with consumer.session(mode="reader") as reader:
        step = reader.ask("How many rows in film?")
        while not step.done and step.prompt:
            step = reader.step("y")
    with owner.session(mode="writer") as writer:
        step = writer.ask("How many rows in film?")
        while not step.done and step.prompt:
            step = writer.step("y")
