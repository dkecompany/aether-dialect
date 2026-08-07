"""Connection pool configuration from EngineLimits."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import EngineLimits, FederationLimits
from aetherdialect._core_utils import (
    sqlalchemy_pool_kwargs_from_limits,
    validate_federation_pool_capacity,
)


@pytest.mark.fast
def test_pool_arguments_reach_create_engine() -> None:
    limits = EngineLimits(pool_size=2, pool_max_overflow=6, pool_recycle_seconds=900, pool_timeout_seconds=15)
    kwargs = sqlalchemy_pool_kwargs_from_limits(limits)
    assert kwargs == {
        "pool_size": 2,
        "max_overflow": 6,
        "pool_recycle": 900,
        "pool_pre_ping": True,
        "pool_timeout": 15,
    }
    recorded: list[dict[str, object]] = []

    def _record_create(url: str, **kw: object) -> MagicMock:
        recorded.append(dict(kw))
        return MagicMock()

    with (
        patch("aetherdialect._dialect_postgres.create_engine", side_effect=_record_create),
        patch("aetherdialect._dialect_postgres.PostgresDialect._require_pglast"),
        patch(
            "aetherdialect._dialect_postgres.PostgresRuntimeConfig.db_url",
            return_value="postgresql+psycopg://user:pass@localhost/db",
        ),
    ):
        from aetherdialect._config import PostgresRuntimeConfig
        from aetherdialect._dialect_postgres import PostgresDialect

        PostgresDialect(PostgresRuntimeConfig, limits=limits)
    assert recorded
    assert recorded[0]["pool_size"] == 2
    assert recorded[0]["max_overflow"] == 6


@pytest.mark.fast
def test_undersized_pool_reports() -> None:
    members = {
        "a": MagicMock(limits=EngineLimits(pool_size=1, pool_max_overflow=0)),
        "b": MagicMock(limits=EngineLimits(pool_size=1, pool_max_overflow=0)),
    }
    fed_limits = FederationLimits(max_parallel_members=4)
    with patch("aetherdialect._core_utils.notify") as notify_mock:
        validate_federation_pool_capacity(members, fed_limits)
    notify_mock.assert_called()
