"""Ensure short-lived raw connections are returned to pools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._main_execution import MainExecutionOps


@pytest.mark.fast
def test_query_log_read_returns_connection_to_pool() -> None:
    raw = MagicMock()
    sa_engine = MagicMock()
    sa_engine.raw_connection.return_value = raw
    dialect = MagicMock()
    dialect.engine = sa_engine
    dialect.connection = None
    engine = MagicMock()
    engine._schema_graph = MagicMock()
    engine._dialect = dialect
    engine._artifacts_dir = "/tmp/artifacts"
    engine._store = {}
    engine._templates = {}
    with (
        patch("aetherdialect._federation_compose.assert_query_log_warmup_allowed"),
        patch("aetherdialect._sql_to_intent.fetch_query_log", return_value=[]),
        patch("aetherdialect._main_interactive.MainInteractiveOps._run_seed_warmup_sql_history_pipeline"),
    ):
        MainExecutionOps.run_seed_warmup_from_query_log_execution(engine)
    raw.close.assert_called_once()
