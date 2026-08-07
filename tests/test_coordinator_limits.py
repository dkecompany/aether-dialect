"""Federation coordinator DuckDB memory and spill placement."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import duckdb
from aetherdialect._config import FederationLimits
from aetherdialect._constants import DIAGNOSTIC_CODE_COORDINATOR_LIMITS
from aetherdialect._core_utils import pop_federation_limits, push_federation_limits
from aetherdialect._federation import _configure_federation_coordinator_connection


@pytest.mark.fast
def test_memory_and_temp_directory_configured(tmp_path) -> None:
    """Coordinator DuckDB receives memory, temp directory, and thread limits at plan start."""
    custom_temp = tmp_path / "custom_coordinator_temp"
    limits = FederationLimits(
        coordinator_memory_limit_bytes=64 * 1024 * 1024,
        coordinator_threads=2,
        coordinator_temp_dir=str(custom_temp),
        coordinator_spill_max_bytes=128 * 1024 * 1024,
    )
    token = push_federation_limits(limits)
    conn = duckdb.connect(":memory:")
    try:
        with patch("aetherdialect._federation.notify") as notify_mock:
            memory_report, temp_directory, threads, owned_temp = _configure_federation_coordinator_connection(
                conn,
                federation_dir=str(tmp_path / "fed"),
            )
        assert memory_report == "64MB"
        assert temp_directory == str(custom_temp)
        assert threads == 2
        assert owned_temp is False
        assert custom_temp.is_dir()
        memory_setting = str(conn.execute("SELECT current_setting('memory_limit')").fetchone()[0])
        assert "MiB" in memory_setting or "MB" in memory_setting
        temp_setting = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        assert str(temp_setting).replace("\\", "/") == str(custom_temp).replace("\\", "/")
        assert int(conn.execute("SELECT current_setting('threads')").fetchone()[0]) == 2
        spill_setting = str(conn.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0])
        assert "GiB" in spill_setting or "GB" in spill_setting or "MiB" in spill_setting or "MB" in spill_setting
        limit_calls = [
            call for call in notify_mock.call_args_list if call.kwargs.get("code") == DIAGNOSTIC_CODE_COORDINATOR_LIMITS
        ]
        assert len(limit_calls) == 1
        details = dict(limit_calls[0].kwargs.get("details") or ())
        assert details["memory_limit"] == "64MB"
        assert details["temp_directory"] == str(custom_temp)
        assert details["threads"] == "2"
        assert details["phase"] == "plan"
    finally:
        conn.close()
        pop_federation_limits(token)
