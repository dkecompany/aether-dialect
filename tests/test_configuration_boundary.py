"""Configuration boundary: connection identity via env/file, behaviour via limits only."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._config import (
    REMOVED_BEHAVIOUR_ENVIRONMENT_KEYS,
    EngineLimits,
    FederationLimits,
    PolicyConfig,
    PostgresRuntimeConfig,
)
from aetherdialect._constants import DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED
from aetherdialect._main_execution import (
    MainExecutionOps,
)

_BEHAVIOUR_KEY_CASES: tuple[tuple[str, str, str, object], ...] = (
    ("AETHERDIALECT_MAX_QUERY_COST_ROWS", "MAX_QUERY_COST_ROWS", "1", 50_000_000.0),
    ("AETHERDIALECT_MAX_QUERY_COST_BYTES", "MAX_QUERY_COST_BYTES", "1", 50_000_000_000.0),
    ("AETHERDIALECT_STATEMENT_TIMEOUT_MS", "STATEMENT_TIMEOUT_MS", "999999", 30_000),
    ("AETHERDIALECT_LLM_TIMEOUT_MS", "LLM_TIMEOUT_MS", "999999", 60_000),
    ("AETHERDIALECT_PROFILE_TIMEOUT_MS", "PROFILE_TIMEOUT_MS", "999999", 120_000),
    ("AETHERDIALECT_EXPLAIN_TIMEOUT_MS", "EXPLAIN_TIMEOUT_MS", "999999", None),
    ("AETHERDIALECT_LLM_BATCH_ENABLED", "LLM_BATCH_ENABLED", "true", False),
    ("AETHERDIALECT_TABULAR_LLM_ASSIST", "TABULAR_LLM_ASSIST", "false", True),
)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("env_key", "policy_attr", "poison", "expected_default"),
    _BEHAVIOUR_KEY_CASES,
    ids=[case[0] for case in _BEHAVIOUR_KEY_CASES],
)
def test_behaviour_keys_no_longer_read_from_environment(
    env_key: str,
    policy_attr: str,
    poison: str,
    expected_default: object,
) -> None:
    env = {env_key: poison}
    MainExecutionOps._apply_runtime_environments(env)
    assert getattr(PolicyConfig, policy_attr) == expected_default


@pytest.mark.fast
def test_connection_keys_still_read_from_file(tmp_path) -> None:
    path = tmp_path / "conn.toml"
    path.write_text(
        "\n".join(
            (
                "[postgresql]",
                'host = "from-file-host"',
                'database = "from-file-db"',
                'user = "from-file-user"',
                'password = "from-file-password"',
                "",
                "[engine]",
                'selected = "postgresql"',
            ),
        ),
        encoding="utf-8",
    )
    flat, claimed, _named = MainExecutionOps._load_config_file(str(path))
    merged, _diag = MainExecutionOps._merge_configuration_environment(flat, toml_claimed_keys=claimed)
    MainExecutionOps._apply_runtime_environments(merged)
    assert PostgresRuntimeConfig.HOST == "from-file-host"
    assert PostgresRuntimeConfig.DATABASE == "from-file-db"
    assert PostgresRuntimeConfig.USER == "from-file-user"
    assert PostgresRuntimeConfig.PASSWORD == "from-file-password"


@pytest.mark.fast
def test_caller_supplied_limits_not_overlaid_by_file(tmp_path) -> None:
    path = tmp_path / "limits.toml"
    path.write_text(
        "\n".join(
            (
                "[limits]",
                "pool_size = 99",
                "statement_timeout_ms = 45000",
                "",
                "[federation_limits]",
                "max_members = 16",
                "",
                "[federation_limits.member_defaults]",
                "pool_size = 7",
            ),
        ),
        encoding="utf-8",
    )
    caller_engine_limits = EngineLimits(pool_size=2, statement_timeout_ms=30_000)
    caller_fed_limits = FederationLimits(max_members=8)
    file_engine_limits = EngineLimits.from_config_file(path)
    file_fed_limits = FederationLimits.from_config_file(path)
    assert file_engine_limits.pool_size == 99
    assert file_engine_limits.statement_timeout_ms == 45_000
    assert file_fed_limits.max_members == 16
    assert file_fed_limits.member_defaults is not None
    assert file_fed_limits.member_defaults.pool_size == 7
    assert caller_engine_limits.pool_size == 2
    assert caller_engine_limits.statement_timeout_ms == 30_000
    assert caller_fed_limits.max_members == 8


@pytest.mark.fast
def test_removed_key_still_set_emits_diagnostic() -> None:
    env = {"AETHERDIALECT_STATEMENT_TIMEOUT_MS": "45000"}
    with patch("aetherdialect._main_execution.notify") as notify_mock:
        MainExecutionOps.emit_ignored_behaviour_environment_diagnostics(env)
    notify_mock.assert_called_once()
    assert notify_mock.call_args.kwargs["code"] == DIAGNOSTIC_CODE_CONFIGURATION_KEY_IGNORED
    details = dict(notify_mock.call_args.kwargs["details"])
    assert details["key"] == "AETHERDIALECT_STATEMENT_TIMEOUT_MS"
    assert details["replacement"] == REMOVED_BEHAVIOUR_ENVIRONMENT_KEYS["AETHERDIALECT_STATEMENT_TIMEOUT_MS"]
