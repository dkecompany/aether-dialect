"""Ensure dual-backend dialects open exactly one live handle at construction."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import MySQLRuntimeConfig, SnowflakeRuntimeConfig
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect, MySQLDialect, SnowflakeDialect


def _databricks_config() -> SimpleNamespace:
    return SimpleNamespace(
        CATALOG="cat",
        SCHEMA="sch",
        SERVER_HOSTNAME="host",
        HTTP_PATH="/sql",
        ACCESS_TOKEN="tok",
        DEBUG=False,
        has_native_connection=lambda: True,
        sqlalchemy_url=lambda: "databricks://token:tok@host?http_path=%2Fsql&catalog=cat&schema=sch",
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("dialect_cls", "config", "native_patch", "native_attr", "extra_patches"),
    [
        (
            MySQLDialect,
            MySQLRuntimeConfig,
            "aetherdialect._dialect_sqlglot_engines.MySQLDialect._open_mysql_connector",
            "_native_connection",
            (
                ("aetherdialect._config.MySQLRuntimeConfig.PASSWORD", "secret"),
                ("aetherdialect._config.MySQLRuntimeConfig.DATABASE", "app"),
            ),
        ),
        (
            SnowflakeDialect,
            SnowflakeRuntimeConfig,
            "snowflake.connector.connect",
            "_snowflake_connection",
            (
                ("aetherdialect._config.SnowflakeRuntimeConfig.ACCOUNT", "acct"),
                ("aetherdialect._config.SnowflakeRuntimeConfig.USER", "user"),
                ("aetherdialect._config.SnowflakeRuntimeConfig.PASSWORD", "secret"),
                (
                    "aetherdialect._config.SnowflakeRuntimeConfig.snowpark_session_reachable",
                    classmethod(lambda cls: False),
                ),
            ),
        ),
        (
            DatabricksDialect,
            _databricks_config(),
            "databricks.sql.connect",
            "connection",
            (("aetherdialect._dialect.Dialect.__init__", None),),
        ),
    ],
)
def test_one_live_handle_per_engine(
    dialect_cls: type,
    config: object,
    native_patch: str,
    native_attr: str,
    extra_patches: tuple[tuple[str, object], ...],
) -> None:
    create_engine_calls: list[object] = []
    native_connect_calls: list[object] = []
    mock_native = MagicMock(name="native_connection")

    def _record_create_engine(*_args: object, **_kwargs: object) -> MagicMock:
        eng = MagicMock(name="sqlalchemy_engine")
        create_engine_calls.append(eng)
        return eng

    def _record_native_connect(*_args: object, **_kwargs: object) -> MagicMock:
        native_connect_calls.append(mock_native)
        return mock_native

    with ExitStack() as stack:
        stack.enter_context(patch("aetherdialect._utils.require_driver"))
        stack.enter_context(patch("aetherdialect._dialect_sqlglot_helper.require_driver"))
        stack.enter_context(
            patch("aetherdialect._dialect_sqlglot_helper.create_engine", side_effect=_record_create_engine)
        )
        stack.enter_context(
            patch("aetherdialect._dialect_sqlglot_engines.create_engine", side_effect=_record_create_engine)
        )
        stack.enter_context(patch(native_patch, side_effect=_record_native_connect))
        for target, value in extra_patches:
            if value is None:
                stack.enter_context(patch(target, return_value=None))
            else:
                stack.enter_context(patch(target, value))
        if dialect_cls is DatabricksDialect:
            cursor = MagicMock()
            mock_native.cursor.return_value = cursor
        dialect = dialect_cls(config)

    assert len(create_engine_calls) == 0, "SQLAlchemy engine should not open when native connector is selected"
    assert len(native_connect_calls) == 1, "native connector should open exactly once"
    assert getattr(dialect, "engine", None) is None
    assert getattr(dialect, native_attr) is mock_native
