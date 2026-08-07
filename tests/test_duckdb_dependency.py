"""Tests for DuckDB core versus sandbox SQLAlchemy dialect dependencies."""

from __future__ import annotations

import importlib.util

import pytest

from aetherdialect._contracts_base import ConfigError
from aetherdialect._dialect_sqlglot_helper import SqlalchemyExecutionMixin


def test_duckdb_sqlalchemy_dialect_missing_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def _hide_duckdb_engine(name: str, package: str | None = None) -> object | None:
        if name == "duckdb_engine":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _hide_duckdb_engine)
    with pytest.raises(ConfigError, match="duckdb-engine"):
        SqlalchemyExecutionMixin.require_duckdb_sqlalchemy_dialect()
