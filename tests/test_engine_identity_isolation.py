"""Engine identity isolation across coexisting AetherEngine instances."""

from __future__ import annotations

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._core_utils import (
    active_engine_identity,
    pop_engine_identity,
    push_engine_identity,
)
from aetherdialect._dialect import Dialect, DialectRegistry


@pytest.mark.fast
def test_active_engine_identity_requires_pushed_context(unbound_engine_identity: None) -> None:
    with pytest.raises(RuntimeError, match="no active engine identity"):
        active_engine_identity()


@pytest.mark.fast
def test_sqlglot_dialect_tracks_engine_type() -> None:
    assert DialectRegistry.sqlglot_dialect_for_engine("duckdb") == "duckdb"
    assert DialectRegistry.sqlglot_dialect_for_engine("postgresql") == "postgres"


@pytest.mark.fast
def test_active_sqlglot_dialect_follows_identity_context() -> None:
    duck = EngineIdentity("duckdb", EngineConfig.RUNTIME)
    pg = EngineIdentity("postgresql", EngineConfig.RUNTIME)
    token_duck = push_engine_identity(duck)
    assert Dialect.active_sqlglot_dialect() == "duckdb"
    token_pg = push_engine_identity(pg)
    assert Dialect.active_sqlglot_dialect() == "postgres"
    pop_engine_identity(token_pg)
    assert Dialect.active_sqlglot_dialect() == "duckdb"
    pop_engine_identity(token_duck)
