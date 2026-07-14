"""DuckDB dialect-syntax live tests. Exercises per-engine SQL rendering for native LIST arrays, date windows, date_diff, and ILIKE-based case-insensitive match."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from ._dialect_scenarios import dialect_duckdb_scenarios
from ._engine_live import (
    build_engine_t2s,
    build_runner,
    engine_schema,
    skip_unless_configured,
)

_ENGINE = "duckdb"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

_SCENARIOS = dialect_duckdb_scenarios()


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, engine_schema("DUCKDB_DATABASE", "rental_shop"))


@pytest.fixture(scope="module")
def runner(t2s):
    return build_runner(t2s)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
def test_duckdb_dialect_syntax(runner, scenario):
    run_and_assert(runner, scenario, header=f"[duckdb:{scenario.id}] {scenario.question}")
