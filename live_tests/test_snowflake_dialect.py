"""Snowflake dialect-syntax live tests. Exercises ``ARRAY_CONTAINS``, ``DATEDIFF``/``DATEADD`` date handling, and ``LOWER``-based case-insensitive match."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from ._dialect_scenarios import dialect_snowflake_scenarios
from ._engine_live import (
    build_engine_t2s,
    build_runner,
    engine_schema,
    skip_unless_configured,
)

_ENGINE = "snowflake"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

_SCENARIOS = dialect_snowflake_scenarios()


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, engine_schema("SNOWFLAKE_DATABASE", "DVDRENTAL_NEW"))


@pytest.fixture(scope="module")
def runner(t2s):
    return build_runner(t2s)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
def test_snowflake_dialect_syntax(runner, scenario):
    run_and_assert(runner, scenario, header=f"[snowflake:{scenario.id}] {scenario.question}")
