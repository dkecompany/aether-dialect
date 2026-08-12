"""Redshift dialect-syntax live tests. Exercises per-engine SQL rendering for array contains, date windows, date differences, and case-insensitive match. Skipped automatically when Redshift is not configured in the live env file."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .live_support import (
    build_engine_t2s,
    build_runner,
    engine_schema,
    skip_unless_configured,
)
from .mydb_scenarios import dialect_redshift_scenarios

_ENGINE = "redshift"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

_SCENARIOS = dialect_redshift_scenarios()


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, engine_schema("REDSHIFT_DATABASE", "rental_shop"))


@pytest.fixture(scope="module")
def runner(t2s):
    return build_runner(t2s)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
def test_redshift_dialect_syntax(runner, scenario):
    run_and_assert(runner, scenario, header=f"[redshift:{scenario.id}] {scenario.question}")
