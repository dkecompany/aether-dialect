"""BigQuery dialect-syntax live tests. Exercises ``IN UNNEST`` array membership, ``DATE_DIFF``/``DATE_TRUNC`` date handling with ``DATE(...)`` normalization, and ``LOWER``-based matching."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .live_support import (
    build_engine_t2s,
    build_runner,
    engine_schema,
    skip_unless_configured,
)
from .mydb_scenarios import dialect_bigquery_scenarios

_ENGINE = "bigquery"
_SKIP_REASON = skip_unless_configured(_ENGINE)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

_SCENARIOS = dialect_bigquery_scenarios()


@pytest.fixture(scope="module")
def t2s():
    return build_engine_t2s(_ENGINE, engine_schema("BIGQUERY_DATASET", "rental_shop"))


@pytest.fixture(scope="module")
def runner(t2s):
    return build_runner(t2s)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
def test_bigquery_dialect_syntax(runner, scenario):
    run_and_assert(runner, scenario, header=f"[bigquery:{scenario.id}] {scenario.question}")
