"""PostgreSQL core pipeline live tests. Runs the full rental_shop scenario bundle plus stateful sequences once on the session-scoped PostgreSQL runner from ``conftest.py``. Per-engine modules keep only connection smoke and dialect-syntax coverage."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert, run_sequence_and_assert

from .mydb_scenarios import (
    bundled_rental_shop_live_scenarios,
    stateful_scenarios,
)

_BUNDLE = bundled_rental_shop_live_scenarios()
_STATEFUL = stateful_scenarios()


@pytest.mark.parametrize("scenario", _BUNDLE, ids=[s.id for s in _BUNDLE])
def test_core_pipeline_bundle(runner, scenario):
    """Run the full rental_shop scenario bundle on PostgreSQL."""
    run_and_assert(runner, scenario, header=f"[core:{scenario.id}] {scenario.question}")


@pytest.mark.parametrize("seq", _STATEFUL, ids=[s.id for s in _STATEFUL])
def test_core_pipeline_stateful(runner, seq):
    """Run stateful sequence scenarios on PostgreSQL."""
    run_sequence_and_assert(runner, seq)
