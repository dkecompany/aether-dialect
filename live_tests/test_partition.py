"""Live tests for partition-related scenarios (``dvdrental_new`` / Delta ``rental_pt`` when present)."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import partition_scenarios

_scenarios = partition_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_partition(runner, scenario):
    """Run partition-scenario questions; Databricks may inject partition predicates when modeled."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
