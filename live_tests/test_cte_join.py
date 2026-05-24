"""CTE-with-joins live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import cte_join_scenarios

_scenarios = cte_join_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_cte_join(runner, scenario):
    """Run a CTE-join scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
