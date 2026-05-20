"""Column names assertion live tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import column_names_scenarios

_scenarios = column_names_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_column_names(runner, scenario):
    """Run scenarios asserting expected output column headers."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
