"""Row value check assertion live tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import row_value_check_scenarios

_scenarios = row_value_check_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_row_value_check(runner, scenario):
    """Run scenarios with custom row-value assertions."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
