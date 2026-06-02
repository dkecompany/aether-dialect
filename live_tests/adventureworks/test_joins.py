"""Cross-table join live tests against the local AdventureWorks database."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .aw_scenarios import join_scenarios

_scenarios = join_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_joins(runner, scenario):
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
