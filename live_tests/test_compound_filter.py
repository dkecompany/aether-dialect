"""Compound filter (multi-condition WHERE) live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import compound_filter_scenarios

_scenarios = compound_filter_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_compound_filter(runner, scenario):
    """Run a compound filter scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
