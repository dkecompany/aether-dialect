"""Aggregation + filter + join compound live pipeline tests."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import agg_filter_join_scenarios

_scenarios = agg_filter_join_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_agg_filter_join(runner, scenario):
    """Run an aggregation+filter+join scenario and assert expectations."""
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
