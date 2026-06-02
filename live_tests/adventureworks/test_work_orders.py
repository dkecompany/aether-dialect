"""Work order and routing live tests against the local AdventureWorks database."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .aw_scenarios import routing_scenarios, work_order_scenarios

_wo_scenarios = work_order_scenarios()
_rt_scenarios = routing_scenarios()


@pytest.mark.live
@pytest.mark.parametrize("scenario", _wo_scenarios, ids=[s.id for s in _wo_scenarios])
def test_work_orders(runner, scenario):
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")


@pytest.mark.live
@pytest.mark.parametrize("scenario", _rt_scenarios, ids=[s.id for s in _rt_scenarios])
def test_routing(runner, scenario):
    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")
