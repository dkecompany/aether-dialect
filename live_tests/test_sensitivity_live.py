"""Live sensitivity enforcement against bundled rental_shop overrides."""

from __future__ import annotations

import pytest

from aetherdialect._live_testing import run_and_assert

from .mydb_scenarios import sensitivity_enforcement_scenarios

_scenarios = sensitivity_enforcement_scenarios()


@pytest.mark.parametrize("scenario", _scenarios, ids=[s.id for s in _scenarios])
def test_sensitivity_enforcement(runner_enforce_sensitivity, scenario):
    """Run a sensitivity scenario with overrides enforced (no selectability relax)."""
    run_and_assert(
        runner_enforce_sensitivity,
        scenario,
        header=f"[{scenario.id}] {scenario.question}",
    )
