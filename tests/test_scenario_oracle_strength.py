"""Fast checks that high-traffic live scenarios carry exact row oracles."""

from __future__ import annotations

import pytest
from live_tests.mydb_scenarios import (
    CORE_ISOLATED_LIVE_SCENARIO_IDS,
    core_isolated_live_scenarios,
    scenario_has_exact_row_oracle,
)


@pytest.mark.fast
def test_core_isolated_subset_has_exact_oracles() -> None:
    scenarios = core_isolated_live_scenarios()
    assert len(scenarios) == len(CORE_ISOLATED_LIVE_SCENARIO_IDS)
    missing = [s.id for s in scenarios if not scenario_has_exact_row_oracle(s.expected)]
    assert not missing, f"scenarios lacking exact row oracles: {missing}"
