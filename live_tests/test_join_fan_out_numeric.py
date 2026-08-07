"""Live numeric fan-out regression — deferred until real parent/child data is wired."""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_numeric_parent_sum_equals_unjoined_total_or_refuses() -> None:
    """SUM(parent.amount) with a multiplying join must refuse or match the unjoined parent total."""
    raise NotImplementedError(
        "numeric fan-out regression executes after the bundled corpus is available; "
        "assert AggregateJoinFanOutError or unjoined SUM(parent.amount) equality"
    )
