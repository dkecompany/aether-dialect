"""Live numeric fan-out checks for multiplying joins against parent totals."""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_numeric_parent_sum_equals_unjoined_total_or_refuses() -> None:
    """SUM(parent.amount) with a multiplying join must refuse or match the unjoined parent total."""
    raise NotImplementedError(
        "requires parent/child corpus rows; assert AggregateJoinFanOutError or unjoined SUM(parent.amount) equality"
    )
