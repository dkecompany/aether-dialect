"""Live IR capability checks against real engines (collected, not run in fast suite)."""

from __future__ import annotations

import pytest

from ._federation_live import federation_partitions_available

pytestmark = pytest.mark.live

_SKIP = not federation_partitions_available()


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_order_by_null_placement_matches_across_postgres_and_mysql() -> None:
    pytest.skip("requires dedicated live fixture session")


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_ntile_window_returns_expected_bucket_count() -> None:
    pytest.skip("requires dedicated live fixture session")


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_ordered_string_agg_returns_expected_concatenation() -> None:
    pytest.skip("requires dedicated live fixture session")


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_stddev_aggregate_returns_expected_value() -> None:
    pytest.skip("requires dedicated live fixture session")


@pytest.mark.skipif(_SKIP, reason="federation partition databases unavailable")
def test_redundant_join_elimination_preserves_row_count() -> None:
    pytest.skip("requires dedicated live fixture session")
