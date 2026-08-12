"""Tests for profiling sample value retention."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._schema_profile import collect_profiling_frequent_values, normalized_value_overlap_sets


@pytest.mark.fast
def test_empty_string_is_a_distinct_value() -> None:
    """Empty strings remain in frequent-value and overlap samples."""
    assert collect_profiling_frequent_values(["", "active", ""]) == ["", "active"]

    left = ColumnMetadata(name="status", data_type="text", value_overlap_sample=["", "active"])
    right = ColumnMetadata(name="status", data_type="text", value_overlap_sample=["", "active"])
    left_set, right_set, _ = normalized_value_overlap_sets(left, right)
    assert "" in left_set
    assert left_set == right_set

    left = ColumnMetadata(
        name="status",
        data_type="varchar",
        value_type="string",
        value_overlap_sample=["", "active"],
    )
    right = ColumnMetadata(
        name="status",
        data_type="varchar",
        value_type="string",
        value_overlap_sample=["", "inactive"],
    )
    left_set, right_set, _ = normalized_value_overlap_sets(left, right)
    assert "" in left_set
    assert "" in right_set
    assert left_set & right_set == {""}
