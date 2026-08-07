"""Tests for profiling sample value retention."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._core_utils import (
    _normalize_overlap_sample_value,
    normalized_value_overlap_sets,
)
from aetherdialect._schema_catalog import collect_profiling_frequent_values


@pytest.mark.fast
def test_empty_string_is_a_distinct_value() -> None:
    """Empty strings remain in frequent-value and overlap samples."""
    assert collect_profiling_frequent_values(["", "active", ""]) == ["", "active"]

    assert _normalize_overlap_sample_value("", case_fold=False) == ""
    assert _normalize_overlap_sample_value("", case_fold=True) == ""

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
