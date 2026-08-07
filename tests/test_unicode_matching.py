"""Tests for Unicode-normalised SQL value matching."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr, WhereParam
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    normalize_text_value,
    normalized_value_overlap_sets,
    value_overlap_ratio_for_columns,
)
from aetherdialect._intent_resolve import _match_frequent_value, _resolve_where_list_cascade

_COMPOSED = "caf\u00e9"
_DECOMPOSED = "caf\u0065\u0301"


@pytest.mark.fast
def test_decomposed_and_composed_forms_match() -> None:
    """NFKC-normalised overlap and literal matching treats composed and decomposed forms as equal."""
    assert normalize_text_value(_DECOMPOSED) == normalize_text_value(_COMPOSED)

    left = ColumnMetadata(
        name="city",
        data_type="varchar",
        value_type="string",
        value_overlap_sample=[_COMPOSED, "paris"],
    )
    right = ColumnMetadata(
        name="city",
        data_type="varchar",
        value_type="string",
        value_overlap_sample=[_DECOMPOSED, "paris"],
    )
    left_set, right_set, _ = normalized_value_overlap_sets(left, right)
    assert normalize_text_value(_COMPOSED) in left_set
    assert normalize_text_value(_COMPOSED) in right_set
    assert left_set & right_set == {normalize_text_value(_COMPOSED), "paris"}
    assert value_overlap_ratio_for_columns(left, right) == 1.0

    col_meta = ColumnMetadata(
        name="city",
        data_type="varchar",
        value_type="string",
        frequent_values=[_COMPOSED],
    )
    assert _match_frequent_value(_DECOMPOSED, col_meta) == _COMPOSED

    film = TableMetadata(
        name="venue",
        columns={"city": col_meta},
        foreign_keys=[],
        primary_key="",
    )
    schema = SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"venue": film},
    )
    fp = WhereParam(
        left_expr=NormalizedExpr.from_column("venue.city"),
        op="=",
        value_type="string",
        raw_value=_DECOMPOSED,
    )
    result, changed = _resolve_where_list_cascade([fp], schema, "")
    assert changed
    assert result[0].raw_value == _COMPOSED
