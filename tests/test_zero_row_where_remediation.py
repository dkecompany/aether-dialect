"""Unit tests for zero-row filter literal remediation helpers."""

from __future__ import annotations

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._utils_intent import (
    morph_variants,
    patch_where_literal_on_intent,
    zero_row_where_remediation_candidates,
    zero_row_where_suggestions,
)


def _schema_with_feature_values(*values: str) -> SchemaGraph:
    column = ColumnMetadata(
        name="feature_name",
        data_type="varchar",
        frequent_values=list(values),
    )
    table = TableMetadata(
        name="item_feature",
        columns={"feature_name": column},
        primary_key=[],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"item_feature": table},
        join_paths_multi={},
        effective_structural_hash="test",
    )


def test_morph_variants_trailer_includes_trailers() -> None:
    assert "trailers" in morph_variants("trailer")


def test_remediation_candidates_deleted_scenes_prefers_underscore_join() -> None:
    cached = ["trailers", "deleted_scenes", "commentaries"]
    candidates = zero_row_where_remediation_candidates("deleted scenes", cached)
    assert candidates[0] == "deleted_scenes"


def test_remediation_candidates_trailer_maps_to_trailers() -> None:
    cached = ["trailers", "deleted_scenes"]
    candidates = zero_row_where_remediation_candidates("trailer", cached)
    assert candidates == ["trailers"]


def test_patch_filter_literal_updates_param_values() -> None:
    where_param = WhereParam(
        left_expr=NormalizedExpr(column_ref="item_feature.feature_name"),
        op="=",
        param_key="p0",
        raw_value="trailer",
    )
    intent = RuntimeIntent(
        tables=["item_feature"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([where_param]),
        param_values={"p0": "trailer"},
    )
    patched = patch_where_literal_on_intent(intent, where_param, "trailers")
    assert patched.param_values["p0"] == "trailers"
    assert (patched.where.leaves() if patched.where else [])[0].raw_value == "trailers"


def test_zero_row_where_suggestions_uses_levenshtein_distance() -> None:
    where_param = WhereParam(
        left_expr=NormalizedExpr(column_ref="item_feature.feature_name"),
        op="=",
        param_key="p0",
        raw_value="traler",
    )
    intent = RuntimeIntent(
        tables=["item_feature"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([where_param]),
        param_values={"p0": "traler"},
    )
    suggestions = zero_row_where_suggestions(intent, _schema_with_feature_values("trailers"))
    assert suggestions == ["No rows for feature_name='traler'. Did you mean 'trailers'?"]


def test_zero_row_where_suggestions_skips_distant_values() -> None:
    where_param = WhereParam(
        left_expr=NormalizedExpr(column_ref="item_feature.feature_name"),
        op="=",
        param_key="p0",
        raw_value="completely unrelated",
    )
    intent = RuntimeIntent(
        tables=["item_feature"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([where_param]),
        param_values={"p0": "completely unrelated"},
    )
    suggestions = zero_row_where_suggestions(intent, _schema_with_feature_values("trailers"))
    assert suggestions == []
