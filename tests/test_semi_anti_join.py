"""Tests for semi-join and anti-join CTE emission rendering and validation."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from aetherdialect._constants import anti_join_presence_column
from aetherdialect._contracts_base import ProbeCtePlacementError, coerce_cte_emission
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import _resolve_joins_fresh
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    _join_kind_for_edge,
    build_deterministic_sql,
    inject_join_into_deterministic_sql,
)
from aetherdialect._validation_execute import (
    enforce_probe_cte_anchor_placement_post_resolution,
    validate_cte_emission_reclassification,
)
from aetherdialect._validation_schema import (
    validate_cte_emission_shapes,
    validate_probe_cte_anchor_placement,
)
from tests.test_sql_gen import _pg_render


def _parent_child_schema() -> SchemaGraph:
    parent_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
    }
    child_cols = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
        "parent_id": ColumnMetadata(
            name="parent_id",
            data_type="integer",
            sensitivity="none",
            is_foreign_key=True,
            fk_target=("parent", "id"),
        ),
        "status": ColumnMetadata(name="status", data_type="text", sensitivity="none"),
    }
    fk = FKEdge(
        src_table="child",
        src_cols=["parent_id"],
        dst_table="parent",
        dst_cols=["id"],
    )
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns=parent_cols,
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns=child_cols,
            primary_key=["id"],
            foreign_keys=[fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="semi_anti_test",
    )


def _forbidden_tokens(sql: str) -> list[str]:
    upper = sql.upper()
    hits: list[str] = []
    if re.search(r"\bEXISTS\s*\(", upper):
        hits.append("EXISTS")
    if re.search(r"\bEXCEPT\b", upper):
        hits.append("EXCEPT")
    if re.search(r"\bLATERAL\b", upper):
        hits.append("LATERAL")
    if re.search(r"\bDISTINCT\s+ON\b", upper):
        hits.append("DISTINCT ON")
    return hits


def _assert_no_legacy_semi_anti_tokens(sql: str) -> None:
    assert _forbidden_tokens(sql) == [], sql


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("join_table", "join_table"),
        ("scalar_subquery", "join_table"),
        ("semi_join", "semi_join"),
        ("anti_join", "anti_join"),
        ("unknown", "join_table"),
    ],
)
def test_coerce_cte_emission(raw: str, expected: str) -> None:
    assert coerce_cte_emission(raw) == expected


def test_join_kind_for_edge_forces_probe_kinds() -> None:
    assert _join_kind_for_edge("probe", "driver", ["id"], None, right_emission="anti_join").strip() == "LEFT"
    assert _join_kind_for_edge("probe", "driver", ["id"], None, right_emission="semi_join").strip() == "INNER"


def test_semi_join_renders_distinct_without_exists() -> None:
    schema = _parent_child_schema()
    semi = RuntimeCteStep(
        cte_name="active_parents",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "active_parents"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )
    det = build_deterministic_sql(intent, None, schema, _pg_render())
    joined = inject_join_into_deterministic_sql(
        det,
        [[], ["parent.id->active_parents.parent_id"]],
        schema=schema,
        edge_kinds_ordered=[[], ["catalog_fk"]],
        dialect=_pg_render(),
        cte_emissions={"active_parents": "semi_join"},
    )
    low = joined.lower()
    assert "select distinct" in low
    assert "inner join" in low
    _assert_no_legacy_semi_anti_tokens(det)
    _assert_no_legacy_semi_anti_tokens(joined)


def test_anti_join_renders_left_join_and_presence_null_without_except() -> None:
    schema = _parent_child_schema()
    presence = anti_join_presence_column("has_child")
    anti = RuntimeCteStep(
        cte_name="has_child",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "has_child"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    det = build_deterministic_sql(intent, None, schema, _pg_render())
    assert presence in det
    assert "1 AS" in det
    assert "IS NULL" in det.upper()
    joined = inject_join_into_deterministic_sql(
        det,
        [[], ["parent.id->has_child.parent_id"]],
        schema=schema,
        edge_kinds_ordered=[[], ["catalog_fk"]],
        dialect=_pg_render(),
        cte_emissions={"has_child": "anti_join"},
    )
    low = joined.lower()
    assert "left join" in low
    assert presence.lower() in low
    assert "is null" in low
    _assert_no_legacy_semi_anti_tokens(det)
    _assert_no_legacy_semi_anti_tokens(joined)


def test_semi_join_rejects_payload_column() -> None:
    schema = _parent_child_schema()
    semi = RuntimeCteStep(
        cte_name="bad_semi",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.status"))],
        output_columns=["status"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "bad_semi"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any(i.issue_id == "semi_join_projection_shape_bad_semi" for i in issues)


def test_semi_join_accepts_key_shape_and_intersection_shape() -> None:
    schema = _parent_child_schema()
    key_semi = RuntimeCteStep(
        cte_name="key_probe",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    key_intent = RuntimeIntent(
        tables=["parent", "key_probe"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[key_semi],
    )
    assert not validate_cte_emission_shapes(key_intent, schema)

    intersection_semi = RuntimeCteStep(
        cte_name="tuple_probe",
        emission="semi_join",
        tables=["child"],
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("child.parent_id")),
            SelectCol(expr=NormalizedExpr.from_column("child.status")),
        ],
        output_columns=["parent_id", "status"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intersection_intent = RuntimeIntent(
        tables=["parent", "tuple_probe"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr.from_column("parent.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[intersection_semi],
    )
    assert not validate_cte_emission_shapes(intersection_intent, schema)


def test_set_difference_rejects_matching_arity_with_mismatched_types() -> None:
    schema = _parent_child_schema()
    anti = RuntimeCteStep(
        cte_name="other_set",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.status"))],
        output_columns=["status"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "other_set"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any(i.issue_id == "set_difference_type_other_set_0" for i in issues)


def test_set_difference_warns_when_probe_type_is_unresolvable() -> None:
    schema = _parent_child_schema()
    anti = RuntimeCteStep(
        cte_name="other_set",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "other_set"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.unknown_col"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any(i.issue_id == "set_difference_type_unresolved_other_set" for i in issues)


def test_set_difference_arity_mismatch_rejected() -> None:
    schema = _parent_child_schema()
    anti = RuntimeCteStep(
        cte_name="other_set",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "other_set"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr.from_column("parent.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any("set difference" in i.message.lower() or "arity" in i.message.lower() for i in issues)


def test_semi_join_does_not_multiply_outer_rows() -> None:
    """Semi-join inner join to distinct keys preserves one outer row per driver key."""
    duckdb = pytest.importorskip("duckdb")
    schema = _parent_child_schema()
    semi = RuntimeCteStep(
        cte_name="active_parents",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "active_parents"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )
    det = build_deterministic_sql(intent, None, schema, _pg_render())
    joined = inject_join_into_deterministic_sql(
        det,
        [[], ["parent.id->active_parents.parent_id"]],
        schema=schema,
        edge_kinds_ordered=[[], ["catalog_fk"]],
        dialect=_pg_render(),
        cte_emissions={"active_parents": "semi_join"},
    )
    conn = duckdb.connect()
    conn.execute("CREATE TABLE parent AS SELECT * FROM (VALUES (1), (2), (3)) AS t(id)")
    conn.execute(
        "CREATE TABLE child AS SELECT * FROM (VALUES (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)) AS t(id, parent_id)"
    )
    rows = conn.execute(joined).fetchall()
    assert len(rows) == 2
    assert sorted(row[0] for row in rows) == [1, 2]


def test_set_difference_forces_distinct_outer_projection() -> None:
    """Set-difference anti-join compiles the outer block with SELECT DISTINCT."""
    schema = _parent_child_schema()
    anti = RuntimeCteStep(
        cte_name="other_set",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "other_set"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    sql = build_deterministic_sql(intent, None, schema, _pg_render())
    assert "SELECT DISTINCT" in sql.upper()
    _assert_no_legacy_semi_anti_tokens(sql)


def test_probe_cte_cannot_be_main_anchor() -> None:
    anti = RuntimeCteStep(
        cte_name="probe",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["probe", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
    )
    issues = validate_probe_cte_anchor_placement(intent)
    assert any("anchor" in i.message.lower() for i in issues)


def test_probe_cte_left_operand_in_cte_scope_missed_before_join_resolution() -> None:
    probe = RuntimeCteStep(
        cte_name="probe",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    wrapper = RuntimeCteStep(
        cte_name="wrapper",
        tables=["parent", "probe"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        chosen_join_path_signature=[],
    )
    intent = RuntimeIntent(
        tables=["parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[probe, wrapper],
    )
    pre_issues = validate_probe_cte_anchor_placement(intent)
    assert not any(i.issue_id == "probe_cte_left_operand_probe" for i in pre_issues)


def test_probe_cte_left_operand_in_cte_scope_enforced_after_join_resolution() -> None:
    probe = RuntimeCteStep(
        cte_name="probe",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    wrapper = RuntimeCteStep(
        cte_name="wrapper",
        tables=["parent", "probe"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        chosen_join_path_signature=["probe.parent_id->parent.id"],
    )
    intent = RuntimeIntent(
        tables=["parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[probe, wrapper],
    )
    with pytest.raises(ProbeCtePlacementError, match="left operand"):
        enforce_probe_cte_anchor_placement_post_resolution(intent)


def test_resolve_joins_fresh_enforces_probe_anchor_in_cte_scope() -> None:
    probe = RuntimeCteStep(
        cte_name="probe",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    wrapper = RuntimeCteStep(
        cte_name="wrapper",
        tables=["probe", "parent"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[probe, wrapper],
    )
    schema = _parent_child_schema()
    join_candidates = {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]}
    cte_join_hints = {
        "wrapper": {
            "candidates": [
                {
                    "candidate_id": "J01",
                    "join_path_signature": ["parent.id->probe.parent_id"],
                    "edge_kinds": [],
                }
            ]
        }
    }
    cmap = {"J00": [], "J01": ["parent.id->probe.parent_id"]}
    det = "WITH probe AS (SELECT child.parent_id FROM child), wrapper AS (SELECT parent.id FROM probe, parent) SELECT parent.id FROM parent"

    with patch(
        "aetherdialect._pipeline.inject_join_into_deterministic_sql",
        side_effect=lambda det_sql, *_args, **_kwargs: det_sql,
    ):
        with pytest.raises(ProbeCtePlacementError, match="left operand"):
            _resolve_joins_fresh(
                det,
                intent,
                cmap,
                cte_join_hints,
                "q",
                join_candidates,
                schema=schema,
                dialect=PostgresDialect.__new__(PostgresDialect),
            )


def test_probe_cte_left_operand_emits_intent_issue_not_join_error() -> None:
    anti = RuntimeCteStep(
        cte_name="probe",
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "probe"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
        chosen_join_path_signature=["probe.parent_id->parent.id"],
    )
    issues = validate_probe_cte_anchor_placement(
        intent,
        join_signature=list(intent.chosen_join_path_signature or []),
    )
    assert any(i.issue_id == "probe_cte_left_operand_probe" for i in issues)
    assert all(i.category.name == "CTE_STRUCTURE" for i in issues if "left_operand" in i.issue_id)


def test_model_declared_scalar_subquery_is_forbidden() -> None:
    cte = RuntimeCteStep(
        cte_name="avg_cte",
        tables=["parent"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "parent.id"))],
        output_columns=["avg_id"],
        emission="scalar_subquery",
        grain="scalar",
    )
    intent = RuntimeIntent(
        tables=["avg_cte"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("avg_cte.avg_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    issues = validate_cte_emission_reclassification(intent, _parent_child_schema())
    assert any(i.issue_id == "cte_emission_model_declared_scalar_subquery_avg_cte" for i in issues)
    assert all(i.severity == "error" for i in issues if "model_declared_scalar_subquery" in i.issue_id)


def test_scalar_subquery_on_non_scalar_cte_emits_reclassification_error() -> None:
    cte = RuntimeCteStep(
        cte_name="wide_cte",
        tables=["parent", "child"],
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr.from_column("child.id")),
        ],
        output_columns=["parent_id", "child_id"],
        emission="scalar_subquery",
        grain="row_level",
    )
    intent = RuntimeIntent(
        tables=["parent"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    issues = validate_cte_emission_reclassification(intent, _parent_child_schema())
    assert any(i.issue_id == "cte_emission_model_declared_scalar_subquery_wide_cte" for i in issues)
    assert all(i.severity == "error" for i in issues)


def test_wrongly_declared_semi_join_reclassified_to_join_table() -> None:
    """A payload column on a declared semi_join is reclassified to join_table."""
    semi = RuntimeCteStep(
        cte_name="bad_semi",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.status"))],
        output_columns=["status"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "bad_semi"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )
    issues = validate_cte_emission_reclassification(intent, _parent_child_schema())
    assert any(i.issue_id == "cte_emission_reclassified_bad_semi" for i in issues)
    assert any(i.severity == "error" and "semi_join" in i.message and "join_table" in i.message for i in issues)
