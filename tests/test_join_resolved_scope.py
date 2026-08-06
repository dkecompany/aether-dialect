"""Resolved join scope includes bridge tables for validation but not clause allowance."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FailureCategory, JoinPathKeyTypeError, NoJoinPathError, WhereParam
from aetherdialect._contracts_core import (
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import join_resolved_scope_tables
from aetherdialect._pipeline import _resolve_joins_fresh, generate_and_validate_sql
from aetherdialect._schema_graph import assert_intent_in_scope
from aetherdialect._validation_execute import validate_semantics
from aetherdialect._validation_schema import (
    validate_filters_schema,
    validate_join_path_key_types,
    validate_join_path_reachability,
    validate_select_cols_schema,
)


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _bridge_schema() -> SchemaGraph:
    bridge_edge = {
        "src_table": "a",
        "src_cols": ["id"],
        "dst_table": "bridge",
        "dst_cols": ["aid"],
    }
    second = {
        "src_table": "bridge",
        "src_cols": ["cid"],
        "dst_table": "c",
        "dst_cols": ["id"],
    }
    path = [bridge_edge, second]
    return SchemaGraph(
        tables={
            "a": TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "c": TableMetadata(name="c", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
            "bridge": TableMetadata(
                name="bridge",
                columns={"aid": _col("aid"), "cid": _col("cid")},
                primary_key=["aid"],
                foreign_keys=[],
            ),
            "island": TableMetadata(name="island", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[]),
        },
        join_paths_multi={"a": {"c": [path]}},
        effective_structural_hash="h",
    )


def test_bridge_join_path_expands_resolved_scope() -> None:
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    assert join_resolved_scope_tables(sig, ["a", "c"]) == ["a", "bridge", "c"]


def test_resolved_scope_includes_bridge_after_join_resolution() -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    join_candidates = {
        "candidates": [
            {
                "candidate_id": "J01",
                "join_path_signature": sig,
                "edge_kinds": ["catalog_fk", "catalog_fk"],
                "edge_count": 2,
            }
        ]
    }
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    det_sql = "SELECT a.id FROM a, c"
    _resolve_joins_fresh(
        det_sql,
        intent,
        {},
        None,
        "list ids",
        join_candidates,
        schema=schema,
        join_preset_scope={"main": "J01"},
    )
    assert "bridge" in intent.resolved_join_tables
    assert set(intent.tables) == {"a", "c"}


def test_unreachable_bridge_in_resolved_scope_fails_reachability_before_render() -> None:
    schema = _bridge_schema()
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        resolved_join_tables=["a", "bridge", "c", "island"],
    )
    issues = validate_join_path_reachability(intent, schema, "main query")
    assert any(i.severity == "error" and "island" in i.message for i in issues)


def test_resolve_joins_raises_when_resolved_scope_is_unreachable() -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id", "island.id->bridge.cid"]
    join_candidates = {
        "candidates": [
            {
                "candidate_id": "J01",
                "join_path_signature": sig,
                "edge_kinds": ["catalog_fk", "catalog_fk", "catalog_fk"],
                "edge_count": 3,
            }
        ]
    }
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(NoJoinPathError):
        _resolve_joins_fresh(
            "SELECT a.id FROM a, c",
            intent,
            {},
            None,
            "list ids",
            join_candidates,
            schema=schema,
            join_preset_scope={"main": "J01"},
        )


def test_select_on_bridge_table_still_rejected_when_not_in_intent_tables() -> None:
    schema = _bridge_schema()
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("bridge.aid"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        resolved_join_tables=["a", "bridge", "c"],
    )
    allowed = set(intent.tables)
    issues = validate_select_cols_schema(intent.select_cols, schema, allowed, context="main query")
    assert any("not in allowed" in i.message for i in issues)


def test_where_on_bridge_table_still_rejected_when_not_in_intent_tables() -> None:
    schema = _bridge_schema()
    fp = WhereParam(
        left_expr=NormalizedExpr.from_column("bridge.aid"),
        op="=",
        value_type="string",
        raw_value="1",
    )
    issues = validate_filters_schema([fp], schema, {"a", "c"}, context="main query")
    assert any("not in allowed" in i.message for i in issues)


def test_bridge_table_in_resolved_scope_fails_aetherspace_check() -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=sig,
        resolved_join_tables=["a", "bridge", "c"],
    )
    assert not assert_intent_in_scope(intent, frozenset({"a", "c"}), frozenset(), schema)


def test_validate_semantics_unreachable_resolved_join_scope() -> None:
    schema = _bridge_schema()
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        resolved_join_tables=["a", "bridge", "c", "island"],
    )
    result = validate_semantics(intent, schema)
    errors = [i for i in result.issues if i.severity == "error"]
    assert any("island" in i.message for i in errors)


def test_validate_semantics_reachability_expands_join_signature_without_resolved_tables() -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id", "island.id->bridge.cid"]
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=sig,
    )
    result = validate_semantics(intent, schema)
    errors = [i for i in result.issues if i.severity == "error"]
    assert any("island" in i.message for i in errors)


def _typed_bridge_schema() -> SchemaGraph:
    bridge_edge = {
        "src_table": "a",
        "src_cols": ["id"],
        "dst_table": "bridge",
        "dst_cols": ["aid"],
    }
    second = {
        "src_table": "bridge",
        "src_cols": ["cid"],
        "dst_table": "c",
        "dst_cols": ["id"],
    }
    path = [bridge_edge, second]
    return SchemaGraph(
        tables={
            "a": TableMetadata(
                name="a",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "c": TableMetadata(
                name="c",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "bridge": TableMetadata(
                name="bridge",
                columns={
                    "aid": ColumnMetadata(name="aid", data_type="text", sensitivity="none"),
                    "cid": ColumnMetadata(name="cid", data_type="integer", sensitivity="none"),
                },
                primary_key=["aid"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={"a": {"c": [path]}},
        effective_structural_hash="typed_bridge",
    )


def test_join_path_key_types_refuses_incompatible_bridge_columns() -> None:
    schema = _typed_bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    issues = validate_join_path_key_types(sig, schema, "main query")
    assert any(i.severity == "error" and "type-compatible" in i.message for i in issues)


def test_resolve_joins_raises_on_incompatible_bridge_join_key_types() -> None:
    schema = _typed_bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]
    join_candidates = {
        "candidates": [
            {
                "candidate_id": "J01",
                "join_path_signature": sig,
                "edge_kinds": ["catalog_fk", "catalog_fk"],
                "edge_count": 2,
            }
        ]
    }
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(JoinPathKeyTypeError):
        _resolve_joins_fresh(
            "SELECT a.id FROM a, c",
            intent,
            {},
            None,
            "list ids",
            join_candidates,
            schema=schema,
            join_preset_scope={"main": "J01"},
        )


@patch("aetherdialect._pipeline._run_sql_validation_cascade", return_value=(True, None, None, []))
@patch(
    "aetherdialect._pipeline.finalize_substitute_sql",
    return_value="SELECT a.id FROM a JOIN bridge ON a.id = bridge.aid JOIN c ON bridge.cid = c.id",
)
@patch("aetherdialect._pipeline.build_deterministic_sql", return_value="SELECT a.id FROM a, c")
@patch(
    "aetherdialect._pipeline._join_signatures_for_deterministic_from_anchor",
    return_value=([], {}),
)
def test_generate_and_validate_sql_refuses_denied_bridge_post_resolution(
    _mock_anchor: MagicMock,
    _mock_build: MagicMock,
    _mock_finalize: MagicMock,
    _mock_validate: MagicMock,
) -> None:
    schema = _bridge_schema()
    sig = ["a.id->bridge.aid", "bridge.cid->c.id"]

    def _fake_resolve(
        deterministic_sql: str,
        intent: RuntimeIntent,
        cmap: dict[str, list[str]],
        cte_join_hints: dict[str, dict[str, object]] | None,
        q_norm: str,
        join_candidates: dict[str, object],
        **kwargs: object,
    ) -> tuple[str, dict[str, str]]:
        intent.chosen_join_candidate_id = "J01"
        intent.chosen_join_path_signature = sig
        intent.resolved_join_tables = ["a", "bridge", "c"]
        return deterministic_sql, {}

    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    join_candidates = {
        "candidates": [
            {
                "candidate_id": "J01",
                "join_path_signature": sig,
                "edge_kinds": ["catalog_fk", "catalog_fk"],
                "edge_count": 2,
            }
        ]
    }
    with patch("aetherdialect._pipeline._resolve_joins_fresh", side_effect=_fake_resolve):
        out = generate_and_validate_sql(
            "list a ids via c",
            intent,
            schema,
            join_candidates,
            {"J01": sig},
            MagicMock(),
            {},
            cte_join_hints=None,
            space_allowed_tables=frozenset({"a", "c"}),
            join_preset_scope={"main": "J01"},
            persist_template_learning=False,
        )
    assert out.success is False
    assert out.error_kind == FailureCategory.DENIED_REFERENCE.value
    assert "aetherspace" in (out.sql_validation_error or "").lower()
