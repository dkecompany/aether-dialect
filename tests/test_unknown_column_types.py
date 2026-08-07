"""Unrecognized column types map to unknown and stay out of inference paths."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    COMPOSE_FIELDS,
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE,
    GROUND_FIELDS,
    INTERPRET_FIELDS,
    SCHEMA_FIELD_RAW_TYPE,
)
from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import data_type_to_value_type, refusal_diagnostic_code_for_intent_issue
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._schema_catalog import (
    _profile_column,
    compute_semantic_profile_join_neighbors,
    emit_schema_unknown_type_unusable_warnings,
    llm_classification_column_scope,
)
from aetherdialect._schema_graph import infer_missing_fks
from aetherdialect._validation_execute import validate_semantics

_LLM_SCHEMA_FIELD_SLICES = (INTERPRET_FIELDS, GROUND_FIELDS, COMPOSE_FIELDS)


def _recording_engine() -> tuple[MagicMock, list[str]]:
    executed: list[str] = []
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: Any) -> MagicMock:
        executed.append(str(sql))
        result = MagicMock()
        if "COUNT(DISTINCT" in str(sql):
            result.fetchone.return_value = (10, 5, 0)
        elif "MAX(freq)" in str(sql):
            result.fetchone.return_value = (2,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    return engine, executed


def _sql_references_column(statements: list[str], column: str, dialect: DuckDBDialect) -> list[str]:
    quoted = dialect.quote_identifier(column).lower()
    bare = column.lower()
    hits: list[str] = []
    for stmt in statements:
        low = stmt.lower()
        if quoted in low or f" {bare} " in f" {low} " or f".{bare}" in low:
            hits.append(stmt)
    return hits


def _sites_graph(*, shape: ColumnMetadata | None = None) -> SchemaGraph:
    shape_col = shape or ColumnMetadata(name="shape", data_type="geometry", description="Map outline")
    return SchemaGraph(
        tables={
            "sites": TableMetadata(
                name="sites",
                columns={
                    "site_id": ColumnMetadata(
                        name="site_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "shape": shape_col,
                },
                primary_key=["site_id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="test",
    )


@pytest.mark.fast
def test_unknown_not_string() -> None:
    assert data_type_to_value_type("geometry") == "unknown"
    col = ColumnMetadata(name="shape", data_type="geometry")
    assert col.value_type == "unknown"


@pytest.mark.fast
def test_unknown_hard_unusable_ignores_usable_override() -> None:
    col = ColumnMetadata(
        name="shape",
        data_type="geometry",
        usable_override=True,
        distinct_count=50,
        distinct_ratio=0.5,
        null_ratio=0.0,
    )
    assert col.value_type == "unknown"
    assert col.is_usable is False
    assert col.is_visible is False

    pk_unknown = ColumnMetadata(
        name="geo_id",
        data_type="geometry",
        is_primary_key=True,
        distinct_count=100,
    )
    assert pk_unknown.value_type == "unknown"
    assert pk_unknown.is_usable is False

    graph = _sites_graph(shape=col)
    notified: list[tuple[str, str]] = []

    def capture_notify(message: str, *, code: str = "", **_kwargs: Any) -> None:
        notified.append((message, code))

    with patch("aetherdialect._schema_catalog.notify", side_effect=capture_notify):
        emit_schema_unknown_type_unusable_warnings(graph)

    assert any(code == DIAGNOSTIC_CODE_SCHEMA_UNKNOWN_TYPE_UNUSABLE for _msg, code in notified)
    assert any("sites.shape" in msg for msg, _code in notified)


@pytest.mark.fast
def test_stats_unusable_still_honours_usable_override() -> None:
    col = ColumnMetadata(
        name="status",
        data_type="varchar",
        value_type="string",
        usable_override=True,
        distinct_count=1,
    )
    assert col.is_usable is True
    assert col.is_visible is True


@pytest.mark.fast
def test_unknown_absent_from_classify_and_ask_payload() -> None:
    graph = _sites_graph()
    scope = llm_classification_column_scope(graph)
    assert "shape" not in scope["sites"]
    assert "site_id" in scope["sites"]

    interpret_payload = json.loads(graph.schema_payload_json(INTERPRET_FIELDS, owner_master_scope=False))
    assert "shape" not in interpret_payload.get("sites", {}).get("columns", {})

    for fields in (GROUND_FIELDS, COMPOSE_FIELDS):
        payload = json.loads(graph.schema_payload_json(fields, owner_master_scope=False))
        assert "shape" not in payload["sites"]["columns"]
        assert "site_id" in payload["sites"]["columns"]

    master_payload = json.loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    assert "shape" in master_payload["sites"]["columns"]


@pytest.mark.fast
def test_unknown_column_excluded_from_inference() -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    col = ColumnMetadata(name="shape", data_type="geometry", description="Map outline")

    _profile_column(dialect, engine, col, "sites", row_count=10)

    assert col.profile_skipped_reason == "unknown"
    assert _sql_references_column(executed, "shape", dialect) == []
    assert col.frequent_values == []
    assert col.value_overlap_sample == []

    overlap = ["10", "20", "30", "40", "50"]
    tables = {
        "customer": TableMetadata(
            name="customer",
            columns={
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    value_overlap_sample=overlap,
                )
            },
            primary_key=["customer_id"],
            foreign_keys=[],
        ),
        "orders": TableMetadata(
            name="orders",
            columns={
                "order_id": ColumnMetadata(
                    name="order_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                ),
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="geometry",
                    value_type="unknown",
                    value_overlap_sample=overlap,
                ),
            },
            primary_key=["order_id"],
            foreign_keys=[],
        ),
    }
    assert infer_missing_fks(tables) == []

    graph = SchemaGraph(
        tables={
            "anchors": TableMetadata(
                name="anchors",
                columns={
                    "anchor_id": ColumnMetadata(
                        name="anchor_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                        is_unique=True,
                        value_overlap_sample=["east", "west"],
                    )
                },
                primary_key=["anchor_id"],
                foreign_keys=[],
            ),
            "sites": TableMetadata(
                name="sites",
                columns={
                    "site_id": ColumnMetadata(
                        name="site_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "shape": col,
                },
                primary_key=["site_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="test",
    )
    compute_semantic_profile_join_neighbors(graph)
    assert graph.tables["sites"].columns["shape"].semantic_join_neighbors == []


@pytest.mark.fast
def test_filter_on_unknown_refused() -> None:
    col = ColumnMetadata(
        name="shape",
        data_type="geometry",
        description="Map outline",
    )
    schema = _sites_graph(shape=col)
    intent = RuntimeIntent(
        tables=["sites"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("sites.site_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="and",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("sites.shape"),
                    op="=",
                    value_type="string",
                    raw_value="polygon",
                ),
            ),
        ),
    )
    result = validate_semantics(intent, schema)
    errors = [issue for issue in result.issues if issue.severity == "error"]
    refusal_issues = [issue for issue in errors if issue.issue_id == "unsupported_column_type"]
    assert refusal_issues
    assert (
        refusal_diagnostic_code_for_intent_issue(refusal_issues[0]) == DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE
    )
    assert "Map outline" in refusal_issues[0].message
    assert "cannot be filtered or aggregated" in refusal_issues[0].message


@pytest.mark.fast
def test_no_raw_data_type_in_any_llm_schema_slice() -> None:
    graph = _sites_graph()
    for fields in _LLM_SCHEMA_FIELD_SLICES:
        payload = json.loads(graph.schema_payload_json(fields, owner_master_scope=False))
        for table_body in payload.values():
            for col_body in table_body.get("columns", {}).values():
                assert SCHEMA_FIELD_RAW_TYPE not in col_body
                assert "data_type" not in col_body

    master_payload = json.loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    shape_body = master_payload["sites"]["columns"]["shape"]
    assert shape_body["type"] == "unknown"
    assert SCHEMA_FIELD_RAW_TYPE not in shape_body
