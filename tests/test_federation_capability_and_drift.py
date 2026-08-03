"""Federation capability intersection and declared-mapping drift wiring."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import WhereParam, FederationConfigError, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    intersect_member_dialect_capabilities,
    parse_federation_manifest,
    parse_federation_mappings,
    plan_federated_intent,
    rescore_declared_mapping_drift,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(name: str, *, source_id: str, overlap: list[str] | None = None) -> SchemaGraph:
    columns = {
        "id": ColumnMetadata(
            name="id",
            data_type="integer",
            sensitivity="none",
            value_overlap_sample=list(overlap or []),
        ),
    }
    table = TableMetadata(
        name=name,
        columns=columns,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}",
        effective_structural_hash=f"eff_{source_id}",
        profiling_hash=f"profile_{source_id}",
    )


@pytest.mark.fast
def test_intersect_member_dialect_capabilities_includes_ilike_when_wrap_available() -> None:
    caps = intersect_member_dialect_capabilities(
        engine_types_by_source={"a": "postgresql", "b": "bigquery"},
    )
    assert "ilike" in caps["where_ops"]
    assert "not ilike" in caps["where_ops"]
    assert "=" in caps["where_ops"]
    assert "=" in caps["having_ops"]


@pytest.mark.fast
def test_plan_federated_intent_allows_ilike_with_mixed_member_capabilities() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_caps",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "bigquery", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    schema = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"name": ColumnMetadata(name="name", data_type="text", sensitivity="none")},
                primary_key=["name"],
                foreign_keys=[],
                source_id="a",
            ),
        },
        join_paths_multi=recompute_join_paths_multi({}),
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.name"),
                    op="ilike",
                    value_type="string",
                    raw_value="%x%",
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, schema, manifest)
    assert plan.ineligible_reason is None


@pytest.mark.fast
def test_federation_member_lacking_ilike_semantics_names_blocked_member() -> None:
    from unittest.mock import MagicMock

    from aetherdialect._federation import _federation_member_lacking_ilike_semantics

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_no_ilike",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "bigquery", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    blocked = MagicMock()
    blocked.supports_ilike = False
    blocked.supports_case_insensitive_wrap = False
    ok = MagicMock()
    ok.supports_ilike = True
    ok.supports_case_insensitive_wrap = True
    lacking = _federation_member_lacking_ilike_semantics(
        manifest,
        dialects_by_source={"a": ok, "b": blocked},
    )
    assert lacking == "b"


@pytest.mark.fast
def test_federation_refuses_ilike_when_member_lacks_native_and_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    from aetherdialect._federation import _federation_unsupported_operator_reason

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_no_ilike",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "bigquery", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )

    def _member_semantics(engine_type: str) -> bool:
        return engine_type != "bigquery"

    monkeypatch.setattr("aetherdialect._federation.member_supports_ilike_semantics", _member_semantics)
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.name"),
                    op="ilike",
                    value_type="string",
                    raw_value="%x%",
                ),
            ]
        ),
    )
    reason = _federation_unsupported_operator_reason(intent, manifest)
    assert reason is not None
    assert "'b'" in reason or "bigquery" in reason
    assert "ilike" in reason


@pytest.mark.fast
def test_ilike_renders_native_on_postgresql_and_wrap_on_bigquery() -> None:
    from aetherdialect._contracts_core import WhereParam
    from aetherdialect._dialect import get_dialect_class
    from aetherdialect._sql_gen import _render_predicate_clause

    pred = WhereParam(
        left_expr=NormalizedExpr.from_column("left_t.name"),
        op="ilike",
        value_type="string",
        param_key="pat",
    )
    pg = object.__new__(get_dialect_class("postgresql"))
    bq = object.__new__(get_dialect_class("bigquery"))
    pg_sql = _render_predicate_clause(pred, pg)
    bq_sql = _render_predicate_clause(pred, bq)
    assert "ILIKE" in pg_sql.upper()
    assert "LOWER" not in pg_sql.upper()
    assert "ILIKE" not in bq_sql.upper()
    assert "LOWER" in bq_sql.upper()
    assert "LIKE" in bq_sql.upper()
    assert "LOWER(:pat)" in bq_sql.replace(" ", "")


@pytest.mark.fast
def test_compose_caches_dialect_capability_intersection() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_caps_compose",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "bigquery", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": _graph("ta", source_id="a"),
        "b": _graph("tb", source_id="b"),
    }
    composite = compose_composite_graph(members, manifest)
    cached = getattr(composite, "_dialect_capability_cache", None)
    assert cached is not None
    assert "ilike" in cached["where_ops"]


@pytest.mark.fast
def test_rescore_declared_mapping_drift_reports_value_overlap_collapse() -> None:
    members = {
        "a": _graph("entity_a", source_id="a", overlap=["1", "2", "3"]),
        "b": _graph("entity_b", source_id="b", overlap=["9", "8", "7"]),
    }
    mappings = parse_federation_mappings(
        {
            "version": 2,
            "logical_columns": [
                {
                    "logical": "entity_id",
                    "unify_in_graph": True,
                    "members": ["entity_a.id", "entity_b.id"],
                },
            ],
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "entity_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    drift = rescore_declared_mapping_drift(mappings, members)
    assert any("value overlap rescoring drift" in msg for msg in drift)


@pytest.mark.fast
def test_compose_invokes_declared_mapping_drift_rescoring() -> None:
    from unittest.mock import patch

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_drift",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": _graph("ta", source_id="a"),
        "b": _graph("tb", source_id="b"),
    }
    with patch(
        "aetherdialect._federation.rescore_declared_mapping_drift",
        return_value=("declared drift signal",),
    ) as mock_rescore:
        with pytest.raises(FederationConfigError, match="declared mapping drift"):
            compose_composite_graph(members, manifest)
    mock_rescore.assert_called_once()
