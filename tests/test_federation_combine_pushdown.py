"""Federated combine must not over-reduce, over-push, or drop ordering on member SQL."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import (
    _collect_member_reducing_edges,
    member_stage_for_source,
    plan_federated_intent,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _left_join_schema() -> SchemaGraph:
    tables = {
        "ta": TableMetadata(
            name="ta",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
        "tb": TableMetadata(
            name="tb",
            columns={
                "a_id": ColumnMetadata(
                    name="a_id",
                    data_type="integer",
                    sensitivity="none",
                    is_unique=True,
                ),
                "status": ColumnMetadata(name="status", data_type="text", sensitivity="none"),
            },
            primary_key=[],
            foreign_keys=[],
            source_id="b",
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _left_join_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_left_combine",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.a_id", "kind": "left", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )


@pytest.mark.fast
def test_left_combine_does_not_reduce_preserved_member() -> None:
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    sources = {"a", "b"}
    source_by_table = dict(manifest.table_namespace)
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    reducing = _collect_member_reducing_edges(
        manifest,
        FederationMappings(version="0.2.3"),
        sources,
        intent,
        source_by_table,
        schema=schema,
    )
    assert not reducing.get("a")
    assert reducing.get("b")


_PRESERVED_MEMBER_ROWS = [{"id": 1}, {"id": 2}, {"id": 3}]


def _member_rows_for_intent(intent: RuntimeIntent) -> list[dict[str, int]]:
    """Simulate member SQL execution: apply any semijoin IN filter on ta.id."""
    rows = list(_PRESERVED_MEMBER_ROWS)
    if intent.where is None:
        return rows
    for fp in intent.where.leaves() or []:
        if fp.op != "in":
            continue
        col = fp.left_expr.column_ref or ""
        if col in {"id", "ta.id"}:
            keys = set((intent.param_values or {}).get(fp.param_key, []))
            rows = [row for row in rows if row["id"] in keys]
    return rows


@pytest.mark.fast
def test_left_combine_semijoin_fallback_skips_preserved_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty reducing_edges must not let the semijoin fallback filter the preserved side."""
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, schema, manifest)
    assert plan.ineligible_reason is None
    preserved_step = next(step for step in plan.steps if step.source_id == "a")
    member_stage = member_stage_for_source(plan, "a")
    assert member_stage is not None
    assert not member_stage.reducing_edges

    member_intent = replace(
        preserved_step.sub_intent,
        tables=["ta"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
    )
    prep = type(
        "Prep",
        (),
        {
            "sub_intent": member_intent,
            "sql": "SELECT id FROM ta",
            "structural_defaults": None,
        },
    )()
    executed = {"b": pd.DataFrame({"a_id": [1]})}

    def _fake_generate_and_validate_sql(_q_norm: str, regen_intent: RuntimeIntent, *_args: object, **_kwargs: object):
        return type("Out", (), {"success": True, "sql": "SELECT id FROM ta"})()

    def _fake_execute_guarded_sql(
        _dialect: object,
        _sql: str,
        _bind: object,
        *,
        intent: RuntimeIntent | None = None,
        **_kwargs: object,
    ) -> list[tuple[int, ...]]:
        assert intent is not None
        return [(row["id"],) for row in _member_rows_for_intent(intent)]

    monkeypatch.setattr("aetherdialect._pipeline_execute.generate_and_validate_sql", _fake_generate_and_validate_sql)
    monkeypatch.setattr("aetherdialect._pipeline_execute.execute_guarded_sql", _fake_execute_guarded_sql)
    monkeypatch.setattr("aetherdialect._federation_execute.validate_federated_sub_intent", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "aetherdialect._federation_execute.validate_sql",
        lambda *a, **k: (True, None, None, None),
    )
    mock_dialect = MagicMock()
    mock_dialect.finalize_render.return_value = "SELECT id FROM ta"

    from aetherdialect._pipeline_execute import _execute_federation_source_step

    frame = _execute_federation_source_step(
        preserved_step,
        prepared_by_source={"a": prep},
        composite_schema=schema,
        dialect_map={},
        dialect=mock_dialect,
        manifest=manifest,
        executed=executed,
        plan=plan,
        semijoin_cap=100,
        q_norm="q",
        join_candidates={},
        cmap={},
        store={},
        gate_kwargs={},
    )
    assert frame is not None
    assert len(frame) == len(_PRESERVED_MEMBER_ROWS)


@pytest.mark.fast
def test_nullable_side_predicate_appears_in_residual_not_member_sql() -> None:
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    nullable_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("tb.status"),
        op="=",
        value_type="string",
        raw_value="open",
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([nullable_filter]),
    )
    plan = plan_federated_intent(intent, schema, manifest)
    assert plan.ineligible_reason is None
    nullable_step = next(step for step in plan.steps if step.source_id == "b")
    assert nullable_step.sub_intent.where is None
    assert plan.residual is not None
    assert plan.residual.where is not None
    residual_leaves = plan.residual.where.leaves() or []
    assert any(fp.raw_value == "open" for fp in residual_leaves)


@pytest.mark.fast
def test_single_member_order_by_moves_to_residual_when_combine_present() -> None:
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("ta.id"))],
        where=None,
    )
    plan = plan_federated_intent(intent, schema, manifest)
    assert plan.ineligible_reason is None
    assert plan.combine
    preserved_step = next(step for step in plan.steps if step.source_id == "a")
    assert not preserved_step.sub_intent.order_by_cols
    assert plan.residual is not None
    assert len(plan.residual.order_by_cols) == 1
    assert plan.residual.order_by_cols[0].expr.column_ref == "ta.id"
