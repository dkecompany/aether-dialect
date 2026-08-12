"""Spanning CTE bodies with unreplayed clauses are refused at plan time."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi


def _members() -> tuple[dict[str, SchemaGraph], object]:
    def _graph(table: str, source_id: str) -> SchemaGraph:
        tables = {
            table: TableMetadata(
                name=table,
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id=source_id,
            )
        }
        return SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=f"sg_{source_id}_{table}",
            effective_structural_hash=f"eff_{source_id}_{table}",
        )

    members = {"a": _graph("ta", "a"), "b": _graph("tb", "b")}
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_span_cte",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={**members["a"].tables, **members["b"].tables},
        join_paths_multi=recompute_join_paths_multi({**members["a"].tables, **members["b"].tables}),
        schema_graph_id="sg_composite",
        effective_structural_hash="eff_composite",
    )
    return members, manifest, composite


@pytest.mark.fast
def test_spanning_cte_with_where_is_ineligible() -> None:
    _, manifest, composite = _members()
    cte = RuntimeCteStep(
        cte_name="span_cte",
        tables=["ta", "tb"],
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("ta.status"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                    raw_value="open",
                ),
            ],
        ),
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
        param_values={"p1": "open"},
    )
    plan = plan_federated_intent(intent, composite, manifest, FederationMappings(version="0.2.3"))
    assert plan.ineligible_reason is not None
    assert "span_cte" in plan.ineligible_reason
    assert "cross-source CTE" in plan.ineligible_reason
