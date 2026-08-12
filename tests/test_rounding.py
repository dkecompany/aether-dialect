"""Federated rounding is applied once at the coordinator with stated per-dialect modes."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED
from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import (
    emit_federation_rounding_mode_mixed_diagnostics,
    plan_federated_intent,
    render_federation_residual_sql,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql
from tests.federation_helpers import enriched_manifest


def _amount_graph(table: str, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none", is_primary_key=True),
                "amt": ColumnMetadata(name="amt", data_type="decimal(12,2)", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


_MANIFEST = {
    "federation_id": "fed_round",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _round_sum_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="scalar",
        select_cols=[
            SelectCol(
                expr=NormalizedExpr(
                    add_groups=[
                        MulGroup(
                            coefficient=1.0,
                            multiply=["left_t.amt"],
                            agg_func="sum",
                            scalar_func="round",
                            scalar_func_args=[2],
                        )
                    ],
                ),
            ),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_rounding_happens_once_in_federated_query() -> None:
    member_graphs = {
        "a": _amount_graph("left_t", "a"),
        "b": _amount_graph("right_t", "b"),
    }
    manifest = enriched_manifest(member_graphs, _MANIFEST, member_graphs=member_graphs)
    composite = compose_composite_graph(member_graphs, manifest)
    intent = _round_sum_intent()
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert plan.residual is not None

    dialect = DialectRegistry.get_class("duckdb").__new__(DialectRegistry.get_class("duckdb"))
    member_sql = [build_deterministic_sql(step.sub_intent, schema=composite, dialect=dialect) for step in plan.steps]
    assert member_sql
    for sql in member_sql:
        assert "ROUND(" not in sql.upper()
    assert any("SUM(" in sql.upper() for sql in member_sql)

    residual_sql = render_federation_residual_sql("SELECT 1 AS fed", plan.residual, schema=composite)
    assert "ROUND(" in residual_sql.upper()
    assert all("ROUND(" not in sql.upper() for sql in member_sql)

    assert DialectRegistry.engine_rounding_mode("duckdb") == "half_up"
    assert DialectRegistry.engine_rounding_mode("sqlite") == "half_even"


@pytest.mark.fast
def test_mixed_rounding_modes_emit_diagnostic() -> None:
    mixed_manifest = {
        **_MANIFEST,
        "sources": [
            {"source_id": "a", "engine": "mysql", "role": "owner"},
            {"source_id": "b", "engine": "sqlite", "role": "owner"},
        ],
    }
    member_graphs = {
        "a": _amount_graph("left_t", "a"),
        "b": _amount_graph("right_t", "b"),
    }
    manifest = parse_federation_manifest(mixed_manifest, include_derived_roster=True)
    composite = compose_composite_graph(member_graphs, manifest)
    intent = _round_sum_intent()
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None

    with patch("aetherdialect._federation_plan.notify") as notify_mock:
        emit_federation_rounding_mode_mixed_diagnostics(manifest, plan, intent, schema=composite)

    codes = [call.kwargs.get("code") for call in notify_mock.call_args_list]
    assert DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED in codes
    mismatch_call = next(
        call for call in notify_mock.call_args_list if call.kwargs.get("code") == DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED
    )
    details = dict(mismatch_call.kwargs.get("details") or ())
    assert details.get("logical_column") == "amt"
