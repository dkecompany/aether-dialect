"""Scalar-grain federation members contribute no coordinator frame."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._contracts_base import FederationDeclarationError, NormalizedExpr
from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SelectCol, SourceStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import (
    plan_federated_intent,
    validate_federation_scalar_grain_member_frames,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, *, source_id: str) -> SchemaGraph:
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


_MANIFEST = {
    "federation_id": "fed_scalar_grain_l28",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_scalar_grain_member_refuses_multi_member_plan_at_plan_time(monkeypatch) -> None:
    """Scalar-grain member required for combine must be refused during planning."""
    from aetherdialect._federation_plan import _build_source_sub_intent

    def _force_scalar_member_a(*args, **kwargs):
        built = _build_source_sub_intent(*args, **kwargs)
        if built is not None and built.source_id == "a":
            return replace(built, sub_intent=replace(built.sub_intent, grain="scalar"))
        return built

    monkeypatch.setattr(
        "aetherdialect._federation_plan._build_source_sub_intent",
        _force_scalar_member_a,
    )

    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("right_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(FederationDeclarationError, match=r"scalar grain.*src_a"):
        plan_federated_intent(intent, composite, manifest)


@pytest.mark.fast
def test_single_member_scalar_plan_skips_frame_validation() -> None:
    """Degenerate single-member scalar plans do not require coordinator member frames."""
    scalar_sub = RuntimeIntent(
        tables=["left_t"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=scalar_sub, projected_keys=("left_t.id",)),),
        grain="scalar",
    )
    validate_federation_scalar_grain_member_frames(plan)
