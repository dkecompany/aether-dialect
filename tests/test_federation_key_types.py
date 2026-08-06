"""Tests for exact numeric join-key typing at the federation coordinator."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    plan_federated_intent,
    render_federation_glue,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import enriched_manifest


def _decimal_key_column(*, precision: int = 18, scale: int = 2) -> ColumnMetadata:
    return ColumnMetadata(
        name="id",
        data_type=f"DECIMAL({precision},{scale})",
        sensitivity="none",
        is_primary_key=True,
        numeric_precision=precision,
        numeric_scale=scale,
        is_exact_numeric=True,
    )


def _member_graph(
    table: str,
    source_id: str,
    *,
    key_column: ColumnMetadata,
) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": key_column},
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
    "federation_id": "fed_decimal_keys",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def _decimal_key_federation(
    *,
    left_key: ColumnMetadata,
    right_key: ColumnMetadata,
) -> tuple[SchemaGraph, object]:
    member_graphs = {
        "a": _member_graph("left_t", "a", key_column=left_key),
        "b": _member_graph("right_t", "b", key_column=right_key),
    }
    manifest = enriched_manifest(member_graphs, _MANIFEST, member_graphs=member_graphs)
    composite = compose_composite_graph(member_graphs, manifest)
    return composite, manifest


@pytest.mark.fast
def test_exact_key_not_cast_to_double() -> None:
    composite, manifest = _decimal_key_federation(
        left_key=_decimal_key_column(precision=18, scale=2),
        right_key=_decimal_key_column(precision=18, scale=2),
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    upper = glue.upper()
    assert "DECIMAL(18, 2)" in upper
    assert "CAST" in upper
    assert "DOUBLE" not in upper


@pytest.mark.fast
def test_mixed_exactness_key_refused_at_composition() -> None:
    exact_key = _decimal_key_column()
    approx_key = ColumnMetadata(
        name="id",
        data_type="DOUBLE PRECISION",
        sensitivity="none",
        is_primary_key=True,
        is_exact_numeric=False,
    )
    member_graphs = {
        "a": _member_graph("left_t", "a", key_column=exact_key),
        "b": _member_graph("right_t", "b", key_column=approx_key),
    }
    manifest = enriched_manifest(member_graphs, _MANIFEST, member_graphs=member_graphs)
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(member_graphs, manifest)
    message = str(exc_info.value)
    assert "left_t.id" in message
    assert "right_t.id" in message
    assert "DECIMAL" in message.upper()
    assert "DOUBLE" in message.upper()
