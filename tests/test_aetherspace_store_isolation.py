"""Template stores are isolated per aetherspace name."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import PipelineSession
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps


def _schema() -> SchemaGraph:
    table = TableMetadata(
        name="orders",
        columns={"order_id": ColumnMetadata(name="order_id", data_type="integer", sensitivity="none")},
        primary_key=["order_id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"orders": table},
        join_paths_multi=recompute_join_paths_multi({"orders": table}),
        effective_structural_hash="eff_space_iso",
        schema_graph_id="graph_space_iso",
    )


@pytest.mark.fast
def test_accept_in_space_a_invisible_in_space_b(tmp_path) -> None:
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _schema()
    graph_id = schema.schema_graph_id

    for space in ("space_a", "space_b"):
        store_dir = artifacts_dir / "intent_templates" / "spaces" / space
        store_dir.mkdir(parents=True)
        store = TemplateOps.empty_template_store(graph_id)
        store._store_dir = str(store_dir)
        TemplateOps.save_template_store(store)

    owner = MagicMock()
    owner._artifacts_dir = str(artifacts_dir)
    owner._schema_graph = schema
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._rejected = {}
    owner._schema_terms = set()

    def _session(**kwargs):
        space = str(kwargs.get("space", "master")).strip().lower()
        return PipelineSession(
            owner,
            mode=kwargs.get("mode", "writer"),
            space_name=space,
        )

    owner.session = _session

    sess_a = owner.session(mode="writer", space="space_a")
    _, store_a, templates_a, _, _ = sess_a._resources()
    intent = RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    TemplateOps.insert_template(
        store_a,
        templates_a,
        schema,
        "orders count in space a",
        intent,
        "SELECT order_id FROM orders",
        dialect=MagicMock(),
        record_accept=True,
    )
    TemplateOps.save_template_store(store_a)

    _, store_b, templates_b, _, _ = owner.session(mode="writer", space="space_b")._resources()
    assert (
        TemplateOps.resolve_template_for_question("orders count in space a", templates_b, template_store=store_b)
        is None
    )
    resolved_a = TemplateOps.resolve_template_for_question(
        "orders count in space a", templates_a, template_store=store_a
    )
    assert resolved_a is not None
