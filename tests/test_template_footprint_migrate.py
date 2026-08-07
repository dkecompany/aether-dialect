"""Footprint-preserving template id migration across schema_graph_id changes."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._templates import TemplateOps, TemplateRefs
from aetherdialect._utils import intent_key


def _col(name: str, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _schema(
    *,
    schema_graph_id: str,
    tables: dict[str, TableMetadata],
    deny_columns: dict[str, set[str]] | None = None,
) -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        effective_structural_hash=f"hash-{schema_graph_id}",
        schema_graph_id=schema_graph_id,
        deny_columns=deny_columns or {},
    )


def _orders_schema(schema_graph_id: str, *, extra_table: bool = False, drop_amount: bool = False) -> SchemaGraph:
    cols = {"id": _col("id"), "amount": _col("amount", "numeric")}
    if drop_amount:
        cols.pop("amount")
    tables = {
        "orders": TableMetadata(
            name="orders",
            columns=cols,
            primary_key=["id"],
            foreign_keys=[],
            row_count=1,
        )
    }
    if extra_table:
        tables["customers"] = TableMetadata(
            name="customers",
            columns={"id": _col("id")},
            primary_key=["id"],
            foreign_keys=[],
            row_count=1,
        )
    return _schema(schema_graph_id=schema_graph_id, tables=tables)


def _template(schema: SchemaGraph, *, tid: str = "T0001") -> Template:
    intent = ConcreteIntent(
        intent_id="x",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"id": "orders"},
    )
    rt = intent.to_runtime_skeleton()
    tmpl = Template(
        id=tid,
        effective_structural_hash=schema.effective_structural_hash,
        schema_graph_id=schema.schema_graph_id,
        intent_signature=intent,
        intent_key=intent_key(rt),
        tables_used=["orders"],
        sql_param="SELECT orders.id FROM orders",
        sql_fp="fp1",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
        schema_column_types={"orders.id": "integer"},
    )
    return TemplateRefs.stamp_template_footprint(tmpl)


@pytest.mark.fast
def test_additive_table_keeps_same_template_id() -> None:
    old = _orders_schema("sg_old")
    new = _orders_schema("sg_new", extra_table=True)
    tmpl = _template(old)
    store: dict = {"templates": {tmpl.id: tmpl.to_dict()}}
    report = TemplateOps.reconcile_template_store(store, new)
    assert tmpl.id in report.kept_template_ids
    kept = Template.from_dict({**store["templates"][tmpl.id], "id": tmpl.id})
    assert kept.id == tmpl.id
    assert kept.schema_graph_id == "sg_new"


@pytest.mark.fast
def test_unrelated_deny_keeps_id() -> None:
    old = _orders_schema("sg_old")
    new = replace(
        _orders_schema("sg_new"),
        deny_columns={"customers": {"secret"}},
    )
    # customers table absent — deny on missing table should not affect orders footprint
    tmpl = _template(old)
    store: dict = {"templates": {tmpl.id: tmpl.to_dict()}}
    report = TemplateOps.reconcile_template_store(store, new)
    assert tmpl.id in report.kept_template_ids


@pytest.mark.fast
def test_footprint_column_dropped_orphans() -> None:
    old = _orders_schema("sg_old")
    intent = ConcreteIntent(
        intent_id="x",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"amount": "orders"},
    )
    tmpl = Template(
        id="T_AMT",
        effective_structural_hash=old.effective_structural_hash,
        schema_graph_id=old.schema_graph_id,
        intent_signature=intent,
        intent_key="ik",
        tables_used=["orders"],
        sql_param="SELECT orders.amount FROM orders",
        sql_fp="fp2",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q2"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
        schema_column_types={"orders.amount": "numeric"},
        footprint_tables=("orders",),
        footprint_columns=("orders.amount",),
    )
    dropped_schema = _orders_schema("sg_new", drop_amount=True)
    store: dict = {"templates": {tmpl.id: tmpl.to_dict()}}
    report = TemplateOps.reconcile_template_store(store, dropped_schema)
    assert tmpl.id in report.dropped_template_ids
    assert tmpl.id not in store["templates"]


@pytest.mark.fast
def test_agent_ref_still_resolves_after_additive_ingest() -> None:
    old = _orders_schema("sg_old")
    new = _orders_schema("sg_new", extra_table=True)
    tmpl = _template(old)
    store: dict = {"templates": {tmpl.id: tmpl.to_dict()}}
    TemplateOps.reconcile_template_store(store, new)
    templates = {tid: Template.from_dict({**raw, "id": tid}) for tid, raw in store["templates"].items()}
    resolved = TemplateOps.resolve_template_ref(tmpl.id, templates)
    assert resolved is not None
    assert resolved.id == tmpl.id
    assert resolved.schema_graph_id == "sg_new"
