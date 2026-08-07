"""Remapped templates must re-render identically or be invalidated."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, TemplateStats, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata
from aetherdialect._core_utils import reset_diagnostic_collector, set_diagnostic_collector
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", sensitivity="none")


def _schema(*, graph_id: str, table: str, column: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={column: _col(column)},
            primary_key=[column],
            foreign_keys=[],
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=graph_id,
        effective_structural_hash=graph_id,
    )


def _template(*, tid: str, table: str, column: str, graph_id: str) -> Template:
    concrete = ConcreteIntent(
        intent_id=f"intent_{tid}",
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={column: table},
    )
    return Template(
        id=tid,
        effective_structural_hash=graph_id,
        schema_graph_id=graph_id,
        intent_signature=concrete,
        intent_key=f"key_{tid}",
        tables_used=[table],
        sql_param=f"SELECT {table}.{column} FROM {table}",
        sql_fp=f"fp_{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig=f"sig_{tid}",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
        schema_column_types={f"{table}.{column}": "integer"},
    )


def _seed(artifacts_dir: str, schema: SchemaGraph, templates: dict[str, Template]) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store(schema.schema_graph_id)
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev


@pytest.mark.fast
def test_remapped_template_rerenders_identically(tmp_path) -> None:
    old_id = "sg_old000000000001__aaaa1111"
    new_id = "sg_new000000000002__bbbb2222"
    old_schema = _schema(graph_id=old_id, table="orders", column="amt")
    new_schema = _schema(graph_id=new_id, table="sales_orders", column="amount")
    tmpl = _template(tid="T0001", table="orders", column="amt", graph_id=old_id)
    _seed(str(tmp_path), old_schema, {"T0001": tmpl})

    remapped, destroyed = TemplateOps._apply_schema_rename_migration_to_store(
        str(tmp_path),
        new_schema,
        renamed_tables=(("orders", "sales_orders"),),
        renamed_columns=(("orders", "amt", "amount"),),
    )
    assert remapped == 1
    assert destroyed == 0
    view = TemplateOps._load_partitioned_view_unlocked(
        TemplateOps.template_store_dir_for_space(str(tmp_path), "master")
    )
    assert view is not None
    raw = view.get_template_raw("T0001")
    assert raw is not None
    assert raw["sql_param"] == 'SELECT "sales_orders"."amount"\nFROM "sales_orders"'


@pytest.mark.fast
def test_divergent_remap_invalidated(tmp_path) -> None:
    old_id = "sg_old000000000003__cccc3333"
    new_id = "sg_new000000000004__dddd4444"
    old_schema = _schema(graph_id=old_id, table="orders", column="amt")
    new_schema = _schema(graph_id=new_id, table="sales_orders", column="amount")
    tmpl = _template(tid="T0001", table="orders", column="amt", graph_id=old_id)
    _seed(str(tmp_path), old_schema, {"T0001": tmpl})
    diags: list = []
    diag_token = set_diagnostic_collector(diags)
    try:
        with patch(
            "aetherdialect._templates.build_deterministic_sql",
            return_value="SELECT amount FROM sales_orders WHERE 1=0",
        ):
            remapped, destroyed = TemplateOps._apply_schema_rename_migration_to_store(
                str(tmp_path),
                new_schema,
                renamed_tables=(("orders", "sales_orders"),),
                renamed_columns=(("orders", "amt", "amount"),),
            )
    finally:
        reset_diagnostic_collector(diag_token)

    assert remapped == 0
    assert destroyed == 1
    view = TemplateOps._load_partitioned_view_unlocked(
        TemplateOps.template_store_dir_for_space(str(tmp_path), "master")
    )
    assert view is not None
    assert "T0001" not in view.partition_map
    assert any(d.code == DIAGNOSTIC_CODE_TEMPLATE_REMAP_DIVERGED for d in diags)
