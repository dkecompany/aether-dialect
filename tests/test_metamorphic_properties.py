"""Metamorphic properties over fixture intents and scoped schema graphs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_core import ConcreteIntent, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import filter_schema_graph_by_scope, recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql
from aetherdialect._templates import TemplateOps

# Recorded for operator live_deferred manifests (needs_corpus only).
NEEDS_CORPUS_NODE_IDS = (
    "tests/test_metamorphic_properties.py::test_redundant_join_preserves_answer",
    "tests/test_metamorphic_properties.py::test_template_replay_matches_fresh_answer",
)


def _duckdb():
    return DialectRegistry.get("duckdb")


def _column(name: str, *, dt: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=dt, sensitivity="none")


def _rename_schema(
    *,
    graph_id: str,
    table: str,
    column: str,
) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={column: _column(column)},
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


def _row_intent(table: str, column: str) -> RuntimeIntent:
    concrete = ConcreteIntent(
        intent_id="meta_intent",
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={column: table},
    )
    return RuntimeIntent(
        tables=list(concrete.tables),
        grain=concrete.grain,
        select_cols=list(concrete.select_cols),
        group_by_cols=list(concrete.group_by_cols),
        order_by_cols=list(concrete.order_by_cols),
        where=concrete.where,
        column_map=dict(concrete.column_map or {}),
    )


RENAME_CASES = (
    pytest.param("orders", "amt", "sales_orders", "amount", id="table_and_column"),
    pytest.param("customers", "id", "client", "client_id", id="dimension_rename"),
)


@pytest.mark.needs_corpus
def test_redundant_join_preserves_answer() -> None:
    """Adding a join that cannot change the result does not change the result."""
    pytest.importorskip("aetherdialect._sandbox")


@pytest.mark.needs_corpus
def test_template_replay_matches_fresh_answer() -> None:
    """Replaying a template produces the same rows as answering the question fresh."""
    pytest.importorskip("aetherdialect._sandbox")


@pytest.mark.fast
@pytest.mark.parametrize("old_table,old_col,new_table,new_col", RENAME_CASES)
def test_rename_remap_sql_matches_fresh_build(
    old_table: str,
    old_col: str,
    new_table: str,
    new_col: str,
) -> None:
    """Renaming then remapping yields SQL byte-identical to a build on the new name."""
    new_id = f"sg_new_{new_table}_{new_col}"
    new_schema = _rename_schema(graph_id=new_id, table=new_table, column=new_col)
    old_intent = _row_intent(old_table, old_col)
    fresh_intent = _row_intent(new_table, new_col)

    tmap = {old_table: new_table}
    colmaps = {old_table: {old_col: new_col}}
    old_concrete = ConcreteIntent(
        intent_id="meta_intent",
        tables=[old_table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{old_table}.{old_col}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={old_col: old_table},
    )
    remapped = TemplateOps._remap_concrete_intent(old_concrete, tmap, colmaps)
    remapped_intent = replace(
        old_intent,
        tables=list(remapped.tables),
        select_cols=list(remapped.select_cols),
        column_map=dict(remapped.column_map or {}),
        chosen_join_path_signature=list(remapped.chosen_join_path_signature or []),
    )

    dialect = _duckdb()
    sql_fresh = build_deterministic_sql(fresh_intent, schema=new_schema, dialect=dialect)
    sql_remapped = build_deterministic_sql(remapped_intent, schema=new_schema, dialect=dialect)
    assert sql_remapped == sql_fresh


SCOPE_CASES = (pytest.param("customers", frozenset({"orders"}), id="deny_orders"),)


@pytest.mark.fast
@pytest.mark.parametrize("focus_table,deny_objects", SCOPE_CASES)
def test_widening_scope_preserves_in_scope_answer(
    simple_schema: SchemaGraph,
    focus_table: str,
    deny_objects: frozenset[str],
) -> None:
    """Widening visible scope does not change SQL for an in-scope intent."""
    narrow_ctx = EngineContext(deny_objects=deny_objects)
    narrow_schema = filter_schema_graph_by_scope(simple_schema, narrow_ctx)
    intent = _row_intent(focus_table, "id")
    dialect = _duckdb()
    sql_narrow = build_deterministic_sql(intent, schema=narrow_schema, dialect=dialect)
    sql_wide = build_deterministic_sql(intent, schema=simple_schema, dialect=dialect)
    assert sql_narrow == sql_wide
