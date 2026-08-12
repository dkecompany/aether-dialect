"""Unit tests for include/allow/deny reflection mode, view key projection, and cover reduction."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    ReflectMode,
    SchemaAccessError,
    TableKind,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, InferenceTag, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import (
    apply_deny_objects_filter,
    effective_reflect_mode,
    raise_if_schema_unusable,
)
from aetherdialect._schema_reflect import (
    apply_cover_reduction,
    project_view_keys_from_bases,
    reflect_schema_graph_for_context,
    view_fully_covers_table,
)
from aetherdialect._utils import scope_hash_fp


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer")


def _table(
    name: str,
    columns: list[str],
    *,
    pk: list[str] | None = None,
    fks: list[FKEdge] | None = None,
) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={c: _col(c) for c in columns},
        primary_key=list(pk or []),
        foreign_keys=list(fks or []),
        kind=TableKind.TABLE,
    )


def _view(name: str, columns: list[str], definition: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={c: _col(c) for c in columns},
        primary_key=[],
        foreign_keys=[],
        kind=TableKind.VIEW,
        view_definition=definition,
    )


def _graph(*tables: TableMetadata) -> SchemaGraph:
    return SchemaGraph(tables={t.name: t for t in tables}, join_paths_multi={})


@pytest.mark.parametrize(
    ("ctx", "expected_mode"),
    [
        (EngineContext(include="tables"), ReflectMode.SINGLE_KIND),
        (EngineContext(include="views"), ReflectMode.SINGLE_KIND),
        (EngineContext(allow_objects=frozenset({"v_a", "t_b"}), include="tables"), ReflectMode.ALLOW_LIST),
        (EngineContext(deny_objects=frozenset({"t_x"}), include="tables"), ReflectMode.BOTH_THEN_DENY),
    ],
)
def test_effective_reflect_mode_matrix(ctx: EngineContext, expected_mode: ReflectMode) -> None:
    assert effective_reflect_mode(ctx) == expected_mode


def test_engine_context_rejects_include_both() -> None:
    with pytest.raises(ConfigError, match="include must be 'tables' or 'views'"):
        EngineContext(include="both")


def test_scope_hash_encodes_reflect_mode() -> None:
    allow_ctx = EngineContext(allow_objects=frozenset({"a"}), include="tables")
    deny_ctx = EngineContext(deny_objects=frozenset({"x"}), include="tables")
    single_tables = EngineContext(include="tables")
    single_views = EngineContext(include="views")
    assert scope_hash_fp(allow_ctx) != scope_hash_fp(deny_ctx)
    assert scope_hash_fp(single_tables) != scope_hash_fp(single_views)
    assert scope_hash_fp(single_tables) != scope_hash_fp(allow_ctx)


class _StubDialect:
    def __init__(self, tables: SchemaGraph, views: SchemaGraph) -> None:
        self._tables = tables
        self._views = views

    def reflect_schema_graph(
        self,
        *,
        include: str,
        allow_objects: frozenset[str] | None = None,
        deny_objects: frozenset[str] | None = None,
        sql_file: str | None = None,
    ) -> SchemaGraph:
        _ = allow_objects, deny_objects, sql_file
        if include == "views":
            return SchemaGraph(
                tables={k: v for k, v in self._views.tables.items()},
                join_paths_multi=dict(self._views.join_paths_multi),
            )
        return SchemaGraph(
            tables={k: v for k, v in self._tables.tables.items()},
            join_paths_multi=dict(self._tables.join_paths_multi),
        )


def test_allow_list_ignores_include_kind() -> None:
    tables = _graph(_table("table_y", ["id"], pk=["id"]))
    views = _graph(_view("view_a", ["id"], "SELECT id FROM table_x"))
    dialect = _StubDialect(tables, views)
    ctx = EngineContext(allow_objects=frozenset({"view_a", "table_y"}), include="tables")
    sg = reflect_schema_graph_for_context(dialect, ctx)
    assert set(sg.tables) == {"view_a", "table_y"}


def test_deny_only_reflects_both_kinds_then_removes() -> None:
    tables = _graph(
        _table("table_x", ["id"], pk=["id"]),
        _table("table_y", ["id"], pk=["id"]),
    )
    views = _graph(_view("view_a", ["id"], "SELECT id FROM table_x"))
    dialect = _StubDialect(tables, views)
    ctx = EngineContext(deny_objects=frozenset({"table_x"}), include="tables")
    sg = reflect_schema_graph_for_context(dialect, ctx)
    assert set(sg.tables) == {"view_a", "table_y"}


def test_pk_projection_onto_view() -> None:
    orders = _table("orders", ["id"], pk=["id"])
    items = _table(
        "items",
        ["id", "order_id"],
        pk=["id"],
        fks=[FKEdge(src_table="items", src_cols=["order_id"], dst_table="orders", dst_cols=["id"])],
    )
    v_items = _view("v_items", ["id", "order_id"], "SELECT id, order_id FROM items")
    sg = _graph(v_items)
    project_view_keys_from_bases(sg, {"orders": orders, "items": items})
    assert "id" in sg.tables["v_items"].primary_key


def test_fk_projection_to_public_table() -> None:
    orders = _table("orders", ["id"], pk=["id"])
    items = _table(
        "items",
        ["id", "order_id"],
        pk=["id"],
        fks=[FKEdge(src_table="items", src_cols=["order_id"], dst_table="orders", dst_cols=["id"])],
    )
    v_items = _view("v_items", ["id", "order_id"], "SELECT id, order_id FROM items")
    sg = _graph(orders, v_items)
    project_view_keys_from_bases(sg, {"items": items})
    edges = sg.tables["v_items"].foreign_keys
    assert any(
        e.dst_table == "orders" and e.src_cols == ["order_id"] and e.inference_tag == InferenceTag.VIEW_LINEAGE
        for e in edges
    )


def test_fk_projection_dropped_without_covering_destination() -> None:
    orders = _table("orders", ["id"], pk=["id"])
    items = _table(
        "items",
        ["id", "order_id"],
        pk=["id"],
        fks=[FKEdge(src_table="items", src_cols=["order_id"], dst_table="orders", dst_cols=["id"])],
    )
    v_items = _view("v_items", ["id", "order_id"], "SELECT id, order_id FROM items")
    sg = _graph(v_items)
    project_view_keys_from_bases(sg, {"orders": orders, "items": items})
    assert sg.tables["v_items"].foreign_keys == []


def test_cover_reduction_removes_fully_covered_tables() -> None:
    a = _table("a", ["a_id", "a_val"])
    b = _table("b", ["b_id", "b_val"])
    v_ab = _view(
        "v_ab",
        ["a_id", "a_val", "b_id", "b_val"],
        "SELECT a.a_id, a.a_val, b.b_id, b.b_val FROM a JOIN b ON TRUE",
    )
    sg = _graph(a, b, v_ab)
    removed = apply_cover_reduction(sg)
    assert removed == 2
    assert set(sg.tables) == {"v_ab"}


def test_partial_cover_keeps_base_table() -> None:
    a = _table("a", ["a_id", "a_val"])
    v_partial = _view("v_partial", ["a_id"], "SELECT a_id FROM a")
    sg = _graph(a, v_partial)
    assert apply_cover_reduction(sg) == 0
    assert "a" in sg.tables


def test_view_fully_covers_table_requires_all_columns() -> None:
    base = _table("a", ["a_id", "a_val"])
    full = _view("v_full", ["a_id", "a_val"], "SELECT a_id, a_val FROM a")
    partial = _view("v_partial", ["a_id"], "SELECT a_id FROM a")
    assert view_fully_covers_table(full, base)
    assert not view_fully_covers_table(partial, base)


def test_views_only_usability_with_projected_fks() -> None:
    orders = _table("orders", ["id"], pk=["id"])
    items = _table(
        "items",
        ["id", "order_id"],
        pk=["id"],
        fks=[FKEdge(src_table="items", src_cols=["order_id"], dst_table="orders", dst_cols=["id"])],
    )
    v_orders = _view("v_orders", ["id"], "SELECT id FROM orders")
    v_items = _view("v_items", ["id", "order_id"], "SELECT id, order_id FROM items")
    sg = _graph(v_orders, v_items)
    apply_cover_reduction(sg)
    project_view_keys_from_bases(sg, {"orders": orders, "items": items})
    raise_if_schema_unusable(sg, EngineContext(allow_objects=frozenset({"v_orders", "v_items"})))


def test_unknown_deny_object_raises() -> None:
    sg = _graph(_table("t1", ["id"]))
    ctx = EngineContext(deny_objects=frozenset({"missing_table"}), include="tables")
    with pytest.raises(SchemaAccessError, match="deny_objects references unknown relation"):
        apply_deny_objects_filter(sg, ctx, strict=True)
