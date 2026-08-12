"""Consumer open uses owner-cache subset + privilege probe (no force- live reclassify)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from aetherdialect._constants_runtime import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._main_init import MainInitOps
from aetherdialect._schema_finalize import build_schema_graph_with_diff
from aetherdialect._schema_graph import (
    assert_consumer_intent_in_scope,
    consumer_graph_is_permission_subset,
    subset_schema_graph_for_visible_tables,
)


@pytest.mark.fast
def test_build_schema_graph_accepts_persist_schema_cache_flag() -> None:
    params = inspect.signature(build_schema_graph_with_diff).parameters
    assert "persist_schema_cache" in params
    assert params["persist_schema_cache"].default is True


@pytest.mark.fast
def test_consumer_init_source_skips_force_live_for_consumer() -> None:
    src = inspect.getsource(MainInitOps.initialize_aether_engine)
    assert "open_consumer_schema_from_owner_cache" in src
    assert "force_live_schema_reflect=pending_migration_map is not None or is_consumer" not in src
    assert "force_live_schema_reflect=pending_migration_map is not None" in src


@pytest.mark.fast
def test_dialect_default_selectable_filter_is_identity() -> None:
    dialect = Dialect.__new__(Dialect)
    names = ["customer", "payment", "film"]
    assert dialect.filter_selectable_relation_names("public", names) == names


@pytest.mark.fast
def test_postgres_selectable_filter_keeps_allowed_only() -> None:
    dialect = PostgresDialect.__new__(PostgresDialect)
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [("customer",), ("payment",)]
    dialect.engine = engine
    kept = dialect.filter_selectable_relation_names("app_schema", ["customer", "payment", "film", "actor"])
    assert kept == ["customer", "payment"]


@pytest.mark.fast
def test_subset_schema_graph_prefers_base_description() -> None:
    owner_a = TableMetadata(
        name="a",
        columns={"id": ColumnMetadata(name="id", data_type="integer", description="enriched", base_description="base")},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
        description="table enriched",
        base_description="table base",
    )
    owner_b = TableMetadata(
        name="b",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )
    owner = SchemaGraph(join_paths_multi={}, tables={"a": owner_a, "b": owner_b}, schema_graph_id="sg1")
    subset = subset_schema_graph_for_visible_tables(owner, frozenset({"a"}), prefer_base_description=True)
    assert set(subset.tables) == {"a"}
    assert subset.tables["a"].description == "table base"
    assert subset.tables["a"].columns["id"].description == "base"


@pytest.mark.fast
def test_permission_subset_tolerates_cross_scope_fk_targets() -> None:
    owner_a = TableMetadata(
        name="a",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "b_id": ColumnMetadata(
                name="b_id",
                data_type="integer",
                is_foreign_key=True,
                fk_target=("b", "id"),
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )
    owner_b = TableMetadata(
        name="b",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )
    owner = SchemaGraph(
        join_paths_multi={},
        tables={"a": owner_a, "b": owner_b},
        schema_graph_id="sg_perm000000000001__abcd1234",
    )
    consumer_a = TableMetadata(
        name="a",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "b_id": ColumnMetadata(name="b_id", data_type="integer"),
        },
        primary_key=["id"],
        foreign_keys=[],
        row_count=1,
    )
    consumer = SchemaGraph(
        join_paths_multi={},
        tables={"a": consumer_a},
        schema_graph_id="sg_perm000000000001__abcd1234",
    )
    assert consumer_graph_is_permission_subset(owner, consumer) is True


@pytest.mark.fast
def test_permission_denied_user_message_is_locate_neutral() -> None:
    assert "Unable to locate the requested data" in PERMISSION_DENIED_USER_MESSAGE
    assert "access" not in PERMISSION_DENIED_USER_MESSAGE.lower()


@pytest.mark.fast
def test_assert_consumer_declared_tables_only_ignores_join_reachability() -> None:
    graph = SchemaGraph(
        join_paths_multi={},
        tables={
            "a": TableMetadata(
                name="a",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
                row_count=1,
            ),
            "secret": TableMetadata(
                name="secret",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
                row_count=1,
            ),
        },
    )
    intent = RuntimeIntent(
        tables=["a"],
        select_cols=[SelectCol.from_dict({"expr": "a.id"})],
        resolved_join_tables=["a", "secret"],
    )
    ctx = EngineContext()
    visible = frozenset({"a", "a.id"})
    assert assert_consumer_intent_in_scope(intent, ctx, graph, visible, declared_tables_only=True) is True
    assert assert_consumer_intent_in_scope(intent, ctx, graph, visible, declared_tables_only=False) is False
