"""Tests for named EngineContext execution-RBAC specs."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    OwnerOnlyOperationError,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import (
    MainExecutionOps,
)
from aetherdialect._utils import scope_hash_fp


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="text")


def _table(name: str, *columns: str) -> TableMetadata:
    cols = {c: _col(c) for c in columns}
    return TableMetadata(name=name, columns=cols, primary_key=(), foreign_keys=())


def _sample_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": _table("orders", "id", "customer_id", "secret"),
            "customers": _table("customers", "id", "email"),
        },
        join_paths_multi={},
    )


def _master_context(*, allow: frozenset[str] = frozenset(), deny: frozenset[str] = frozenset()) -> EngineContext:
    return EngineContext(
        allow_objects=allow,
        deny_columns=deny,
    )


class TestNamedEngineContextSpecValidation:
    def test_rejects_sql_file(self) -> None:
        with pytest.raises(ConfigError, match="sql_file"):
            MainExecutionOps.validate_named_engine_context_spec(EngineContext(sql_file="schema.sql"))

    def test_rejects_notes_file(self) -> None:
        with pytest.raises(ConfigError, match="notes_file"):
            MainExecutionOps.validate_named_engine_context_spec(EngineContext(notes_file="notes.txt"))

    def test_rejects_include_override(self) -> None:
        with pytest.raises(ConfigError, match="include"):
            MainExecutionOps.validate_named_engine_context_spec(EngineContext(include="both"))

    def test_identical_specs_share_scope_hash(self) -> None:
        left = EngineContext(deny_columns=frozenset({"orders.secret"}))
        right = EngineContext(deny_columns=frozenset({"orders.secret"}))
        assert scope_hash_fp(left) == scope_hash_fp(right)


class TestNamedContextSubsetValidation:
    def test_rejects_widened_allow(self) -> None:
        master = _master_context(allow=frozenset({"orders"}))
        named = EngineContext(allow_objects=frozenset({"orders", "customers"}))
        with pytest.raises(ConfigError, match="widens master scope"):
            MainExecutionOps.validate_named_context_subset(master, named, _sample_graph())

    def test_rejects_named_context_deny_not_superset_of_master(self) -> None:
        master = _master_context(deny=frozenset({"orders.secret"}))
        named = EngineContext(deny_columns=frozenset())
        with pytest.raises(ConfigError, match="inherit all master deny_columns"):
            MainExecutionOps.validate_named_context_subset(master, named, _sample_graph())

    def test_accepts_valid_subset(self) -> None:
        master = _master_context(allow=frozenset({"orders", "customers"}))
        named = EngineContext(allow_objects=frozenset({"orders"}))
        MainExecutionOps.validate_named_context_subset(master, named, _sample_graph())

    def test_intersects_allow_and_unions_deny(self) -> None:
        master = _master_context(deny=frozenset({"customers.email"}))
        named = EngineContext(
            allow_objects=frozenset({"orders"}),
            deny_columns=frozenset({"customers.email"}),
        )
        eff = MainExecutionOps.effective_execution_context(master, named, "team_a")
        assert eff.allow_objects == frozenset({"orders"})
        assert eff.deny_columns == frozenset({"customers.email"})


class TestNamedContextPersistence:
    def test_round_trip_and_list(self, tmp_path: Path) -> None:
        engine_dir = str(tmp_path)
        ctx = EngineContext(
            allow_objects=frozenset({"orders"}),
            deny_columns=frozenset({"customers.email"}),
        )
        path = MainExecutionOps.save_named_schema_context(engine_dir, "team_a", ctx)
        assert path == MainExecutionOps._named_schema_context_path(engine_dir, "team_a")
        loaded = MainExecutionOps.load_named_schema_context(engine_dir, "team_a")
        assert loaded == ctx
        assert MainExecutionOps.list_named_schema_context_names(engine_dir) == ("team_a",)

    def test_export_master_and_named(self, tmp_path: Path) -> None:
        engine_dir = str(tmp_path)
        master = _master_context(deny=frozenset({"orders.secret"}))
        MainExecutionOps.save_named_schema_context(
            engine_dir,
            "team_a",
            EngineContext(allow_objects=frozenset({"orders"})),
        )
        master_doc = MainExecutionOps.build_named_schema_context_export(engine_dir, "master", master)
        named_doc = MainExecutionOps.build_named_schema_context_export(engine_dir, "team_a", master)
        assert master_doc["name"] == "master"
        assert named_doc["name"] == "team_a"
        assert named_doc["allow_objects"] == ["orders"]


class TestResolveEngineContextPlan:
    def test_consumer_object_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OwnerOnlyOperationError):
            MainExecutionOps.resolve_engine_context_plan(
                EngineContext(allow_objects=frozenset({"orders"})),
                str(tmp_path),
                schema_role="consumer",
                load_master=_master_context(),
                prepare_master=None,
            )

    def test_str_unknown_raises(self, tmp_path: Path) -> None:
        master = _master_context()
        MainExecutionOps.write_schema_context_cache(str(tmp_path), master)
        with pytest.raises(ConfigError, match="unknown engine context"):
            MainExecutionOps.resolve_engine_context_plan(
                "missing",
                str(tmp_path),
                schema_role="owner",
                load_master=master,
                prepare_master=None,
            )

    def test_str_loads_named(self, tmp_path: Path) -> None:
        master = _master_context(allow=frozenset({"orders", "customers"}))
        MainExecutionOps.write_schema_context_cache(str(tmp_path), master)
        named_spec = EngineContext(allow_objects=frozenset({"orders"}))
        MainExecutionOps.save_named_schema_context(str(tmp_path), "team_a", named_spec)
        m, active, name = MainExecutionOps.resolve_engine_context_plan(
            "team_a",
            str(tmp_path),
            schema_role="consumer",
            load_master=master,
            prepare_master=None,
        )
        assert m == master
        assert active == named_spec
        assert name == "team_a"

    def test_object_defines_master(self, tmp_path: Path) -> None:
        master = _master_context()
        m, active, name = MainExecutionOps.resolve_engine_context_plan(
            master,
            str(tmp_path),
            schema_role="owner",
            load_master=None,
            prepare_master=master,
        )
        assert m == master
        assert active == master
        assert name == "master"
