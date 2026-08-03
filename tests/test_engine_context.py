"""Tests for named EngineContext execution-RBAC specs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    OwnerOnlyOperationError,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import scope_hash_fp
from aetherdialect._main_execution import (
    _effective_execution_context,
    _named_schema_context_path,
    export_named_schema_context_json,
    list_named_schema_context_names,
    load_named_schema_context,
    resolve_engine_context_plan,
    save_named_schema_context,
    validate_named_context_subset,
    validate_named_engine_context_spec,
    validate_space_subset_of_execution_context,
    write_schema_context_cache,
)


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
            validate_named_engine_context_spec(EngineContext(sql_file="schema.sql"))

    def test_rejects_notes_file(self) -> None:
        with pytest.raises(ConfigError, match="notes_file"):
            validate_named_engine_context_spec(EngineContext(notes_file="notes.txt"))

    def test_rejects_include_override(self) -> None:
        with pytest.raises(ConfigError, match="include"):
            validate_named_engine_context_spec(EngineContext(include="both"))

    def test_identical_specs_share_scope_hash(self) -> None:
        left = EngineContext(deny_columns=frozenset({"orders.secret"}))
        right = EngineContext(deny_columns=frozenset({"orders.secret"}))
        assert scope_hash_fp(left) == scope_hash_fp(right)


class TestNamedContextSubsetValidation:
    def test_rejects_widened_allow(self) -> None:
        master = _master_context(allow=frozenset({"orders"}))
        named = EngineContext(allow_objects=frozenset({"orders", "customers"}))
        with pytest.raises(ConfigError, match="widens master scope"):
            validate_named_context_subset(master, named, _sample_graph())

    def test_rejects_named_context_deny_not_superset_of_master(self) -> None:
        master = _master_context(deny=frozenset({"orders.secret"}))
        named = EngineContext(deny_columns=frozenset())
        with pytest.raises(ConfigError, match="inherit all master deny_columns"):
            validate_named_context_subset(master, named, _sample_graph())

    def test_accepts_valid_subset(self) -> None:
        master = _master_context(allow=frozenset({"orders", "customers"}))
        named = EngineContext(allow_objects=frozenset({"orders"}))
        validate_named_context_subset(master, named, _sample_graph())

    def test_intersects_allow_and_unions_deny(self) -> None:
        master = _master_context(deny=frozenset({"customers.email"}))
        named = EngineContext(
            allow_objects=frozenset({"orders"}),
            deny_columns=frozenset({"customers.email"}),
        )
        eff = _effective_execution_context(master, named, "team_a")
        assert eff.allow_objects == frozenset({"orders"})
        assert eff.deny_columns == frozenset({"customers.email"})


class TestNamedContextPersistence:
    def test_round_trip_and_list(self, tmp_path: Path) -> None:
        engine_dir = str(tmp_path)
        ctx = EngineContext(
            allow_objects=frozenset({"orders"}),
            deny_columns=frozenset({"customers.email"}),
        )
        path = save_named_schema_context(engine_dir, "team_a", ctx)
        assert path == _named_schema_context_path(engine_dir, "team_a")
        loaded = load_named_schema_context(engine_dir, "team_a")
        assert loaded == ctx
        assert list_named_schema_context_names(engine_dir) == ("team_a",)

    def test_export_master_and_named(self, tmp_path: Path) -> None:
        engine_dir = str(tmp_path)
        master = _master_context(deny=frozenset({"orders.secret"}))
        save_named_schema_context(
            engine_dir,
            "team_a",
            EngineContext(allow_objects=frozenset({"orders"})),
        )
        master_path = export_named_schema_context_json(engine_dir, "master", master)
        named_path = export_named_schema_context_json(engine_dir, "team_a", master)
        assert master_path.is_file()
        assert named_path.is_file()
        payload = json.loads(named_path.read_text(encoding="utf-8"))
        assert payload["name"] == "team_a"
        assert "version" in payload


class TestResolveEngineContextPlan:
    def test_consumer_object_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OwnerOnlyOperationError):
            resolve_engine_context_plan(
                EngineContext(allow_objects=frozenset({"orders"})),
                str(tmp_path),
                schema_role="consumer",
                load_master=_master_context(),
                prepare_master=None,
            )

    def test_str_unknown_raises(self, tmp_path: Path) -> None:
        master = _master_context()
        write_schema_context_cache(str(tmp_path), master)
        with pytest.raises(ConfigError, match="unknown engine context"):
            resolve_engine_context_plan(
                "missing",
                str(tmp_path),
                schema_role="owner",
                load_master=master,
                prepare_master=None,
            )

    def test_str_loads_named(self, tmp_path: Path) -> None:
        master = _master_context(allow=frozenset({"orders", "customers"}))
        write_schema_context_cache(str(tmp_path), master)
        named_spec = EngineContext(allow_objects=frozenset({"orders"}))
        save_named_schema_context(str(tmp_path), "team_a", named_spec)
        m, active, name = resolve_engine_context_plan(
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
        m, active, name = resolve_engine_context_plan(
            master,
            str(tmp_path),
            schema_role="owner",
            load_master=None,
            prepare_master=master,
        )
        assert m == master
        assert active == master
        assert name == "master"


class TestSpaceSubsetOfContext:
    def test_rejects_space_table_outside_context(self) -> None:
        master = _master_context(allow=frozenset({"orders", "customers"}))
        eff = _effective_execution_context(
            master,
            EngineContext(allow_objects=frozenset({"orders"})),
            "team_a",
        )
        with pytest.raises(ConfigError, match="exceed the active engine context"):
            validate_space_subset_of_execution_context(
                frozenset({"customers"}),
                frozenset(),
                eff,
                _sample_graph(),
            )
