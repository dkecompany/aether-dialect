"""Denied relations must be excluded before reflection and profiling."""

from __future__ import annotations

import copy
import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

import aetherdialect._schema_overrides
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import apply_deny_objects_filter, assign_schema_graph_hashes
from aetherdialect._schema_overrides import build_schema_graph, save_schema_to_cache

pytestmark = [pytest.mark.fast, pytest.mark.usefixtures("stub_schema_llm_classifier")]


def _mk_col(name: str, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _mk_table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": _mk_col("id")},
        primary_key=["id"],
        foreign_keys=[],
    )


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _save_with_probe(
    sg: SchemaGraph,
    engine_context: EngineContext,
    notes_content: str,
    probe_hash: str,
    cache_path: str,
) -> None:
    sg.notes_sha256 = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest()
    assign_schema_graph_hashes(sg, engine_context, sg.notes_sha256)
    sg.ddl_probe_hash = probe_hash
    save_schema_to_cache(sg, cache_path)


class _FullRebuildStubDialect(Dialect):
    name = "stub"

    def __init__(self, reflected_sg: SchemaGraph) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg
        self.reflect_calls = 0
        self.reflect_deny_objects: frozenset[str] | None = None
        self.profiled_tables: list[tuple[str, ...]] = []
        self.build_events: list[str] = []

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return ""

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        self.reflect_calls += 1
        self.build_events.append("reflect")
        self.reflect_deny_objects = frozenset(deny_objects or ())
        reflected = copy.deepcopy(self._reflected)
        if deny_objects:
            deny = {str(x).lower() for x in deny_objects}
            for name in list(reflected.tables.keys()):
                if name.lower() in deny:
                    reflected.tables.pop(name, None)
        return reflected

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.build_events.append("profile")
        self.profiled_tables.append(tuple(sorted(sg.tables.keys())))

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


class _PartialRebuildStubDialect(Dialect):
    name = "stub"

    def __init__(self, reflected_sg: SchemaGraph, probe_value: str) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg
        self._probe_value = probe_value
        self.reflect_only_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe_value

    def reflect_only(self, engine_context: EngineContext) -> SchemaGraph:
        self.reflect_only_calls += 1
        return copy.deepcopy(self._reflected)

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        raise AssertionError("reflect_schema_graph should not run on partial-rebuild path")

    def profile_schema(self, sg: SchemaGraph) -> None:
        for t in sg.tables.values():
            t.row_count = max(t.row_count, 1)

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.mark.fast
def test_full_rebuild_passes_deny_objects_into_reflection(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = copy.deepcopy(schema_graph)
    template.tables["secret"] = _mk_table("secret")
    dialect = _FullRebuildStubDialect(template)
    ctx = EngineContext(deny_objects=frozenset({"secret"}))

    out = build_schema_graph(dialect, ctx, notes_content="n")

    assert dialect.reflect_calls == 1
    assert dialect.reflect_deny_objects == frozenset({"secret"})
    assert "secret" not in out.tables


@pytest.mark.fast
def test_full_rebuild_does_not_profile_denied_objects(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = copy.deepcopy(schema_graph)
    template.tables["secret"] = _mk_table("secret")
    dialect = _FullRebuildStubDialect(template)
    ctx = EngineContext(deny_objects=frozenset({"secret"}))

    build_schema_graph(dialect, ctx, notes_content="n")

    assert dialect.profiled_tables, "profile_schema must run on full rebuild"
    assert "secret" not in dialect.profiled_tables[0]


@pytest.mark.fast
def test_full_rebuild_deny_filter_runs_before_profiling(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = copy.deepcopy(schema_graph)
    template.tables["secret"] = _mk_table("secret")
    dialect = _FullRebuildStubDialect(template)
    ctx = EngineContext(deny_objects=frozenset({"secret"}))

    phase_events: list[str] = []
    real_apply_deny = apply_deny_objects_filter
    real_add_profiling = aetherdialect._schema_overrides._add_profiling_data

    def _track_profile(*args: Any, **kwargs: Any) -> None:
        phase_events.append("profile")
        return real_add_profiling(*args, **kwargs)

    def _track_deny(sg: SchemaGraph, engine_context: EngineContext) -> None:
        phase_events.append("deny")
        return real_apply_deny(sg, engine_context)

    monkeypatch.setattr(aetherdialect._schema_overrides, "_add_profiling_data", _track_profile)
    monkeypatch.setattr(aetherdialect._schema_overrides, "apply_deny_objects_filter", _track_deny)

    build_schema_graph(dialect, ctx, notes_content="n")

    assert phase_events.count("profile") == 1
    assert phase_events.count("deny") == 1
    assert phase_events.index("deny") < phase_events.index("profile")


@pytest.mark.fast
def test_partial_probe_path_applies_deny_objects_after_reflect(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = EngineContext(deny_objects=frozenset({"products"}))
    _save_with_probe(schema_graph, ctx, "notes", probe_hash="probe-OLD", cache_path=cache_path)

    new_struct = copy.deepcopy(schema_graph)
    new_struct.tables["customers"].columns["probe_col"] = _mk_col("probe_col", "varchar")
    dialect = _PartialRebuildStubDialect(new_struct, probe_value="probe-NEW")

    deny_calls: list[tuple[str, ...]] = []
    real_apply_deny = apply_deny_objects_filter

    def _track_deny(sg: SchemaGraph, engine_context: EngineContext) -> None:
        deny_calls.append(tuple(sorted(sg.tables.keys())))
        real_apply_deny(sg, engine_context)

    monkeypatch.setattr(aetherdialect._schema_overrides, "apply_deny_objects_filter", _track_deny)

    out = build_schema_graph(dialect, ctx, notes_content="notes")

    assert dialect.reflect_only_calls == 1
    assert len(deny_calls) == 1, "partial probe rebuild must apply deny_objects after reflection"
    assert "products" in deny_calls[0], "deny filter must see reflected objects before removal"
    assert "products" not in out.tables
    assert "probe_col" in out.tables["customers"].columns
