"""Tests for classification + subset fast path."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest

import aetherdialect._schema_finalize
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import ConfigError, EngineContext, FederationContext
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect import Dialect
from aetherdialect._schema_finalize import build_schema_graph
from aetherdialect._schema_graph import (
    assign_schema_graph_hashes,
    classify_scope_change,
    compute_dialect_probe,
    filter_schema_graph_by_scope,
    schema_context_from_descriptor,
    scope_descriptor_for,
)
from aetherdialect._schema_reflect import save_schema_to_cache
from aetherdialect._utils_artifacts import read_gzip_json

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _ProbeStubDialect(Dialect):
    name = "stub"

    def __init__(self, probe_value: str = "DIALECT_DIGEST") -> None:
        super().__init__(MagicMock())
        self._probe_value = probe_value
        self.reflect_calls = 0
        self.profile_calls = 0

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe_value

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        self.reflect_calls += 1
        raise AssertionError("reflect_schema_graph should not run in this test")

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.profile_calls += 1

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _save_with_probe(
    sg: SchemaGraph,
    ctx: EngineContext,
    notes_content: str,
    probe_hash: str,
    cache_path: str,
) -> None:
    sg.notes_sha256 = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest()
    assign_schema_graph_hashes(sg, ctx, sg.notes_sha256)
    sg.ddl_probe_hash = probe_hash
    save_schema_to_cache(sg, cache_path)


def test_classify_identical_when_all_fields_match() -> None:
    a = EngineContext(deny_columns=frozenset({"customers.email"}))
    b = EngineContext(deny_columns=frozenset({"customers.email"}))
    assert classify_scope_change(a, b) == "identical"


def test_classify_subset_when_new_adds_deny() -> None:
    old = EngineContext()
    new = EngineContext(deny_columns=frozenset({"customers.email"}))
    assert classify_scope_change(old, new) == "subset"


def test_classify_superset_when_new_removes_deny() -> None:
    old = EngineContext(deny_columns=frozenset({"customers.email"}))
    new = EngineContext()
    assert classify_scope_change(old, new) == "superset"


def test_classify_subset_when_new_narrows_allow_objects() -> None:
    old = EngineContext(allow_objects=frozenset({"a", "b", "c"}))
    new = EngineContext(allow_objects=frozenset({"a", "b"}))
    assert classify_scope_change(old, new) == "subset"


def test_classify_superset_when_old_universal_new_universal_deny_dropped() -> None:
    old = EngineContext(deny_columns=frozenset({"orders.status", "customers.email"}))
    new = EngineContext(deny_columns=frozenset({"orders.status"}))
    assert classify_scope_change(old, new) == "superset"


def test_classify_orthogonal_when_different_denies() -> None:
    old = EngineContext(deny_columns=frozenset({"customers.email"}))
    new = EngineContext(deny_columns=frozenset({"orders.status"}))
    assert classify_scope_change(old, new) == "orthogonal"


def test_engine_context_rejects_include_both() -> None:
    with pytest.raises(ConfigError, match="include must be 'tables' or 'views'"):
        EngineContext(include="both")


def test_federation_context_rejects_include_both() -> None:
    with pytest.raises(ConfigError, match="include must be 'tables' or 'views'"):
        FederationContext(include="both")


def test_classify_orthogonal_include_tables_vs_views() -> None:
    old = EngineContext(include="tables")
    new = EngineContext(include="views")
    assert classify_scope_change(old, new) == "orthogonal"


def test_scope_descriptor_round_trip() -> None:
    ctx = EngineContext(
        allow_objects=frozenset({"customers", "orders"}),
        deny_columns=frozenset({"products.price"}),
        allow_columns=frozenset({"*.name"}),
        include="views",
    )
    desc = scope_descriptor_for(ctx)
    rebuilt = schema_context_from_descriptor(desc)
    assert rebuilt.allow_objects == ctx.allow_objects
    assert rebuilt.deny_columns == ctx.deny_columns
    assert rebuilt.allow_columns == ctx.allow_columns
    assert rebuilt.include == ctx.include


def test_filter_drops_deny_objects(schema_graph: SchemaGraph) -> None:
    new_ctx = EngineContext(deny_objects=frozenset({"orders", "products"}))
    filtered = filter_schema_graph_by_scope(schema_graph, new_ctx)
    assert set(filtered.tables) == {"customers"}
    assert "orders" in schema_graph.tables


def test_filter_does_not_narrow_allow_objects(schema_graph: SchemaGraph) -> None:
    new_ctx = EngineContext(allow_objects=frozenset({"customers"}))
    filtered = filter_schema_graph_by_scope(schema_graph, new_ctx)
    assert set(filtered.tables) == set(schema_graph.tables)


def test_filter_strips_denied_columns(schema_graph: SchemaGraph) -> None:
    new_ctx = EngineContext(deny_columns=frozenset({"customers.email"}))
    filtered = filter_schema_graph_by_scope(schema_graph, new_ctx)
    assert "email" not in filtered.tables["customers"].columns
    assert "email" in schema_graph.tables["customers"].columns


def test_cache_subset_path_filters_in_memory_no_reflect(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx_saved = EngineContext()
    dialect = _ProbeStubDialect()
    probe = compute_dialect_probe(dialect, ctx_saved)
    _save_with_probe(schema_graph, ctx_saved, "notes", probe, cache_path)

    classify_calls: list[Any] = []
    monkeypatch.setattr(
        aetherdialect._schema_finalize,
        "apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )

    new_ctx = EngineContext(deny_columns=frozenset({"customers.email"}))
    out = build_schema_graph(dialect, new_ctx, notes_content="notes")

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    assert classify_calls == []
    assert "email" not in out.tables["customers"].columns
    raw = read_gzip_json(cache_path)
    assert raw["scope_descriptor"]["deny_columns"] == ["customers.email"]


def test_cache_subset_path_with_notes_change_reruns_classifier(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx_saved = EngineContext()
    dialect = _ProbeStubDialect()
    probe = compute_dialect_probe(dialect, ctx_saved)
    _save_with_probe(schema_graph, ctx_saved, "notes-OLD", probe, cache_path)

    classify_calls: list[Any] = []
    boolean_calls: list[int] = []
    monkeypatch.setattr(
        "aetherdialect._schema_profile.apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )
    monkeypatch.setattr(
        "aetherdialect._schema_profile.apply_boolean_coercion_pass",
        lambda sg: boolean_calls.append(1),
    )
    monkeypatch.setattr(
        "aetherdialect._schema_profile.apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )
    monkeypatch.setattr(
        "aetherdialect._schema_profile.apply_boolean_coercion_pass",
        lambda sg: boolean_calls.append(1),
    )

    new_ctx = EngineContext(deny_columns=frozenset({"customers.email"}))
    out = build_schema_graph(dialect, new_ctx, notes_content="notes-NEW")

    assert dialect.reflect_calls == 0
    assert classify_calls == ["notes-NEW"]
    assert boolean_calls == [1]
    assert out.notes_sha256 == hashlib.sha256(b"notes-NEW").hexdigest()


def test_cache_subset_via_deny_objects(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    ctx_saved = EngineContext()
    dialect = _ProbeStubDialect()
    probe = compute_dialect_probe(dialect, ctx_saved)
    _save_with_probe(schema_graph, ctx_saved, "n", probe, cache_path)

    new_ctx = EngineContext(deny_objects=frozenset({"products"}))
    out = build_schema_graph(dialect, new_ctx, notes_content="n")

    assert dialect.reflect_calls == 0
    assert set(out.tables) == {"customers", "orders"}
    raw = read_gzip_json(cache_path)
    assert sorted(raw["tables"].keys()) == ["customers", "orders"]


def test_cache_without_scope_descriptor_skips_subset_path(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    """Caches lacking ``scope_descriptor`` cannot use the subset path and fall back to fingerprint validation."""
    ctx_saved = EngineContext()
    dialect = _ProbeStubDialect()
    probe = compute_dialect_probe(dialect, ctx_saved)
    _save_with_probe(schema_graph, ctx_saved, "n", probe, cache_path)

    raw = read_gzip_json(cache_path)
    raw["scope_descriptor"] = None
    from aetherdialect._utils_artifacts import write_gzip_json_atomic

    write_gzip_json_atomic(cache_path, raw, sort_keys=True)

    new_ctx = EngineContext(deny_columns=frozenset({"customers.email"}))
    with pytest.raises(AssertionError):
        build_schema_graph(dialect, new_ctx, notes_content="n")


def test_full_rebuild_profiles_only_columns_after_scope_trim(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    if os.path.isfile(cache_path):
        os.remove(cache_path)

    template = deepcopy(schema_graph)
    profiled_snapshots: list[int] = []
    reflect_kw: list[tuple[Any, Any]] = []

    class _ReflectDialect(_ProbeStubDialect):
        def reflect_schema_graph(
            self,
            *,
            include: Any = "tables",
            allow_objects: Any = None,
            deny_objects: Any = None,
            sql_file: Any = None,
        ) -> SchemaGraph:
            self.reflect_calls += 1
            reflect_kw.append((include, allow_objects))
            return deepcopy(template)

        def profile_schema(self, sg: SchemaGraph) -> None:
            super().profile_schema(sg)
            profiled_snapshots.append(sum(len(t.columns) for t in sg.tables.values()))

    dialect = _ReflectDialect()
    ctx = EngineContext(
        allow_objects=frozenset({"customers"}),
        allow_columns=frozenset({"customers.customer_id"}),
    )
    build_schema_graph(dialect, ctx, notes_content=None)
    assert dialect.reflect_calls == 2
    assert reflect_kw == [("tables", None), ("views", None)]
    assert profiled_snapshots == [1]


def test_full_build_applies_deny_objects_filter(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = deepcopy(schema_graph)

    class _ReflectDialect(_ProbeStubDialect):
        def reflect_schema_graph(
            self,
            *,
            include: Any = "tables",
            allow_objects: Any = None,
            deny_objects: Any = None,
            sql_file: Any = None,
        ) -> SchemaGraph:
            self.reflect_calls += 1
            return deepcopy(template)

    dialect = _ReflectDialect()
    ctx = EngineContext(deny_objects=frozenset({"products"}))
    monkeypatch.setattr(
        aetherdialect._schema_finalize,
        "apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: None,
    )
    monkeypatch.setattr(
        "aetherdialect._schema_profile.apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: None,
    )
    out = build_schema_graph(dialect, ctx, notes_content="n")
    assert dialect.reflect_calls == 2
    assert "customers" in out.tables
    assert "orders" in out.tables
    assert "products" not in out.tables
