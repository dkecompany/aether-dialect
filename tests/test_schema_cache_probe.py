"""Tests for the DDL probe + cache-hit fast path in ``build_schema_graph``."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect import _schema as schema_mod
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import SchemaContext, SchemaGraph
from aetherdialect._core_utils import read_gzip_json
from aetherdialect._dialect import Dialect
from aetherdialect._schema import (
    _save_schema_to_cache,
    _sql_file_content_sha256,
    assign_schema_graph_hashes,
    build_schema_graph,
    compute_dialect_probe,
)


class _ProbeStubDialect(Dialect):
    """Minimal Dialect subclass with controllable probe + reflection hooks for tests."""

    name = "stub"

    def __init__(
        self,
        probe_value: str = "",
        reflect_result: SchemaGraph | None = None,
        raise_in_probe: bool = False,
    ) -> None:
        super().__init__(MagicMock())
        self._probe_value = probe_value
        self._reflect_result = reflect_result
        self._raise_in_probe = raise_in_probe
        self.reflect_calls = 0
        self.profile_calls = 0
        self.probe_calls = 0

    def compute_ddl_probe(self, schema_context: SchemaContext) -> str:
        self.probe_calls += 1
        if self._raise_in_probe:
            raise RuntimeError("boom")
        return self._probe_value

    def reflect_schema_graph(self, *, include: Any = "tables", allow_objects: Any = None) -> SchemaGraph:
        self.reflect_calls += 1
        if self._reflect_result is None:
            raise AssertionError("reflect_schema_graph should not be called in this test")
        return self._reflect_result

    def profile_schema(self, sg: SchemaGraph) -> None:
        self.profile_calls += 1

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Redirect ``EngineConfig.SCHEMA_JSON_PATH`` to a temp gz file for the test."""

    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _save_with_probe(
    sg: SchemaGraph,
    schema_context: SchemaContext,
    notes_content: str,
    probe_hash: str,
    cache_path: str,
) -> None:
    """Stamp hashes + probe on *sg* and persist to *cache_path*."""

    sg.notes_sha256 = hashlib.sha256((notes_content or "").encode("utf-8")).hexdigest()
    assign_schema_graph_hashes(sg, schema_context, sg.notes_sha256)
    sg.ddl_probe_hash = probe_hash
    _save_schema_to_cache(sg, cache_path)


def test_ddl_probe_hash_round_trips_to_dict(schema_graph: SchemaGraph) -> None:
    schema_graph.ddl_probe_hash = "abc123"
    d = schema_graph.to_dict()
    assert d["ddl_probe_hash"] == "abc123"
    rebuilt = SchemaGraph.from_dict(d)
    assert rebuilt.ddl_probe_hash == "abc123"


def test_ddl_probe_hash_persists_in_cache_file(schema_graph: SchemaGraph, cache_path: str) -> None:
    ctx = SchemaContext()
    _save_with_probe(
        schema_graph,
        ctx,
        notes_content="",
        probe_hash="deadbeef",
        cache_path=cache_path,
    )
    raw = read_gzip_json(cache_path)
    assert raw["ddl_probe_hash"] == "deadbeef"


def test_compute_dialect_probe_changes_with_sql_file_content(
    tmp_path: Any,
) -> None:
    f1 = tmp_path / "a.sql"
    f2 = tmp_path / "b.sql"
    f1.write_text("SELECT 1;", encoding="utf-8")
    f2.write_text("SELECT 2;", encoding="utf-8")
    dialect = _ProbeStubDialect(probe_value="DIALECT_DIGEST")
    ctx1 = SchemaContext(sql_file=str(f1))
    ctx2 = SchemaContext(sql_file=str(f2))
    p1 = compute_dialect_probe(dialect, ctx1)
    p2 = compute_dialect_probe(dialect, ctx2)
    assert p1 and p2
    assert p1 != p2


def test_compute_dialect_probe_returns_empty_when_dialect_returns_empty() -> None:
    dialect = _ProbeStubDialect(probe_value="")
    assert compute_dialect_probe(dialect, SchemaContext()) == ""


def test_compute_dialect_probe_swallows_dialect_exception() -> None:
    dialect = _ProbeStubDialect(raise_in_probe=True)
    assert compute_dialect_probe(dialect, SchemaContext()) == ""


def test_sql_file_content_sha256_missing_file_returns_empty_digest(
    tmp_path: Any,
) -> None:
    empty = hashlib.sha256(b"").hexdigest()
    assert _sql_file_content_sha256(None) == empty
    assert _sql_file_content_sha256(str(tmp_path / "nope.sql")) == empty


def test_cache_hit_via_probe_skips_reflect_and_profile(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SchemaContext()
    dialect = _ProbeStubDialect(probe_value="DIALECT_DIGEST")
    combined_probe = compute_dialect_probe(dialect, ctx)
    _save_with_probe(
        schema_graph,
        ctx,
        notes_content="notes-A",
        probe_hash=combined_probe,
        cache_path=cache_path,
    )

    classify_calls = []
    monkeypatch.setattr(
        schema_mod,
        "apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )

    out = build_schema_graph(dialect, ctx, notes_content="notes-A")

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    assert classify_calls == []
    assert out.ddl_probe_hash == combined_probe
    assert len(out.tables) == len(schema_graph.tables)


def test_notes_only_refresh_reruns_classifier_only(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SchemaContext()
    dialect = _ProbeStubDialect(probe_value="DIALECT_DIGEST")
    combined_probe = compute_dialect_probe(dialect, ctx)
    _save_with_probe(
        schema_graph,
        ctx,
        notes_content="notes-OLD",
        probe_hash=combined_probe,
        cache_path=cache_path,
    )

    classify_calls: list[str | None] = []
    boolean_calls: list[int] = []
    monkeypatch.setattr(
        schema_mod,
        "apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )
    monkeypatch.setattr(
        schema_mod,
        "apply_boolean_coercion_pass",
        lambda sg: boolean_calls.append(1),
    )

    out = build_schema_graph(dialect, ctx, notes_content="notes-NEW")

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    assert classify_calls == ["notes-NEW"]
    assert boolean_calls == [1]
    new_notes_hash = hashlib.sha256(b"notes-NEW").hexdigest()
    assert out.notes_sha256 == new_notes_hash

    raw = read_gzip_json(cache_path)
    assert raw["notes_hash"] == new_notes_hash
    assert raw["ddl_probe_hash"] == combined_probe


def test_probe_match_with_scope_mismatch_does_not_take_fast_path(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When probe matches but scope is *orthogonal* (neither subset nor superset), legacy validation runs."""

    ctx_saved = SchemaContext(deny_columns=frozenset({"customers.email"}))
    dialect = _ProbeStubDialect(probe_value="DIALECT_DIGEST")
    combined_probe = compute_dialect_probe(dialect, ctx_saved)
    _save_with_probe(
        schema_graph,
        ctx_saved,
        notes_content="n",
        probe_hash=combined_probe,
        cache_path=cache_path,
    )

    classify_calls: list[Any] = []
    monkeypatch.setattr(
        schema_mod,
        "apply_column_roles_llm",
        lambda sg, notes_content=None, **kwargs: classify_calls.append(notes_content),
    )

    ctx_new = SchemaContext(deny_columns=frozenset({"orders.status"}))
    with pytest.raises(AssertionError):
        build_schema_graph(dialect, ctx_new, notes_content="n")
    assert classify_calls == []


def test_legacy_cache_without_probe_falls_to_legacy_branch_and_backfills(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SchemaContext()
    _save_with_probe(schema_graph, ctx, notes_content="n", probe_hash="", cache_path=cache_path)
    raw_before = read_gzip_json(cache_path)
    assert raw_before["ddl_probe_hash"] == ""

    dialect = _ProbeStubDialect(probe_value="DIALECT_DIGEST")
    combined_probe = compute_dialect_probe(dialect, ctx)

    out = build_schema_graph(dialect, ctx, notes_content="n")

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    assert out.ddl_probe_hash == combined_probe
    raw_after = read_gzip_json(cache_path)
    assert raw_after["ddl_probe_hash"] == combined_probe


def test_dialect_with_empty_probe_uses_legacy_branch(
    schema_graph: SchemaGraph,
    cache_path: str,
) -> None:
    ctx = SchemaContext()
    _save_with_probe(schema_graph, ctx, notes_content="n", probe_hash="", cache_path=cache_path)
    dialect = _ProbeStubDialect(probe_value="")

    out = build_schema_graph(dialect, ctx, notes_content="n")

    assert dialect.reflect_calls == 0
    assert dialect.profile_calls == 0
    assert out.ddl_probe_hash == ""
    raw_after = read_gzip_json(cache_path)
    assert raw_after["ddl_probe_hash"] == ""


def test_base_dialect_compute_ddl_probe_returns_empty(
    schema_graph: SchemaGraph,
) -> None:
    """The base ``Dialect.compute_ddl_probe`` ABC default returns an empty string."""

    class _Bare(Dialect):
        name = "bare"

        def __init__(self) -> None:
            super().__init__(MagicMock())

        def reflect_schema_graph(self, *, include: Any = "tables", allow_objects: Any = None) -> SchemaGraph:
            raise NotImplementedError

        def profile_schema(self, sg: SchemaGraph) -> None:
            raise NotImplementedError

        def ast_validate(self, sql: str) -> tuple[bool, str]:
            return True, ""

        def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
            return True, ""

    d = _Bare()
    assert d.compute_ddl_probe(SchemaContext()) == ""
