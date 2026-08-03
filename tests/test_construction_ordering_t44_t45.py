"""Schema construction must finalize overrides before hashing and must not stamp migration tier early."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

import aetherdialect._schema_overrides
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import EngineContext, MigrationTier
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._core_utils import read_artifact_manifest, write_artifact_manifest
from aetherdialect._dialect import Dialect
from aetherdialect._schema_graph import assign_schema_graph_hashes, compute_dialect_probe
from aetherdialect._schema_overrides import save_schema_to_cache

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _ProbeCacheHitDialect(Dialect):
    """Dialect whose DDL probe matches a pre-seeded cache (probe cache- hit path)."""

    name = "stub"

    def __init__(self, *, probe: str) -> None:
        super().__init__(MagicMock())
        self._probe = probe

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return self._probe

    def reflect_schema_graph(self, **kwargs: Any) -> SchemaGraph:
        raise AssertionError("reflect_schema_graph should not run on cache hit")

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


class _FingerprintCacheHitDialect(Dialect):
    """Dialect with no DDL probe so the legacy fingerprint cache-hit path is used."""

    name = "stub"

    def __init__(self) -> None:
        super().__init__(MagicMock())

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return ""

    def reflect_only(self, schema_context: EngineContext) -> SchemaGraph:
        raise AssertionError("reflect_only should not run on fingerprint cache hit")

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params: Any = None, **kwargs: Any) -> tuple[bool, str]:
        return True, ""


@pytest.fixture
def cache_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    p = str(tmp_path / "schema_graph.json.gz")
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", p)
    return p


def _patch_build_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aetherdialect._schema_overrides, "notify_schema_path_health", lambda sg: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "validate_scope_against_graph", lambda sg, c: None)
    monkeypatch.setattr(aetherdialect._schema_overrides, "raise_if_schema_unusable", lambda sg, c: None)


def _save_probe_cache(schema_graph: SchemaGraph, cache_path: str, *, notes: str, probe: str) -> None:
    ctx = EngineContext()
    schema_graph.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)
    schema_graph.ddl_probe_hash = probe
    save_schema_to_cache(schema_graph, cache_path)


def _save_fingerprint_cache(schema_graph: SchemaGraph, cache_path: str, *, notes: str) -> None:
    ctx = EngineContext()
    schema_graph.notes_sha256 = hashlib.sha256(notes.encode()).hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)
    schema_graph.ddl_probe_hash = ""
    save_schema_to_cache(schema_graph, cache_path)


def _install_call_order_tracker(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    order: list[str] = []
    real_assign = aetherdialect._schema_overrides.assign_schema_graph_hashes
    real_finalize = aetherdialect._schema_overrides.finalize_with_overrides

    def _track_assign(*args: Any, **kwargs: Any) -> None:
        order.append("assign_schema_graph_hashes")
        return real_assign(*args, **kwargs)

    def _track_finalize(*args: Any, **kwargs: Any) -> bool:
        order.append("finalize_with_overrides")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(aetherdialect._schema_overrides, "assign_schema_graph_hashes", _track_assign)
    monkeypatch.setattr(aetherdialect._schema_overrides, "finalize_with_overrides", _track_finalize)
    return order


def _assert_finalize_before_assign(order: list[str]) -> None:
    assign_positions = [i for i, name in enumerate(order) if name == "assign_schema_graph_hashes"]
    finalize_positions = [i for i, name in enumerate(order) if name == "finalize_with_overrides"]
    assert finalize_positions, f"expected finalize_with_overrides in call order, got {order!r}"
    assert assign_positions, f"expected assign_schema_graph_hashes in call order, got {order!r}"
    assert min(finalize_positions) < min(assign_positions), (
        "finalize_with_overrides must run before assign_schema_graph_hashes; got "
        f"{order!r}"
    )


@pytest.mark.fast
def test_probe_cache_hit_finalizes_overrides_before_hash_assignment(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T44: probe cache-hit path must hash only after override replay."""
    notes = ""
    ctx = EngineContext()
    probe = compute_dialect_probe(_ProbeCacheHitDialect(probe="probe_seed"), ctx)
    _save_probe_cache(schema_graph, cache_path, notes=notes, probe=probe)
    _patch_build_tail(monkeypatch)
    order = _install_call_order_tracker(monkeypatch)

    dialect = _ProbeCacheHitDialect(probe=probe)
    aetherdialect._schema_overrides.build_schema_graph_with_diff(dialect, ctx, notes_content=notes)

    _assert_finalize_before_assign(order)


@pytest.mark.fast
def test_fingerprint_cache_hit_finalizes_overrides_before_hash_assignment(
    schema_graph: SchemaGraph,
    cache_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T44: fingerprint self-check cache-hit path must hash only after override replay."""
    notes = ""
    ctx = EngineContext()
    _save_fingerprint_cache(schema_graph, cache_path, notes=notes)
    _patch_build_tail(monkeypatch)
    order = _install_call_order_tracker(monkeypatch)

    dialect = _FingerprintCacheHitDialect()
    aetherdialect._schema_overrides.build_schema_graph_with_diff(dialect, ctx, notes_content=notes)

    _assert_finalize_before_assign(order)


@pytest.mark.fast
def test_save_schema_to_cache_preserves_migration_tier_until_policy_runs(
    schema_graph: SchemaGraph,
    tmp_path: Any,
) -> None:
    """T45: save_schema_to_cache must not claim NO_CHANGE before apply_migration_policy."""
    cache_path = str(tmp_path / "schema_graph.json.gz")
    artifacts_dir = str(tmp_path)
    ctx = EngineContext()
    schema_graph.notes_sha256 = hashlib.sha256(b"").hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)

    write_artifact_manifest(
        artifacts_dir,
        structural_hash=schema_graph.structural_hash,
        profiling_hash=schema_graph.profiling_hash,
        scope_hash=schema_graph.scope_hash,
        effective_structural_hash=schema_graph.effective_structural_hash,
        schema_graph_id=schema_graph.schema_graph_id,
        notes_hash=schema_graph.notes_hash,
        semantic_edges_hash=schema_graph.semantic_edges_hash,
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )

    save_schema_to_cache(schema_graph, cache_path)

    manifest = read_artifact_manifest(artifacts_dir)
    assert manifest is not None
    assert manifest.last_migration_tier != MigrationTier.NO_CHANGE.value
    assert manifest.last_migration_tier == MigrationTier.SOFT_REFRESH.value


@pytest.mark.fast
def test_fresh_save_schema_to_cache_leaves_migration_tier_unset(
    schema_graph: SchemaGraph,
    tmp_path: Any,
) -> None:
    """T45: first cache write must not pre-stamp NO_CHANGE on an empty manifest."""
    cache_path = str(tmp_path / "schema_graph.json.gz")
    artifacts_dir = str(tmp_path)
    ctx = EngineContext()
    schema_graph.notes_sha256 = hashlib.sha256(b"").hexdigest()
    assign_schema_graph_hashes(schema_graph, ctx, schema_graph.notes_sha256)

    save_schema_to_cache(schema_graph, cache_path)

    manifest = read_artifact_manifest(artifacts_dir)
    assert manifest is not None
    assert manifest.last_migration_tier != MigrationTier.NO_CHANGE.value
    assert manifest.last_migration_tier == ""
