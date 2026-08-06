"""Tests for aetherspace scoping, persistence, and management guardrails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import AETHERSPACE_ARTIFACT_VERSION
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    NormalizedExpr,
    SpaceContext,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._main_execution import (
    MainExecutionOps,
)
from aetherdialect._pipeline import generate_and_validate_sql
from aetherdialect._schema_graph import assert_intent_in_scope


def _film_only_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.rating"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _film_with_customer_join_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["film", "customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("film.rating")),
            SelectCol(expr=NormalizedExpr.from_column("customer.email")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


class TestAetherspaceScoping:
    def test_space_limits_select_columns(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            graph = sb.engine._schema_graph
            sb.engine.aetherspace(
                "films_only",
                SpaceContext(tables=frozenset({"film"}), columns=frozenset({"film.rating"})),
            )
            intent = _film_with_customer_join_intent()
            allowed_tables, allowed_columns = sb.engine._resolve_aetherspace("films_only")[1:3]
            assert not assert_intent_in_scope(intent, allowed_tables, allowed_columns, graph)

    def test_in_space_select_passes_when_join_tables_out_of_space(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            graph = sb.engine._schema_graph
            sb.engine.aetherspace("films_only", SpaceContext(tables=frozenset({"film"})))
            intent = RuntimeIntent(
                tables=["film"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.rating"))],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            )
            allowed_tables, allowed_columns = sb.engine._resolve_aetherspace("films_only")[1:3]
            assert assert_intent_in_scope(intent, allowed_tables, allowed_columns, graph)

    def test_generate_and_validate_sql_rejects_out_of_space_select(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            graph = sb.engine._schema_graph
            store = sb.engine._store
            dialect = sb.engine._dialect
            sb.engine.aetherspace("films_only", SpaceContext(tables=frozenset({"film"})))
            _, space_tables, space_columns, _, _ = sb.engine._resolve_aetherspace("films_only")
            out = generate_and_validate_sql(
                "customer email with film rating",
                _film_with_customer_join_intent(),
                graph,
                {"candidates": []},
                {"J00": []},
                dialect,
                store,
                space_allowed_tables=space_tables,
                space_allowed_columns=space_columns,
            )
            assert out.success is False
            assert "aetherspace" in (out.sql_validation_error or "").lower()


class TestAetherspacePersistence:
    def test_export_aetherspace_round_trip(self, tmp_path) -> None:
        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path), cleanup_artifacts=False) as sb:
            sb.engine.aetherspace("films_only", SpaceContext(tables=frozenset({"film"})))
            path = sb.engine.export_aetherspace("films_only")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["name"] == "films_only"
            assert "film" in payload["tables"]

    def test_export_context_round_trip(self, tmp_path) -> None:
        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path), cleanup_artifacts=False) as sb:
            path = sb.engine.export_context("master")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["name"] == "master"

    def test_empty_space_after_migration_raises(self, tmp_path) -> None:
        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path), cleanup_artifacts=False) as sb:
            engine_dir = str(sb.engine._artifacts_dir)
            snap = MainExecutionOps.subset_graph_for_space(
                sb.engine._schema_graph,
                SpaceContext(tables=frozenset({"film"})),
            )
            MainExecutionOps.save_aetherspace_snapshot(engine_dir, "orphan", snap)
            MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
                engine_dir,
                dropped_tables=("film",),
            )
            with pytest.raises(ConfigError, match="space empty after schema migration"):
                sb.engine._resolve_aetherspace("orphan")


class TestRunInteractiveSpaceValidation:
    def test_session_rejects_space_outside_engine_context(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            sb.engine.aetherspace("wide", SpaceContext(tables=frozenset({"film", "customer"})))
            sb.engine._runtime_config = replace(
                sb.engine._runtime_config,
                execution_context=EngineContext(allow_objects=frozenset({"film"})),
            )
            with pytest.raises(ConfigError, match="exceed the active engine context"):
                with sb.engine.session(space="wide"):
                    pass

    def test_validate_space_subset_matches_run_interactive_guard(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            sb.engine.aetherspace("wide", SpaceContext(tables=frozenset({"film", "customer"})))
            _, space_tables, space_columns, _, _ = sb.engine._resolve_aetherspace("wide")
            exec_ctx = EngineContext(allow_objects=frozenset({"film"}))
            with pytest.raises(ConfigError, match="exceed the active engine context"):
                MainExecutionOps.validate_space_subset_of_execution_context(
                    space_tables,
                    space_columns,
                    exec_ctx,
                    sb.engine._schema_graph,
                )

    def test_session_unknown_space_raises(self) -> None:
        with AetherEngine.offline_sandbox() as sb:
            with pytest.raises(ConfigError, match="unknown aetherspace"):
                with sb.engine.session(space="missing_space"):
                    pass


class TestSpaceContextNotesFileSandbox:
    @staticmethod
    def _catalog_notes_path() -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "data" / "sandbox_space_catalog_notes.txt"

    @pytest.mark.fast
    def test_notes_file_round_trips_into_aetherspace_notes(self, tmp_path: Path) -> None:
        notes = self._catalog_notes_path()
        from aetherdialect._constants import SANDBOX_CATALOG_SPACE_TABLES

        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path / "arts"), cleanup_artifacts=False) as sb:
            desc = sb.engine.aetherspace(
                "catalog_notes",
                SpaceContext(tables=SANDBOX_CATALOG_SPACE_TABLES, notes_file=str(notes)),
            )
            assert desc.notes is not None
            assert "catalog inventory" in desc.notes.lower()
            resolved = sb.engine.aetherspace("catalog_notes")
            assert resolved.notes == desc.notes

    @pytest.mark.fast
    def test_notes_hash_in_snapshot_detects_notes_change(self, tmp_path: Path) -> None:
        notes = tmp_path / "catalog_notes_copy.txt"
        source = self._catalog_notes_path().read_text(encoding="utf-8")
        notes.write_text(source, encoding="utf-8")
        expected_first = hashlib.sha256(source.encode("utf-8")).hexdigest()
        from aetherdialect._constants import SANDBOX_CATALOG_SPACE_TABLES

        arts = tmp_path / "arts"
        with AetherEngine.offline_sandbox(artifacts_dir=str(arts), cleanup_artifacts=False) as sb:
            sb.engine.aetherspace(
                "catalog_notes",
                SpaceContext(tables=SANDBOX_CATALOG_SPACE_TABLES, notes_file=str(notes)),
            )
            snap = MainExecutionOps.load_aetherspace_snapshot(str(sb.engine._artifacts_dir), "catalog_notes")
            assert snap is not None
            assert snap["version"] == AETHERSPACE_ARTIFACT_VERSION
            assert snap["notes_hash"] == expected_first
            assert snap["notes"] is not None
            changed = source + "\nextra line for hash change\n"
            notes.write_text(changed, encoding="utf-8")
            expected_second = hashlib.sha256(changed.encode("utf-8")).hexdigest()
            assert expected_second != expected_first
            assert snap["notes_hash"] != expected_second
            with pytest.raises(ConfigError, match="custom notes"):
                sb.engine.aetherspace(
                    "catalog_notes",
                    SpaceContext(tables=SANDBOX_CATALOG_SPACE_TABLES, notes_file=str(notes)),
                )

    @pytest.mark.fast
    def test_defining_space_with_notes_does_not_rebuild_schema(self, tmp_path: Path) -> None:
        notes = self._catalog_notes_path()
        from aetherdialect._constants import SANDBOX_CATALOG_SPACE_TABLES

        arts = tmp_path / "arts"
        with AetherEngine.offline_sandbox(artifacts_dir=str(arts), cleanup_artifacts=False) as sb:
            engine_dir = Path(str(sb.engine._artifacts_dir))
            schema_path = engine_dir / "schema_graph.json.gz"
            assert schema_path.is_file()
            before_mtime = schema_path.stat().st_mtime_ns
            before_size = schema_path.stat().st_size
            before_structural = str(sb.engine._schema_graph.effective_structural_hash or "")
            before_notes = str(
                getattr(sb.engine._schema_graph, "notes_sha256", None)
                or getattr(sb.engine._schema_graph, "notes_hash", None)
                or ""
            )
            before_graph_id = id(sb.engine._schema_graph)
            sb.engine.aetherspace(
                "catalog_notes",
                SpaceContext(tables=SANDBOX_CATALOG_SPACE_TABLES, notes_file=str(notes)),
            )
            assert id(sb.engine._schema_graph) == before_graph_id
            assert str(sb.engine._schema_graph.effective_structural_hash or "") == before_structural
            after_notes = str(
                getattr(sb.engine._schema_graph, "notes_sha256", None)
                or getattr(sb.engine._schema_graph, "notes_hash", None)
                or ""
            )
            assert after_notes == before_notes
            assert schema_path.stat().st_mtime_ns == before_mtime
            assert schema_path.stat().st_size == before_size
