"""Tests for aetherspace scoping, persistence, and management guardrails."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import ConfigError
from aetherdialect._contracts_base import (
    EngineContext,
    NormalizedExpr,
    SpaceContext,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._main_execution import (
    apply_structural_migration_to_aetherspace_snapshots,
    save_aetherspace_snapshot,
    subset_graph_for_space,
    validate_space_subset_of_execution_context,
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
        filters_param=[],
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
        filters_param=[],
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
                filters_param=[],
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

    def test_export_aetherengine_round_trip(self, tmp_path) -> None:
        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path), cleanup_artifacts=False) as sb:
            path = sb.engine.export_aetherengine("master")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["name"] == "master"

    def test_empty_space_after_migration_raises(self, tmp_path) -> None:
        with AetherEngine.offline_sandbox(artifacts_dir=str(tmp_path), cleanup_artifacts=False) as sb:
            engine_dir = str(sb.engine._artifacts_dir)
            snap = subset_graph_for_space(
                sb.engine._schema_graph,
                SpaceContext(tables=frozenset({"film"})),
            )
            save_aetherspace_snapshot(engine_dir, "orphan", snap)
            apply_structural_migration_to_aetherspace_snapshots(
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
                validate_space_subset_of_execution_context(
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
