"""Sandbox production-parity gaps: stages, migration, consumer baseline, malformed fixtures, warmup guard."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._constants import CONSUMER_ALLOW_OBJECTS
from aetherdialect._contracts_base import ConfigError, MigrationPreview, RuntimeConfig
from aetherdialect._main_execution import load_runtime_config
from aetherdialect._sandbox import (
    _apply_sandbox_consumer_execution_scope,
    _baseline_dir_for_preset,
    _consumer_reader_schema_context,
    _owner_writer_schema_context,
    _sandbox_memory_engine_dir,
    Sandbox,
)

_EXPECTED_UNEXERCISED_PRODUCTION_STAGES: frozenset[str] = frozenset(
    {
        "live_reflection_and_profiling",
        "probe_mismatch_partial_rebuild",
        "cold_build_descriptions_and_classification",
        "composite_composition_replay_skip",
        "warmup_and_question_simulation",
        "model_turns_outside_recorded_fixtures",
    },
)

_STAGE_DOC_TOKENS: dict[str, tuple[str, ...]] = {
    "live_reflection_and_profiling": ("reflection", "profiling"),
    "probe_mismatch_partial_rebuild": ("probe", "mismatch"),
    "cold_build_descriptions_and_classification": ("description", "classification"),
    "composite_composition_replay_skip": ("composite", "composition"),
    "warmup_and_question_simulation": ("warmup", "qsim"),
    "model_turns_outside_recorded_fixtures": ("recorded", "fixture"),
}


def _write_minimal_graph_gz(path: Path, schema_graph_id: str) -> None:
    payload = {
        "schema_graph_id": schema_graph_id,
        "tables": {},
        "join_paths_multi": {},
        "effective_structural_hash": schema_graph_id,
        "structural_hash": schema_graph_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))


def _minimal_sandbox_host(tmp_path: Path) -> tuple[Path, str]:
    extract = tmp_path / "bundle"
    owner_dir = extract / "artifacts_baseline" / "owner"
    consumer_dir = extract / "artifacts_baseline" / "consumer"
    _write_minimal_graph_gz(owner_dir / "schema_graph.json.gz", "owner_baseline_sg")
    _write_minimal_graph_gz(consumer_dir / "schema_graph.json.gz", "consumer_baseline_sg")
    artifacts = str(tmp_path / "artifacts")
    return extract, artifacts


@pytest.mark.fast
def test_sandbox_unexercised_production_stages_are_catalogued() -> None:
    """Every skipped production stage must be named in code and docs (not silently omitted)."""
    from aetherdialect._constants import SANDBOX_UNEXERCISED_PRODUCTION_STAGES

    catalogued = frozenset(SANDBOX_UNEXERCISED_PRODUCTION_STAGES)
    assert catalogued == _EXPECTED_UNEXERCISED_PRODUCTION_STAGES

    sandbox_doc = (Path(__file__).resolve().parents[1] / "docs" / "SANDBOX.md").read_text(encoding="utf-8")
    lowered = sandbox_doc.lower()
    for stage, tokens in _STAGE_DOC_TOKENS.items():
        assert stage in catalogued
        assert all(token in lowered for token in tokens), f"{stage!r} not documented in SANDBOX.md"


@pytest.mark.fast
def test_sandbox_owner_can_preview_migration_for_user_corpus_variant() -> None:
    """Sandbox owner must exercise preview_migration_map on a declared corpus variant."""
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="duckdb"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/migration_preview",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            schema_role="owner",
            consumer_visible_objects=None,
            context_name="master",
            data_quality_report=None,
        )
        engine = AetherEngine()
    engine._sandbox_mode = True

    with patch("aetherdialect.aetherdialect.preview_schema_migration") as preview:
        preview.return_value = MigrationPreview(
            tier="compatible",
            affected_tables=(),
            affected_columns=(),
            skeleton_path="",
        )
        result = engine.preview_migration_map()
    assert isinstance(result, MigrationPreview)
    preview.assert_called_once()


@pytest.mark.fast
def test_sandbox_consumer_baseline_seeded_after_owner_uses_consumer_graph(tmp_path: Path) -> None:
    """Consumer role must seed consumer baseline even when owner baseline already exists."""
    extract, artifacts = _minimal_sandbox_host(tmp_path)
    sandbox = Sandbox.__new__(Sandbox)
    sandbox._extract_path = extract
    sandbox._artifacts_dir = artifacts
    sandbox._seed_role_baseline(role="owner", include="tables")
    sandbox._seed_role_baseline(role="consumer", include="tables")

    engine_dir = _sandbox_memory_engine_dir(artifacts)
    from aetherdialect._schema_graph import load_schema_graph_snapshot

    graph = load_schema_graph_snapshot(str(engine_dir / "schema_graph.json.gz"))
    assert graph is not None
    assert str(graph.schema_graph_id) == "consumer_baseline_sg"


@pytest.mark.fast
def test_sandbox_consumer_scope_narrowed_at_construction() -> None:
    """Consumer engine_context must be narrowed at construction, not only execution_context."""
    owner_ctx = _owner_writer_schema_context()
    consumer_ctx = _consumer_reader_schema_context()
    assert consumer_ctx.allow_objects == owner_ctx.allow_objects

    owner = MagicMock()
    consumer_ctx = _consumer_reader_schema_context()
    llm_exec = load_runtime_config(merged_env={})
    owner._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir="/tmp/sandbox_consumer_scope",
        engine_context=consumer_ctx,
        llm_execution=llm_exec,
        execution_context=consumer_ctx,
    )
    _apply_sandbox_consumer_execution_scope(owner, restricted=False)
    narrowed = frozenset(owner._runtime_config.execution_context.allow_objects or ())
    assert narrowed == CONSUMER_ALLOW_OBJECTS
    assert frozenset(owner._runtime_config.engine_context.allow_objects or ()) == CONSUMER_ALLOW_OBJECTS


@pytest.mark.fast
def test_bundled_mock_fixtures_include_malformed_repair_exercises() -> None:
    """Mock corpus must include deliberately malformed outputs that exercise repair paths."""
    from aetherdialect._constants import SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS

    questions = tuple(SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS)
    assert len(questions) >= 1
    normalized = {q.strip().lower() for q in questions}
    assert all(q for q in normalized)


@pytest.mark.fast
def test_malformed_mock_fixture_entries_declare_repair_stage() -> None:
    """Malformed fixture specs must declare compose-stage breakage and repair replay."""
    from aetherdialect._constants import SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS

    specs = tuple(SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS)
    assert len(specs) >= 1
    for spec in specs:
        assert str(spec.get("question", "")).strip()
        assert str(spec.get("malformed_output", "")).strip()
        assert str(spec.get("repair_output", "")).strip() or spec.get("repair_response")


@pytest.mark.fast
def test_federation_run_seed_warmup_blocked_in_sandbox() -> None:
    """AetherFederation.run_seed_warmup must use the same production-API guard as AetherEngine."""
    fed = AetherFederation.__new__(AetherFederation)
    fed._sandbox_mode = True
    fed._closed = False
    fed._schema_graph = MagicMock()
    fed._dialect = MagicMock()
    fed._artifacts_dir = Path("/tmp/fed_warmup")
    fed._store = MagicMock()
    fed._templates = {}
    fed._federation_manifest = None
    fed._federation_member_graphs = {}
    fed._federation_mappings = None
    fed._federation_dialects = {}
    fed._federation_source_runtimes = {}
    fed._federation_storage_dir = None

    with (
        patch("aetherdialect.aetherdialect.llm_credentials_configured", return_value=True),
        patch("aetherdialect.aetherdialect.seed_warmup_run_once") as warmup,
    ):
        with pytest.raises(ConfigError, match="run_seed_warmup"):
            fed.run_seed_warmup("/tmp/seed_questions.txt")
        warmup.assert_not_called()


@pytest.mark.fast
def test_aether_engine_run_seed_warmup_blocked_in_sandbox() -> None:
    """Control — AetherEngine.run_seed_warmup is already gated in sandbox mode."""
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="duckdb"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/engine_warmup",
            store=MagicMock(),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={},
            schema_role="owner",
            consumer_visible_objects=None,
            context_name="master",
            data_quality_report=None,
        )
        engine = AetherEngine()
    engine._sandbox_mode = True
    with pytest.raises(ConfigError, match="run_seed_warmup"):
        engine.run_seed_warmup("/tmp/seed_questions.txt")


@pytest.mark.fast
def test_baseline_dir_for_preset_resolves_consumer_and_owner_dirs(tmp_path: Path) -> None:
    """Bundled baselines must expose distinct consumer and owner directories."""
    extract, _ = _minimal_sandbox_host(tmp_path)
    owner_dir = _baseline_dir_for_preset(extract, "owner_writer")
    consumer_dir = _baseline_dir_for_preset(extract, "consumer_reader")
    assert owner_dir is not None and consumer_dir is not None
    assert owner_dir.name == "owner"
    assert consumer_dir.name == "consumer"
