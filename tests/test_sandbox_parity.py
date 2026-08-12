"""Sandbox production parity: stages, migration, consumer baseline, malformed fixtures, warmup guard."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation, EngineContext, Sandbox
from aetherdialect._contracts_base import ConfigError
from aetherdialect._contracts_core import MigrationPreview

_baseline_dir_for_preset = Sandbox._baseline_dir_for_preset
_consumer_reader_schema_context = Sandbox._consumer_reader_schema_context
_owner_writer_schema_context = Sandbox._owner_writer_schema_context
_sandbox_memory_engine_dir = Sandbox._sandbox_memory_engine_dir

_EXPECTED_UNEXERCISED_PRODUCTION_STAGES: frozenset[str] = frozenset(
    {
        "live_reflection_and_profiling",
        "probe_mismatch_partial_rebuild",
        "cold_build_descriptions_and_classification",
        "member_cold_reflect_profile_and_member_drift_migration_pending",
        "warmup_and_question_simulation",
        "model_turns_outside_recorded_fixtures",
    },
)

_STAGE_DOC_TOKENS: dict[str, tuple[str, ...]] = {
    "live_reflection_and_profiling": ("reflection", "profiling"),
    "probe_mismatch_partial_rebuild": ("probe", "mismatch"),
    "cold_build_descriptions_and_classification": ("description", "classification"),
    "member_cold_reflect_profile_and_member_drift_migration_pending": (
        "member",
        "drift",
        "migration",
    ),
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
    """Skipped production stages are named in code and docs."""
    from aetherdialect._constants_runtime import SANDBOX_UNEXERCISED_PRODUCTION_STAGES

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
            skeleton_document={},
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

    engine_dir = Sandbox._sandbox_memory_engine_dir(artifacts)
    from aetherdialect._schema_graph import load_schema_graph_snapshot

    graph = load_schema_graph_snapshot(str(engine_dir / "schema_graph.json.gz"))
    assert graph is not None
    assert str(graph.schema_graph_id) == "consumer_baseline_sg"


@pytest.mark.fast
def test_sandbox_consumer_scope_matches_owner_at_construction(tmp_path: Path) -> None:
    """Consumer engine_context defaults to the full owner scope, resolved at construction."""
    from aetherdialect import Sandbox

    owner_ctx = Sandbox._owner_writer_schema_context()
    consumer_ctx = Sandbox._consumer_reader_schema_context()
    assert consumer_ctx.allow_objects == owner_ctx.allow_objects

    contexts_at_init: list[frozenset[str]] = []

    class FakeEngine:
        def __init__(self, schema_context: EngineContext, **kwargs: object) -> None:
            del kwargs
            contexts_at_init.append(frozenset(schema_context.allow_objects or ()))
            from aetherdialect._contracts_core import RuntimeConfig
            from aetherdialect._utils_artifacts import load_runtime_config

            llm_exec = load_runtime_config(merged_env={})
            self._schema_graph = MagicMock(schema_literal_json="{}")
            self._runtime_config = RuntimeConfig(
                engine="duckdb",
                artifacts_dir="/tmp/sandbox_consumer_scope",
                engine_context=schema_context,
                llm_execution=llm_exec,
                execution_context=schema_context,
            )
            self._schema_role = "consumer"

    import aetherdialect._sandbox
    from tests._sandbox_csv_bundle import write_main_csv_ddl_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_main_csv_ddl_bundle(bundle, tables=(("customer", "customer_id"),))

    original = aetherdialect._sandbox.Sandbox._aether_engine_cls
    aetherdialect._sandbox.Sandbox._aether_engine_cls = lambda: FakeEngine
    try:
        with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
            sandbox.load_dataset("main")
            sandbox.engine(role="consumer")
    finally:
        aetherdialect._sandbox.Sandbox._aether_engine_cls = original

    assert contexts_at_init == [frozenset()]


@pytest.mark.fast
def test_bundled_mock_fixtures_include_malformed_repair_exercises() -> None:
    """Mock corpus must include deliberately malformed outputs that exercise repair paths."""
    from aetherdialect._constants_runtime import SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS

    questions = tuple(SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS)
    assert len(questions) >= 1
    normalized = {q.strip().lower() for q in questions}
    assert all(q for q in normalized)


@pytest.mark.fast
def test_malformed_mock_fixture_entries_declare_repair_stage() -> None:
    """Malformed fixture specs must declare compose-stage breakage and repair replay."""
    from aetherdialect._constants_runtime import SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS

    specs = tuple(SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS)
    assert len(specs) >= 1
    for spec in specs:
        assert str(spec.get("question", "")).strip()
        assert str(spec.get("malformed_output", "")).strip()
        assert str(spec.get("repair_output", "")).strip() or spec.get("repair_response")


@pytest.mark.fast
def test_malformed_compose_repair_uses_intent_format_task() -> None:
    """Malformed compose override is one-shot; format-repair uses intent_format task."""
    import json

    from aetherdialect._constants_runtime import (
        INTENT_COMPOSE_SYSTEM,
        SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS,
        SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS,
    )
    from aetherdialect._contracts_base import MockFixtureMissingError
    from aetherdialect._intent_loop import build_intent_format_repair_prompt
    from aetherdialect._llm_provider import MockProvider, SandboxRuntimeState

    question = SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS[0]
    spec = SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS[0]
    malformed = str(spec["malformed_output"]).strip()
    repair = str(spec["repair_output"]).strip()
    fixtures = Path(__file__).resolve().parent / "_malformed_mock_fixtures.json"
    fixtures.write_text(json.dumps({"fixtures": []}), encoding="utf-8")
    runtime = SandboxRuntimeState()
    token = SandboxRuntimeState.bind_sandbox_runtime(runtime)
    try:
        provider = MockProvider(str(fixtures))
        compose_user = json.dumps({"question": question})
        first = provider.chat_text(INTENT_COMPOSE_SYSTEM, compose_user, task="intent", max_retries=0, timeout=1.0)
        assert first.strip() == malformed
        with pytest.raises(MockFixtureMissingError):
            provider.chat_text(INTENT_COMPOSE_SYSTEM, compose_user, task="intent", max_retries=0, timeout=1.0)
        repair_prompt = build_intent_format_repair_prompt(question, malformed, "JSON parse failed")
        repaired = provider.chat_text(
            INTENT_COMPOSE_SYSTEM,
            repair_prompt,
            task="intent_format",
            max_retries=0,
            timeout=1.0,
        )
        assert repaired.strip() == repair
        json.loads(repaired)
    finally:
        SandboxRuntimeState.reset_sandbox_runtime(token)
        fixtures.unlink(missing_ok=True)


@pytest.mark.fast
def test_malformed_mock_fixture_replay_returns_invalid_json_first() -> None:
    """Compose replay must return malformed JSON before the repair response."""
    import json

    from aetherdialect._constants_runtime import (
        INTENT_COMPOSE_SYSTEM,
        SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS,
        SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS,
    )
    from aetherdialect._llm_provider import MockProvider, SandboxRuntimeState

    question = SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS[0]
    spec = SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS[0]
    fixtures = Path(__file__).resolve().parent / "_malformed_mock_fixtures.json"
    fixtures.write_text(json.dumps({"fixtures": []}), encoding="utf-8")
    runtime = SandboxRuntimeState()
    token = SandboxRuntimeState.bind_sandbox_runtime(runtime)
    try:
        provider = MockProvider(str(fixtures))
        user = json.dumps({"question": question})
        first = provider.chat_text(INTENT_COMPOSE_SYSTEM, user, task="intent", max_retries=0, timeout=1.0)
        with pytest.raises(json.JSONDecodeError):
            json.loads(first)
        assert first.strip() == str(spec["malformed_output"]).strip()
        repaired = provider.chat_text(
            INTENT_COMPOSE_SYSTEM,
            str(spec["malformed_output"]),
            task="intent",
            max_retries=0,
            timeout=1.0,
        )
        json.loads(repaired)
    finally:
        SandboxRuntimeState.reset_sandbox_runtime(token)
        fixtures.unlink(missing_ok=True)


@pytest.mark.fast
def test_sandbox_handle_applies_bundled_migration_variant() -> None:
    """User-facing handle must preview and apply a bundled migration corpus variant."""
    from aetherdialect._sandbox import Sandbox

    if not Sandbox.data_zip_path().exists():
        pytest.skip("sandbox bundle not present")
    with Sandbox.create_offline_sandbox(AetherEngine) as handle:
        preview = handle.preview_migration_corpus_variant()
        assert preview.tier in {"compatible", "remap", "destructive"}
        migrated = handle.apply_migration_corpus_variant()
        assert migrated is handle.engine
        with handle.session() as session:
            step = session.accept_until_done("How many books do we have?")
        assert step.done
        assert step.sql
        assert step.error is None


@pytest.mark.fast
def test_federation_run_qsim_blocked_in_sandbox() -> None:
    """AetherFederation.run_qsim must use the same production-API guard as AetherEngine."""
    fed = AetherFederation.__new__(AetherFederation)
    fed._sandbox_mode = True
    fed._closed = False
    fed._schema_graph = MagicMock()
    fed._dialect = MagicMock()
    fed._artifacts_dir = Path("/tmp/fed_qsim")
    fed._store = MagicMock()
    fed._templates = {}
    fed._federation_manifest = None
    fed._federation_mappings = None
    fed._schema_stats = {"table_count": 10, "total_filterable": 50}

    with (
        patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True),
        patch("aetherdialect.aetherdialect.qsim_run_once") as qsim,
    ):
        with pytest.raises(ConfigError, match="run_qsim"):
            fed.run_qsim()
        qsim.assert_not_called()


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
        patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True),
        patch("aetherdialect.aetherdialect.seed_warmup_run_once") as warmup,
    ):
        with pytest.raises(ConfigError, match="warmup is not supported on AetherFederation"):
            fed.run_seed_warmup("/tmp/seed_questions.txt")
        warmup.assert_not_called()


@pytest.mark.fast
def test_aether_engine_run_seed_warmup_blocked_in_sandbox() -> None:
    """AetherEngine.run_seed_warmup is gated in sandbox mode."""
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
    owner_dir = Sandbox._baseline_dir_for_preset(extract, "owner_writer")
    consumer_dir = Sandbox._baseline_dir_for_preset(extract, "consumer_reader")
    assert owner_dir is not None and consumer_dir is not None
    assert owner_dir.name == "owner"
    assert consumer_dir.name == "consumer"


def _write_minimal_sandbox_bundle(root: Path) -> None:
    from tests._sandbox_csv_bundle import write_main_csv_ddl_bundle

    write_main_csv_ddl_bundle(root, tables=(("customer", "customer_id"),))


@pytest.mark.fast
def test_sandbox_engine_session_ask_uses_pipeline_session(tmp_path: Path) -> None:
    """Sandbox.engine().session().ask must be a PipelineSession contract."""
    from aetherdialect import SessionStep
    from aetherdialect._contracts_core import RuntimeConfig
    from aetherdialect._main_session import PipelineSession
    from aetherdialect._utils_artifacts import load_runtime_config

    ask_questions: list[str] = []

    class FakeEngine:
        def __init__(self, schema_context: EngineContext, **kwargs: object) -> None:
            del kwargs
            llm_exec = load_runtime_config(merged_env={})
            self._schema_graph = MagicMock(schema_literal_json="{}")
            self._runtime_config = RuntimeConfig(
                engine="duckdb",
                artifacts_dir="/tmp/sandbox_session_parity",
                engine_context=schema_context,
                llm_execution=llm_exec,
                execution_context=schema_context,
            )
            self._schema_role = "owner"
            self._sandbox_mode = True
            self._closed = False
            self._store = {}
            self._templates = {}
            self._rejected = {}
            self._schema_terms = set()
            self.dialect = "duckdb"

        def session(self, **kwargs: object) -> PipelineSession:
            del kwargs
            session = PipelineSession(self, mode="reader")

            def _ask(question: str) -> SessionStep:
                ask_questions.append(question)
                return SessionStep(done=True, prompt=None, kind="result", sql="SELECT 1")

            session.ask = _ask
            return session

    import aetherdialect._sandbox

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_sandbox_bundle(bundle)
    original = aetherdialect._sandbox.Sandbox._aether_engine_cls
    aetherdialect._sandbox.Sandbox._aether_engine_cls = lambda: FakeEngine
    try:
        with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
            sandbox.load_dataset("main")
            engine = sandbox.engine(role="owner")
            session = engine.session()
            assert isinstance(session, PipelineSession)
            assert callable(session.ask)
            step = session.ask("How many customers?")
            assert step.done is True
            assert ask_questions == ["How many customers?"]
    finally:
        aetherdialect._sandbox.Sandbox._aether_engine_cls = original


@pytest.mark.fast
def test_aether_engine_warmup_entrypoints_blocked_in_sandbox() -> None:
    """Warmup and qsim entrypoints raise when `_sandbox_mode` is set."""
    with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
        init.return_value = MagicMock(
            runtime_config=MagicMock(engine="duckdb"),
            llm_config=MagicMock(),
            schema_graph=MagicMock(),
            dialect=MagicMock(),
            artifacts_dir="/tmp/engine_warmup_parity",
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

    with (
        patch("aetherdialect.aetherdialect.seed_warmup_run_once") as warmup,
        patch("aetherdialect.aetherdialect.run_seed_warmup_from_history_execution") as from_history,
        patch("aetherdialect.aetherdialect.run_seed_warmup_from_query_log_execution") as from_log,
        patch("aetherdialect.aetherdialect.qsim_run_once") as qsim,
    ):
        with pytest.raises(ConfigError, match="run_seed_warmup"):
            engine.run_seed_warmup("/tmp/seed_questions.txt")
        with pytest.raises(ConfigError, match="run_seed_warmup_from_history"):
            engine.run_seed_warmup_from_history("/tmp/history.sql")
        with pytest.raises(ConfigError, match="run_seed_warmup_from_query_log"):
            engine.run_seed_warmup_from_query_log()
        with pytest.raises(ConfigError, match="run_qsim"):
            engine.run_qsim()
        warmup.assert_not_called()
        from_history.assert_not_called()
        from_log.assert_not_called()
        qsim.assert_not_called()


@pytest.mark.fast
def test_sandbox_provider_selects_fixture_llm_offline(tmp_path: Path) -> None:
    """Provider ``sandbox`` selects fixture LLM replay (`uses_network` false)."""
    from aetherdialect._config import EngineConfig
    from aetherdialect._llm_provider import LLMProvider, MockProvider

    assert EngineConfig.is_sandbox_llm_provider("sandbox") is True
    assert EngineConfig.normalize_llm_provider("sandbox") == "sandbox"

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_sandbox_bundle(bundle)
    with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
        assert sandbox.llm_mode == "sandbox"
        assert sandbox.uses_network is False

    with (
        patch.object(EngineConfig, "LLM_PROVIDER", "sandbox"),
        patch.object(MockProvider, "get") as get_mock,
    ):
        get_mock.return_value.chat_text.return_value = '{"ok": true}'
        text = LLMProvider.chat("system", "user", task="intent", max_retries=0, timeout=1.0)
        assert text == '{"ok": true}'
        get_mock.assert_called_once()
        get_mock.return_value.chat_text.assert_called_once()
