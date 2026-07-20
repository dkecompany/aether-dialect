"""Smoke tests for the stable ``aetherdialect`` import surface and key façade contracts."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect
from aetherdialect import AetherEngine, AsyncPipelineSession
from aetherdialect._config import ConfigError, EngineConfig
from aetherdialect._contracts_base import (
    AuditEvent,
    EngineContext,
    LLMConfig,
    RuntimeConfig,
    SessionActiveError,
    SessionStep,
)
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._templates import empty_template_store


def _minimal_engine(**overrides: object) -> AetherEngine:
    """Construct a ``AetherEngine`` shell without running ``initialize_aether_engine``."""
    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/aether_api",
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=MagicMock(effective_structural_hash="hash1"),
        _dialect=MagicMock(),
        _artifacts_dir=Path("/tmp/aether_api"),
        _store=empty_template_store("hash1"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_role="owner",
        _consumer_visible_objects=None,
        _schema_stats={"table_count": 3, "total_filterable": 10},
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for k, v in defaults.items():
        setattr(obj, str(k), v)
    return obj


def test_package_all_matches_documented_exports() -> None:
    """``__all__`` stays a curated subset of the public façade."""
    allowed = {
        "AetherEngine",
        "AetherSpace",
        "AsyncPipelineSession",
        "AuditEvent",
        "ConfigError",
        "ConnectionError",
        "ConfigSnapshot",
        "DatabasePingFailed",
        "Diagnostic",
        "EngineContext",
        "LlmTransientFailure",
        "MigrationPendingError",
        "MigrationPreview",
        "MockFixtureMissingError",
        "OwnerOnlyOperationError",
        "PERMISSION_DENIED_USER_MESSAGE",
        "PipelineSession",
        "QSimSummarySnapshot",
        "RetryableError",
        "SchemaAccessError",
        "SchemaStatsSnapshot",
        "SeedWarmupSummarySnapshot",
        "SessionActiveError",
        "SessionStep",
        "SpaceContext",
        "StatementTimeoutError",
        "__version__",
    }
    assert set(aetherdialect.__all__) == allowed


def test_renamed_public_symbols_exported() -> None:
    """Historical rename: ``AetherEngine`` and ``EngineContext`` replace legacy symbols."""
    assert hasattr(aetherdialect, "AetherEngine")
    assert hasattr(aetherdialect, "EngineContext")
    assert hasattr(aetherdialect, "SpaceContext")
    assert hasattr(aetherdialect, "AetherSpace")
    assert not hasattr(aetherdialect, "Text2SQL")
    assert not hasattr(aetherdialect, "SchemaContext")
    assert hasattr(AetherEngine, "sandbox_doctor")
    assert hasattr(AetherEngine, "assert_sandbox_complete")
    assert hasattr(AetherEngine, "sandbox_questions")


def test_select_engine_rejects_unknown_aether_key() -> None:
    from aetherdialect._main_execution import _select_engine_name

    with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_ENGINE"):
        _select_engine_name(
            {
                "AETHERDIALECT_ENGINE": "not_a_registered_engine",
                "PGDATABASE": "d",
                "PGUSER": "u",
                "PGPASSWORD": "p",
            },
        )


def test_configure_llm_rejects_unknown_provider_key() -> None:
    from aetherdialect._main_execution import _configure_llm_from_environment

    with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_LLM_PROVIDER"):
        _configure_llm_from_environment({"OPENAI_API_KEY": "sk", "AETHERDIALECT_LLM_PROVIDER": "anthropic"})


def test_audit_sink_invoked_on_init(tmp_path: Path) -> None:
    """``audit_sink`` receives an ``AuditEvent`` when construction succeeds."""
    from aetherdialect._main_execution import AetherEngineInitResult

    events: list[AuditEvent] = []

    def sink(ev: AuditEvent) -> None:
        events.append(ev)

    sd = os.path.join(str(tmp_path), "intent_templates")
    os.makedirs(sd, exist_ok=True)
    with patch.object(EngineConfig, "TEMPLATE_STORE_DIR", sd):
        store = empty_template_store("h")
        with patch("aetherdialect.aetherdialect.initialize_aether_engine") as init:
            init.return_value = AetherEngineInitResult(
                runtime_config=RuntimeConfig(
                    engine="postgresql",
                    artifacts_dir=str(tmp_path),
                    engine_context=EngineContext(),
                    llm_execution=load_runtime_config(merged_env=dict(os.environ)),
                ),
                llm_config=LLMConfig(provider="openai"),
                schema_graph=MagicMock(effective_structural_hash="h"),
                dialect="postgresql",
                artifacts_dir=str(tmp_path),
                store=store,
                templates={},
                rejected={},
                schema_terms=set(),
                schema_stats={"table_count": 1, "total_filterable": 5},
            )
            engine = AetherEngine(EngineContext(), artifacts_dir=str(tmp_path), audit_sink=sink)
            assert engine is not None
    assert any(e.event_type == "init" for e in events)


def test_pipeline_session_active_error_is_public() -> None:
    assert issubclass(SessionActiveError, Exception)


def test_session_step_dataclass_shape() -> None:
    st = SessionStep(
        done=True,
        prompt=None,
        kind="ok",
        sql=None,
        data=None,
        message=None,
        error=None,
        diagnostics=(),
        intent_summary=None,
        status=None,
        reply_shape=None,
        semantic_warnings=(),
    )
    assert st.done is True


def test_async_session_wraps_pipeline_session() -> None:
    engine = _minimal_engine()
    inner = engine.session()
    async_sess = AsyncPipelineSession(inner)
    assert async_sess._inner is inner


def test_pipeline_session_ask_rejects_non_str() -> None:
    engine = _minimal_engine()
    with engine.session() as s:
        with pytest.raises(TypeError):
            s.ask(123)


def test_clear_template_store_triggers_reinit() -> None:
    engine = _minimal_engine()
    with (
        patch("aetherdialect.aetherdialect.clear_template_store_only", return_value=True) as clr,
        patch("aetherdialect.aetherdialect.initialize_aether_engine") as init,
    ):
        from aetherdialect._main_execution import AetherEngineInitResult

        init.return_value = AetherEngineInitResult(
            runtime_config=engine._runtime_config,
            llm_config=engine._llm_config,
            schema_graph=engine._schema_graph,
            dialect=engine._dialect,
            artifacts_dir=str(engine._artifacts_dir),
            store=empty_template_store("hash1"),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats=dict(engine._schema_stats),
        )
        assert engine.clear_template_store() is True
    clr.assert_called_once()
    init.assert_called_once()


def test_apply_migration_map_classmethod_exists() -> None:
    assert callable(getattr(AetherEngine, "apply_migration_map", None))


def test_no_cross_module_underscore_imports() -> None:
    """Package modules must not import ``_``-prefixed symbols from sibling modules."""
    import ast

    root = Path(__file__).resolve().parents[1] / "src" / "aetherdialect"
    violations: list[str] = []

    class _FunctionImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_function = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.in_function += 1
            self.generic_visit(node)
            self.in_function -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.in_function += 1
            self.generic_visit(node)
            self.in_function -= 1

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self.in_function and (node.level > 0 or (node.module and node.module.startswith("aetherdialect."))):
                violations.append(f"{path.name}:{node.lineno} function-scoped import from {node.module}")
            self.generic_visit(node)

    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None and node.level == 0:
                    continue
                is_internal = node.level > 0 or (
                    node.module is not None
                    and (node.module == "aetherdialect" or node.module.startswith("aetherdialect."))
                )
                if not is_internal:
                    continue
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name} from {node.module}")
        _FunctionImportVisitor().visit(tree)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Name) and target.id == node.value.id:
                violations.append(f"{path.name}:{node.lineno} self-alias {target.id} = {node.value.id}")
    assert violations == [], violations
