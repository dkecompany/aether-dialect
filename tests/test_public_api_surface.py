"""Smoke tests for the stable ``aetherdialect`` import surface and key façade contracts."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect
from aetherdialect._config import EngineConfig, load_runtime_config
from aetherdialect._contracts_base import (
    AuditEvent,
    ConfigError,
    LLMConfig,
    RuntimeConfig,
    SchemaContext,
    SessionActiveError,
    SessionStep,
)
from aetherdialect.text2sql import AsyncPipelineSession, Text2SQL
from aetherdialect._templates import empty_template_store


def _minimal_t2s(**overrides: object) -> Text2SQL:
    """Construct a ``Text2SQL`` shell without running ``initialize_text2sql``."""

    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/t2s_api",
            schema_context=SchemaContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=MagicMock(effective_structural_hash="hash1"),
        _dialect=MagicMock(),
        _artifacts_dir=Path("/tmp/t2s_api"),
        _store=empty_template_store("hash1"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_stats={"table_count": 3, "total_filterable": 10},
    )
    defaults.update(overrides)
    obj = Text2SQL.__new__(Text2SQL)
    for k, v in defaults.items():
        setattr(obj, str(k), v)
    return obj


def test_package_all_matches_documented_exports() -> None:
    """``__all__`` stays a curated subset of the public façade."""

    allowed = {
        "AsyncPipelineSession",
        "AuditEvent",
        "ConfigError",
        "ConfigSnapshot",
        "DatabasePingFailed",
        "Diagnostic",
        "LlmExecutionConfig",
        "LlmTransientFailure",
        "MigrationPendingError",
        "MigrationPreview",
        "PipelineSession",
        "QSimSummarySnapshot",
        "RetryableError",
        "RuntimeConfig",
        "SchemaAccessError",
        "SchemaContext",
        "SchemaStatsSnapshot",
        "SeedWarmupSummarySnapshot",
        "SessionActiveError",
        "SessionStep",
        "StatementTimeoutError",
        "Text2SQL",
        "__version__",
    }
    assert set(aetherdialect.__all__) == allowed


def test_select_engine_rejects_unknown_aether_key() -> None:
    from aetherdialect._main_execution import _select_engine_name

    with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_ENGINE"):
        _select_engine_name(
            {
                "AETHERDIALECT_ENGINE": "mysql",
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

    from aetherdialect._main_execution import Text2SQLInitResult

    events: list[AuditEvent] = []

    def sink(ev: AuditEvent) -> None:
        events.append(ev)

    sd = os.path.join(str(tmp_path), "intent_templates")
    os.makedirs(sd, exist_ok=True)
    with patch.object(EngineConfig, "TEMPLATE_STORE_DIR", sd):
        store = empty_template_store("h")
        with patch("aetherdialect.text2sql.initialize_text2sql") as init:
            init.return_value = Text2SQLInitResult(
                runtime_config=RuntimeConfig(
                    engine="postgresql",
                    artifacts_dir=str(tmp_path),
                    schema_context=SchemaContext(),
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
            t2s = Text2SQL(SchemaContext(), artifacts_dir=str(tmp_path), audit_sink=sink)
            assert t2s is not None
    assert any(e.event_type == "init" for e in events)


def test_pipeline_session_active_error_is_public() -> None:
    assert issubclass(SessionActiveError, Exception)


def test_emit_otel_span_returns_context_manager() -> None:
    t2s = _minimal_t2s()
    ctx = t2s.emit_otel_span("test_span", k=1)
    assert isinstance(ctx, AbstractContextManager)
    with ctx:
        pass


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
    t2s = _minimal_t2s()
    inner = t2s.session()
    async_sess = AsyncPipelineSession(inner)
    assert async_sess._inner is inner  # type: ignore[attr-defined]


def test_pipeline_session_ask_rejects_non_str() -> None:
    t2s = _minimal_t2s()
    with t2s.session() as s:
        with pytest.raises(TypeError):
            s.ask(123)  # type: ignore[arg-type]


def test_clear_template_store_triggers_reinit() -> None:
    t2s = _minimal_t2s()
    with (
        patch("aetherdialect.text2sql.clear_template_store_only", return_value=True) as clr,
        patch("aetherdialect.text2sql.initialize_text2sql") as init,
    ):
        from aetherdialect._main_execution import Text2SQLInitResult

        init.return_value = Text2SQLInitResult(
            runtime_config=t2s._runtime_config,  # type: ignore[attr-defined]
            llm_config=t2s._llm_config,  # type: ignore[attr-defined]
            schema_graph=t2s._schema_graph,  # type: ignore[attr-defined]
            dialect=t2s._dialect,  # type: ignore[attr-defined]
            artifacts_dir=str(t2s._artifacts_dir),  # type: ignore[attr-defined]
            store=empty_template_store("hash1"),
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats=dict(t2s._schema_stats),  # type: ignore[attr-defined]
        )
        assert t2s.clear_template_store() is True
    clr.assert_called_once()
    init.assert_called_once()


def test_apply_migration_map_classmethod_exists() -> None:
    assert callable(getattr(Text2SQL, "apply_migration_map", None))
