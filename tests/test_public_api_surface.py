"""Smoke tests for the stable ``aetherdialect`` import surface and key façade contracts."""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect
from aetherdialect import AetherEngine, AsyncPipelineSession
from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import (
    AuditEvent,
    ConfigError,
    EngineContext,
    LLMConfig,
    RuntimeConfig,
    SessionActiveError,
    SessionStep,
)
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._templates import TemplateOps

_API_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "API_REFERENCE.md"


def _api_reference_text() -> str:
    return _API_REFERENCE.read_text(encoding="utf-8")


def _parse_exported_symbols_table(md: str) -> set[str]:
    section = md.split("## Exported symbols", 1)[1].split("\n## ", 1)[0]
    return {match.group(1) for match in re.finditer(r"^\| `([^`]+)` \|", section, re.MULTILINE)}


def _parse_aether_engine_method_names(md: str) -> set[str]:
    engine_section = md.split("## AetherEngine", 1)[1].split("\n## FederationContext", 1)[0]
    methods_section = engine_section.split("### Methods", 1)[1]
    names: set[str] = set()
    for line in methods_section.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|", 2)[1].strip()
        match = re.match(r"`([^`]+)`", cell)
        if match:
            name = match.group(1).split("(", 1)[0].rstrip(".")
            if name and name != "inspect_tabular_upload":
                names.add(name)
    return names


def _parse_offline_sandbox_param_names(md: str) -> set[str]:
    section = md.split("### `AetherEngine.offline_sandbox` quick path", 1)[1]
    table = section.split("| `SandboxHandle` member", 1)[0]
    names: set[str] = set()
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|", 2)[1]
        names.update(re.findall(r"`([^`]+)`", cell))
    return names


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
        _store=TemplateOps.empty_template_store("hash1"),
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
        _construction_phase_callback=None,
        _ask_phase_callback=None,
        _token_provider=None,
        _tenant_slug=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for k, v in defaults.items():
        setattr(obj, str(k), v)
    return obj


def _parse_exception_docs(md: str) -> set[str]:
    section = md.split("## Exceptions", 1)[1].split("\n---\n", 1)[0]
    return {match.group(1) for match in re.finditer(r"^\| `([^`]+)` \|", section, re.MULTILINE)}


def test_every_exception_exported_and_documented() -> None:
    """Every public library exception is exported and has a one-line API_REFERENCE entry."""
    import inspect

    documented = _parse_exception_docs(_api_reference_text())
    exported = {
        name
        for name in aetherdialect.__all__
        if inspect.isclass(getattr(aetherdialect, name, None))
        and issubclass(getattr(aetherdialect, name), BaseException)
    }
    missing_docs = sorted(exported - documented)
    missing_exports = sorted(documented - exported)
    assert missing_docs == [], f"exported exceptions missing API_REFERENCE rows: {missing_docs}"
    assert missing_exports == [], f"API_REFERENCE documents unexported exceptions: {missing_exports}"
    for name in sorted(exported):
        row_match = re.search(
            rf"^\| `{re.escape(name)}` \| [^|]+ \| ([^|]+) \|",
            _api_reference_text(),
            re.MULTILINE,
        )
        assert row_match is not None, f"missing exception row for {name}"
        when_raised = row_match.group(1).strip()
        assert when_raised and when_raised != "-", f"{name} needs when-raised documentation"


def test_no_message_constants_exported() -> None:
    assert "PERMISSION_DENIED_USER_MESSAGE" not in aetherdialect.__all__
    assert not hasattr(aetherdialect, "PERMISSION_DENIED_USER_MESSAGE")


def test_package_all_matches_documented_exports() -> None:
    """``__all__`` matches the curated public export list."""
    allowed = set(aetherdialect.__all__)
    assert "AetherError" in allowed
    assert "DatabaseConnectionError" in allowed
    assert "ConnectionError" not in allowed
    assert "PERMISSION_DENIED_USER_MESSAGE" not in allowed
    assert set(aetherdialect.__all__) == allowed
    assert not hasattr(aetherdialect, "cancel_active_federation_turn")
    assert not hasattr(aetherdialect, "compose_federation_dry_run")
    assert hasattr(aetherdialect, "FederationMemberExecutionError")
    assert hasattr(aetherdialect, "FederationCapExceededError")
    assert callable(getattr(aetherdialect.PipelineSession, "cancel_active_federation_turn", None))
    assert callable(getattr(aetherdialect.AsyncPipelineSession, "cancel_active_federation_turn", None))


def test_package_exports_match_api_reference_symbols() -> None:
    """``__all__`` and API_REFERENCE exported-symbols table stay aligned."""
    documented = _parse_exported_symbols_table(_api_reference_text())
    exported = set(aetherdialect.__all__)
    missing_from_docs = sorted(exported - documented)
    extra_in_docs = sorted(documented - exported)
    assert missing_from_docs == [], f"exported but not documented in API_REFERENCE: {missing_from_docs}"
    assert extra_in_docs == [], f"documented in API_REFERENCE but not exported: {extra_in_docs}"


def test_aether_engine_documented_methods_exist() -> None:
    """Every AetherEngine method named in API_REFERENCE exists on the class."""
    documented = _parse_aether_engine_method_names(_api_reference_text())
    missing = sorted(name for name in documented if not hasattr(AetherEngine, name))
    assert missing == [], f"API_REFERENCE documents missing AetherEngine methods: {missing}"


def test_offline_sandbox_documented_params_exist() -> None:
    """offline_sandbox parameter table matches the classmethod signature."""
    documented = _parse_offline_sandbox_param_names(_api_reference_text())
    actual = set(inspect.signature(AetherEngine.offline_sandbox).parameters)
    bogus = sorted(documented - actual)
    undocumented = sorted(actual - documented)
    assert bogus == [], f"API_REFERENCE documents bogus offline_sandbox params: {bogus}"
    assert undocumented == [], f"offline_sandbox params missing from API_REFERENCE: {undocumented}"


def test_renamed_public_symbols_exported() -> None:
    """Historical rename: ``AetherEngine`` and ``EngineContext`` replace legacy symbols."""
    assert hasattr(aetherdialect, "AetherEngine")
    assert hasattr(aetherdialect, "AetherFederation")
    assert hasattr(aetherdialect, "EngineContext")
    assert hasattr(aetherdialect, "Sandbox")
    assert hasattr(aetherdialect, "SpaceContext")
    assert hasattr(aetherdialect, "AetherSpace")
    assert not hasattr(aetherdialect, "Text2SQL")
    assert not hasattr(aetherdialect, "SchemaContext")
    assert hasattr(AetherEngine, "sandbox_doctor")
    assert hasattr(AetherEngine, "assert_sandbox_complete")
    assert hasattr(AetherEngine, "sandbox_questions")


def test_select_engine_rejects_unknown_aether_key() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_ENGINE"):
        MainExecutionOps._select_engine_name(
            {
                "AETHERDIALECT_ENGINE": "not_a_registered_engine",
                "PGDATABASE": "d",
                "PGUSER": "u",
                "PGPASSWORD": "p",
            },
        )


def test_configure_llm_rejects_unknown_provider_key() -> None:
    from aetherdialect._main_execution import MainExecutionOps

    with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_LLM_PROVIDER"):
        MainExecutionOps._configure_llm_from_environment(
            {"OPENAI_API_KEY": "sk", "AETHERDIALECT_LLM_PROVIDER": "anthropic"}
        )


def test_audit_sink_invoked_on_init(tmp_path: Path) -> None:
    """``audit_sink`` receives an ``AuditEvent`` when construction succeeds."""
    from aetherdialect._contracts_base import AetherEngineInitResult

    events: list[AuditEvent] = []

    def sink(ev: AuditEvent) -> None:
        events.append(ev)

    sd = os.path.join(str(tmp_path), "intent_templates")
    os.makedirs(sd, exist_ok=True)
    with patch.object(EngineConfig, "TEMPLATE_STORE_DIR", sd):
        store = TemplateOps.empty_template_store("h")
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
        notices=(),
        data_truncated=False,
    )
    assert st.done is True
    assert st.notices == ()
    assert st.data_truncated is False


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
        from aetherdialect._contracts_base import AetherEngineInitResult

        init.return_value = AetherEngineInitResult(
            runtime_config=engine._runtime_config,
            llm_config=engine._llm_config,
            schema_graph=engine._schema_graph,
            dialect=engine._dialect,
            artifacts_dir=str(engine._artifacts_dir),
            store=TemplateOps.empty_template_store("hash1"),
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


@pytest.mark.fast
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
