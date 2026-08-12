"""Owner-only gates for consumer engines across session and facade entry points."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED, SESSION_KIND_RESULT
from aetherdialect._constants_runtime import SANDBOX_MEMBER_SPACE_TABLES
from aetherdialect._contracts_base import Diagnostic, OwnerOnlyOperationError, SchemaRole, SpaceContext
from aetherdialect._contracts_core import SessionStep
from aetherdialect._federation_execute import inspect_persisted_federation
from aetherdialect._main_session import PipelineSession
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import sanitize_session_step_for_egress
from aetherdialect.aetherdialect import AetherEngine, AetherFederation


def _make_consumer_federation() -> AetherFederation:
    fed = AetherFederation.__new__(AetherFederation)
    fed._schema_role = SchemaRole.CONSUMER
    fed._closed = False
    fed._federation_mapping_suggestions = ()
    return fed


@pytest.mark.fast
def test_consumer_mapping_suggestions_raises_owner_only() -> None:
    fed = _make_consumer_federation()
    with pytest.raises(OwnerOnlyOperationError, match="mapping_suggestions"):
        fed.mapping_suggestions()


def _make_consumer_engine(
    *,
    member: str = "storefront",
    graph_id: str = "sg_test000000000001__abcd1234",
) -> AetherEngine:
    visible = SANDBOX_MEMBER_SPACE_TABLES[member]
    engine = AetherEngine.__new__(AetherEngine)
    engine._schema_role = SchemaRole.CONSUMER
    engine._consumer_visible_objects = visible
    engine._schema_graph = MagicMock()
    engine._schema_graph.schema_graph_id = graph_id
    engine._schema_graph.effective_structural_hash = "eff"
    engine._artifacts_dir = Path("/tmp/artifacts")
    engine._store = TemplateOps.empty_template_store(graph_id)
    engine._templates = {}
    engine._rejected = {}
    engine._dialect = None
    engine._pipeline_writer_lock = threading.Lock()
    engine._runtime_config = MagicMock()
    engine._runtime_config.llm_execution = None
    engine._audit_sink = None
    engine._closed = False
    engine._sandbox_closed = False
    engine._context_name = "master"
    engine._execution_engine = None
    engine._native_connection = None
    return engine


def _resolve_space_patch(engine: AetherEngine, member: str) -> MagicMock:
    space_desc = MagicMock()
    space_desc.uid = member
    tables = SANDBOX_MEMBER_SPACE_TABLES[member]
    return patch.object(
        AetherEngine,
        "_resolve_aetherspace",
        return_value=(space_desc, tables, frozenset(), frozenset(), frozenset()),
    )


@pytest.mark.fast
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
@pytest.mark.parametrize(
    "door",
    [
        "facade_session",
        "pipeline_session",
        "run_interactive",
    ],
)
def test_consumer_writer_allowed_by_door(member: str, door: str, tmp_path: Path) -> None:
    engine = _make_consumer_engine(member=member)
    engine._artifacts_dir = tmp_path
    engine._schema_terms = set()
    engine._schema_stats = {"total_filterable": 1}

    if door == "facade_session":
        with _resolve_space_patch(engine, member):
            session = engine.session(mode="writer")
        assert isinstance(session, PipelineSession)
        assert session._session_mode == "writer"
        return

    if door == "pipeline_session":
        session = PipelineSession(engine)
        assert isinstance(session, PipelineSession)
        assert session._session_mode == "writer"
        return

    with (
        _resolve_space_patch(engine, member),
        patch("aetherdialect.aetherdialect.AetherEngine._ensure_llm"),
        patch("aetherdialect.aetherdialect.input", return_value=""),
        patch("aetherdialect.aetherdialect.push_diagnostic_sink", return_value=MagicMock()),
        patch("aetherdialect.aetherdialect.pop_diagnostic_sink"),
        patch("aetherdialect.aetherdialect.notify"),
        patch("aetherdialect.aetherdialect.echo_user_text"),
        patch("aetherdialect.aetherdialect.Sandbox.require_sandbox_adoption"),
        patch("aetherdialect.aetherdialect.diagnostic_print_listener"),
        patch("aetherdialect.aetherdialect.terminated"),
    ):
        engine.run_interactive()


@pytest.mark.fast
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        ("delete_aetherspace", lambda engine: engine.delete_aetherspace(name="demo")),
        (
            "aetherspace",
            lambda engine: engine.aetherspace(
                name="demo",
                space_context=SpaceContext(tables=frozenset({"customer"})),
            ),
        ),
        (
            "apply_knowledge",
            lambda engine: engine.apply_knowledge(
                "master",
                {"domain_knowledge": []},
            ),
        ),
        ("refresh", lambda engine: engine.refresh()),
        ("refresh", lambda engine: engine.refresh(reflect=False)),
        ("run_qsim", lambda engine: engine.run_qsim(num_intents=1, num_questions=1)),
    ],
)
def test_consumer_owner_only_operations_raise(member: str, operation: str, invoke) -> None:
    engine = _make_consumer_engine(member=member)
    with pytest.raises(OwnerOnlyOperationError, match=operation):
        invoke(engine)


@pytest.mark.fast
def test_consumer_inspect_persisted_raises_permission_error(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="owner role"):
        inspect_persisted_federation(str(tmp_path), "fed_test", schema_role=SchemaRole.CONSUMER)


@pytest.mark.fast
def test_consumer_session_step_egress_hides_member_source_ids() -> None:
    from aetherdialect._contracts_core import SessionError, SessionOutcome

    step = SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        sql={"storefront_db": "SELECT 1", "catalog_db": "SELECT 2"},
        error=SessionError(code=SessionOutcome.EXECUTION_FAILED, source_id="storefront_db"),
        diagnostics=(
            Diagnostic(
                stage="execution",
                level="info",
                code=DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
                message="Federated turn queried sources: storefront_db,catalog_db",
                details=(("phase", "execution"), ("sources_queried", "storefront_db,catalog_db")),
                source_id="storefront_db",
            ),
        ),
    )
    redacted = sanitize_session_step_for_egress(step)
    payload = str(redacted)
    assert "storefront_db" not in payload
    assert "catalog_db" not in payload
    assert redacted.error is not None
    assert redacted.error.source_id == "member_0"
    assert set(redacted.sql or {}) == {"member_0", "member_1"}


@pytest.mark.fast
def test_pipeline_session_egress_hides_member_source_ids_for_consumer() -> None:
    from aetherdialect._contracts_core import SessionError, SessionOutcome

    engine = _make_consumer_engine()
    session = PipelineSession(engine, mode="reader")
    raw_step = SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        sql={"crm_db": "SELECT 3"},
        error=SessionError(code=SessionOutcome.EXECUTION_FAILED, source_id="crm_db"),
    )
    redacted = session._egress_session_step(raw_step)
    assert "crm_db" not in str(redacted)
    assert redacted.error is not None
    assert redacted.error.source_id == "member_0"
