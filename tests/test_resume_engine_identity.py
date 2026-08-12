"""SQL confirm resume binds engine identity like ask."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._contracts_core import PipelineSuspended
from aetherdialect._main_session import PipelineSession
from aetherdialect._utils import active_engine_identity


@pytest.mark.fast
def test_sql_confirm_resume_has_identity() -> None:
    seen: dict[str, object] = {}
    identity = EngineIdentity(engine_type="duckdb", runtime_config=SimpleNamespace())
    owner = SimpleNamespace(
        _dialect=MagicMock(),
        _artifacts_dir=None,
        _pipeline_writer_lock=None,
        _runtime_config=SimpleNamespace(llm_execution=SimpleNamespace()),
        _schema_graph=MagicMock(),
        _schema_role="owner",
        _phase_callback=None,
        limits=EngineLimits(),
        _engine_identity=identity,
        _sandbox_runtime=None,
        _audit_emit=None,
    )
    sess = PipelineSession(owner)
    sess._suspended = PipelineSuspended("execute", "confirm?", payload=None)
    sess._session_busy = True
    sess._turn_cancel_event = MagicMock()

    def _fake_dispatch(session: object, ex: object) -> None:
        seen["identity"] = active_engine_identity()

    with (
        patch.object(PipelineSession, "_owner_engine_identity", return_value=identity),
        patch(
            "aetherdialect._main_interactive.MainInteractiveOps.dispatch_pipeline_resume", side_effect=_fake_dispatch
        ),
        patch(
            "aetherdialect._main_session.MainSessionSerdeOps._session_domain_knowledge_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch(
            "aetherdialect._main_session.llm_execution_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch(
            "aetherdialect._utils.llm_usage_session_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch.object(PipelineSession, "_completed_step", return_value=MagicMock(done=True)),
    ):
        sess._resume_from_suspend()

    assert seen.get("identity") is identity
