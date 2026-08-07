"""reuse_saved_question binds engine identity for dialect resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._core_utils import active_engine_identity
from aetherdialect._dialect import Dialect
from aetherdialect._main_execution import PipelineSession


@pytest.mark.fast
def test_reuse_resolves_active_dialect() -> None:
    seen: dict[str, object] = {}
    identity = EngineIdentity(engine_type="duckdb", runtime_config=SimpleNamespace())
    dialect = MagicMock(spec=Dialect)
    owner = SimpleNamespace(
        _dialect=dialect,
        _artifacts_dir=None,
        _pipeline_writer_lock=None,
        _runtime_config=SimpleNamespace(llm_execution=SimpleNamespace()),
        _schema_graph=MagicMock(),
        _schema_role="owner",
        _ask_phase_callback=None,
        limits=EngineLimits(),
        _engine_identity=identity,
        _sandbox_runtime=None,
        _audit_emit=None,
    )
    sess = PipelineSession(owner)
    sess._session_mode = "reader"

    def _fake_force(*_a, **_k):
        seen["identity"] = active_engine_identity()

    with (
        patch.object(PipelineSession, "_owner_engine_identity", return_value=identity),
        patch.object(PipelineSession, "_resources", return_value=(MagicMock(), {}, (), (), MagicMock())),
        patch("aetherdialect._main_execution.force_reuse_saved_question", side_effect=_fake_force),
        patch(
            "aetherdialect._main_execution.MainExecutionOps._owner_business_knowledge_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch(
            "aetherdialect._main_execution.llm_execution_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch(
            "aetherdialect._main_execution.llm_usage_session_scope",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch("aetherdialect._main_execution.MainExecutionOps._consumer_sql_gate_kwargs", return_value={}),
        patch("aetherdialect._main_execution.MainExecutionOps._federation_reuse_kwargs", return_value={}),
        patch(
            "aetherdialect._main_execution.MainExecutionOps._persist_template_learning_for_pipeline_session",
            return_value=False,
        ),
        patch.object(PipelineSession, "_completed_step", return_value=MagicMock(done=True)),
        patch.object(PipelineSession, "_audit_ask_emit"),
        patch.object(PipelineSession, "_emit_turn_llm_usage", return_value=()),
    ):
        sess.reuse_saved_question("old q", "new q", {"p1": 1})

    assert seen.get("identity") is identity
