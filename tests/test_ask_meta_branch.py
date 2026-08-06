"""Ask path branches to metadata answers for schema_catalog / business_knowledge routes."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect
from aetherdialect._contracts_base import QuestionRoute, QuestionValidationResult, SessionStep
from aetherdialect._main_execution import MainExecutionOps, PipelineSession
from aetherdialect._templates import TemplateOps


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._schema_graph.schema_graph_id = "sg-meta"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._sandbox_closed = False
    owner._artifacts_dir = "/tmp/aether_meta"
    owner._pipeline_writer_lock = threading.Lock()
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = MagicMock()
    owner._ask_phase_callback = None
    owner._business_knowledge = MagicMock()
    owner._business_knowledge.scope_kwargs.return_value = {"entries": (), "digest": None}
    owner._dialect = MagicMock()
    owner.dialect = "postgresql"
    owner.limits = MagicMock()
    owner._federation_manifest = None
    owner._sandbox_runtime = None
    return owner


def _reuse_none(*_a: object, **_k: object) -> MagicMock:
    return MagicMock(reuse_type="none", best_template=None, reuse_history_index=None)


@pytest.mark.fast
def test_schema_catalog_ask_returns_meta_step() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner, mode="writer")
    meta_step = SessionStep(
        done=True,
        prompt=None,
        kind="meta",
        sql=None,
        message="There are 3 tables.",
        meta_payload={"response_kind": "schema_catalog", "headline": "There are 3 tables."},
    )
    validation = QuestionValidationResult(
        accepted=True,
        route=QuestionRoute.SCHEMA_CATALOG,
        corrected="how many tables",
    )

    def _load(schema, store, templates, rejected, schema_terms, dialect=None):
        return dialect or MagicMock(), schema, store, templates, rejected, schema_terms

    with (
        patch("aetherdialect._pipeline.EngineConfig.llm_credentials_configured", return_value=True),
        patch("aetherdialect._main_execution.load_pipeline_resources", side_effect=_load),
        patch("aetherdialect._main_execution.validate_question", return_value=validation),
        patch("aetherdialect._main_execution.match_question_level_template_reuse", side_effect=_reuse_none),
        patch.object(MainExecutionOps, "answer_metadata_question", return_value=meta_step) as answer_mock,
        patch("aetherdialect._main_execution.normalize_question_via_llm") as normalize_mock,
    ):
        step = sess.ask("how many tables")
    assert step.kind == "meta"
    assert step.done is True
    assert step.sql is None
    assert step.message == "There are 3 tables."
    answer_mock.assert_called_once()
    normalize_mock.assert_not_called()
    codes = {d.code for d in step.diagnostics}
    assert "meta.route.schema_catalog" in codes


@pytest.mark.fast
def test_analytical_ask_does_not_emit_meta_diagnostics() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner, mode="writer")
    validation = QuestionValidationResult(
        accepted=True,
        route=QuestionRoute.ANALYTICAL,
        corrected="list orders",
    )

    def _load(schema, store, templates, rejected, schema_terms, dialect=None):
        return dialect or MagicMock(), schema, store, templates, rejected, schema_terms

    with (
        patch("aetherdialect._pipeline.EngineConfig.llm_credentials_configured", return_value=True),
        patch("aetherdialect._main_execution.load_pipeline_resources", side_effect=_load),
        patch("aetherdialect._main_execution.validate_question", return_value=validation),
        patch("aetherdialect._main_execution.match_question_level_template_reuse", side_effect=_reuse_none),
        patch.object(MainExecutionOps, "answer_metadata_question") as answer_mock,
        patch(
            "aetherdialect._main_execution.normalize_question_via_llm",
            side_effect=RuntimeError("stop_after_analytical_branch"),
        ),
    ):
        step = sess.ask("list orders")
    answer_mock.assert_not_called()
    codes = {d.code for d in step.diagnostics}
    assert not any(str(c).startswith("meta.") for c in codes)
    assert step.done is True


@pytest.mark.fast
def test_metadata_answer_stays_internal() -> None:
    assert "answer_metadata_question" not in aetherdialect.__all__
    assert hasattr(MainExecutionOps, "answer_metadata_question")
