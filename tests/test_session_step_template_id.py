"""SessionStep.template_id exposes matched or accepted template id."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import SESSION_KIND_RESULT
from aetherdialect._contracts_base import SessionStep
from aetherdialect._contracts_core import (
    ConcreteIntent,
    GenerationPath,
    RuntimeIntent,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._main_execution import PipelineSession


def _tmpl() -> Template:
    return Template(
        id="T0777",
        intent_signature=ConcreteIntent(
            intent_id="i",
            tables=["t1"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        intent_key="k",
        tables_used=["t1"],
        sql_param="SELECT COUNT(*) FROM t1",
        sql_fp="fp777",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=True, num_where=0),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{}],
            questions=["how many"],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
    )


@pytest.mark.fast
def test_terminal_success_exposes_template_id() -> None:
    tmpl = _tmpl()
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.tables = {}
    owner._store = {}
    owner._templates = {tmpl.id: tmpl}
    owner._rejected = {}
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._dialect = MagicMock()
    owner._dialect.name = "postgresql"
    owner._dialect.config = owner._runtime_config
    owner._audit_emit = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._artifacts_dir = None
    owner._business_knowledge = None
    owner._ask_phase_callback = None
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._schema_terms = set()
    sess = PipelineSession(owner, mode="writer")
    sess._turn_question = "how many"
    sess._last_turn_outcome = {
        "outcome": "success",
        "sql": tmpl.sql_param,
        "rows": [(3,)],
        "columns": ["count"],
        "matched_template": tmpl,
        "intent": RuntimeIntent(
            tables=["t1"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        "generation_path": GenerationPath.EXACT_QUESTION_REUSE,
    }
    step = sess._completed_step()
    assert step.kind == SESSION_KIND_RESULT
    assert step.template_id == "T0777"


@pytest.mark.fast
def test_meta_step_template_id_none() -> None:
    step = SessionStep(
        done=True,
        prompt=None,
        kind="meta",
        sql=None,
        meta_payload={"response_kind": "schema_catalog"},
        template_id=None,
    )
    assert step.template_id is None
    assert step.sql is None
