"""Cross-cutting embeddable-readiness interaction checks (fast, mocked)."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AsyncPipelineSession, BusinessKnowledgeEntry
from aetherdialect._constants import (
    PERMISSION_DENIED_USER_MESSAGE,
    PIPELINE_SUSPEND_ID_SQL,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_META,
)
from aetherdialect._contracts_base import (
    AccessError,
    JoinEdge,
    NormalizedExpr,
    ParameterBinding,
    PipelineSuspended,
    PredicateGroup,
    QuestionRoute,
    QuestionValidationResult,
    SessionStep,
    WhereParam,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedSqlBundle,
    FederatedStatementRecord,
    GenerationPath,
    InteractiveTailSnapshot,
    RuntimeIntent,
    SqlFeedbackSuspendContext,
    SqlGenerationOutcome,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._core_utils import business_knowledge_scope
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect
from aetherdialect._intent_expr import parse_intent_response
from aetherdialect._main_execution import MainExecutionOps, PipelineSession
from aetherdialect._templates import TemplateOps


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._schema_graph.schema_graph_id = "sg-ix"
    owner._store = TemplateOps.empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._sandbox_closed = False
    owner._artifacts_dir = "/tmp/aether_ix"
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


def _engine_schema() -> SchemaGraph:
    customers = TableMetadata(
        name="customers",
        columns={
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                role="identifier",
                is_primary_key=True,
                distinct_count=10,
                distinct_ratio=1.0,
                row_count=10,
                null_ratio=0.0,
            )
        },
        primary_key=["customer_id"],
        foreign_keys=[],
        description="Customers",
    )
    return SchemaGraph(
        tables={"customers": customers},
        join_paths_multi={},
        schema_graph_id="sg-ix-meta",
        effective_structural_hash="hash-ix",
    )


@pytest.mark.fast
def test_semantic_comma_from_accepted() -> None:
    dx = DatabricksDialect.__new__(DatabricksDialect)
    parsed = dx.parse_select("SELECT 1 FROM a")
    carriers = dx.ordered_join_carrier_froms(parsed)
    edge = JoinEdge(table="b", alias=None, kind="INNER", on_terms=(("a", "x", "b", "x"),))
    assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], [edge]) is True
    out = dx.emit_sql(parsed).lower()
    assert " from a, b" in out or " from a,b" in out
    assert "cross join" not in out


@pytest.mark.fast
def test_intent_null_where_accepted() -> None:
    payload = {
        "tables": ["customers"],
        "grain": "row_level",
        "select": [{"expr": "customers.customer_id"}],
        "group_by": [],
        "order_by": [],
        "where": None,
        "having": None,
    }
    parsed = parse_intent_response(json.dumps(payload), "list customers")
    assert parsed is not None
    assert parsed.where is None
    assert parsed.having is None


@pytest.mark.fast
def test_async_ask_step() -> None:
    inner = MagicMock()
    inner.ask.return_value = SessionStep(done=False, prompt="y/n?", kind="awaiting_intent_confirm")
    inner.step.return_value = SessionStep(done=True, prompt=None, kind="result", sql="SELECT 1")
    ap = AsyncPipelineSession(inner)

    async def _run() -> None:
        first = await ap.ask("list customers")
        assert first.done is False
        second = await ap.step("y")
        assert second.done is True
        assert second.kind == "result"

    asyncio.run(_run())
    inner.ask.assert_called_once_with("list customers")
    inner.step.assert_called_once_with("y")


@pytest.mark.fast
def test_pparams_on_sql_suspend() -> None:
    intent_sig = ConcreteIntent(
        intent_id="i1",
        tables=["payment"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("payment.payment_date"),
                    op="=",
                    value_type="date",
                    param_key="p1",
                )
            ]
        ),
        limit_param_key="s1",
    )
    tmpl = Template(
        id="T0009",
        intent_signature=intent_sig,
        intent_key="k9",
        tables_used=["payment"],
        sql_param="SELECT * FROM payment WHERE payment_date = :p1 LIMIT :s1",
        sql_fp="fp9",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "2024-01-01", "s1": 10}],
            questions=["payments on date"],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
        param_display_names={"p1": "payment date", "s1": "row limit"},
    )
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
    intent = RuntimeIntent(
        tables=["payment"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        param_values={"p1": "2024-01-01", "s1": 10},
        sql_param=tmpl.sql_param,
    )
    tail = InteractiveTailSnapshot(
        q_norm="payments on date",
        intent=intent,
        schema=owner._schema_graph,
        store=owner._store,
        templates=owner._templates,
        rejected=owner._rejected,
        schema_terms=set(),
        dialect=owner._dialect,
        semantic_warnings=(),
        has_union_match=False,
        cols_changed=False,
        matched_template=tmpl,
        union_select_cols=None,
        structural_match_templates=(),
        ikey="k9",
        intent_sim=0.0,
    )
    gen_out = SqlGenerationOutcome(
        sql=tmpl.sql_param,
        success=True,
        generation_path=GenerationPath.EXACT_QUESTION_REUSE,
        matched_template=tmpl,
    )
    ctx = SqlFeedbackSuspendContext(
        tail=tail,
        execution_intent=intent,
        sql=tmpl.sql_param,
        preview_rows=((1,),),
        sql_parameters=(("p1", "2024-01-01"), ("s1", 10)),
        suspended_at=None,
        tmpl_sd=None,
        gen_out=gen_out,
        matched_rejected_template=None,
        force_feedback=False,
    )
    sess = PipelineSession(owner, mode="reader", space_name="master")
    step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "Is this correct?", ctx))
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    handles = [b.handle for b in step.parameters]
    assert handles == ["p1"]
    assert isinstance(step.parameters[0], ParameterBinding)
    assert step.parameters[0].column_expr == "payment.payment_date"


@pytest.mark.fast
def test_ask_schema_catalog_meta_count() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner, mode="writer")
    schema = _engine_schema()
    dump = MainExecutionOps.build_meta_schema_dump(schema)
    answer = {
        "response_kind": "schema_catalog",
        "headline": "One table.",
        "counts": {
            "tables": dump["inventory"]["table_count"],
            "columns": None,
            "members": None,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [],
        "relationships": [],
        "notes": [],
    }
    meta_step = SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_META,
        sql=None,
        message=MainExecutionOps.format_meta_schema_message(answer),
        meta_payload=answer,
    )
    validation = QuestionValidationResult(
        accepted=True, route=QuestionRoute.SCHEMA_CATALOG, corrected="how many tables"
    )

    def _load(schema_g, store, templates, rejected, schema_terms, dialect=None):
        return dialect or MagicMock(), schema or schema_g, store, templates, rejected, schema_terms

    with (
        patch("aetherdialect._pipeline.EngineConfig.llm_credentials_configured", return_value=True),
        patch("aetherdialect._main_execution.load_pipeline_resources", side_effect=_load),
        patch("aetherdialect._main_execution.validate_question", return_value=validation),
        patch("aetherdialect._main_execution.match_question_level_template_reuse", side_effect=_reuse_none),
        patch.object(MainExecutionOps, "answer_metadata_question", return_value=meta_step),
        patch("aetherdialect._main_execution.normalize_question_via_llm") as normalize_mock,
    ):
        step = sess.ask("how many tables")
    assert step.kind == SESSION_KIND_META
    assert step.sql is None
    assert step.meta_payload is not None
    assert step.meta_payload["counts"]["tables"] == dump["inventory"]["table_count"]
    assert str(dump["inventory"]["table_count"]) in (step.message or "")
    normalize_mock.assert_not_called()
    assert "meta.route.schema_catalog" in {d.code for d in step.diagnostics}


@pytest.mark.fast
def test_ask_business_knowledge_prose() -> None:
    entries = (BusinessKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),)
    llm_answer = {"response_kind": "business_knowledge", "message": "ARR is annual recurring revenue."}
    with (
        business_knowledge_scope(entries=entries, digest="d1"),
        patch("aetherdialect._main_execution.LLMProvider.json", return_value=llm_answer),
    ):
        step = MainExecutionOps.answer_metadata_question(
            MagicMock(), "what is ARR", QuestionRoute.BUSINESS_KNOWLEDGE, None, None, None
        )
    assert step.kind == SESSION_KIND_META
    assert step.message == "ARR is annual recurring revenue."
    assert step.meta_payload == {"response_kind": "business_knowledge"}


@pytest.mark.fast
def test_analytical_ask_no_meta_diagnostics() -> None:
    owner = _session_owner()
    sess = PipelineSession(owner, mode="writer")
    validation = QuestionValidationResult(accepted=True, route=QuestionRoute.ANALYTICAL, corrected="list orders")

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
    assert not any(str(d.code).startswith("meta.") for d in step.diagnostics)


@pytest.mark.fast
def test_execute_template_after_accept_returns_rows() -> None:
    tmpl = Template(
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
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.tables = {}
    owner._store = {}
    owner._templates = {tmpl.id: tmpl}
    owner._rejected = {}
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    owner._sandbox_closed = False
    owner._pipeline_writer_lock = None
    owner._artifacts_dir = None
    owner._business_knowledge = None
    owner._ask_phase_callback = None
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._schema_terms = set()
    sess = PipelineSession(owner, mode="reader", space_name="master")
    sess.note_turn_outcome(
        outcome="success",
        sql="SELECT COUNT(*) FROM t1",
        rows=[(3,)],
        columns=("count",),
        matched_template=tmpl,
    )
    step = sess._completed_step()
    assert step.template_id == "T0777"
    engine = MagicMock()
    engine.execute_template = MagicMock(return_value=SimpleNamespace(rows=((3,),)))
    result = engine.execute_template(step.template_id, {})
    assert result.rows == ((3,),)
    engine.execute_template.assert_called_once_with("T0777", {})


@pytest.mark.fast
def test_federation_multi_member_sql_dict() -> None:
    bundle = FederatedSqlBundle(
        statements=(
            FederatedStatementRecord(source_id="crm", engine="postgresql", statement="SELECT 1 FROM crm.t"),
            FederatedStatementRecord(source_id="ops", engine="duckdb", statement="SELECT 2 FROM ops.t"),
        ),
        display_sql="/* federated */",
    )
    sql = MainExecutionOps._resolved_session_step_sql(
        None,
        federated_bundle=bundle,
        generation_path=GenerationPath.FEDERATION_PLAN,
    )
    assert isinstance(sql, dict)
    assert sql["crm"] == "SELECT 1 FROM crm.t"
    assert sql["ops"] == "SELECT 2 FROM ops.t"


@pytest.mark.fast
def test_owner_warehouse_access_error_no_contact_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    noted: dict[str, object] = {}

    def _capture(_choice_port: object, **kwargs: object) -> None:
        noted.update(kwargs)

    monkeypatch.setattr("aetherdialect._main_execution.note_interactive_turn", _capture)
    port = SimpleNamespace(_owner=SimpleNamespace(_schema_role="owner"))
    MainExecutionOps._note_access_error_turn(
        port, AccessError("execute", "warehouse privilege missing", reason="warehouse")
    )
    assert noted.get("outcome") == "validation_failed"
    assert noted.get("error") == "warehouse privilege missing"
    assert "contact your administrator" not in str(noted.get("error") or "").lower()
    assert str(noted.get("error") or "") != PERMISSION_DENIED_USER_MESSAGE
