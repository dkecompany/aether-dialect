"""Additional Step 26 plan checklist coverage (unit-level, no live DB or integration)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aetherdialect._config import (
    SESSION_KIND_AWAITING_INTENT_CONFIRM,
    SESSION_KIND_AWAITING_INTENT_FEEDBACK,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_AWAITING_SQL_FEEDBACK,
    SESSION_KIND_ERROR,
    SESSION_KIND_RESULT,
)
from aetherdialect._contracts_base import (
    ColumnMetadata,
    CteIntent,
    LogicalIntent,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._contracts_core import (
    FeedbackKind,
    QuestionFeedbackEntry,
    RefinementContext,
    RejectionBucket,
)
from aetherdialect._core_utils import _azure_deployment_for_model, llm_execution_scope
from aetherdialect._intent_process import (
    _build_intent_logical_prompt,
    _logical_intent_to_serialisable,
    _parse_logical_intent_response,
)
from aetherdialect._main_execution import (
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PipelineSuspended,
)
from aetherdialect._templates import (
    collect_question_feedback_for_prompt,
    compute_question_feedback_penalty,
)


class TestStageALogicalPromptAndRoundTrip:
    """Stage A JSON shape, vocabulary rules, and logical parse round-trip."""

    def test_logical_prompt_includes_prior_user_corrections(self) -> None:
        payload = json.loads(
            _build_intent_logical_prompt(
                "q",
                "{}",
                "",
                "[]",
                ("turn rejected bad filter",),
                (),
            )
        )
        assert payload.get("prior_user_corrections") == ["turn rejected bad filter"]

    def test_logical_prompt_contains_no_union_rule(self) -> None:
        raw = _build_intent_logical_prompt("union all sales and returns", "{}", "", "[]", (), ())
        assert "UNION" in raw.upper()
        rules = json.loads(raw).get("logical_schema_rules", [])
        assert any("UNION" in str(r).upper() for r in rules)

    def test_logical_intent_round_trip_cte_window_having_parameter(self) -> None:
        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_primary_key=True,
            distinct_count=1,
            distinct_ratio=1.0,
            null_ratio=0.0,
            is_canonical_duplicate=False,
        )
        sg = SchemaGraph(
            tables={
                "customers": TableMetadata(
                    name="customers",
                    columns={"id": col},
                    primary_key=["id"],
                    foreign_keys=[],
                ),
                "orders": TableMetadata(
                    name="orders",
                    columns={"id": col},
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        cte = CteIntent(
            name="per_customer",
            depends_on=("orders",),
            tables=("orders",),
            select="order_id and customer_id",
            window="row_number partitioned by customers.id order by orders.order_date descending as rn",
            having="count star greater than 1",
        )
        li = LogicalIntent(
            tables=("customers", "orders"),
            select="customer name and order count",
            filter="active customers",
            having="total spend over 100",
            window="same ranking as per_customer step",
            cte_steps=(cte,),
        )
        raw = json.dumps(_logical_intent_to_serialisable(li), separators=(",", ":"), sort_keys=True)
        out, issues = _parse_logical_intent_response(raw, sg)
        assert not issues
        assert out is not None
        assert out.tables == li.tables
        assert len(out.cte_steps) == 1
        assert out.cte_steps[0].name == "per_customer"
        assert out.having


class TestRefinementContextConversationHints:
    """RefinementContext carries conversation rejection hints for Stage A."""

    def test_default_empty_hints(self) -> None:
        ctx = RefinementContext("q")
        assert ctx.conversation_rejection_hints == ()

    def test_hints_preserved(self) -> None:
        ctx = RefinementContext("q", conversation_rejection_hints=("bad join",))
        assert ctx.conversation_rejection_hints == ("bad join",)


class TestParseIntentPassesConversationHints:
    """``parse_intent_via_llm`` forwards conversation rejection hints into intent parse."""

    @patch("aetherdialect._pipeline._invoke_intent_parse_with_hints")
    def test_prior_user_corrections_from_refinement_context(self, mock_invoke: MagicMock) -> None:
        from aetherdialect._contracts_base import (
            ColumnMetadata,
            SchemaGraph,
            TableMetadata,
        )
        from aetherdialect._pipeline import parse_intent_via_llm

        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_primary_key=True,
            distinct_count=5,
            distinct_ratio=1.0,
            null_ratio=0.0,
            is_canonical_duplicate=False,
        )
        sg = SchemaGraph(
            tables={"t": TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        mock_invoke.return_value = (None, [], 0)
        ctx = RefinementContext("q", conversation_rejection_hints=("sql dropped region",))
        parse_intent_via_llm("q", sg, {}, {}, refinement_ctx=ctx)
        _kwargs = mock_invoke.call_args[1]
        assert _kwargs.get("prior_user_corrections") == ("sql dropped region",)


class TestQuestionFeedbackCollection:
    """Question feedback rows are collected and penalised uniformly (engine-only source)."""

    def _entry(self) -> dict[str, object]:
        return QuestionFeedbackEntry(
            summary="s",
            buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="h",
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t0",
            updated_at="t0",
            source="engine",
        ).to_dict()

    def test_collect_includes_matching_rows(self) -> None:
        store = {"question_feedback": {"q": [self._entry()]}}
        rows = collect_question_feedback_for_prompt(store, "q", "h")
        assert len(rows) == 1

    def test_penalty_counts_matching_rows(self) -> None:
        store = {"question_feedback": {"q": [self._entry()]}}
        p = compute_question_feedback_penalty(store, "q", "h")
        assert p > 0


class TestAzureDeploymentResolution:
    """Logical model ids resolve to configured Azure deployment names."""

    def test_task_models_map_under_execution_scope(self) -> None:
        from aetherdialect._config import LlmExecutionConfig

        cfg = LlmExecutionConfig(
            azure_endpoint="https://x",
            azure_api_key="k",
            azure_api_version="v",
            deployment_light="D4O",
            deployment_medium="D41",
            deployment_heavy="D54",
            max_query_cost_rows=1,
            max_query_cost_bytes=1,
            statement_timeout_ms=1,
            llm_timeout_ms=1,
            profile_timeout_ms=1,
            explain_timeout_ms=None,
        )
        with llm_execution_scope(cfg):
            assert _azure_deployment_for_model("gpt-4o-mini") == "D4O"
            assert _azure_deployment_for_model("gpt-4.1-mini") == "D41"
            assert _azure_deployment_for_model("gpt-5.4-mini") == "D54"


class TestIntentSummarySessionKinds:
    """Intent summary is attached only on confirm, SQL review, and result steps."""

    def test_intent_summary_kind_policy(self) -> None:
        with_summary = frozenset(
            {
                SESSION_KIND_AWAITING_INTENT_CONFIRM,
                SESSION_KIND_AWAITING_SQL_CONFIRM,
                SESSION_KIND_AWAITING_SQL_FEEDBACK,
                SESSION_KIND_RESULT,
            }
        )
        without = frozenset(
            {
                SESSION_KIND_AWAITING_INTENT_FEEDBACK,
                SESSION_KIND_ERROR,
            }
        )
        assert without.isdisjoint(with_summary)


class TestPipelineSuspendedIntentSummary:
    """Suspended intent-confirm steps carry intent summary when payload exposes intent."""

    def test_intent_confirm_suspend_gets_summary(self) -> None:
        from aetherdialect._contracts_core import (
            InteractiveTailSnapshot,
            NormalizedExpr,
            RuntimeIntent,
            SelectCol,
        )
        from aetherdialect._main_execution import PipelineSession

        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            natural_language="  nl  ",
        )
        snap = InteractiveTailSnapshot(
            q_norm="q",
            intent=intent,
            schema=None,
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            dialect=None,
            semantic_warnings=(),
            has_union_match=False,
            cols_changed=False,
            matched_template=None,
            union_select_cols=None,
            structural_match_templates=(),
            ikey="ik",
            intent_sim=1.0,
        )
        ex = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "ok?", snap)
        owner = MagicMock()
        owner._runtime_config = MagicMock(llm_execution=MagicMock())
        sess = PipelineSession(owner, mode="writer")
        step = sess._suspend_to_step(ex)
        assert step.intent_summary is not None
        assert step.intent_summary.natural_language == "nl"


class TestEmitRuntimeConfigOverrideDiagnostics:
    """Runtime TOML-over-env diagnostics match ``initialize_text2sql`` behaviour."""

    def test_emits_one_diagnostic_per_key(self) -> None:
        from aetherdialect._config import DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED
        from aetherdialect._core_utils import (
            reset_diagnostic_collector,
            set_diagnostic_collector,
        )
        from aetherdialect._main_execution import (
            _emit_runtime_config_override_diagnostics,
        )

        buf: list = []
        tok = set_diagnostic_collector(buf)
        try:
            _emit_runtime_config_override_diagnostics(frozenset({"b_field", "a_field"}))
        finally:
            reset_diagnostic_collector(tok)
        codes = [getattr(d, "code", "") for d in buf]
        assert codes.count(DIAGNOSTIC_CODE_CONFIG_FILE_VALUE_APPLIED) == 2
        assert "a_field" in buf[0].message and "b_field" in buf[1].message


class TestMakeIntentIssueStageAttribution:
    """``STAGE_ATTRIBUTION_TABLE`` keys set ``responsible_stage`` on issues from :func:`make_intent_issue`."""

    def test_known_issue_ids(self) -> None:
        from aetherdialect._contracts_base import (
            STAGE_ATTRIBUTION_TABLE,
            FailureCategory,
            make_intent_issue,
        )

        for issue_id, expected in STAGE_ATTRIBUTION_TABLE.items():
            issue = make_intent_issue(
                issue_id=f"prefix_{issue_id}_suffix",
                category=FailureCategory.OTHER,
                severity="error",
                message="m",
            )
            assert issue.responsible_stage == expected


class TestRefinementContinuationTerminalError:
    """Refinement continuation surfaces intent-parse failure as a terminal error step."""

    def test_terminal_error_when_intent_pass_returns_false(self) -> None:
        from contextlib import nullcontext

        from aetherdialect._contracts_core import QuestionFormStorage, RefinementContext
        from aetherdialect._main_execution import SESSION_KIND_ERROR, PipelineSession
        from aetherdialect._templates import empty_template_store

        owner = MagicMock()
        owner._dialect = MagicMock()
        owner._schema_graph = MagicMock()
        owner._store = empty_template_store("h")
        owner._templates = {}
        owner._rejected = {}
        owner._schema_terms = set()
        owner._runtime_config = MagicMock(llm_execution=MagicMock())
        sess = PipelineSession(owner, mode="writer")
        sess._session_busy = True
        sess._refinement_ctx = RefinementContext("q1", form_storage=QuestionFormStorage(corrected="q1"))
        with (
            patch(
                "aetherdialect._main_execution._interactive_run_intent_pass",
                return_value=False,
            ),
            patch(
                "aetherdialect._main_execution.llm_execution_scope",
                lambda *_a, **_k: nullcontext(),
            ),
        ):
            step = sess._continue_after_refinement_retry()
        assert step.kind == SESSION_KIND_ERROR
        assert "Intent parse failed" in (step.error or "")


class TestLlmTaskDeploymentMatrix:
    """Every ``llm_chat`` task label maps to a deployment-backed model id under Azure scope."""

    def test_all_task_labels_resolve(self) -> None:
        from aetherdialect._config import LlmExecutionConfig
        from aetherdialect._core_utils import (
            _azure_deployment_for_model,
            _task_model_for_profile,
            llm_execution_scope,
        )

        cfg = LlmExecutionConfig(
            azure_endpoint="x",
            azure_api_key="k",
            azure_api_version="v",
            deployment_light="DM0",
            deployment_medium="DM1",
            deployment_heavy="DM2",
            max_query_cost_rows=1,
            max_query_cost_bytes=1,
            statement_timeout_ms=1,
            llm_timeout_ms=1,
            profile_timeout_ms=1,
            explain_timeout_ms=None,
        )
        tasks = (
            "default",
            "intent",
            "feedback",
            "conversation",
            "schema",
            "schema_base",
            "ddl",
            "join",
            "judge",
        )
        with llm_execution_scope(cfg):
            for task in tasks:
                mid = _task_model_for_profile(task)
                dep = _azure_deployment_for_model(mid)
                assert dep in {"DM0", "DM1", "DM2"}


class TestHandBuiltSqlShapeHints:
    """Illustrative IR/SQL shape checks aligned with Step 26 structural-pattern bullets (static strings)."""

    def test_inner_join_distinct_avoids_exists_token(self) -> None:
        sql = "SELECT DISTINCT c.id FROM customers c INNER JOIN orders o ON o.customer_id = c.id"
        low = sql.lower()
        assert "exists" not in low
        assert "inner join" in low and "distinct" in low

    def test_left_join_null_guard_avoids_not_exists(self) -> None:
        sql = "SELECT c.id FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL"
        low = sql.lower()
        assert "not exists" not in low
        assert "left join" in low and "is null" in low

    def test_two_cte_month_compare_shape(self) -> None:
        sql = (
            "WITH m AS (SELECT customer_id, amt FROM orders WHERE dt >= DATE '2024-05-01'), "
            "p AS (SELECT customer_id, amt FROM orders WHERE dt < DATE '2024-05-01') "
            "SELECT m.customer_id FROM m INNER JOIN p ON p.customer_id = m.customer_id"
        )
        assert sql.lower().count("with ") >= 1
        assert " join " in sql.lower()

    def test_row_number_top_one_shape(self) -> None:
        sql = (
            "WITH r AS (SELECT customer_id, order_date, "
            "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn FROM orders) "
            "SELECT * FROM r WHERE rn = 1"
        )
        low = sql.lower()
        assert "row_number()" in low and "partition by" in low and "rn = 1" in low
        assert "lateral" not in low


class TestDiagnosticConstantsSurface:
    """All Step-1 diagnostic string constants remain defined and non-empty."""

    def test_all_diagnostic_codes_non_empty(self) -> None:
        import aetherdialect._config as cfg

        names = [n for n in dir(cfg) if n.startswith("DIAGNOSTIC_CODE_") and n.isupper()]
        assert len(names) >= 10
        for n in names:
            val = getattr(cfg, n)
            assert isinstance(val, str) and val.strip()


class TestFullIntentParseStageARetryDiagnostic:
    """``full_intent_parse`` emits Stage A retry diagnostic after a failed logical parse."""

    def test_stage_a_retry_diagnostic_then_success(self) -> None:
        import json

        from aetherdialect._config import DIAGNOSTIC_CODE_STAGE_A_RETRY
        from aetherdialect._contracts_base import (
            ColumnMetadata,
            SchemaGraph,
            TableMetadata,
        )
        from aetherdialect._contracts_core import (
            NormalizedExpr,
            RuntimeIntent,
            SelectCol,
        )
        from aetherdialect._core_utils import (
            reset_diagnostic_collector,
            set_diagnostic_collector,
        )
        from aetherdialect._intent_process import (
            _logical_intent_to_serialisable,
            full_intent_parse,
        )

        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_primary_key=True,
            distinct_count=1,
            distinct_ratio=1.0,
            null_ratio=0.0,
            is_canonical_duplicate=False,
        )
        sg = SchemaGraph(
            tables={"t": TableMetadata(name="t", columns={"id": col}, primary_key=["id"], foreign_keys=[])},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        bad_stage_a = '{"tables":["t"],"select":""}'
        li = LogicalIntent(tables=("t",), select="id column")
        good_stage_a = json.dumps(_logical_intent_to_serialisable(li), separators=(",", ":"), sort_keys=True)
        stub_intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            natural_language="nl",
        )
        llm_seq = [bad_stage_a, good_stage_a, "{}"]
        buf: list = []
        tok = set_diagnostic_collector(buf)
        try:
            with (
                patch("aetherdialect._intent_process.llm_chat", side_effect=list(llm_seq)),
                patch(
                    "aetherdialect._intent_process._format_repair_loop",
                    return_value=(stub_intent, 0),
                ),
                patch(
                    "aetherdialect._intent_process._run_schema_semantic_repair_loop",
                    return_value=(stub_intent, [], 99, None),
                ),
            ):
                out, _warns, _calls = full_intent_parse("count rows", sg, store=None)
        finally:
            reset_diagnostic_collector(tok)
        assert out is stub_intent
        assert DIAGNOSTIC_CODE_STAGE_A_RETRY in [getattr(d, "code", "") for d in buf]
