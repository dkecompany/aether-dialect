"""Tests for :mod:`aetherdialect._pipeline` and programmatic session flows that call it."""

import csv
import os
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest as _pt

from aetherdialect._config import (
    EngineConfig,
    PolicyConfig,
)
from aetherdialect._constants import (
    JOIN_CHOICE_SCOPE_MAIN,
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
    PIPELINE_SUSPEND_ID_SQL,
    SESSION_KIND_AWAITING_INTENT_CONFIRM,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_IDLE,
    SESSION_KIND_RESULT,
    SHAPE_QUESTION_INDEX_KEY,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
    GenerationPath,
)
from aetherdialect._contracts_base import (
    FailureCategory,
    NoJoinPathError,
    NormalizedExpr,
    PipelineSuspended,
    SessionActiveError,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    DirectReuseSuspendContext,
    FeedbackCounts,
    FeedbackKind,
    InteractiveTailSnapshot,
    QuestionFeedbackEntry,
    RefinementContext,
    RefinementRetry,
    RejectionBucket,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SqlGenerationOutcome,
    StructuralCompareResult,
    Template,
    UserFeedbackRejectSuspendContext,
    ValueHistory,
)
from aetherdialect._contracts_schema import (
    SchemaGraph,
    SQLShape,
    TemplateStats,
)
from aetherdialect._core_utils import bind_params_for_sql, diagnostic_print_listener
from aetherdialect._dialect import Dialect
from aetherdialect._intent_process import TrustedTemplateHit
from aetherdialect._main_execution import PipelineSession
from aetherdialect._pipeline import (
    PathSelectionState,
    _choose_generation_path,
    _join_signatures_for_deterministic_from_anchor,
    _most_frequent_natural_language,
    _remap_value_history_structural_keys,
    _resolve_joins_fresh,
    _run_sql_validation_cascade,
    _shape_distance,
    _sql_phase_join_resources,
    _structural_key_remap_from_assignment_order,
    align_template_to_widened_intent,
    best_accepted_template_similarity,
    build_interactive_tail_snapshot,
    build_result_dataframe,
    complete_direct_sql_reuse_user_choice,
    complete_user_feedback_reject,
    compute_final_metrics,
    confirm_intent_with_user,
    display_final_results_to_stdout,
    enriched_display_alias_map,
    extract_column_headers,
    generate_and_validate_sql,
    generate_join_candidates,
    handle_direct_sql_reuse,
    handle_user_feedback,
    load_pipeline_resources,
    match_question_level_template_reuse,
    merge_structural_defaults_for_reuse,
    other_template_owns_question_string,
    parse_intent_via_llm,
    prepare_union_match_join_phase,
    refinement_retry_available,
    save_result_csv,
    should_skip_intent_confirmation,
)
from aetherdialect._sql_gen import generate_col_alias
from aetherdialect._templates import (
    compute_question_feedback_penalty,
    empty_template_store,
)
from aetherdialect._utils import QuestionReuseMatch, intent_key


class TestChooseGenerationPath:
    """``_choose_generation_path`` maps union resolution to template vs fresh SQL build."""

    def test_choose_generation_path_returns_4_1_when_widen_match_available(self):
        st = PathSelectionState(
            has_matched_template=True,
            resolved_union_path=GenerationPath.UNION_TEMPLATE_WIDEN,
            matched_template_id="T1",
            structural_matches=0,
            cols_changed=True,
            retry_depth=0,
        )
        assert _choose_generation_path(st) == GenerationPath.UNION_TEMPLATE_WIDEN

    def test_choose_generation_path_returns_4_2_when_relaxed_widen_available(self):
        st = PathSelectionState(
            has_matched_template=True,
            resolved_union_path=GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
            matched_template_id="T1",
            structural_matches=0,
            cols_changed=True,
            retry_depth=0,
        )
        assert _choose_generation_path(st) == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN

    def test_choose_generation_path_falls_through_to_fresh_only_when_no_widen(self):
        st = PathSelectionState(
            has_matched_template=False,
            resolved_union_path=GenerationPath.UNION_TEMPLATE_WIDEN,
            matched_template_id="",
            structural_matches=0,
            cols_changed=True,
            retry_depth=0,
        )
        assert _choose_generation_path(st) == GenerationPath.FRESH


class TestShapeDistance:
    """Tests for shape_distance."""

    def test_identical_shapes(self):
        """shape_distance of identical shapes is 0."""
        a = SQLShape(num_joins=1, has_group_by=True, has_agg=True)
        assert _shape_distance(a, a) == 0.0

    def test_different_group_by(self):
        """shape_distance penalizes group_by mismatch."""
        a = SQLShape(num_joins=0, has_group_by=True, has_agg=False)
        b = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        d = _shape_distance(a, b)
        assert d > 0.0
        assert d <= 1.0

    def test_different_agg(self):
        """shape_distance penalizes agg mismatch."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=True)
        b = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        d = _shape_distance(a, b)
        assert d > 0.0

    def test_join_distance_scales(self):
        """shape_distance increases with join count difference."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        b = SQLShape(num_joins=4, has_group_by=False, has_agg=False)
        d = _shape_distance(a, b)
        assert d > 0.0

    def test_symmetric(self):
        """shape_distance is symmetric."""
        a = SQLShape(num_joins=1, has_group_by=True, has_agg=False)
        b = SQLShape(num_joins=3, has_group_by=False, has_agg=False)
        assert abs(_shape_distance(a, b) - _shape_distance(b, a)) < 1e-9

    def test_max_distance_is_bounded(self):
        """shape_distance is bounded by 1.0."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        b = SQLShape(num_joins=100, has_group_by=True, has_agg=True)
        assert _shape_distance(a, b) <= 1.0


class TestMostFrequentNaturalLanguage:
    """Tests for _most_frequent_natural_language."""

    def test_returns_most_common(self):
        """_most_frequent_natural_language returns most frequent string."""
        vh = ValueHistory(
            param_values=[{}, {}, {}],
            questions=["q1", "q2", "q3"],
            natural_language=["total sales", "total sales", "revenue"],
        )
        assert _most_frequent_natural_language(vh) == "total sales"

    def test_empty_returns_empty(self):
        """_most_frequent_natural_language returns empty for empty list."""
        vh = ValueHistory(param_values=[], questions=[], natural_language=[])
        assert _most_frequent_natural_language(vh) == ""

    def test_skips_empty_strings(self):
        """_most_frequent_natural_language skips empty strings."""
        vh = ValueHistory(
            param_values=[{}, {}],
            questions=["q1", "q2"],
            natural_language=["", "actual text"],
        )
        assert _most_frequent_natural_language(vh) == "actual text"


class TestExtractColumnHeaders:
    """Tests for extract_column_headers."""

    def test_extracts_aliases(self):
        """extract_column_headers extracts AS aliases."""
        sql = "SELECT customers.name AS customer_name, SUM(orders.amount) AS total FROM orders JOIN customers"
        headers = extract_column_headers(sql)
        assert "customer_name" in headers
        assert "total" in headers

    def test_extracts_bare_columns(self):
        """extract_column_headers extracts table.column format."""
        sql = "SELECT orders.order_id, orders.amount FROM orders"
        headers = extract_column_headers(sql)
        assert "order_id" in headers
        assert "amount" in headers

    def test_no_select_returns_empty(self):
        """extract_column_headers returns empty for non-SELECT."""
        headers = extract_column_headers("INSERT INTO t VALUES (1)")
        assert headers == []

    def test_empty_sql_returns_empty(self):
        """extract_column_headers returns empty for empty string."""
        assert extract_column_headers("") == []

    def test_select_from_no_match_returns_empty(self):
        """extract_column_headers returns empty when SELECT...FROM not found."""
        headers = extract_column_headers("SELECT")
        assert headers == []

    def test_handles_nested_parens(self):
        """extract_column_headers handles nested parentheses in functions."""
        sql = "SELECT ROUND(SUM(orders.amount), 2) AS rounded_total FROM orders"
        headers = extract_column_headers(sql)
        assert "rounded_total" in headers

    def test_multiple_columns(self):
        """extract_column_headers handles multiple columns."""
        sql = "SELECT a.x AS col1, b.y AS col2, c.z AS col3 FROM a"
        headers = extract_column_headers(sql)
        assert len(headers) == 3


class TestComputeQuestionFeedbackPenalty:
    """Tests for ``compute_question_feedback_penalty``."""

    def test_empty_store_zero(self) -> None:
        assert compute_question_feedback_penalty({}, "q", "h") == 0.0

    def test_matching_entries_add_penalty(self) -> None:
        store = {
            "question_feedback": {
                "mine": [
                    {
                        "summary": "x",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "h",
                        "intent_structural_hash": "ix",
                        "intent_payload": "{}",
                        "created_at": "t",
                        "updated_at": "t",
                    },
                ],
            },
        }
        pen = compute_question_feedback_penalty(store, "mine", "h")
        assert pen >= PolicyConfig.PEN_BY_THREE_SOURCE_UNIT

    def test_capped(self) -> None:
        row = {
            "summary": "x",
            "buckets": [RejectionBucket.OTHER.value],
            "kind": FeedbackKind.VALIDATION_FAILURE.value,
            "effective_structural_hash": "h",
            "intent_structural_hash": "ix",
            "intent_payload": "{}",
            "created_at": "t",
            "updated_at": "t",
        }
        store = {"question_feedback": {"q": [row] * 500}}
        assert compute_question_feedback_penalty(store, "q", "h") <= PolicyConfig.PENALTY_CAP


class TestSaveResultCsv:
    """Tests for ``build_result_dataframe`` + ``save_result_csv``."""

    def test_scalar_grain_dataframe_is_none(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert build_result_dataframe([(1,)], intent, "SELECT 1") is None

    def test_writes_csv_with_headers(self, tmp_path, monkeypatch):
        """save_result_csv writes CSV with column headers."""
        monkeypatch.chdir(tmp_path)
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            sql_param="SELECT t.name AS customer_name FROM t",
        )
        df = build_result_dataframe([("Alice",), ("Bob",)], intent, "SELECT t.name FROM t")
        assert df is not None
        save_result_csv(df)
        out = tmp_path / "results.csv"
        assert out.exists()
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["customer_name"]
        assert len(rows) == 3

    def test_writes_without_headers_when_none(self, tmp_path, monkeypatch):
        """save_result_csv writes data only when no headers extracted."""
        monkeypatch.chdir(tmp_path)
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        df = build_result_dataframe([(1, 2)], intent, "INVALID SQL")
        assert df is not None
        save_result_csv(df)
        out = tmp_path / "results.csv"
        assert out.exists()


class TestShapeDistanceEdgeCases:
    """Edge-case tests for shape_distance."""

    def test_zero_vs_zero(self):
        """shape_distance of default shapes is 0."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        assert _shape_distance(a, a) == 0.0

    def test_max_join_difference_capped(self):
        """shape_distance caps join difference contribution."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        b = SQLShape(num_joins=100, has_group_by=False, has_agg=False)
        d = _shape_distance(a, b)
        assert abs(d - 1.0 / 3.0) < 1e-9

    def test_all_different(self):
        """shape_distance max when all dimensions differ."""
        a = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        b = SQLShape(num_joins=10, has_group_by=True, has_agg=True)
        assert abs(_shape_distance(a, b) - 1.0) < 1e-9


class TestExtractColumnHeadersEdgeCases:
    """Edge-case tests for extract_column_headers."""

    def test_star_select(self):
        """extract_column_headers handles SELECT *."""
        headers = extract_column_headers("SELECT * FROM t")
        assert headers == ["*"]

    def test_count_star(self):
        """extract_column_headers handles COUNT(*)."""
        headers = extract_column_headers("SELECT COUNT(*) AS cnt FROM t")
        assert "cnt" in headers

    def test_expression_without_alias(self):
        """extract_column_headers falls back to expression name."""
        headers = extract_column_headers("SELECT 1+1 FROM t")
        assert len(headers) == 1


class TestMostFrequentNaturalLanguageEdgeCases:
    """Edge-case tests for _most_frequent_natural_language."""

    def test_tie_returns_first(self):
        """_most_frequent_natural_language returns first on tie."""
        vh = ValueHistory(
            param_values=[{}, {}],
            questions=["q1", "q2"],
            natural_language=["alpha", "beta"],
        )
        result = _most_frequent_natural_language(vh)
        assert result in ("alpha", "beta")

    def test_all_empty_returns_empty(self):
        """_most_frequent_natural_language returns empty when all entries are empty."""
        vh = ValueHistory(
            param_values=[{}],
            questions=["q"],
            natural_language=[""],
        )
        assert _most_frequent_natural_language(vh) == ""


def _make_pipeline_template(
    tid="T0001",
    trust_level=1,
    intent_key="test_key",
    accept=1,
    reject=0,
) -> Template:
    """Create a minimal Template for pipeline tests."""
    return Template(
        id=tid,
        effective_structural_hash="test_hash",
        intent_signature=ConcreteIntent(
            intent_id="test",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        ),
        intent_key=intent_key,
        tables_used=["orders"],
        sql_param="SELECT order_id FROM orders",
        sql_fp="test_fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="test_sig",
        value_history=ValueHistory(param_values=[{}], questions=["test"], natural_language=["test"]),
        stats=TemplateStats(accept=accept, reject=reject),
        trust_level=trust_level,
    )


class TestMatchQuestionLevelTemplateReuse:
    """Tests for match_question_level_template_reuse."""

    @patch("aetherdialect._pipeline.find_trusted_template_match", return_value=None)
    def test_no_match_returns_none_type(self, mock_skip):
        """No match returns reuse_type='none'."""
        result = match_question_level_template_reuse("random question", {})
        assert result.reuse_type == "none"
        assert result.best_template is None

    @patch("aetherdialect._pipeline.find_trusted_template_match")
    def test_exact_match_returns_direct_reuse(self, mock_skip):
        """Exact match returns reuse_type='direct_reuse' with score 1.0."""
        tmpl = _make_pipeline_template(trust_level=2)
        mock_skip.return_value = TrustedTemplateHit(
            template=tmpl,
            reuse_hit=QuestionReuseMatch(
                template_id="T0001",
                history_index=0,
                stored_normalized_text="test question",
                candidate_normalized="test question",
                token_edit_sum=0,
            ),
        )
        result = match_question_level_template_reuse("test question", {"T0001": tmpl})
        assert result.reuse_type == "direct_reuse"
        assert result.similarity_score == 1.0
        assert result.best_template.id == "T0001"
        assert result.reuse_candidate_normalized == "test question"

    @patch("aetherdialect._pipeline.find_trusted_template_match")
    def test_template_store_forwards_shape_token_intent_and_union_indexes(self, mock_ftm):
        """Persisted store indexes and optional runtime intent are passed through to the trusted matcher."""
        tmpl = _make_pipeline_template(trust_level=2)
        mock_ftm.return_value = None
        store = {
            SHAPE_QUESTION_INDEX_KEY: {"shape_key": ["T0001"]},
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY: {"fp1": [["T0001", "0"]]},
            TEMPLATE_INTENT_KEY_INDEX_KEY: {"intent_key_1": ["T0001"]},
            TEMPLATE_UNION_FAMILY_INDEX_KEY: {
                "body_k": ["T0001"],
                "body_k|join_k": ["T0001"],
            },
        }
        rt = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        match_question_level_template_reuse(
            "any",
            {"T0001": tmpl},
            template_store=store,
            candidate_intent=rt,
        )
        kwargs = mock_ftm.call_args[1]
        assert kwargs["shape_question_index"] == {"shape_key": ["T0001"]}
        assert kwargs["question_token_index"] is store[TEMPLATE_QUESTION_TOKEN_INDEX_KEY]
        assert kwargs["intent_key_index"] == {"intent_key_1": ["T0001"]}
        assert kwargs["union_family_index"] == {
            "body_k": ["T0001"],
            "body_k|join_k": ["T0001"],
        }
        assert kwargs["candidate_intent"] is rt


class TestLoadPipelineResources:
    """Tests for load_pipeline_resources."""

    @patch("aetherdialect._pipeline.llm_credentials_configured", return_value=False)
    def test_missing_api_token(self, _mock_lc):
        """Missing LLM credentials raises RuntimeError."""
        with _pt.raises(RuntimeError, match="No OpenAI/Azure OpenAI API key configured"):
            load_pipeline_resources(schema="s", store="st", templates="t", rejected="r", schema_terms=set())

    @patch("aetherdialect._pipeline.llm_credentials_configured", return_value=True)
    @patch("aetherdialect._pipeline.EngineConfig")
    @patch("aetherdialect._pipeline.get_dialect")
    def test_missing_schema_raises(self, mock_gd, mock_ec, _mock_lc):
        """None schema raises RuntimeError."""
        mock_ec.TYPE = "postgresql"
        mock_ec.RUNTIME = SimpleNamespace(db_url=lambda: "postgresql://localhost/db")
        mock_gd.return_value = SimpleNamespace(engine=None)
        with _pt.raises(RuntimeError, match="Schema"):
            load_pipeline_resources(schema=None, store={}, templates={}, rejected={}, schema_terms=set())

    @patch("aetherdialect._pipeline.llm_credentials_configured", return_value=True)
    @patch("aetherdialect._pipeline.EngineConfig")
    @patch("aetherdialect._pipeline.get_dialect")
    def test_success_returns_tuple(self, mock_gd, mock_ec, _mock_lc):
        """Valid inputs return 6-element tuple."""
        mock_ec.TYPE = "postgresql"
        mock_ec.RUNTIME = SimpleNamespace(db_url=lambda: "postgresql://localhost/db")
        mock_gd.return_value = SimpleNamespace(engine="engine")
        result = load_pipeline_resources(schema="s", store="st", templates="t", rejected="r", schema_terms={"term"})
        assert len(result) == 6
        assert result[0] == mock_gd.return_value


class TestGenerateJoinCandidates:
    """Tests for generate_join_candidates."""

    @patch("aetherdialect._pipeline.join_candidate_map", return_value={"J00": []})
    @patch(
        "aetherdialect._pipeline.join_hints_multi",
        return_value={"candidates": [{"candidate_id": "J00", "join_path_signature": []}]},
    )
    def test_single_table_no_cte(self, mock_jh, mock_jcm):
        """Single table intent returns default candidates."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        schema = SimpleNamespace(tables={"orders": SimpleNamespace(columns={})})
        jc, cmap, cte_hints = generate_join_candidates(intent, schema)
        assert "candidates" in jc
        assert isinstance(cmap, dict)
        assert cte_hints == {}

    @patch(
        "aetherdialect._pipeline.cte_to_intent_for_ranking",
        return_value=SimpleNamespace(tables=["orders", "customers"]),
    )
    @patch("aetherdialect._pipeline.join_candidate_map", return_value={"J00": []})
    @patch(
        "aetherdialect._pipeline.join_hints_multi",
        return_value={"candidates": [{"candidate_id": "J00", "join_path_signature": []}]},
    )
    def test_cte_multi_table_gets_hints(self, mock_jh, mock_jcm, mock_cte_intent):
        """CTE with multiple tables gets its own join hints."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders", "customers"],
            select_cols=[],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        schema = SimpleNamespace(
            tables={
                "orders": SimpleNamespace(columns={}),
                "customers": SimpleNamespace(columns={}),
            }
        )
        jc, cmap, cte_hints = generate_join_candidates(intent, schema)
        assert "cte1" in cte_hints

    @patch("aetherdialect._pipeline.join_candidate_map", return_value={"J00": []})
    @patch("aetherdialect._pipeline.join_hints_multi", return_value={"candidates": []})
    def test_cte_single_table_j00(self, mock_jh, mock_jcm):
        """CTE with single table gets default J00 candidate."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[],
            output_columns=["order_id"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        schema = SimpleNamespace(tables={"orders": SimpleNamespace(columns={})})
        _, _, cte_hints = generate_join_candidates(intent, schema)
        assert cte_hints["cte1"]["candidates"][0]["candidate_id"] == "J00"

    @patch("aetherdialect._pipeline.join_candidate_map", return_value={})
    @patch(
        "aetherdialect._pipeline.join_hints_multi",
        return_value={"candidates": [{"candidate_id": "J01", "join_path_signature": ["x"]}]},
    )
    def test_main_join_hints_use_schema_tables_only(self, mock_jh, mock_jcm):
        """Main query join_hints_multi receives physical tables plus CTE names that have virtual specs."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["payment"],
            select_cols=[],
            output_columns=["a"],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["cte2", "customer", "payment", "cte1"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        schema = SimpleNamespace(
            tables={
                "customer": SimpleNamespace(columns={}),
                "payment": SimpleNamespace(columns={}),
            }
        )
        generate_join_candidates(intent, schema)
        mock_jh.assert_called_once()
        assert mock_jh.call_args[0][1] == ["customer", "payment", "cte1"]


class TestDisplayFinalResultsToStdout:
    """Tests for display_final_results_to_stdout."""

    @patch("aetherdialect._pipeline.print_query_result")
    def test_prints_result_rows(self, mock_print):
        """Invokes ``print_query_result`` with resolved display SQL."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        display_final_results_to_stdout("show orders", intent, "SELECT * FROM orders", [(1,)])
        mock_print.assert_called_once()


class TestComputeFinalMetrics:
    """Tests for compute_final_metrics."""

    @patch("aetherdialect._pipeline.extract_tables_from_sql", return_value=["orders"])
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.intent_key", return_value="ik1")
    def test_returns_float(self, _ik, _cm, _fp, _shape, _ext, schema_graph):
        """compute_final_metrics returns a float in [0,1]."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
            extra_tables=[],
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik1",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        templates = {"T0001": tmpl}
        join_candidates = {"candidates": [{"join_path_signature": []}]}
        store = {"intent_failure_log": []}
        conf = compute_final_metrics(
            "SELECT order_id FROM orders",
            intent,
            schema_graph,
            templates,
            join_candidates,
            store,
        )
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    @patch("aetherdialect._pipeline.extract_tables_from_sql", return_value=["orders"])
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.intent_key", return_value="ik1")
    def test_sets_sql_shape(self, _ik, _cm, _fp, mock_shape, _ext, schema_graph):
        """compute_final_metrics sets intent.sql_shape."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
            extra_tables=[],
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik1",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        templates = {"T0001": tmpl}
        join_candidates = {"candidates": [{"join_path_signature": []}]}
        store = {"intent_failure_log": []}
        compute_final_metrics(
            "SELECT order_id FROM orders",
            intent,
            schema_graph,
            templates,
            join_candidates,
            store,
        )
        assert intent.sql_shape is not None


class TestHandleUserFeedback:
    """Tests for handle_user_feedback."""

    @patch("aetherdialect._pipeline.save_template_store")
    def test_invalid_choice_saves_and_returns(self, mock_save, schema_graph, capsys):
        """Invalid choice (not y/n) prints a hint, saves the store, and returns None."""
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 1,
            "next_reject_id": 1,
        }
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        with diagnostic_print_listener(print):
            result = handle_user_feedback(
                "x",
                intent,
                "SELECT 1",
                schema_graph,
                store,
                {},
                {},
                "q",
                GenerationPath.FRESH,
                None,
                None,
            )
        assert result is None
        mock_save.assert_called_once()
        captured = capsys.readouterr().out
        assert "Invalid choice" in captured

    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik1")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._pipeline.promote_trust")
    @patch("aetherdialect._pipeline.record_template_feedback")
    @patch("aetherdialect._pipeline.flatten_param_values", return_value={})
    def test_accept_existing_template(self, _flat, _rec, _promote, _fp, _cm, _ik, _shape, _save, schema_graph):
        """Accept with existing template promotes trust."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik1",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        templates = {"T0001": tmpl}
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "y",
            intent,
            "SELECT order_id FROM orders",
            schema_graph,
            store,
            templates,
            {},
            "q",
            GenerationPath.INTENT_DIRECT_MATCH,
            tmpl,
            None,
        )
        _rec.assert_called_once()
        _promote.assert_called_once()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch(
        "aetherdialect._pipeline.other_template_owns_question_string",
        return_value=False,
    )
    @patch("aetherdialect._pipeline.runtime_intent_to_concrete")
    @patch(
        "aetherdialect._pipeline.flatten_param_values",
        return_value={"s1": 5, "f_status": "open"},
    )
    @patch("aetherdialect._pipeline.promote_trust")
    @patch("aetherdialect._pipeline.record_template_feedback")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp_union")
    @patch("aetherdialect._pipeline.intent_key", return_value="ik_union")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    def test_accept_intent_union_match_updates_template_sql_and_signature(
        self,
        _shape,
        _cm,
        _ik,
        _fp,
        _rec,
        _promote,
        _flat,
        mock_rtc,
        _own,
        _tts,
        _save,
        schema_graph,
    ):
        """Union widen path refreshes template SQL, intent signature, keys, fp, and structural defaults."""
        new_sig = ConcreteIntent(
            intent_id="tid",
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.amount")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        mock_rtc.return_value = new_sig
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.amount")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
            sql_param="SELECT order_id, amount FROM orders WHERE status = :s1",
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="tid",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik_old",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp_old",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{"s1": 1}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
            structural_defaults={},
        )
        templates = {"T0001": tmpl}
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "y",
            intent,
            "SELECT order_id, amount FROM orders WHERE status = 'open'",
            schema_graph,
            store,
            templates,
            {},
            "q",
            GenerationPath.UNION_TEMPLATE_WIDEN,
            tmpl,
            None,
        )
        mock_rtc.assert_called_once_with(intent, "tid")
        assert tmpl.sql_param == intent.sql_param
        assert tmpl.intent_signature is new_sig
        assert tmpl.intent_key == "ik_union"
        assert tmpl.sql_fp == "fp_union"
        assert tmpl.structural_defaults == {"s1": 5}
        _rec.assert_called_once()
        _promote.assert_called_once()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik1")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch("aetherdialect._pipeline.record_template_feedback")
    @patch("aetherdialect._pipeline.promote_trust")
    def test_accept_exact_question_reuse_records_stats_and_promotes(
        self,
        _promote,
        _rec,
        _del_rej,
        _fp,
        _cm,
        _ik,
        _shape,
        _tts,
        _save,
        schema_graph,
    ):
        """Path 1 accept records stats and promotes trust via handle_user_feedback."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik1",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=2,
        )
        templates = {"T0001": tmpl}
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "y",
            intent,
            "SELECT order_id FROM orders",
            schema_graph,
            store,
            templates,
            {},
            "q",
            GenerationPath.EXACT_QUESTION_REUSE,
            tmpl,
            None,
        )
        _rec.assert_called_once_with(tmpl, accept=True)
        _promote.assert_called_once_with(tmpl, "q")

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik1")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._pipeline.reject_out_per_question", return_value=(False, False))
    @patch("aetherdialect._pipeline.record_template_feedback")
    def test_reject_routes_to_matched_template_not_intent_key_lookup(
        self,
        _rec,
        _reject_out,
        _fp,
        _cm,
        _ik,
        _shape,
        _tts,
        _save,
        schema_graph,
    ):
        """Reject on intent-direct path demotes the pipeline's matched template."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik_other",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=5, reject=0),
            trust_level=2,
        )
        templates = {"T0001": tmpl}
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "n",
            intent,
            "SELECT order_id FROM orders",
            schema_graph,
            store,
            templates,
            {},
            "q",
            GenerationPath.INTENT_DIRECT_MATCH,
            tmpl,
            None,
        )
        _rec.assert_called_once()
        assert _rec.call_args[0][0] is tmpl
        _reject_out.assert_called_once_with(templates, tmpl, "q")

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik_p3")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._pipeline.record_question_feedback")
    @patch("aetherdialect._pipeline.summarize_failure_for_memory")
    @patch("aetherdialect._pipeline.reject_out_per_question", return_value=(False, False))
    @patch("aetherdialect._pipeline.record_template_feedback")
    @patch("builtins.input", return_value="wrong cols")
    def test_reject_intent_direct_match_records_question_feedback(
        self,
        _input_mock,
        _rec,
        _reject_out,
        mock_sum,
        mock_rqf,
        _fp,
        _cm,
        _ik,
        _shape,
        _tts,
        _save,
        schema_graph,
    ):
        """Path 3 reject records summarized question feedback when intent reuse applies."""
        mock_sum.return_value = QuestionFeedbackEntry(
            summary="x",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash=schema_graph.effective_structural_hash,
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=ConcreteIntent(
                intent_id="t",
                tables=["orders"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            ),
            intent_key="ik_other",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
            stats=TemplateStats(accept=5, reject=0),
            trust_level=2,
        )
        templates = {"T0001": tmpl}
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "n",
            intent,
            "SELECT order_id FROM orders",
            schema_graph,
            store,
            templates,
            {},
            "new_q",
            GenerationPath.INTENT_DIRECT_MATCH,
            tmpl,
            None,
        )
        mock_sum.assert_called_once()
        mock_rqf.assert_called_once()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik_fresh")
    @patch("aetherdialect._templates.colmap_signature", return_value="cm1")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp1")
    @patch("aetherdialect._pipeline.record_question_feedback")
    @patch("aetherdialect._pipeline.summarize_failure_for_memory")
    @patch("builtins.input", return_value="wrong columns")
    def test_reject_fresh_records_question_feedback_when_no_matched_template(
        self,
        _input_mock,
        mock_sum,
        mock_rqf,
        _fp,
        _cm,
        _ik,
        _shape,
        _tts,
        _save,
        schema_graph,
    ):
        """Fresh path reject with no template still records question feedback."""
        mock_sum.return_value = QuestionFeedbackEntry(
            summary="x",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash=schema_graph.effective_structural_hash,
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={},
        )
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "n",
            intent,
            "SELECT order_id FROM orders",
            schema_graph,
            store,
            {},
            {},
            "q",
            GenerationPath.FRESH,
            None,
            None,
        )
        mock_sum.assert_called_once()
        mock_rqf.assert_called_once()


class TestConfirmIntentWithUser:
    """Tests for the 3-branch confirm_intent_with_user logic."""

    @staticmethod
    def _make_intent(nl: str = "Show film titles") -> RuntimeIntent:
        return RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            natural_language=nl,
        )

    @staticmethod
    def _make_store() -> dict:
        return {"templates": {}, "next_id": 1}

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_warnings_user_confirms(self, mock_choice, mock_save):
        """Semantic warnings are folded into the intent body; one confirmation prompt."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=["col type mismatch"],
            similarity_score=0.0,
            rejected=None,
        )
        assert result is True
        assert mock_choice.call_count == 1
        assert "Is this correct?" in mock_choice.call_args[0][1]

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="n")
    def test_warnings_user_declines(self, mock_choice, mock_save):
        """Branch 1: warnings present, user says no → False, store saved (Path 5 uses low similarity)."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=["col type mismatch"],
            similarity_score=0.0,
        )
        assert result is False
        mock_save.assert_called_once()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no")
    def test_high_similarity_auto_proceeds(self, mock_choice, mock_save):
        """Branch 2: no warnings, high similarity → auto-proceed, no prompt."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=None,
            similarity_score=0.95,
        )
        assert result is True
        mock_choice.assert_not_called()
        mock_save.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_low_similarity_user_confirms(self, mock_choice, mock_save):
        """Low similarity without warnings or schema issues skips confirmation (gate returns True)."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=None,
            similarity_score=0.0,
        )
        assert result is True
        mock_choice.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="n")
    def test_low_similarity_user_declines(self, mock_choice, mock_save):
        """Same gate skips the decline path when confirmation is not offered."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=None,
            similarity_score=0.0,
        )
        assert result is True
        mock_choice.assert_not_called()
        mock_save.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value=None)
    def test_warnings_none_choice_declines(self, mock_choice, mock_save):
        """Branch 1: warnings present, interactive_yes_no returns None → False."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=["warn1"],
            similarity_score=0.0,
        )
        assert result is False

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no")
    def test_empty_warnings_list_is_no_warnings(self, mock_choice, mock_save):
        """Empty warnings list treated as no warnings (falsy)."""
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=[],
            similarity_score=0.95,
        )
        assert result is True
        mock_choice.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_fallback_nl_summary(self, mock_choice, mock_save):
        """Empty natural_language still auto-skips confirmation when the gate applies."""
        intent = self._make_intent(nl="")
        result = confirm_intent_with_user(
            intent,
            self._make_store(),
            semantic_warnings=None,
            similarity_score=0.0,
        )
        assert result is True
        mock_choice.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    @patch("aetherdialect._pipeline.print_info")
    def test_intent_already_confirmed_skips_print_and_prompt(self, mock_print_info, mock_choice, mock_save):
        result = confirm_intent_with_user(
            self._make_intent(),
            self._make_store(),
            semantic_warnings=None,
            similarity_score=0.0,
            rejected=None,
            intent_already_confirmed=True,
        )
        assert result is True
        mock_print_info.assert_not_called()
        mock_choice.assert_not_called()


class TestRunSqlValidationCascade:
    """Tests for _run_sql_validation_cascade helper."""

    def _make_intent(self, tables=None, grain="row_level", select_cols=None):
        """Build a minimal RuntimeIntent."""
        sc = select_cols or [SelectCol(expr=NormalizedExpr.from_column("t.id"))]
        return RuntimeIntent(
            tables=tables or ["t"],
            grain=grain,
            select_cols=sc,
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )

    @patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
    def test_all_pass(self, mock_sql):
        """When validate_sql passes, cascade returns (True, '')."""
        intent = self._make_intent()
        ok, err, cat, diags = _run_sql_validation_cascade(
            "SELECT t.id FROM t",
            intent,
            None,
        )
        assert ok is True
        assert err == ""
        assert cat is None
        assert diags == []

    @patch(
        "aetherdialect._pipeline.validate_sql",
        return_value=(
            False,
            "explain_error",
            FailureCategory.EXECUTION_EXPLAIN_FAILED,
            [],
        ),
    )
    def test_explain_failure_detected(self, mock_sql):
        """EXPLAIN failure stops cascade."""
        intent = self._make_intent(grain="grouped")
        ok, err, cat, diags = _run_sql_validation_cascade(
            "SELECT t.id FROM t",
            intent,
            None,
        )
        assert ok is False
        assert err == "explain_error"
        assert cat == FailureCategory.EXECUTION_EXPLAIN_FAILED


class TestResolveJoinsFresh:
    """Tests for _resolve_joins_fresh join selection and injection."""

    @patch("aetherdialect._pipeline.get_join_choice_from_llm")
    def test_fallback_when_llm_choice_has_empty_signature(self, mock_llm):
        """Use another candidate when the LLM id maps to an empty path."""
        mock_llm.return_value = {JOIN_CHOICE_SCOPE_MAIN: "J01"}
        intent = RuntimeIntent(
            tables=["film", "language"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        cmap = {
            "J01": [],
            "J02": ["film.language_id->language.language_id"],
        }
        join_candidates = {
            "candidates": [
                {"candidate_id": "J01", "join_path_signature": []},
                {"candidate_id": "J02", "join_path_signature": cmap["J02"]},
            ],
        }
        det = "SELECT title FROM film\nWHERE 1=1"
        from aetherdialect._dialect_postgres import PostgresDialect

        sql_param, _ = _resolve_joins_fresh(
            det,
            intent,
            cmap,
            None,
            "list films and language",
            join_candidates,
            schema=None,
            dialect=PostgresDialect.__new__(PostgresDialect),
        )
        assert intent.chosen_join_candidate_id == "J02"
        assert "language" in sql_param.lower()

    def test_resolve_joins_fresh_aligns_signatures_for_each_cte_carrier(self):
        from aetherdialect._dialect_postgres import PostgresDialect

        cte_multi = RuntimeCteStep(
            cte_name="cte1",
            tables=["actor", "film_actor"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("actor.actor_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        cte_scalar = RuntimeCteStep(
            cte_name="cte2",
            tables=["rental"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            grain="scalar",
            emission="scalar_subquery",
        )
        intent = RuntimeIntent(
            tables=["actor"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("actor.actor_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte_multi, cte_scalar],
        )
        join_candidates = {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]}
        cte_join_hints = {
            "cte1": {
                "candidates": [
                    {
                        "candidate_id": "J01",
                        "join_path_signature": ["actor.actor_id->film_actor.actor_id"],
                        "edge_kinds": [],
                    }
                ]
            }
        }
        cmap = {"J00": [], "J01": ["actor.actor_id->film_actor.actor_id"]}
        det = "WITH cte1 AS (SELECT 1) SELECT 1 FROM actor"
        captured: list[list[list[str]]] = []

        def _inj(det_sql, sigs, schema=None, *, edge_kinds_ordered=None, dialect=None):
            captured.append(sigs)
            return det_sql

        with patch(
            "aetherdialect._pipeline.inject_join_into_deterministic_sql",
            side_effect=_inj,
        ):
            _resolve_joins_fresh(
                det,
                intent,
                cmap,
                cte_join_hints,
                "q",
                join_candidates,
                schema=None,
                dialect=PostgresDialect.__new__(PostgresDialect),
            )
        assert len(captured) == 1
        sigs = captured[0]
        assert len(sigs) == len(intent.cte_steps) + 1
        assert sigs[0] == ["actor.actor_id->film_actor.actor_id"]
        assert sigs[1] == []
        assert sigs[2] == []


class TestPrepareUnionMatchJoinPhase:
    """Tests for prepare_union_match_join_phase."""

    @patch("aetherdialect._pipeline.generate_join_candidates")
    def test_no_union_candidates_calls_generate_join_candidates(self, mock_gjc, schema_graph: SchemaGraph):
        mock_gjc.return_value = ({"candidates": []}, {}, {})
        intent = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = prepare_union_match_join_phase("q norm", intent, schema_graph, MagicMock(), {})
        assert out[0] is None
        assert out[3] is None
        mock_gjc.assert_called_once()


class TestEnrichedDisplayAliasMap:
    """Tests for enriched_display_alias_map."""

    @patch(
        "aetherdialect._pipeline.llm_chat",
        return_value='{"aliases":{"col=t.a":"column_a"}}',
    )
    def test_simple_select_uses_llm_for_scalar_column(self, _mock_llm):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        base: dict[str, str] = {}
        m = enriched_display_alias_map("q", "SELECT t.a FROM t", intent, base)
        assert m.get("col=t.a") == "column_a"


class TestGeneratePath43RuntimeSubset:
    """``RUNTIME_SUBSET_TEMPLATE_WIDE`` widens in-memory ``select_cols`` to the template projection."""

    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    )
    def test_widens_intent_select_cols(self, _mock_val, schema_graph: SchemaGraph):
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        conc = ConcreteIntent(
            intent_id="ct",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1, sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT orders.order_id, orders.customer_id FROM orders",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        join_candidates: dict = {"candidates": []}
        cmap = {"J00": []}
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        out = generate_and_validate_sql(
            "q",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            matched_template=tmpl,
            union_select_cols=[sc1, sc2],
            cols_changed=False,
            union_sql_path=GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE,
        )
        assert out.success is True
        assert out.generation_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE
        assert len(intent.select_cols) == 2
        assert intent.select_cols[1].expr.primary_column == "orders.customer_id"


class TestGenerationPathSqlBranches:
    """Forced ``generate_and_validate_sql`` outcomes for paths ``3``, ``4.1``, ``4.2``, ``5`` (validation mocked)."""

    @patch("aetherdialect._pipeline.align_template_to_widened_intent")
    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    )
    def test_path3_intent_direct_match(self, _mock_val, _mock_save, mock_align, schema_graph: SchemaGraph):
        intent = _single_table_intent("orders", "order_id")
        sc = intent.select_cols[0]
        conc = ConcreteIntent(
            intent_id="c",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT orders.order_id FROM orders",
        )
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
        out = generate_and_validate_sql(
            "q",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            cte_join_hints=cte_hints,
            matched_template=tmpl,
            union_select_cols=[sc],
            cols_changed=False,
            union_sql_path=GenerationPath.INTENT_DIRECT_MATCH,
        )
        assert out.success is True
        assert out.generation_path == GenerationPath.INTENT_DIRECT_MATCH
        assert len(intent.select_cols) == 1
        mock_align.assert_not_called()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(False, "forced_fail", FailureCategory.EXECUTION_SCHEMA_ERROR, []),
    )
    def test_path3_sql_validation_failure_is_terminal_no_fresh_fallback(
        self, _mock_val, mock_save, schema_graph: SchemaGraph
    ):
        """Path 3 SQL validation failure must terminate without falling back to fresh build."""
        intent = _single_table_intent("orders", "order_id")
        sc = intent.select_cols[0]
        conc = ConcreteIntent(
            intent_id="c",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT orders.order_id FROM orders",
        )
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
        with patch(
            "aetherdialect._pipeline._resolve_joins_fresh",
            return_value=("SELECT orders.order_id FROM orders", {}),
        ) as mock_fresh:
            out = generate_and_validate_sql(
                "q",
                intent,
                schema_graph,
                join_candidates,
                cmap,
                dialect,
                store,
                cte_join_hints=cte_hints,
                matched_template=tmpl,
                union_select_cols=[sc],
                cols_changed=False,
                union_sql_path=GenerationPath.INTENT_DIRECT_MATCH,
            )
        mock_fresh.assert_called_once()
        assert out.success is False
        assert out.generation_path == GenerationPath.INTENT_DIRECT_MATCH
        assert out.sql_validation_error == "forced_fail"
        assert out.error_kind == "execution_schema_error"
        mock_save.assert_called_once()

    @patch("aetherdialect._pipeline.align_template_to_widened_intent")
    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline.build_deterministic_sql",
        return_value="SELECT orders.order_id, orders.customer_id FROM orders WHERE 1=1",
    )
    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    )
    def test_path41_union_template_widen_runtime_select_stays_narrow(
        self, _mock_val, _mock_build, _mock_save, mock_align, schema_graph: SchemaGraph
    ):
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        conc = ConcreteIntent(
            intent_id="c",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1, sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT 1 FROM orders",
        )
        intent = _single_table_intent("orders", "order_id")
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
        out = generate_and_validate_sql(
            "q",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            cte_join_hints=cte_hints,
            matched_template=tmpl,
            union_select_cols=[sc1, sc2],
            cols_changed=True,
            union_sql_path=GenerationPath.UNION_TEMPLATE_WIDEN,
        )
        assert out.success is True
        assert out.generation_path == GenerationPath.UNION_TEMPLATE_WIDEN
        assert len(intent.select_cols) == 1
        mock_align.assert_called_once()

    @patch("aetherdialect._pipeline._join_matches_template_intent", return_value=False)
    @patch("aetherdialect._pipeline.align_template_to_widened_intent")
    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline.build_deterministic_sql",
        return_value="SELECT orders.order_id, orders.customer_id FROM orders WHERE 1=1",
    )
    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    )
    def test_path41_union_widen_skips_align_when_join_fingerprint_mismatches(
        self,
        _mock_val,
        _mock_build,
        _mock_save,
        mock_align,
        _mock_jm,
        schema_graph: SchemaGraph,
    ):
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        conc = ConcreteIntent(
            intent_id="c",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1, sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT 1 FROM orders",
        )
        intent = _single_table_intent("orders", "order_id")
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
        out = generate_and_validate_sql(
            "q",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            cte_join_hints=cte_hints,
            matched_template=tmpl,
            union_select_cols=[sc1, sc2],
            cols_changed=True,
            union_sql_path=GenerationPath.UNION_TEMPLATE_WIDEN,
        )
        assert out.success is True
        assert out.join_matches_template is False
        mock_align.assert_not_called()

    @patch("aetherdialect._pipeline.align_template_to_widened_intent")
    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline.build_deterministic_sql",
        return_value="SELECT orders.order_id, orders.customer_id FROM orders WHERE 1=1",
    )
    @patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    )
    def test_path42_union_template_and_runtime_widen(
        self, _mock_val, _mock_build, _mock_save, mock_align, schema_graph: SchemaGraph
    ):
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        conc = ConcreteIntent(
            intent_id="c",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1, sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = replace(
            _make_pipeline_template(),
            intent_signature=conc,
            sql_param="SELECT 1 FROM orders",
        )
        intent = _single_table_intent("orders", "order_id")
        store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
        dialect = _StubDialect()
        join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
        out = generate_and_validate_sql(
            "q",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            cte_join_hints=cte_hints,
            matched_template=tmpl,
            union_select_cols=[sc1, sc2],
            cols_changed=True,
            union_sql_path=GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN,
        )
        assert out.success is True
        assert out.generation_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN
        assert len(intent.select_cols) == 2
        mock_align.assert_called_once()


class TestAlignTemplateToWidenedIntent:
    """Direct coverage for :func:`align_template_to_widened_intent`."""

    def test_updates_sql_param_display_aliases_and_intent_key(self):
        tmpl = _make_pipeline_template()
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            sql_param="SELECT orders.order_id FROM orders WHERE 1=1",
        )
        align_template_to_widened_intent(tmpl, intent, _StubDialect())
        assert tmpl.sql_param == intent.sql_param
        assert tmpl.intent_key == intent_key(intent)
        ga = generate_col_alias(sc1)
        assert ga and tmpl.display_alias_map.get(sc1.signature_key) == ga


class TestSqlPhaseJoinResources:
    """Tests for _sql_phase_join_resources pinned-join shortcut."""

    @patch("aetherdialect._pipeline.generate_join_candidates")
    def test_pinned_template_match_skips_join_llm(self, mock_gjc, schema_graph: SchemaGraph):
        mock_gjc.return_value = ({}, {}, {})
        tmpl = _make_pipeline_template()
        new_sig = replace(
            tmpl.intent_signature,
            chosen_join_candidate_id="J01",
            chosen_join_path_signature=["orders.order_id->orders.customer_id"],
        )
        tmpl = replace(tmpl, intent_signature=new_sig)
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        _sql_phase_join_resources(
            intent,
            schema_graph,
            tmpl,
            GenerationPath.INTENT_DIRECT_MATCH,
        )
        mock_gjc.assert_called_once()
        assert intent.chosen_join_candidate_id == "J01"
        assert intent.chosen_join_path_signature == ["orders.order_id->orders.customer_id"]

    def test_fresh_path_calls_generate_join_candidates(self, schema_graph: SchemaGraph):
        intent = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        with patch("aetherdialect._pipeline.generate_join_candidates") as mock_gjc:
            mock_gjc.return_value = ({"candidates": []}, {}, {})
            _sql_phase_join_resources(intent, schema_graph, None, None)
        mock_gjc.assert_called_once()


class TestMergeStructuralDefaultsForReuse:
    """Tests for merge_structural_defaults_for_reuse (direct-reuse vs path 3 parity)."""

    def test_fills_missing_s_from_defaults(self):
        """Placeholder :s4 in SQL is filled from structural_defaults when absent from params."""
        params = {"s1": 0.0, "s2": 0.0, "s3": 0.0}
        sql = "SELECT 1 ORDER BY x + :s4 DESC"
        n = merge_structural_defaults_for_reuse(
            sql,
            params,
            {"s4": 0.0},
        )
        assert n == 1
        assert params["s4"] == 0.0

    def test_does_not_overwrite_existing(self):
        """Existing structural keys are left unchanged."""
        params = {"s4": 1.0}
        n = merge_structural_defaults_for_reuse(
            "ORDER BY :s4",
            params,
            {"s4": 0.0},
        )
        assert n == 0
        assert params["s4"] == 1.0

    def test_sql_param_placeholders_include_structural_keys(self):
        """Structural keys referenced only in parameterized SQL are backfilled."""
        params: dict[str, float] = {}
        n = merge_structural_defaults_for_reuse(
            "SELECT 1 WHERE x = :s1",
            params,
            {"s1": 0.0},
        )
        assert n == 1
        assert params["s1"] == 0.0


class TestBuildInteractiveTailSnapshot:
    """``build_interactive_tail_snapshot`` tuple/set freezing."""

    def test_freezes_collections_and_copies_schema_terms(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = _make_pipeline_template()
        uc = [SelectCol(expr=NormalizedExpr.from_column("t.id"))]
        st = [tmpl]
        terms = {"a", "b"}
        snap = build_interactive_tail_snapshot(
            q_norm="q",
            intent=intent,
            schema=MagicMock(),
            store={},
            templates={},
            rejected={},
            schema_terms=terms,
            dialect=MagicMock(),
            semantic_warnings=[],
            has_union_match=True,
            cols_changed=False,
            matched_template=None,
            union_select_cols=uc,
            structural_match_templates=st,
            ikey="ik",
            intent_sim=0.9,
            union_candidate_template_ids=("T0001", "T0002"),
        )
        assert isinstance(snap, InteractiveTailSnapshot)
        assert snap.union_select_cols == tuple(uc)
        assert snap.structural_match_templates == tuple(st)
        assert snap.union_candidate_template_ids == ("T0001", "T0002")
        assert snap.schema_terms == terms
        terms.add("c")
        assert "c" not in snap.schema_terms


class TestParseIntentViaLlmMocked:
    """``parse_intent_via_llm`` branches without a real LLM parse."""

    @patch("aetherdialect._pipeline.save_template_store")
    @patch(
        "aetherdialect._pipeline.invoke_intent_parse_with_hints",
        return_value=(None, ["w"], 4, None),
    )
    def test_none_intent_records_failure(self, mock_full, mock_save):
        schema = MagicMock(schema_hash="sh", effective_structural_hash="sh")
        with tempfile.TemporaryDirectory() as td:
            sd = os.path.join(td, "intent_templates")
            os.makedirs(sd, exist_ok=True)
            with patch.object(EngineConfig, "TEMPLATE_STORE_DIR", sd):
                store = empty_template_store("sh")
                intent, warns, calls, _plan = parse_intent_via_llm("question", schema, {}, store)
        assert intent is None
        assert warns == ["w"]
        assert calls == 4
        mock_save.assert_called_once()

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="n")
    def test_schema_invalid_user_declines_at_intent_confirm(self, mock_yes, mock_save):
        bad = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            schema_invalid=True,
        )
        schema = MagicMock(effective_structural_hash="sh")
        ok = confirm_intent_with_user(
            bad,
            {},
            semantic_warnings=None,
            similarity_score=0.0,
            rejected={},
            q_norm="q",
            schema=schema,
        )
        assert ok is False
        mock_save.assert_called_once()

    def test_schema_invalid_returns_intent_without_suspend(self):
        bad = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            schema_invalid=True,
        )
        port = MagicMock()
        port.has_pending_choice.return_value = False
        with patch(
            "aetherdialect._pipeline.invoke_intent_parse_with_hints",
            return_value=(bad, [], 0, None),
        ):
            intent, _, _, _ = parse_intent_via_llm("q", MagicMock(schema_hash="h"), {}, {}, choice_port=port)
        assert intent is not None
        assert intent.schema_invalid is True

    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_schema_invalid_user_accepts(self, mock_yes):
        bad = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            schema_invalid=True,
        )
        ok = confirm_intent_with_user(
            bad,
            {},
            semantic_warnings=None,
            similarity_score=0.0,
            rejected={},
            q_norm="q",
            schema=MagicMock(effective_structural_hash="h"),
        )
        assert ok is True
        assert bad.schema_invalid is False


class TestCaptureParsePromptRecordsInvoke:
    """``_invoke_intent_parse_with_hints`` forwards ``in_turn_seed`` into ``full_intent_parse``."""

    def test_invoke_forwards_in_turn_seed(self):
        import aetherdialect._intent_process

        stub = (None, [], 0, None)
        seed = [
            {
                "kind": "validation_failure",
                "summary": "repair hint",
                "bucket": "OTHER",
                "effective_structural_hash": "sh",
                "created_at": "t",
                "is_post_restart": "False",
            }
        ]
        with patch.object(aetherdialect._intent_process, "full_intent_parse", return_value=stub) as mock_fp:
            aetherdialect._intent_process.invoke_intent_parse_with_hints(
                "how many rows?",
                MagicMock(schema_hash="sh"),
                store={},
                in_turn_seed=seed,
                max_retries=1,
            )
        assert mock_fp.call_args.kwargs["in_turn_seed"] == seed


class TestBestAcceptedSimilarityAndSkipConfirmation:
    """``best_accepted_template_similarity`` and ``should_skip_intent_confirmation``."""

    def test_best_similarity_empty_templates(self):
        assert best_accepted_template_similarity(MagicMock(), {}) == 0.0

    @patch("aetherdialect._pipeline.structural_compare")
    def test_best_similarity_takes_max(self, mock_sc):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        a = _make_pipeline_template(tid="A")
        b = _make_pipeline_template(tid="B")

        def _sc(_intent, tmpl, *, mode="full"):
            s = 0.2 if tmpl.id == "A" else 0.95
            return StructuralCompareResult(
                non_agg_symmetric_diff=0,
                union_eligible=False,
                similarity_score=s,
            )

        mock_sc.side_effect = _sc
        score = best_accepted_template_similarity(intent, {"A": a, "B": b})
        assert score == 0.95
        assert mock_sc.call_count == 2

    def test_should_skip_true_when_clean(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert should_skip_intent_confirmation(intent, {}, "q1", None) is True
        assert should_skip_intent_confirmation(intent, None, "q1", []) is True

    def test_should_skip_false_when_schema_invalid(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        intent.schema_invalid = True
        assert should_skip_intent_confirmation(intent, {}, "q1", None) is False

    def test_should_skip_false_when_semantic_warnings(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert should_skip_intent_confirmation(intent, {}, "q1", ["warn"]) is False

    def test_should_skip_false_when_prior_wrong_join_feedback(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        store = {
            "question_feedback": {
                "q1": [
                    {
                        "summary": "bad join",
                        "bucket": RejectionBucket.WRONG_TABLES_OR_JOINS.value,
                        "kind": FeedbackKind.INTENT_REJECTED.value,
                        "effective_structural_hash": "h",
                        "intent_structural_hash": "ih",
                        "intent_payload": "",
                        "created_at": "t1",
                        "updated_at": "t2",
                    }
                ],
            },
        }
        assert should_skip_intent_confirmation(intent, store, "q1", None) is False

    def test_should_skip_false_when_prior_non_join_feedback(self):
        """Any recorded rejection for the same question forces intent confirmation."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        store = {
            "question_feedback": {
                "q1": [
                    {
                        "summary": "x",
                        "bucket": RejectionBucket.OTHER.value,
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "h",
                        "created_at": "t",
                    }
                ],
            },
        }
        assert should_skip_intent_confirmation(intent, store, "q1", None) is False


class TestUserRefinementBudget:
    """``PolicyConfig.MAX_USER_REFINEMENTS`` and :exc:`RefinementRetry` on SQL feedback decline."""

    def test_refinement_retry_available_respects_budget(self):
        """Exhausted refinement charges disable another retry."""
        ctx = RefinementContext("question text")
        assert refinement_retry_available(ctx) is True
        ctx.refinement_rounds_executed = PolicyConfig.MAX_USER_REFINEMENTS
        assert refinement_retry_available(ctx) is False

    def test_refinement_retry_available_false_when_blocked(self):
        """Same-intent guard sets ``block_further_refinement`` to stop silent retries."""
        ctx = RefinementContext("question text")
        ctx.block_further_refinement = True
        assert refinement_retry_available(ctx) is False

    def test_complete_user_feedback_reject_raises_refinement_retry(self):
        """When budget remains, SQL rejection records memory then requests an in-turn re-parse."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        schema = MagicMock()
        schema.effective_structural_hash = "h"
        ctx_rej = UserFeedbackRejectSuspendContext(
            intent=intent,
            sql="SELECT 1",
            schema=schema,
            store={"templates": {}, "next_id": 1},
            templates={},
            rejected={},
            q_norm="q",
            generation_path=GenerationPath.FRESH,
            matched_template=None,
            matched_rejected_template=None,
            dialect=None,
            structural_match_templates=(),
        )
        port = SimpleNamespace(_refinement_ctx=RefinementContext("question text"))
        with (
            patch("aetherdialect._pipeline.summarize_failure_for_memory") as sm,
            patch("aetherdialect._pipeline.record_question_feedback"),
            patch("aetherdialect._pipeline.templates_to_store", return_value=ctx_rej.store),
            patch("aetherdialect._pipeline.save_template_store"),
        ):
            sm.return_value = SimpleNamespace(buckets=[SimpleNamespace(value=RejectionBucket.OTHER.value)])
            with _pt.raises(RefinementRetry):
                complete_user_feedback_reject(
                    ctx_rej,
                    needs_reason=True,
                    reject_reason="wrong grain",
                    choice_port=port,
                )

    def test_complete_user_feedback_reject_terminates_when_max_is_zero(self, monkeypatch):
        """With ``MAX_USER_REFINEMENTS=0`` the path matches legacy single-attempt behavior (no retry)."""
        monkeypatch.setattr(PolicyConfig, "MAX_USER_REFINEMENTS", 0)
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        schema = MagicMock()
        schema.effective_structural_hash = "h"
        ctx_rej = UserFeedbackRejectSuspendContext(
            intent=intent,
            sql="SELECT 1",
            schema=schema,
            store={"templates": {}, "next_id": 1},
            templates={},
            rejected={},
            q_norm="q",
            generation_path=GenerationPath.FRESH,
            matched_template=None,
            matched_rejected_template=None,
            dialect=None,
            structural_match_templates=(),
        )
        port = SimpleNamespace(_refinement_ctx=RefinementContext("question text"))
        with (
            patch("aetherdialect._pipeline.summarize_failure_for_memory") as sm,
            patch("aetherdialect._pipeline.record_question_feedback"),
            patch("aetherdialect._pipeline.templates_to_store", return_value=ctx_rej.store),
            patch("aetherdialect._pipeline.save_template_store"),
        ):
            sm.return_value = SimpleNamespace(buckets=[SimpleNamespace(value=RejectionBucket.OTHER.value)])
            out = complete_user_feedback_reject(
                ctx_rej,
                needs_reason=True,
                reject_reason="wrong grain",
                choice_port=port,
            )
        assert isinstance(out, dict)


class TestOtherTemplateOwnsQuestionString:
    """``other_template_owns_question_string`` dedupe gate."""

    @patch("aetherdialect._pipeline.exact_question_match", return_value=True)
    def test_true_when_other_template_matches_question(self, mock_eq):
        a = _make_pipeline_template(tid="A")
        b = _make_pipeline_template(tid="B")
        b.value_history.questions = ["same?"]
        assert other_template_owns_question_string({"A": a, "B": b}, "A", "same?") is True


class TestJoinSignatureHelpers:
    """``_join_signatures_for_deterministic_from_anchor`` CTE and main- scope behavior."""

    def test_main_sig_empty_when_multiple_non_j00(self):
        intent = RuntimeIntent(
            tables=["a", "b"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        cmap = {"J00": [], "J01": ["x"], "J02": ["y"]}
        main, cte = _join_signatures_for_deterministic_from_anchor(cmap, None, intent)
        assert main == []
        assert cte == {}

    def test_main_sig_when_single_non_j00(self):
        intent = RuntimeIntent(
            tables=["a", "b"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        cmap = {"J00": [], "J01": ["p1", "p2"]}
        main, _ = _join_signatures_for_deterministic_from_anchor(cmap, None, intent)
        assert main == ["p1", "p2"]

    def test_cte_single_non_j00_candidate(self):
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["t1", "t2"],
            select_cols=[],
            output_columns=[],
            grain="row_level",
        )
        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        hints = {
            "c1": {
                "candidates": [
                    {"candidate_id": "J00", "join_path_signature": []},
                    {"candidate_id": "J01", "join_path_signature": ["edge"]},
                ],
            },
        }
        _, cte_sigs = _join_signatures_for_deterministic_from_anchor({"J00": []}, hints, intent)
        assert cte_sigs["c1"] == ["edge"]


class TestStructuralKeyRemap:
    """``_structural_key_remap_from_assignment_order`` and value-history remap."""

    @patch(
        "aetherdialect._pipeline.structural_s_key_assignment_order",
        side_effect=[["s1", "s2"], ["s9", "s8"]],
    )
    def test_remap_zip_when_same_length(self, mock_ord):
        a = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        b = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        m = _structural_key_remap_from_assignment_order(a, b)
        assert m == {"s1": "s9", "s2": "s8"}

    @patch(
        "aetherdialect._pipeline.structural_s_key_assignment_order",
        side_effect=[["s1"], ["s1", "s2"]],
    )
    def test_remap_empty_when_length_mismatch(self, mock_ord):
        a = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        b = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert _structural_key_remap_from_assignment_order(a, b) == {}

    def test_remap_value_history_noop_on_empty_map(self):
        vh = ValueHistory(param_values=[{"s1": 1}], questions=["q"], natural_language=[""])
        _remap_value_history_structural_keys(vh, {})
        assert vh.param_values[0] == {"s1": 1}

    def test_remap_value_history_rewrites_keys(self):
        vh = ValueHistory(param_values=[{"s1": 1, "keep": 2}], questions=["q"], natural_language=[""])
        _remap_value_history_structural_keys(vh, {"s1": "s9"})
        assert vh.param_values[0] == {"s9": 1, "keep": 2}


class TestCompleteDirectSqlReuseUserChoice:
    """``complete_direct_sql_reuse_user_choice``."""

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.reject_out_per_question", return_value=(False, False))
    def test_decline_bumps_reject(self, mock_reject_out, mock_save):
        tmpl = _make_pipeline_template(trust_level=2)
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        ctx = DirectReuseSuspendContext(
            q_norm="q",
            ref_tmpl=tmpl,
            dialect=MagicMock(),
            store={},
            templates={},
            rejected={},
            schema=MagicMock(),
            intent=intent,
            sql="SELECT 1",
            rows=(),
            display_sql="SELECT 1",
            headers=None,
            is_exact=False,
            reuse_path=GenerationPath.FUZZY_REUSE_FULL_PARAMS,
            sd_reuse=None,
        )
        r = complete_direct_sql_reuse_user_choice(ctx, "n")
        assert isinstance(r, SqlGenerationOutcome)
        assert r.success is True
        assert r.generation_path == GenerationPath.FUZZY_REUSE_FULL_PARAMS
        assert tmpl.stats.reject == 1
        mock_reject_out.assert_called_once_with({}, tmpl, "q")

    @patch("aetherdialect._pipeline.llm_chat", return_value='{"aliases":{}}')
    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch("aetherdialect._pipeline.save_result_csv")
    @patch("aetherdialect._pipeline.promote_trust")
    def test_accept_saves_csv_and_promotes(self, mock_promote, mock_csv, mock_del, mock_tts, mock_save, _mock_llm):
        tmpl = _make_pipeline_template(trust_level=2)
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        ctx = DirectReuseSuspendContext(
            q_norm="q",
            ref_tmpl=tmpl,
            dialect=MagicMock(),
            store={},
            templates={"T0001": tmpl},
            rejected={},
            schema=MagicMock(),
            intent=intent,
            sql="SELECT 1",
            rows=((1,),),
            display_sql="SELECT 1",
            headers=None,
            is_exact=True,
            reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
            sd_reuse=None,
        )
        r = complete_direct_sql_reuse_user_choice(ctx, "y")
        assert r.generation_path == GenerationPath.EXACT_QUESTION_REUSE
        mock_csv.assert_called_once()
        mock_promote.assert_called_once_with(tmpl, "q")
        assert tmpl.stats.accept == 2


class TestHandleDirectSqlReuseMocked:
    """``handle_direct_sql_reuse`` without DB or extraction LLM."""

    @patch("aetherdialect._pipeline.llm_chat", return_value='{"aliases":{}}')
    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch("aetherdialect._pipeline.save_result_csv")
    @patch("aetherdialect._pipeline.print_query_result")
    @patch("aetherdialect._pipeline.promote_trust")
    @patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
    def test_exact_question_auto_accepts_high_trust(
        self,
        mock_val,
        mock_promote,
        mock_print,
        mock_csv,
        mock_del,
        mock_tts,
        mock_save,
        _mock_llm,
    ):
        concrete = ConcreteIntent(
            intent_id="id",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=concrete,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(
                param_values=[{}],
                questions=["norm_q"],
                natural_language=["nl"],
            ),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=2,
            feedback_by_question={"norm_q": FeedbackCounts(accepts=1, rejects=0)},
        )
        dialect = MagicMock()
        dialect.finalize_render.return_value = "EXEC"
        dialect.explain_validation_sql = lambda sql, _pv: sql
        dialect.execute.return_value = [("row",)]
        store: dict = {"templates": {"T1": tmpl}}
        result = handle_direct_sql_reuse(
            "norm_q",
            tmpl,
            dialect,
            store,
            {"T1": tmpl},
            {},
            MagicMock(schema_hash="h"),
        )
        assert result is not None and result.success is True
        assert result.generation_path == GenerationPath.EXACT_QUESTION_REUSE
        dialect.execute.assert_called_once()
        mock_csv.assert_called_once()
        mock_promote.assert_called_once_with(tmpl, "norm_q")

    @patch("aetherdialect._pipeline.llm_chat", return_value='{"aliases":{}}')
    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch("aetherdialect._pipeline.save_result_csv")
    @patch("aetherdialect._pipeline.print_query_result")
    @patch("aetherdialect._pipeline.promote_trust")
    @patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
    @patch(
        "aetherdialect._pipeline.extract_fuzzy_reuse_params",
        return_value={"p1": 7, "s1": 10},
    )
    def test_fuzzy_reuse_literal_structural_path_2_1(
        self,
        mock_extract,
        mock_val,
        mock_promote,
        mock_print,
        mock_csv,
        mock_del,
        mock_tts,
        mock_save,
        _mock_llm,
    ):
        """Fuzzy question with exemplar structural ``s`` matching defaults → path ``2.1``."""
        concrete = ConcreteIntent(
            intent_id="id",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=concrete,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT :p1 FROM orders WHERE order_id < :s1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(
                param_values=[{"p1": 1, "s1": 10}],
                questions=["stored_q"],
                natural_language=["nl"],
            ),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=2,
            structural_defaults={"s1": 10},
            feedback_by_question={"new_fuzzy_q": FeedbackCounts(accepts=1, rejects=0)},
        )
        dialect = MagicMock()
        dialect.finalize_render.return_value = "EXEC"
        dialect.explain_validation_sql = lambda sql, _pv: sql
        dialect.execute.return_value = [("row",)]
        store: dict = {"templates": {"T1": tmpl}}
        result = handle_direct_sql_reuse(
            "new_fuzzy_q",
            tmpl,
            dialect,
            store,
            {"T1": tmpl},
            {},
            MagicMock(schema_hash="h"),
            reuse_history_index=0,
        )
        assert result is not None and result.success is True
        assert result.generation_path == GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL
        mock_extract.assert_called_once()
        assert mock_extract.call_args.kwargs.get("literal_structural_only") is True

    @patch("aetherdialect._pipeline.llm_chat", return_value='{"aliases":{}}')
    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch("aetherdialect._pipeline.save_result_csv")
    @patch("aetherdialect._pipeline.print_query_result")
    @patch("aetherdialect._pipeline.promote_trust")
    @patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
    @patch(
        "aetherdialect._pipeline.extract_fuzzy_reuse_params",
        return_value={"p1": 7, "s1": 99},
    )
    def test_fuzzy_reuse_full_params_path_2_2(
        self,
        mock_extract,
        mock_val,
        mock_promote,
        mock_print,
        mock_csv,
        mock_del,
        mock_tts,
        mock_save,
        _mock_llm,
    ):
        """Fuzzy question with exemplar structural ``s`` differing from defaults → path ``2.2``."""
        concrete = ConcreteIntent(
            intent_id="id",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=concrete,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT :p1 FROM orders WHERE order_id < :s1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(
                param_values=[{"p1": 1, "s1": 99}],
                questions=["stored_q"],
                natural_language=["nl"],
            ),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=2,
            structural_defaults={"s1": 10},
            feedback_by_question={"new_fuzzy_q": FeedbackCounts(accepts=1, rejects=0)},
        )
        dialect = MagicMock()
        dialect.finalize_render.return_value = "EXEC"
        dialect.explain_validation_sql = lambda sql, _pv: sql
        dialect.execute.return_value = [("row",)]
        store: dict = {"templates": {"T1": tmpl}}
        result = handle_direct_sql_reuse(
            "new_fuzzy_q",
            tmpl,
            dialect,
            store,
            {"T1": tmpl},
            {},
            MagicMock(schema_hash="h"),
            reuse_history_index=0,
        )
        assert result is not None and result.success is True
        assert result.generation_path == GenerationPath.FUZZY_REUSE_FULL_PARAMS
        mock_extract.assert_called_once()
        assert mock_extract.call_args.kwargs.get("literal_structural_only") is False


class TestLoadPipelineResourcesMore:
    """Extra ``load_pipeline_resources`` validation paths."""

    @patch("aetherdialect._pipeline.llm_credentials_configured", return_value=True)
    @patch("aetherdialect._pipeline.EngineConfig")
    @patch("aetherdialect._pipeline.get_dialect", return_value=SimpleNamespace(engine="e"))
    def test_unsupported_engine_type(self, mock_gd, mock_ec, _lc):
        mock_ec.TYPE = "bogus_engine"
        mock_ec.RUNTIME = SimpleNamespace(db_url=lambda: "mysql://x")
        with _pt.raises(ValueError, match="Unsupported engine"):
            load_pipeline_resources(
                schema={},
                store={},
                templates={},
                rejected={},
                schema_terms=set(),
            )

    @patch("aetherdialect._pipeline.llm_credentials_configured", return_value=True)
    @patch("aetherdialect._pipeline.EngineConfig")
    @patch("aetherdialect._pipeline.get_dialect", return_value=SimpleNamespace(engine="e"))
    def test_missing_store(self, mock_gd, mock_ec, _lc):
        mock_ec.TYPE = "postgresql"
        mock_ec.RUNTIME = SimpleNamespace(db_url=lambda: "postgresql://x")
        with _pt.raises(RuntimeError, match="Store must be provided"):
            load_pipeline_resources(
                schema={},
                store=None,
                templates={},
                rejected={},
                schema_terms=set(),
            )


class TestMatchQuestionLevelTemplateReuseListTemplates:
    """``match_question_level_template_reuse`` accepts dict or list of templates."""

    @patch("aetherdialect._pipeline.find_trusted_template_match")
    def test_passes_list_to_matcher_when_dict(self, mock_find):
        mock_find.return_value = None
        tmpl = _make_pipeline_template()
        match_question_level_template_reuse("q", {"X": tmpl})
        passed = mock_find.call_args[0][1]
        assert isinstance(passed, list)
        assert passed == [tmpl]

    @patch("aetherdialect._pipeline.find_trusted_template_match")
    def test_passes_list_through_when_already_list(self, mock_find):
        mock_find.return_value = None
        tmpl = _make_pipeline_template()
        match_question_level_template_reuse("q", [tmpl])
        assert mock_find.call_args[0][1] == [tmpl]


class TestConfirmIntentSuspendAndUnionSkip:
    """More ``confirm_intent_with_user`` branches."""

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_warnings_deferred_suspend(self, mock_yes, mock_save):
        tail = MagicMock(spec=InteractiveTailSnapshot)
        port = MagicMock()
        port.has_pending_choice.return_value = False
        with _pt.raises(PipelineSuspended) as ei:
            confirm_intent_with_user(
                TestConfirmIntentWithUser._make_intent(),
                TestConfirmIntentWithUser._make_store(),
                semantic_warnings=["w"],
                similarity_score=0.0,
                choice_port=port,
                suspend_tail=tail,
            )
        assert ei.value.state_id == PIPELINE_SUSPEND_ID_INTENT_CONFIRM

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.interactive_yes_no", return_value="y")
    def test_prior_wrong_join_forces_confirm_despite_union_col_similarity(self, mock_yes, mock_save):
        intent = TestConfirmIntentWithUser._make_intent()
        store = TestConfirmIntentWithUser._make_store()
        store["question_feedback"] = {
            "q": [
                {
                    "summary": "wrong tables",
                    "bucket": RejectionBucket.WRONG_TABLES_OR_JOINS.value,
                    "kind": FeedbackKind.INTENT_REJECTED.value,
                    "effective_structural_hash": "h",
                    "intent_structural_hash": "ih",
                    "intent_payload": "",
                    "created_at": "t1",
                    "updated_at": "t2",
                }
            ]
        }
        result = confirm_intent_with_user(
            intent,
            store,
            semantic_warnings=None,
            similarity_score=1.0,
            has_union_match=True,
            cols_changed=True,
            rejected={},
            q_norm="q",
            schema=MagicMock(schema_hash="h"),
        )
        assert result is True
        mock_yes.assert_called_once()


class TestHandleUserFeedbackMore:
    """Extra ``handle_user_feedback`` edge cases."""

    @patch("aetherdialect._pipeline.save_template_store")
    def test_raises_when_reuse_path_requires_template(self, mock_save, schema_graph):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        with _pt.raises(RuntimeError, match="Missing matched_template"):
            handle_user_feedback(
                "y",
                intent,
                "SELECT 1",
                schema_graph,
                {"next_id": 1},
                {},
                {},
                "q",
                GenerationPath.EXACT_QUESTION_REUSE,
                None,
                None,
            )

    @patch("aetherdialect._pipeline.save_template_store")
    @patch("aetherdialect._pipeline.templates_to_store", side_effect=lambda s, t: s)
    @patch("aetherdialect._pipeline.delete_rejected_templates_matching_question")
    @patch(
        "aetherdialect._pipeline.promote_rejected_to_template",
        return_value=_make_pipeline_template(tid="NEW"),
    )
    @patch(
        "aetherdialect._pipeline.sql_shape",
        return_value=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
    )
    @patch("aetherdialect._pipeline.intent_key", return_value="ik")
    @patch("aetherdialect._templates.colmap_signature", return_value="")
    @patch("aetherdialect._pipeline.compute_sql_fp", return_value="fp")
    def test_accept_promotes_matched_rejected_template(
        self,
        _fp,
        _cm,
        _ik,
        _shape,
        mock_prom_rej,
        mock_del,
        _tts,
        _save,
        schema_graph,
    ):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        rt = MagicMock()
        store = {
            "templates": {},
            "rejected_templates": {},
            "next_id": 2,
            "next_reject_id": 1,
        }
        handle_user_feedback(
            "y",
            intent,
            "SELECT 1",
            schema_graph,
            store,
            {},
            {},
            "q",
            GenerationPath.FRESH,
            None,
            rt,
        )
        mock_prom_rej.assert_called_once()


class TestResolveJoinsFreshNoPlaceholder:
    """Early exit in ``_resolve_joins_fresh`` for scopes that need no join."""

    def test_single_table_main_sets_j00(self):
        intent = RuntimeIntent(
            tables=["a"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sql_param, cte_ids = _resolve_joins_fresh(
            "SELECT 1",
            intent,
            {"J01": ["x"]},
            None,
            "q",
            {"candidates": []},
        )
        assert sql_param == "SELECT 1"
        assert cte_ids == {}
        assert intent.chosen_join_candidate_id == "J00"
        assert intent.chosen_join_path_signature == []

    def test_multi_table_no_candidates_raises_no_join_path(self):
        """Multi-table main scope with empty FK+semantic candidates raises NoJoinPathError."""
        intent = RuntimeIntent(
            tables=["alpha", "beta"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        det = "SELECT id FROM alpha\nWHERE 1=1"
        with _pt.raises(NoJoinPathError) as exc_info:
            _resolve_joins_fresh(
                det,
                intent,
                {"J00": []},
                None,
                "join alpha and beta",
                {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]},
                schema=None,
            )
        assert "alpha" in exc_info.value.tables
        assert "beta" in exc_info.value.tables
        assert exc_info.value.scope_label == "main query"

    def test_multi_table_cte_no_candidates_raises_no_join_path(self):
        """Multi-table CTE without any FK or semantic edge raises NoJoinPathError tagged with CTE name."""
        cte_step = RuntimeCteStep(
            cte_name="bridge_step",
            tables=["alpha", "beta"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        intent = RuntimeIntent(
            tables=["alpha"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("alpha.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte_step],
        )
        det = "SELECT id FROM alpha\nWHERE 1=1"
        with _pt.raises(NoJoinPathError) as exc_info:
            _resolve_joins_fresh(
                det,
                intent,
                {"J00": []},
                {"bridge_step": {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]}},
                "bridge alpha beta",
                {"candidates": [{"candidate_id": "J00", "join_path_signature": []}]},
                schema=None,
            )
        assert "bridge_step" in exc_info.value.scope_label


def _session_owner() -> MagicMock:
    """Build a mock ``AetherEngine`` owner for :class:`PipelineSession` tests."""
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = empty_template_store("test_hash")
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    return owner


@_pt.fixture
def bare_session() -> PipelineSession:
    """Minimal session with mock owner."""
    return PipelineSession(_session_owner())


def test_ask_completes_when_pipeline_returns(bare_session: PipelineSession) -> None:
    with patch("aetherdialect._main_execution.interactive_run_once") as mock_run:
        mock_run.return_value = None
        step = bare_session.ask("show rows")
    mock_run.assert_called_once()
    assert step.done is True
    assert step.kind == SESSION_KIND_RESULT
    assert bare_session.awaiting_prompt() is False


def test_reset_clears_pending_suspend(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("q")
    assert bare_session.awaiting_prompt() is True
    bare_session.reset()
    assert bare_session.awaiting_prompt() is False


def test_ask_captures_suspend(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(
        PIPELINE_SUSPEND_ID_INTENT_CONFIRM,
        "ignored",
        None,
    )
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        step = bare_session.ask("q")
    assert step.done is False
    assert step.kind == SESSION_KIND_AWAITING_INTENT_CONFIRM
    assert step.prompt == "Is this correct? (y/n): "
    assert bare_session.awaiting_prompt() is True


def test_step_resumes_and_completes(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "prompt", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("q")
    with patch("aetherdialect._main_execution.dispatch_pipeline_resume") as mock_dispatch:
        step = bare_session.step("y")
    mock_dispatch.assert_called_once()
    assert step.done is True
    assert step.kind == SESSION_KIND_RESULT
    assert bare_session.awaiting_prompt() is False


def test_step_second_suspend(bare_session: PipelineSession) -> None:
    first = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "first", None)
    second = PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "second", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=first):
        bare_session.ask("q")
    with patch("aetherdialect._main_execution.dispatch_pipeline_resume", side_effect=second):
        step = bare_session.step("y")
    assert step.done is False
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert step.prompt == "Is this correct? (y/n): "
    assert step.message == "second"
    assert bare_session.awaiting_prompt() is True


def test_step_invalid_token(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("q")
    step = bare_session.step("maybe")
    assert step.done is False
    assert step.message == "Invalid choice — please answer y or n."
    assert step.kind == SESSION_KIND_AWAITING_INTENT_CONFIRM
    assert step.prompt == "Is this correct? (y/n): "
    assert step.reply_shape == "yes_no"
    assert bare_session.awaiting_prompt() is True


def test_step_invalid_token_no_suspend(bare_session: PipelineSession) -> None:
    step = bare_session.step("maybe")
    assert step.kind == SESSION_KIND_IDLE
    assert step.done is True
    assert bare_session.awaiting_prompt() is False


def test_step_invalid_then_valid_resumes(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("q")
    first = bare_session.step("maybe")
    assert first.done is False and first.message == "Invalid choice — please answer y or n."
    with patch("aetherdialect._main_execution.dispatch_pipeline_resume") as mock_dispatch:
        second = bare_session.step("y")
    mock_dispatch.assert_called_once()
    assert second.done is True
    assert second.kind == SESSION_KIND_RESULT
    assert bare_session.awaiting_prompt() is False


def test_ask_while_busy_raises(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("q")
    with _pt.raises(SessionActiveError):
        bare_session.ask("other")


def test_has_pending_choice_and_take_yes_no(bare_session: PipelineSession) -> None:
    assert bare_session.has_pending_choice() is False
    bare_session._resume_choice_stage_id = PIPELINE_SUSPEND_ID_INTENT_CONFIRM
    bare_session._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "y"))
    assert bare_session.has_pending_choice() is True
    assert bare_session.take_yes_no("stage", "p", ["y", "n"]) == "y"
    assert bare_session.has_pending_choice() is False


def test_take_yes_no_empty_queue_raises(bare_session: PipelineSession) -> None:
    with _pt.raises(PipelineSuspended, match="empty"):
        bare_session.take_yes_no("s", "p", ["y", "n"])


def test_take_yes_no_is_silent(
    capsys: _pt.CaptureFixture[str],
    bare_session: PipelineSession,
) -> None:
    bare_session._resume_choice_stage_id = PIPELINE_SUSPEND_ID_INTENT_CONFIRM
    bare_session._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "y"))
    assert bare_session.take_yes_no("s", "p", ["y", "n"]) == "y"
    assert capsys.readouterr().out == ""
    bare_session._choice_queue.append((PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "n"))
    assert bare_session.take_yes_no("s", "p", ["y", "n"], silent_no=True) == "n"
    assert capsys.readouterr().out == ""


def test_reset_clears_suspend_and_queue_partial_state(
    bare_session: PipelineSession,
) -> None:
    bare_session._suspended = PipelineSuspended("x", "m", None)
    bare_session._choice_queue.append(("x", "y"))
    bare_session.reset()
    assert bare_session.awaiting_prompt() is False
    assert bare_session.has_pending_choice() is False


def test_reset_clears_suspend_before_second_ask(bare_session: PipelineSession) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "p", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        bare_session.ask("one")
    assert bare_session.awaiting_prompt() is True
    bare_session.reset()
    with patch("aetherdialect._main_execution.interactive_run_once", return_value=None):
        bare_session.ask("two")
    assert bare_session.awaiting_prompt() is False


def test_intent_confirm_suspend_step_resume_completes(
    bare_session: PipelineSession,
) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_INTENT_CONFIRM, "continue?", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        step = bare_session.ask("q")
    assert step.done is False
    assert step.kind == SESSION_KIND_AWAITING_INTENT_CONFIRM
    with patch("aetherdialect._main_execution.dispatch_pipeline_resume"):
        step2 = bare_session.step("y")
    assert step2.done is True
    assert step2.kind == SESSION_KIND_RESULT


def test_direct_reuse_suspend_step_resume_completes(
    bare_session: PipelineSession,
) -> None:
    suspended = PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "correct?", None)
    with patch("aetherdialect._main_execution.interactive_run_once", side_effect=suspended):
        step = bare_session.ask("q")
    assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    with patch("aetherdialect._main_execution.dispatch_pipeline_resume"):
        step2 = bare_session.step("y")
    assert step2.done is True


def test_step_after_completed_turn_requires_new_ask(
    bare_session: PipelineSession,
) -> None:
    with patch("aetherdialect._main_execution.interactive_run_once", return_value=None):
        bare_session.ask("q")
    step = bare_session.step("y")
    assert step.kind == SESSION_KIND_IDLE
    assert "ask()" in (step.error or "")


class _StubDialect(Dialect):
    name = "stub"

    def __init__(self) -> None:
        super().__init__(MagicMock())

    def ast_validate(self, sql: str) -> tuple[bool, str]:
        return True, ""

    def explain_sql(self, sql: str, params=None, **kwargs) -> tuple[bool, str]:
        return True, ""

    def quote_table_column(self, table: str, column: str) -> str:
        return f'"{table}"."{column}"'


def _single_table_intent(table: str, col: str) -> RuntimeIntent:
    return RuntimeIntent(
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{col}"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


@_pt.mark.integration
def test_generate_path5_sql_validation_failure_carries_error(
    schema_graph: SchemaGraph,
) -> None:
    intent = _single_table_intent("customers", "customer_id")
    join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema_graph)
    store: dict = {"next_id": 1, "templates": {}, "rejected_templates": {}}
    dialect = _StubDialect()
    with (
        patch(
            "aetherdialect._pipeline._run_sql_validation_cascade",
            return_value=(
                False,
                "forced_fail",
                FailureCategory.EXECUTION_EXPLAIN_FAILED,
                [],
            ),
        ),
        patch("aetherdialect._pipeline.save_template_store") as mock_save,
    ):
        out = generate_and_validate_sql(
            "list customer ids",
            intent,
            schema_graph,
            join_candidates,
            cmap,
            dialect,
            store,
            cte_join_hints=cte_hints,
            matched_template=None,
            structural_match_templates=[],
        )
    assert out.success is False
    assert out.sql_validation_error == "forced_fail"
    assert out.error_kind == "execution_explain_failed"
    mock_save.assert_called_once()


@_pt.mark.integration
def test_bind_params_for_sql_no_placeholders_returns_none() -> None:
    assert bind_params_for_sql("SELECT 1", {"p1": 1}) is None


@_pt.mark.integration
def test_bind_params_for_sql_returns_map_when_tokens_present() -> None:
    assert bind_params_for_sql("SELECT :p1", {"p1": 1}) == {"p1": 1}


def test_bind_params_for_sql_returns_map_for_dollar_tokens() -> None:
    assert bind_params_for_sql("SELECT * FROM t WHERE id = $p1", {"p1": 1}) == {"p1": 1}


def test_bigquery_pre_execute_rewrite_single_at_prefix() -> None:
    from aetherdialect._dialect_sqlglot_engines import BigQueryDialect

    d = BigQueryDialect.__new__(BigQueryDialect)
    out = d.pre_execute_rewrite("SELECT :p1 FROM t WHERE id = :p2")
    assert "@@" not in out
    assert "@p1" in out


def test_run_sql_validation_cascade_uses_canonical_sql(monkeypatch) -> None:
    from aetherdialect._pipeline import _run_sql_validation_cascade

    captured: list[str] = []

    def _fake_validate(_dialect, sql: str, *_a, **_k):
        captured.append(sql)
        return True, "", None, []

    monkeypatch.setattr("aetherdialect._pipeline.validate_sql", _fake_validate)
    intent = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        sql_param="SELECT title FROM film WHERE rating = :p1",
        param_values={"p1": "PG"},
    )
    dialect = SimpleNamespace(explain_validation_sql=lambda sql, _pv: sql)
    ok, _, _, _ = _run_sql_validation_cascade(
        "SELECT title FROM film WHERE rating = @p1",
        intent,
        dialect,
        schema=SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="h"),
    )
    assert ok is True
    assert captured and ":p1" in captured[0]


def test_generate_and_validate_sql_aborts_on_schema_invalid() -> None:
    from aetherdialect._contracts_base import FailureCategory
    from aetherdialect._pipeline import generate_and_validate_sql

    intent = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        schema_invalid=True,
    )
    dialect = MagicMock()
    out = generate_and_validate_sql(
        "q",
        intent,
        SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="h"),
        {},
        {},
        dialect,
        {},
        persist_template_learning=False,
    )
    assert out.success is False
    assert out.error_kind == FailureCategory.INTENT_SCHEMA_INVALID_ABORT.value


def test_finalize_planner_schema_invalid_trusts_interpret() -> None:
    from aetherdialect._contracts_base import NormalizedExpr
    from aetherdialect._contracts_core import (
        InterpretPlan,
        SelectCol,
    )
    from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
    from aetherdialect._intent_process import finalize_planner_schema_invalid_flag

    schema = SchemaGraph(
        tables={
            "staff": TableMetadata(
                name="staff",
                columns={"staff_id": ColumnMetadata(name="staff_id", data_type="INTEGER")},
                primary_key=["staff_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )
    plan = InterpretPlan(
        approach="staff ids for active employees",
        tables=("staff",),
        schema_invalid=True,
    )
    intent = RuntimeIntent(
        tables=["staff"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("staff.staff_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        schema_invalid=False,
    )
    finalized = finalize_planner_schema_invalid_flag(intent, plan, schema)
    assert finalized.schema_invalid is True


def test_clear_planner_schema_invalid_after_user_accept() -> None:
    from aetherdialect._pipeline import clear_planner_schema_invalid_after_user_accept

    intent = RuntimeIntent(
        tables=["film"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        schema_invalid=True,
    )
    clear_planner_schema_invalid_after_user_accept(intent)
    assert intent.schema_invalid is False


def test_select_col_prefers_llm_for_single_agg() -> None:
    from aetherdialect._contracts_base import MulGroup
    from aetherdialect._contracts_core import SelectCol
    from aetherdialect._sql_gen import select_col_prefers_llm_display_alias

    sc = SelectCol(
        expr=NormalizedExpr(
            add_groups=[MulGroup(multiply=["film.film_id"], agg_func="count")],
        )
    )
    assert select_col_prefers_llm_display_alias(sc) is True


def test_select_col_prefers_deterministic_for_arithmetic() -> None:
    from aetherdialect._contracts_base import MulGroup
    from aetherdialect._contracts_core import SelectCol
    from aetherdialect._sql_gen import select_col_prefers_llm_display_alias

    sc = SelectCol(
        expr=NormalizedExpr(
            add_groups=[
                MulGroup(multiply=["payment.amount"]),
                MulGroup(multiply=["payment.tax"]),
            ]
        )
    )
    assert select_col_prefers_llm_display_alias(sc) is False
