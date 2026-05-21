"""Tests for intent_process module."""

import json
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest

from aetherdialect._config import MAX_NON_AGG_COL_DIFF, VALID_HAVING_OPS, GenerationPath
from aetherdialect._contracts_base import (
    ColumnMetadata,
    FailureCategory,
    IntentIssue,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ConcreteIntent,
    FilterParam,
    HavingParam,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
    concrete_intent_to_runtime_skeleton,
    runtime_intent_to_concrete,
)
from aetherdialect._core_utils import stable_json
from aetherdialect._intent_process import (
    INTENT_CRITICAL_RULES,
    INTENT_FORMAT_REPAIR_JSON_RULES,
    INTENT_PARSE_RULES_APPEND,
    _apply_post_processing,
    _base_similarity,
    _build_intent_format_repair_prompt,
    _build_intent_parse_prompt,
    _build_intent_semantic_repair_prompt,
    _classify_schema_error,
    _compute_error_signature_issues,
    _compute_error_signature_strings,
    _compute_filters_similarity,
    _compute_having_similarity,
    _compute_order_by_cols_similarity,
    _compute_select_cols_similarity,
    _cte_step_similarity,
    _detect_oscillation,
    _diff_cols_span_disjoint_tables,
    _format_repair_loop,
    _invoke_intent_parse_with_hints,
    _jaccard,
    _logical_intent_to_serialisable,
    _normalize_cte_output_aliases,
    _phase_g_post_validation_passes,
    _resolve_repair_instruction,
    _runtime_intent_case_registry_has_empty_branches,
    _structural_body_matches,
    _summarize_intent_changes,
    _union_family_index_from_templates,
    apply_runtime_post_processing_lite,
    collect_structural_match_templates,
    find_trusted_template_match,
    intent_similarity,
    logical_intent_from_parsed,
    match_template_for_union,
    reconcile_union_family_after_mutation,
    reconcile_union_family_body_join_after_mutation,
    resolve_sql_path,
    select_col_diff,
    structural_compare,
    structural_compare_runtime,
    union_template_compatibility,
)
from aetherdialect._intent_resolve import (
    UnionSelectColumnDelta,
    classify_union_merge_case,
    compute_intent_union,
    join_path_key_concrete,
    join_path_key_runtime,
)
from aetherdialect._utils import body_similarity_key, intent_key


class TestJaccard:
    """Tests for _jaccard."""

    def test_identical_sets(self):
        """Jaccard of identical sets is 1.0."""
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """Jaccard of disjoint sets is 0.0."""
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        """Jaccard of partial overlap."""
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self):
        """Jaccard of two empty sets is 1.0."""
        assert _jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        """Jaccard with one empty set is 0.0."""
        assert _jaccard({"a"}, set()) == 0.0


class TestComputeSimilarities:
    """Tests for compute_*_similarity functions."""

    def test_filters_both_empty(self):
        """Both empty filter lists yield 1.0."""
        assert _compute_filters_similarity([], []) == 1.0

    def test_filters_one_empty(self):
        """One empty filter list yields 0.0."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        assert _compute_filters_similarity([fp], []) == 0.0

    def test_having_both_empty(self):
        """Both empty having lists yield 1.0."""
        assert _compute_having_similarity([], []) == 1.0

    def test_select_cols_identical(self):
        """Identical select cols yield 1.0."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        assert _compute_select_cols_similarity([sc], [sc]) == 1.0

    def test_order_by_cols_both_empty(self):
        """Both empty order_by lists yield 1.0."""
        assert _compute_order_by_cols_similarity([], []) == 1.0


class TestIntentSimilarity:
    """Tests for intent_similarity."""

    def test_identical_intents(self):
        """Identical intents yield 1.0."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        assert intent_similarity(intent, intent) == pytest.approx(1.0)

    def test_completely_different_intents(self):
        """Completely different intents yield low score."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t2.b"))
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc1],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        i2 = RuntimeIntent(
            tables=["t2"],
            grain="row_level",
            select_cols=[sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        score = intent_similarity(i1, i2)
        assert score < 1.0


class TestBuildIntentSemanticRepairPrompt:
    """Tests for build_intent_semantic_repair_prompt."""

    def test_returns_string(self):
        """build_intent_semantic_repair_prompt returns a string."""
        issue = IntentIssue(
            issue_id="i1",
            category=FailureCategory.OTHER,
            severity="error",
            message="missing table",
        )
        result = _build_intent_semantic_repair_prompt("test q", '{"tables":[]}', [issue], [], "schema text")
        assert isinstance(result, str)

    def test_contains_question(self):
        """build_intent_semantic_repair_prompt includes question in output."""
        issue = IntentIssue(
            issue_id="i1",
            category=FailureCategory.OTHER,
            severity="error",
            message="missing",
        )
        result = _build_intent_semantic_repair_prompt("my question", "{}", [issue], [], "schema")
        assert "my question" in result

    def test_contains_issue_message(self):
        """build_intent_semantic_repair_prompt includes issue message."""
        issue = IntentIssue(
            issue_id="i1",
            category=FailureCategory.OTHER,
            severity="warning",
            message="bad column",
        )
        result = _build_intent_semantic_repair_prompt("q", "{}", [], [issue], "schema")
        assert "bad column" in result

    def test_multiple_issues(self):
        """build_intent_semantic_repair_prompt handles multiple issues."""
        issues = [
            IntentIssue(
                issue_id="i1",
                category=FailureCategory.OTHER,
                severity="error",
                message="issue1",
            ),
            IntentIssue(
                issue_id="i2",
                category=FailureCategory.OTHER,
                severity="warning",
                message="issue2",
            ),
        ]
        result = _build_intent_semantic_repair_prompt("q", "{}", [issues[0]], [issues[1]], "schema")
        assert "issue1" in result
        assert "issue2" in result


class TestBuildIntentFormatRepairPrompt:
    """Tests for build_intent_format_repair_prompt."""

    def test_returns_string(self):
        """build_intent_format_repair_prompt returns a string."""
        result = _build_intent_format_repair_prompt("q", "bad json", "expected }")
        assert isinstance(result, str)

    def test_contains_parse_error(self):
        """build_intent_format_repair_prompt includes parse error."""
        result = _build_intent_format_repair_prompt("q", "raw", "missing bracket")
        assert "missing bracket" in result

    def test_includes_full_response(self):
        """build_intent_format_repair_prompt includes full raw response."""
        long_raw = "x" * 5000
        result = _build_intent_format_repair_prompt("q", long_raw, "err")
        assert "x" * 5000 in result

    def test_includes_instructional_placeholder_mapping(self):
        """Format-repair payload uses SSOT field_specifications and output_format."""
        result = _build_intent_format_repair_prompt("q", "{}", "err")
        assert "field_specifications" in result
        assert "output_format" in result

    def test_instructions_concat_json_rules_and_critical_rules(self):
        result = _build_intent_format_repair_prompt("q", "{}", "err")
        data = json.loads(result)
        assert data["instructions"] == list(INTENT_FORMAT_REPAIR_JSON_RULES) + list(INTENT_CRITICAL_RULES)


class TestFormatRepairLoop:
    """Tests for _format_repair_loop."""

    def test_no_llm_when_parse_clean(self):
        """Valid JSON without instructional placeholders skips repair calls."""

        raw = json.dumps(
            {
                "tables": ["film"],
                "grain": "row_level",
                "select_cols": ["film.film_id"],
            }
        )
        with patch("aetherdialect._intent_process.llm_chat") as chat:
            intent, calls = _format_repair_loop("sys", raw, "q", max_retries=3)
        assert calls == 0
        assert chat.call_count == 0
        assert intent is not None

    def test_llm_round_when_instructional_placeholder_in_expr(self):
        """Parsed intent with table_N in expr triggers one format-repair LLM call."""

        bad = json.dumps(
            {
                "tables": ["film"],
                "grain": "row_level",
                "select_cols": ["table_1.film_id"],
            }
        )
        good = json.dumps(
            {
                "tables": ["film"],
                "grain": "row_level",
                "select_cols": ["film.film_id"],
            }
        )
        with patch("aetherdialect._intent_process.llm_chat", return_value=good) as chat:
            intent, calls = _format_repair_loop("sys", bad, "q", max_retries=3)
        assert calls == 1
        assert chat.call_count == 1
        assert intent is not None
        assert intent.select_cols[0].expr.primary_column == "film.film_id"


class TestBaseSimilarity:
    """Tests for _base_similarity."""

    def test_identical_fields(self):
        """Identical fields yield 1.0."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        score = _base_similarity(["t"], ["t"], [sc], [sc], [], [], [], [], [], [], [], [])
        assert score == pytest.approx(1.0)

    def test_completely_different(self):
        """Completely different fields yield low score."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t2.b"))
        score = _base_similarity(["t"], ["t2"], [sc1], [sc2], [], [], [], [], [], [], [], [])
        assert score < 1.0

    def test_all_empty(self):
        """All empty fields yield 1.0."""
        score = _base_similarity([], [], [], [], [], [], [], [], [], [], [], [])
        assert score == pytest.approx(1.0)

    def test_same_tables_rest_different_at_most_tables_weight(self):
        """When only tables overlap, non-table clause similarities vanish except the tables term."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t.b"))
        fp1 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            param_key="p1",
        )
        fp2 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op="=",
            value_type="string",
            param_key="p2",
        )
        g1 = NormalizedExpr.from_column("t.a")
        g2 = NormalizedExpr.from_column("t.c")
        ob1 = OrderByCol(expr=NormalizedExpr.from_column("t.a"), direction="ASC")
        ob2 = OrderByCol(expr=NormalizedExpr.from_column("t.d"), direction="ASC")
        hp1 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op=">",
            value_type="number",
            param_key="h1",
        )
        hp2 = HavingParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op=">",
            value_type="number",
            param_key="h2",
        )
        score = _base_similarity(
            ["t"],
            ["t"],
            [sc1],
            [sc2],
            [g1],
            [g2],
            [ob1],
            [ob2],
            [fp1],
            [fp2],
            [hp1],
            [hp2],
        )
        assert score <= 0.30 + 1e-6

    def test_different_tables_same_else_at_most_complement_of_tables_weight(self):
        """When tables differ but other clauses match, score stays below 0.70."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        score = _base_similarity(
            ["t1"],
            ["t2"],
            [sc],
            [sc],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        assert score <= 0.70 + 1e-6


class TestStructuralBodyMatches:
    """Tests for _structural_body_matches table-set gate."""

    def test_false_when_table_sets_differ(self):
        """Unequal table lists prevent a structural body match."""
        sc = SelectCol(expr=NormalizedExpr.from_column("film.film_id"))
        intent = RuntimeIntent(
            tables=["film", "actor"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        concrete = ConcreteIntent(
            intent_id="c",
            tables=["film"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert _structural_body_matches(intent, concrete) is False

    def test_true_when_tables_and_body_align(self):
        """Equal tables and matching clause skeletons match."""
        sc = SelectCol(expr=NormalizedExpr.from_column("film.film_id"))
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        concrete = ConcreteIntent(
            intent_id="c",
            tables=["film"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert _structural_body_matches(intent, concrete) is True


class TestCteStepSimilarity:
    """Tests for cte_step_similarity."""

    def test_identical_ctes(self):
        """Identical CTE steps yield 1.0."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        assert _cte_step_similarity(cte, cte) == pytest.approx(1.0)

    def test_different_ctes(self):
        """Different CTE steps yield low score."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t2.b"))
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[sc1],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            tables=["t2"],
            select_cols=[sc2],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        assert _cte_step_similarity(cte1, cte2) < 1.0


class TestIntentSimilarityEdgeCases:
    """Edge-case tests for intent_similarity."""

    def test_with_cte_steps(self):
        """intent_similarity with CTE steps weights properly."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        score = intent_similarity(i1, i2)
        assert score == pytest.approx(1.0)

    def test_mismatched_cte_count(self):
        """intent_similarity with different CTE counts still works."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[],
            natural_language="test",
        )
        score = intent_similarity(i1, i2)
        assert 0.0 <= score <= 1.0


class TestComputeSimilaritiesEdgeCases:
    """Edge-case tests for compute_*_similarity functions."""

    def test_filters_identical(self):
        """Identical filter lists yield 1.0."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        assert _compute_filters_similarity([fp], [fp]) == 1.0

    def test_having_one_empty(self):
        """One empty having list yields 0.0."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        assert _compute_having_similarity([hp], []) == 0.0

    def test_select_cols_different(self):
        """Different select cols yield low score."""
        sc1 = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("t.b"))
        score = _compute_select_cols_similarity([sc1], [sc2])
        assert score < 1.0

    def test_order_by_identical(self):
        """Identical order_by lists yield 1.0."""
        ob = OrderByCol(expr=NormalizedExpr.from_column("t.a"), direction="asc")
        assert _compute_order_by_cols_similarity([ob], [ob]) == 1.0

    def test_order_by_one_empty(self):
        """One empty order_by list yields 0.0."""
        ob = OrderByCol(expr=NormalizedExpr.from_column("t.a"), direction="asc")
        assert _compute_order_by_cols_similarity([ob], []) == 0.0


class TestComputeFiltersSimilarity:
    """Tests for compute_filters_similarity."""

    def test_identical_filters_score_one(self):
        """Identical filter lists produce score 1.0."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.col"), op="=", value_type="string")
        assert _compute_filters_similarity([fp], [fp]) == 1.0

    def test_different_filter_group_penalized(self):
        """Different filter_group wiring reduces similarity when signature sets match."""
        fp1 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.c1"),
            op="=",
            value_type="string",
            filter_group=1,
        )
        fp2 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.c2"),
            op="=",
            value_type="string",
            filter_group=2,
        )
        fp1b = FilterParam(
            left_expr=NormalizedExpr.from_column("t.c1"),
            op="=",
            value_type="string",
            filter_group=1,
        )
        fp2b = FilterParam(
            left_expr=NormalizedExpr.from_column("t.c2"),
            op="=",
            value_type="string",
            filter_group=3,
        )
        score = _compute_filters_similarity([fp1, fp2], [fp1b, fp2b])
        assert 0 < score < 1.0

    def test_empty_lists_score_one(self):
        """Both empty lists produce score 1.0."""
        assert _compute_filters_similarity([], []) == 1.0


class TestComputeHavingSimilarity:
    """Tests for compute_having_similarity."""

    def test_identical_having_score_one(self):
        """Identical having lists produce score 1.0."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        assert _compute_having_similarity([hp], [hp]) == 1.0

    def test_different_filter_group_penalized(self):
        """Different filter_group wiring reduces similarity when signature sets match."""
        hp1 = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            filter_group=1,
        )
        hp2 = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.amt"),
            op=">",
            value_type="number",
            filter_group=2,
        )
        hp1b = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            filter_group=1,
        )
        hp2b = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "t.amt"),
            op=">",
            value_type="number",
            filter_group=3,
        )
        score = _compute_having_similarity([hp1, hp2], [hp1b, hp2b])
        assert 0 < score < 1.0

    def test_empty_lists_score_one(self):
        """Both empty lists produce score 1.0."""
        assert _compute_having_similarity([], []) == 1.0


class TestBuildIntentParsePrompt:
    """Tests for _build_intent_parse_prompt."""

    def test_returns_tuple_of_strings(self):
        """Returns (system_message, user_message) tuple."""
        system, user = _build_intent_parse_prompt("how many orders?", "schema text here", ["orders", "customers"])
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_contains_parser_role(self):
        """System message contains parser role description."""
        system, _ = _build_intent_parse_prompt("q", "schema", ["orders"])
        assert "parser" in system.lower() or "json" in system.lower()

    def test_user_contains_table_list(self):
        """User message contains the allowed tables."""
        _, user = _build_intent_parse_prompt("q", "schema", ["orders", "customers"])
        assert "orders" in user
        assert "customers" in user

    def test_user_contains_question(self):
        """User message contains the question (embedded in the JSON task payload)."""
        _, user = _build_intent_parse_prompt("total revenue by customer", "schema", ["orders"])
        assert "total revenue" in user or "task" in user.lower()

    def test_rules_use_canonical_critical_rules_plus_parse_suffix(self):
        _, user = _build_intent_parse_prompt("q", "schema", ["orders"])
        data = json.loads(user)
        assert data["rules"] == list(INTENT_CRITICAL_RULES) + list(INTENT_PARSE_RULES_APPEND)

    def test_rules_include_flat_or_of_and_filters_param_example(self):
        _, user = _build_intent_parse_prompt("q", "schema", ["orders"])
        data = json.loads(user)
        joined = "\n".join(data["rules"])
        compact = joined.replace(" ", "")
        assert "1,1,2,2" in compact
        assert "filter_group" in compact.lower()
        assert (
            "Do not put bool_op on rows that carry filter_group" in joined
            or "Do not emit bool_op on rows that have a filter_group" in joined
        )
        assert "Do not nest filter_group as an array" in joined

    def test_operator_reference_having_ops_match_valid_having_ops(self):
        _, user = _build_intent_parse_prompt("q", "schema", ["t"])
        data = json.loads(user)
        assert data["operator_reference"]["having_ops"] == sorted(VALID_HAVING_OPS)


class TestPlannerSingleOutputWindowRule:
    """Planner Stage-A decomposition includes the no-CTE-for-single-window rule (WF-009)."""

    def test_logical_decomposition_guidance_contains_rule(self):
        from aetherdialect._intent_process import (
            _LOGICAL_DECOMPOSITION_GUIDANCE,
            PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE,
            _build_intent_logical_prompt,
        )

        guidance = stable_json(list(_LOGICAL_DECOMPOSITION_GUIDANCE))
        q = "list rentals with rental id and next rental date for the same inventory item ordered by rental date"
        payload = _build_intent_logical_prompt(q, "{}", "", guidance, (), ())
        assert PLANNER_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE in payload


class TestFindTrustedTemplateMatch:
    """Tests for find_trusted_template_match."""

    def test_empty_templates_returns_none(self):
        """No templates → returns None."""
        assert find_trusted_template_match("question", []) is None

    def test_trust_level_1_exact_match_returns_tuple(self):
        """Trust level 1 with exact fuzzy match returns (None, template)."""
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
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm",
            value_history=ValueHistory(
                param_values=[{}],
                questions=["show all orders"],
                natural_language=["show orders"],
            ),
            stats=TemplateStats(accept=2, reject=0),
            trust_level=1,
        )
        result = find_trusted_template_match("show all orders", [tmpl])
        assert result is not None
        assert result.template is tmpl

    def test_trust_level_2_exact_match_returns_tuple(self):
        """Trust level 2 with exact fuzzy match returns (None, template)."""
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
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm",
            value_history=ValueHistory(
                param_values=[{}],
                questions=["show all orders"],
                natural_language=["show orders"],
            ),
            stats=TemplateStats(accept=10, reject=0),
            trust_level=2,
        )
        result = find_trusted_template_match("show all orders", [tmpl])
        assert result is not None
        assert result.template is tmpl


class TestClassifySchemaError:
    """Tests for _classify_schema_error."""

    def test_unknown_table(self):
        assert _classify_schema_error("Unknown table: payment") == FailureCategory.UNKNOWN_TABLE

    def test_unknown_column(self):
        assert _classify_schema_error("Unknown select column: film.revenue") == FailureCategory.UNKNOWN_COLUMN

    def test_fallback(self):
        assert _classify_schema_error("Some other error") == FailureCategory.SCHEMA_VALIDATION

    def test_unknown_table_case_insensitive(self):
        assert _classify_schema_error("UNKNOWN TABLE: foo") == FailureCategory.UNKNOWN_TABLE

    def test_unknown_column_varied_phrasing(self):
        assert _classify_schema_error("Unknown filter column: x.y") == FailureCategory.UNKNOWN_COLUMN


class TestComputeErrorSignatureIssues:
    """Tests for _compute_error_signature_issues."""

    def test_empty(self):
        assert _compute_error_signature_issues([]) == frozenset()

    def test_deterministic(self):
        issues = [
            IntentIssue(
                issue_id="a",
                category=FailureCategory.OTHER,
                severity="error",
                message="m1",
            ),
            IntentIssue(
                issue_id="b",
                category=FailureCategory.OTHER,
                severity="error",
                message="m2",
            ),
        ]
        sig = _compute_error_signature_issues(issues)
        assert sig == frozenset({(FailureCategory.OTHER, "m1"), (FailureCategory.OTHER, "m2")})

    def test_ignores_issue_id(self):
        i1 = [
            IntentIssue(
                issue_id="x",
                category=FailureCategory.OTHER,
                severity="error",
                message="m",
            )
        ]
        i2 = [
            IntentIssue(
                issue_id="y",
                category=FailureCategory.OTHER,
                severity="error",
                message="m",
            )
        ]
        assert _compute_error_signature_issues(i1) == _compute_error_signature_issues(i2)


class TestComputeErrorSignatureStrings:
    """Tests for _compute_error_signature_strings."""

    def test_empty(self):
        assert _compute_error_signature_strings([]) == frozenset()

    def test_basic(self):
        sig = _compute_error_signature_strings(["err1", "err2"])
        assert sig == frozenset({"err1", "err2"})


class TestDetectOscillation:
    """Tests for _detect_oscillation."""

    def test_too_short(self):
        assert _detect_oscillation([frozenset({"a"})]) is False

    def test_aa_pattern(self):
        s = frozenset({"err1"})
        assert _detect_oscillation([s, s]) is True

    def test_abab_pattern(self):
        a = frozenset({"err1"})
        b = frozenset({"err2"})
        assert _detect_oscillation([a, b, a, b]) is True

    def test_no_oscillation(self):
        a = frozenset({"err1"})
        b = frozenset({"err2"})
        c = frozenset({"err3"})
        assert _detect_oscillation([a, b, c]) is False

    def test_ab_not_oscillation(self):
        a = frozenset({"err1"})
        b = frozenset({"err2"})
        assert _detect_oscillation([a, b]) is False

    def test_aab_not_aa_at_end(self):
        a = frozenset({"err1"})
        b = frozenset({"err2"})
        assert _detect_oscillation([a, a, b]) is False


class TestSemanticRepairPromptCriticalRules:
    """Tests for critical_rules in semantic repair prompt."""

    def test_contains_critical_rules(self):
        issue = IntentIssue(
            issue_id="i1",
            category=FailureCategory.OTHER,
            severity="error",
            message="missing table",
        )
        result = _build_intent_semantic_repair_prompt("test q", '{"tables":[]}', [issue], [], "schema text")
        assert "critical_rules" in result
        assert "having_param" in result
        assert "window_registry" in result
        payload = json.loads(result)
        assert payload["critical_rules"] == list(INTENT_CRITICAL_RULES)


def _col(table: str, column: str) -> SelectCol:
    """Build a bare non-aggregated SelectCol."""
    return SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))


def _agg_col(agg: str, table: str, column: str) -> SelectCol:
    """Build an aggregated SelectCol."""
    return SelectCol(expr=NormalizedExpr.from_agg(agg, f"{table}.{column}"))


def _runtime(
    tables: list[str],
    cols: list[SelectCol],
) -> RuntimeIntent:
    """Build a minimal RuntimeIntent for union matching tests."""
    return RuntimeIntent(
        tables=tables,
        grain="row_level",
        select_cols=cols,
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


def _concrete(
    tables: list[str],
    cols: list[SelectCol],
) -> ConcreteIntent:
    """Build a minimal ConcreteIntent for union matching tests."""
    return ConcreteIntent(
        intent_id="",
        tables=tables,
        grain="row_level",
        select_cols=cols,
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


class TestMaxNonAggColDiff:
    """Tests for MAX_NON_AGG_COL_DIFF constant."""

    def test_value_is_two(self):
        """Threshold for non-aggregate column diff is 2."""
        assert MAX_NON_AGG_COL_DIFF == 2


class TestDiffColsSpanDisjointTables:
    """Tests for _diff_cols_span_disjoint_tables."""

    def test_no_diff_returns_false(self):
        """Identical column sets never span disjoint tables."""
        cols = [_col("customer", "first_name")]
        assert not _diff_cols_span_disjoint_tables(
            cols,
            cols,
            ["customer"],
            ["customer"],
        )

    def test_diff_from_shared_table_returns_false(self):
        """Extra column from a table both intents share is allowed."""
        intent_cols = [_col("customer", "first_name")]
        concrete_cols = [
            _col("customer", "first_name"),
            _col("customer", "email"),
        ]
        assert not _diff_cols_span_disjoint_tables(
            intent_cols,
            concrete_cols,
            ["customer"],
            ["customer"],
        )

    def test_template_col_from_foreign_table_returns_true(self):
        """Column in template from table not in intent is rejected."""
        intent_cols = [_col("customer", "first_name")]
        concrete_cols = [
            _col("customer", "first_name"),
            _col("rental", "rental_date"),
        ]
        assert _diff_cols_span_disjoint_tables(
            intent_cols,
            concrete_cols,
            ["customer"],
            ["customer", "rental"],
        )

    def test_intent_col_from_foreign_table_returns_true(self):
        """Column in intent from table not in template is rejected."""
        intent_cols = [
            _col("customer", "first_name"),
            _col("payment", "amount"),
        ]
        concrete_cols = [_col("customer", "first_name")]
        assert _diff_cols_span_disjoint_tables(
            intent_cols,
            concrete_cols,
            ["customer", "payment"],
            ["customer"],
        )

    def test_bidirectional_both_foreign_returns_true(self):
        """Diff columns from disjoint tables on both sides."""
        intent_cols = [
            _col("customer", "first_name"),
            _col("payment", "amount"),
        ]
        concrete_cols = [
            _col("customer", "first_name"),
            _col("rental", "rental_date"),
        ]
        assert _diff_cols_span_disjoint_tables(
            intent_cols,
            concrete_cols,
            ["customer", "payment"],
            ["customer", "rental"],
        )

    def test_empty_cols_returns_false(self):
        """Empty column lists produce no diff."""
        assert not _diff_cols_span_disjoint_tables([], [], [], [])

    def test_agg_cols_ignored(self):
        """Aggregated columns are excluded from the diff check."""
        intent_cols = [_col("customer", "first_name")]
        concrete_cols = [
            _col("customer", "first_name"),
            _agg_col("count", "rental", "rental_id"),
        ]
        assert not _diff_cols_span_disjoint_tables(
            intent_cols,
            concrete_cols,
            ["customer"],
            ["customer", "rental"],
        )


class TestSelectColDiff:
    """Tests for select_col_diff."""

    def test_identical_cols(self):
        """Matching columns produce zero diff."""
        cols = [_col("t", "a"), _agg_col("sum", "t", "b")]
        agg_match, diff = select_col_diff(cols, cols)
        assert agg_match
        assert diff == 0

    def test_non_agg_diff(self):
        """Non-aggregate symmetric difference counted correctly."""
        i_cols = [_col("t", "a")]
        c_cols = [_col("t", "a"), _col("t", "b"), _col("t", "c")]
        agg_match, diff = select_col_diff(i_cols, c_cols)
        assert agg_match
        assert diff == 2

    def test_agg_mismatch(self):
        """Aggregation function mismatch flagged."""
        i_cols = [_agg_col("sum", "t", "a")]
        c_cols = [_agg_col("avg", "t", "a")]
        agg_match, _ = select_col_diff(i_cols, c_cols)
        assert not agg_match

    def test_exceeds_max_threshold(self):
        """Diff of 3 exceeds the new threshold of 2."""
        i_cols = [_col("t", "a")]
        c_cols = [_col("t", "a"), _col("t", "b"), _col("t", "c"), _col("t", "d")]
        _, diff = select_col_diff(i_cols, c_cols)
        assert diff > MAX_NON_AGG_COL_DIFF


class TestResolveSqlPath:
    """Tests for resolve_sql_path."""

    def test_no_template_is_fresh(self):
        assert resolve_sql_path(matched_template=None, cols_changed=False, union_sql_path=None) is GenerationPath.FRESH

    def test_explicit_path_preserved(self):
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=_concrete(["t"], [_col("t", "a")]),
            intent_key="k",
            tables_used=["t"],
            sql_param="SELECT 1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=[""]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        assert (
            resolve_sql_path(
                matched_template=tmpl,
                cols_changed=True,
                union_sql_path=GenerationPath.UNION_TEMPLATE_WIDEN,
            )
            is GenerationPath.UNION_TEMPLATE_WIDEN
        )

    def test_infers_path_three_when_no_union_sql_path(self):
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=_concrete(["t"], [_col("t", "a")]),
            intent_key="k",
            tables_used=["t"],
            sql_param="SELECT 1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=[""]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        assert (
            resolve_sql_path(matched_template=tmpl, cols_changed=False, union_sql_path=None)
            is GenerationPath.INTENT_DIRECT_MATCH
        )


class TestComputeIntentUnion:
    """Tests for compute_intent_union."""

    def test_identical_cols_no_change(self):
        """Same columns produce cols_changed=False."""
        cols = [_col("t", "a"), _col("t", "b")]
        intent = _runtime(["t"], list(cols))
        concrete = _concrete(["t"], list(cols))
        union_cols, cols_changed, merge_case = compute_intent_union(intent, concrete)
        assert not cols_changed
        assert merge_case is UnionSelectColumnDelta.EQUAL
        assert len(union_cols) == 2

    def test_new_col_detected(self):
        """Extra column in intent produces cols_changed=True."""
        intent = _runtime(
            ["t"],
            [_col("t", "a"), _col("t", "b"), _col("t", "c")],
        )
        concrete = _concrete(
            ["t"],
            [_col("t", "a"), _col("t", "b")],
        )
        union_cols, cols_changed, merge_case = compute_intent_union(intent, concrete)
        assert cols_changed
        assert merge_case is UnionSelectColumnDelta.INTENT_ONLY_EXTRA
        assert len(union_cols) == 3

    def test_concrete_order_preserved(self):
        """Concrete columns appear first in union."""
        intent = _runtime(["t"], [_col("t", "c")])
        concrete = _concrete(
            ["t"],
            [_col("t", "a"), _col("t", "b")],
        )
        union_cols, _, merge_case = compute_intent_union(intent, concrete)
        assert merge_case is UnionSelectColumnDelta.BOTH_EXTRA
        keys = [sc.signature_key for sc in union_cols]
        c_key = _col("t", "c").signature_key
        assert keys[-1] == c_key

    def test_returns_three_tuple(self):
        """Return type is (list, bool, UnionSelectColumnDelta)."""
        intent = _runtime(["t"], [])
        concrete = _concrete(["t"], [])
        result = compute_intent_union(intent, concrete)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestClassifyUnionMergeCase:
    """Tests for classify_union_merge_case."""

    def test_template_only_extra(self):
        """Fewer intent columns than template yields TEMPLATE_ONLY_EXTRA."""
        intent = _runtime(["t"], [_col("t", "a")])
        concrete = _concrete(["t"], [_col("t", "a"), _col("t", "b")])
        assert classify_union_merge_case(intent, concrete) is UnionSelectColumnDelta.TEMPLATE_ONLY_EXTRA

    def test_both_extra(self):
        """Disjoint extras on both sides yields BOTH_EXTRA."""
        intent = _runtime(["t"], [_col("t", "a"), _col("t", "c")])
        concrete = _concrete(["t"], [_col("t", "a"), _col("t", "b")])
        assert classify_union_merge_case(intent, concrete) is UnionSelectColumnDelta.BOTH_EXTRA


class TestMatchTemplateForUnion:
    """Tests for match_template_for_union end-to-end."""

    @staticmethod
    def _make_template(
        tid: str,
        cols: list[SelectCol],
        tables: list[str],
        trust: int = 1,
    ) -> Template:
        """Build a minimal template for matching tests."""
        sig = _concrete(tables, cols)
        sig.chosen_join_candidate_id = "J00"
        sig.chosen_join_path_signature = []
        return Template(
            id=tid,
            effective_structural_hash="h",
            intent_signature=sig,
            intent_key="k",
            tables_used=tables,
            sql_param="SELECT 1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="",
            value_history=ValueHistory(
                param_values=[],
                questions=[],
                natural_language=[],
            ),
            stats=TemplateStats(),
            trust_level=trust,
        )

    def test_same_table_extra_col_matches(self):
        """Extra column from shared table allows union match."""
        intent = _runtime(
            ["customer"],
            [_col("customer", "first_name")],
        )
        tmpl = self._make_template(
            "T0001",
            [_col("customer", "first_name"), _col("customer", "email")],
            ["customer"],
        )
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is not None
        matched, union_cols, cols_changed, sql_path = result
        assert matched.id == "T0001"
        assert not cols_changed
        assert sql_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE

    def test_foreign_table_col_rejected(self):
        """Extra column from foreign table prevents match."""
        intent = _runtime(
            ["customer"],
            [_col("customer", "first_name")],
        )
        tmpl = self._make_template(
            "T0001",
            [_col("customer", "first_name"), _col("rental", "rental_date")],
            ["customer", "rental"],
        )
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is None

    def test_low_trust_skipped(self):
        """Templates with trust_level < 1 are ignored."""
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = self._make_template("T0001", [_col("t", "a")], ["t"], trust=0)
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is None

    def test_agg_mismatch_rejected(self):
        """Mismatched aggregation prevents union match."""
        intent = _runtime(["t"], [_agg_col("sum", "t", "a")])
        tmpl = self._make_template(
            "T0001",
            [_agg_col("avg", "t", "a")],
            ["t"],
        )
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is None

    def test_exceeds_diff_threshold_rejected(self):
        """Diff of 3 exceeds threshold of 2 and prevents match."""
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = self._make_template(
            "T0001",
            [_col("t", "a"), _col("t", "b"), _col("t", "c"), _col("t", "d")],
            ["t"],
        )
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is None

    def test_returns_four_tuple(self):
        """Successful match returns (template, cols, cols_changed, union_sql_path)."""
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = self._make_template("T0001", [_col("t", "a")], ["t"])
        result = match_template_for_union(intent, {"T0001": tmpl})
        assert result is not None
        assert len(result) == 4

    def test_best_match_smallest_diff(self):
        """Among multiple candidates, smallest diff wins."""
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl_far = self._make_template(
            "T0001",
            [_col("t", "a"), _col("t", "b"), _col("t", "c")],
            ["t"],
        )
        tmpl_close = self._make_template(
            "T0002",
            [_col("t", "a"), _col("t", "b")],
            ["t"],
        )
        result = match_template_for_union(
            intent,
            {"T0001": tmpl_far, "T0002": tmpl_close},
        )
        assert result is not None
        assert result[0].id == "T0002"


class TestCollectStructuralMatchTemplates:
    """Tests for collect_structural_match_templates."""

    def test_sorted_by_id_same_intent_key(self):
        """All trusted union-compatible templates with matching intent key appear sorted by id."""
        intent = _runtime(["t"], [_col("t", "a")])
        ik = intent_key(intent)
        tmpl_hi = TestMatchTemplateForUnion._make_template("T0003", [_col("t", "a")], ["t"])
        tmpl_hi = replace(
            tmpl_hi,
            intent_key=ik,
            id="T0003",
            intent_signature=replace(
                tmpl_hi.intent_signature,
                chosen_join_path_signature=["b.b->c.c"],
            ),
        )
        tmpl_lo = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"])
        tmpl_lo = replace(
            tmpl_lo,
            intent_key=ik,
            id="T0001",
            intent_signature=replace(
                tmpl_lo.intent_signature,
                chosen_join_path_signature=["a.a->d.d"],
            ),
        )
        out = collect_structural_match_templates(intent, {"T0003": tmpl_hi, "T0001": tmpl_lo})
        assert [t.id for t in out] == ["T0001", "T0003"]

    def test_excludes_wrong_body_similarity_key(self):
        intent = _runtime(["t"], [_col("t", "a")])
        ik = intent_key(intent)
        tmpl = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "b")], ["t"])
        tmpl = replace(tmpl, intent_key=ik)
        out = collect_structural_match_templates(intent, {"T0001": tmpl})
        assert out == []

    def test_three_templates_same_selects_different_join_all_listed(self):
        """Same intent key and selects; structural list keeps all join variants; union tie picks first."""
        intent = _runtime(["t"], [_col("t", "a")])
        ik = intent_key(intent)
        templates: dict[str, Template] = {}
        for tid in ["T0001", "T0002", "T0003"]:
            tmpl = TestMatchTemplateForUnion._make_template(tid, [_col("t", "a")], ["t"])
            sig = replace(
                tmpl.intent_signature,
                chosen_join_path_signature=[f"{tid}.join"],
            )
            templates[tid] = replace(tmpl, intent_key=ik, intent_signature=sig)
        listed = collect_structural_match_templates(intent, templates)
        assert sorted(t.id for t in listed) == ["T0001", "T0002", "T0003"]
        m = match_template_for_union(intent, templates)
        assert m is not None
        assert m[0].id == "T0001"


class TestCteTablesStructuralBodyMatch:
    """CTE table list participates in structural body matching."""

    def test_differing_cte_tables_rejects_match(self):
        sc = SelectCol(expr=NormalizedExpr.from_column("film.film_id"))
        cte_x = RuntimeCteStep(cte_name="sq", tables=["film"], select_cols=[sc])
        cte_y = RuntimeCteStep(cte_name="sq", tables=["actor"], select_cols=[sc])
        base = _runtime(["film"], [sc])
        left = replace(base, cte_steps=[cte_x])
        right = replace(base, cte_steps=[cte_y])
        concrete = runtime_intent_to_concrete(left, "cid")
        assert not _structural_body_matches(right, concrete)


class TestIntentFingerprintVsSimilarity:
    """``intent_key`` hashing vs ``intent_similarity`` scoring on the same intents."""

    @staticmethod
    def _minimal_intent(*, tables: list[str], select_terms: list[str]) -> RuntimeIntent:
        return RuntimeIntent(
            tables=tables,
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column(t)) for t in select_terms],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )

    def test_identical_intents_share_key_and_unit_similarity(self):
        """Same structural fields yield one key and similarity 1.0."""
        a = self._minimal_intent(tables=["t1"], select_terms=["t1.a"])
        b = self._minimal_intent(tables=["t1"], select_terms=["t1.a"])
        assert intent_key(a) == intent_key(b)
        assert intent_similarity(a, b) == pytest.approx(1.0)

    def test_different_select_changes_key_and_lowers_similarity(self):
        """Select list differences affect both fingerprint and similarity."""
        a = self._minimal_intent(tables=["t1"], select_terms=["t1.a"])
        b = self._minimal_intent(tables=["t1"], select_terms=["t1.b"])
        assert intent_key(a) != intent_key(b)
        assert intent_similarity(a, b) < 1.0


class TestStructuralCompareVsUnionTemplateCompatibility:
    """``structural_compare`` union_eligible aligns with ``union_template_compatibility``."""

    def test_eligible_when_union_row_present(self):
        intent = _runtime(
            ["customer"],
            [_col("customer", "first_name")],
        )
        tmpl = TestMatchTemplateForUnion._make_template(
            "T0001",
            [_col("customer", "first_name"), _col("customer", "email")],
            ["customer"],
        )
        row = union_template_compatibility(intent, tmpl)
        cr_full = structural_compare(intent, tmpl)
        cr_warm = structural_compare(intent, tmpl, mode="warmup_gold_store_check")
        assert row is not None
        assert cr_full.union_eligible is True
        assert cr_warm.union_eligible is True

    def test_ineligible_when_union_row_absent(self):
        intent = _runtime(
            ["customer"],
            [_col("customer", "first_name")],
        )
        tmpl = TestMatchTemplateForUnion._make_template(
            "T0001",
            [_col("customer", "first_name"), _col("rental", "rental_date")],
            ["customer", "rental"],
        )
        row = union_template_compatibility(intent, tmpl)
        cr = structural_compare(intent, tmpl)
        assert row is None
        assert cr.union_eligible is False

    def test_full_mode_sets_similarity_score_warmup_mode_omits(self):
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"])
        cr_full = structural_compare(intent, tmpl, mode="full")
        cr_warm = structural_compare(intent, tmpl, mode="warmup_gold_store_check")
        assert cr_full.similarity_score is not None
        assert cr_warm.similarity_score is None

    def test_path4_subpath_intent_only_extra(self):
        intent = _runtime(["t"], [_col("t", "a"), _col("t", "b")])
        tmpl = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"])
        cr = structural_compare(intent, tmpl, mode="full")
        assert cr.union_eligible is True
        assert cr.union_sql_path == GenerationPath.UNION_TEMPLATE_WIDEN

    def test_path4_subpath_both_extra(self):
        intent = _runtime(["t"], [_col("t", "a"), _col("t", "c")])
        tmpl = TestMatchTemplateForUnion._make_template(
            "T0001",
            [_col("t", "a"), _col("t", "b")],
            ["t"],
        )
        cr = structural_compare(intent, tmpl, mode="full")
        assert cr.union_eligible is True
        assert cr.union_sql_path == GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN

    def test_path4_subpath_template_only_extra(self):
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = TestMatchTemplateForUnion._make_template(
            "T0001",
            [_col("t", "a"), _col("t", "b")],
            ["t"],
        )
        cr = structural_compare(intent, tmpl, mode="full")
        assert cr.union_eligible is True
        assert cr.union_sql_path == GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE

    def test_runtime_compare_matches_template_when_trusted(self):
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"], trust=1)
        cr_t = structural_compare(intent, tmpl, mode="full")
        cr_r = structural_compare_runtime(intent, tmpl.intent_signature, tmpl.sql_fp, mode="full")
        assert cr_t.union_eligible == cr_r.union_eligible
        assert cr_t.union_sql_path == cr_r.union_sql_path
        assert cr_t.similarity_score == cr_r.similarity_score

    def test_runtime_union_eligible_when_template_untrusted(self):
        intent = _runtime(["t"], [_col("t", "a")])
        tmpl = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"], trust=0)
        cr_t = structural_compare(intent, tmpl, mode="warmup_gold_store_check")
        cr_r = structural_compare_runtime(
            intent,
            tmpl.intent_signature,
            tmpl.sql_fp,
            mode="warmup_gold_store_check",
        )
        assert cr_t.union_eligible is False
        assert cr_r.union_eligible is True


class TestResolveRepairInstruction:
    """Tests for _resolve_repair_instruction."""

    def test_known_category_uses_mapping(self):
        issue = IntentIssue(
            issue_id="x",
            category=FailureCategory.UNKNOWN_TABLE,
            severity="error",
            message="raw message",
        )
        assert "schema" in _resolve_repair_instruction(issue).lower()

    def test_unknown_category_falls_back_to_message(self):
        issue = IntentIssue(
            issue_id="x",
            category=FailureCategory.OTHER,
            severity="error",
            message="only this text",
        )
        assert _resolve_repair_instruction(issue) == "only this text"


class TestSummarizeIntentChanges:
    """Tests for _summarize_intent_changes."""

    def test_no_changes(self):
        ri = _runtime(["t"], [_col("t", "a")])
        assert _summarize_intent_changes(ri, ri) == "no_changes"

    def test_reports_changed_top_level_fields(self):
        a = _runtime(["t"], [_col("t", "a")])
        b = replace(a, grain="grouped")
        summary = _summarize_intent_changes(a, b)
        assert "grain" in summary


class TestNormalizeCteOutputAliases:
    """Tests for _normalize_cte_output_aliases."""

    @staticmethod
    def _film_schema() -> SchemaGraph:
        film = TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(name="film_id", data_type="integer", role="identifier"),
            },
            primary_key=["film_id"],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"film": film}, join_paths_multi={}, effective_structural_hash="h")

    def test_empty_cte_steps_returns_same_object_semantics(self):
        intent = _runtime(["film"], [_col("film", "film_id")])
        sg = self._film_schema()
        out = _normalize_cte_output_aliases(intent, sg)
        assert out.cte_steps == []

    def test_remaps_main_expr_when_output_alias_mismatches_derived(self):
        sg = self._film_schema()
        sc = SelectCol(expr=NormalizedExpr.from_column("film.film_id"))
        cte = RuntimeCteStep(
            cte_name="sq",
            tables=["film"],
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=["wrong_alias"],
        )
        main_sel = SelectCol(expr=NormalizedExpr.from_column("sq.wrong_alias"))
        intent = replace(
            _runtime(["film", "sq"], [main_sel]),
            cte_steps=[cte],
        )
        out = _normalize_cte_output_aliases(intent, sg)
        assert out.cte_steps[0].output_columns == ["film_id"]
        assert out.select_cols[0].expr.primary_column == "sq.film_id"


class TestJoinPathKeys:
    """Tests for join_path_key_runtime / join_path_key_concrete."""

    def test_same_signatures_equal_hashes(self):
        sig = _concrete(["t"], [_col("t", "a")])
        sig = replace(sig, chosen_join_path_signature=["p1", "p2"])
        rt = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[_col("t", "a")],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_path_signature=["p1", "p2"],
        )
        assert join_path_key_runtime(rt) == join_path_key_concrete(sig)

    def test_differing_paths_yield_different_keys(self):
        a = replace(
            _concrete(["t"], [_col("t", "a")]),
            chosen_join_path_signature=["old"],
        )
        b = replace(
            _concrete(["t"], [_col("t", "a")]),
            chosen_join_path_signature=["new"],
        )
        assert join_path_key_concrete(a) != join_path_key_concrete(b)


class TestReconcileUnionFamilies:
    """Tests for reconcile_union_family_* helpers."""

    def test_reconcile_union_family_after_mutation_merges_same_instance_key(self):
        sig = _concrete(["t"], [_col("t", "a")])
        sig = replace(sig, chosen_join_candidate_id="J00", chosen_join_path_signature=[])
        base_kw = dict(
            effective_structural_hash="h",
            intent_key="k",
            tables_used=["t"],
            sql_param="SELECT 1",
            sql_fp="shared_fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="",
            stats=TemplateStats(),
            trust_level=1,
        )
        t_lo = Template(
            id="T0001",
            intent_signature=sig,
            value_history=ValueHistory(
                param_values=[{"p1": 1}],
                questions=["q1"],
                natural_language=["n1"],
            ),
            **base_kw,
        )
        t_hi = Template(
            id="T0002",
            intent_signature=sig,
            value_history=ValueHistory(
                param_values=[{"p2": 2}],
                questions=["q2"],
                natural_language=["n2"],
            ),
            **base_kw,
        )
        templates = {"T0001": t_lo, "T0002": t_hi}
        removed = reconcile_union_family_after_mutation(templates)
        assert removed == ["T0002"]
        assert "T0002" not in templates
        keeper = templates["T0001"]
        assert keeper.value_history.questions == ["q1", "q2"]

    def test_reconcile_union_family_with_union_family_index_matches_full_scan(self):
        """Indexed path should remove the same ids as the legacy full scan."""

        def pair() -> dict[str, Template]:
            sig = _concrete(["t"], [_col("t", "a")])
            sig = replace(sig, chosen_join_candidate_id="J00", chosen_join_path_signature=[])
            base_kw = dict(
                effective_structural_hash="h",
                intent_key="k",
                tables_used=["t"],
                sql_param="SELECT 1",
                sql_fp="shared_fp",
                shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
                colmap_sig="",
                stats=TemplateStats(),
                trust_level=1,
            )
            t_lo = Template(
                id="T0001",
                intent_signature=sig,
                value_history=ValueHistory(
                    param_values=[{"p1": 1}],
                    questions=["q1"],
                    natural_language=["n1"],
                ),
                **base_kw,
            )
            t_hi = Template(
                id="T0002",
                intent_signature=sig,
                value_history=ValueHistory(
                    param_values=[{"p2": 2}],
                    questions=["q2"],
                    natural_language=["n2"],
                ),
                **base_kw,
            )
            return {"T0001": t_lo, "T0002": t_hi}

        indexed = pair()
        legacy = pair()
        ufi = _union_family_index_from_templates(indexed)
        removed_i = reconcile_union_family_after_mutation(indexed, union_family_index=ufi)
        removed_l = reconcile_union_family_after_mutation(legacy)
        assert removed_i == removed_l == ["T0002"]
        assert indexed.keys() == legacy.keys() == {"T0001"}
        assert indexed["T0001"].value_history.questions == legacy["T0001"].value_history.questions

    def test_reconcile_body_join_merges_same_body_and_join_path(self):
        sig = _concrete(["t"], [_col("t", "a")])
        sig = replace(sig, chosen_join_candidate_id="J00", chosen_join_path_signature=[])
        base_kw = dict(
            effective_structural_hash="h",
            intent_key="k",
            tables_used=["t"],
            sql_param="SELECT 1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="",
            stats=TemplateStats(),
            trust_level=1,
        )
        t_lo = Template(
            id="T0001",
            intent_signature=sig,
            sql_fp="fp_a",
            value_history=ValueHistory(
                param_values=[{}],
                questions=["q1"],
                natural_language=[""],
            ),
            **base_kw,
        )
        t_hi = Template(
            id="T0002",
            intent_signature=sig,
            sql_fp="fp_b",
            value_history=ValueHistory(
                param_values=[{}],
                questions=["q2"],
                natural_language=[""],
            ),
            **base_kw,
        )
        templates = {"T0001": t_lo, "T0002": t_hi}
        removed = reconcile_union_family_body_join_after_mutation(templates)
        assert removed == ["T0002"]
        assert "T0002" not in templates
        assert templates["T0001"].value_history.questions == ["q1", "q2"]


class TestPhaseGPostValidation:
    """Tests for _phase_g_post_validation_passes."""

    @staticmethod
    def _minimal_schema() -> SchemaGraph:
        tmeta = TableMetadata(
            name="t",
            columns={
                "a": ColumnMetadata(name="a", data_type="varchar", role="categorical"),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": tmeta}, join_paths_multi={}, effective_structural_hash="h")

    def _valid_intent(self) -> RuntimeIntent:
        return RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            natural_language="q",
        )

    def test_passes_when_schema_and_semantics_clean(self):
        assert _phase_g_post_validation_passes(self._valid_intent(), self._minimal_schema()) is True

    def test_fails_on_schema_error(self):
        intent = replace(self._valid_intent(), tables=["does_not_exist"])
        assert _phase_g_post_validation_passes(intent, self._minimal_schema()) is False


class TestApplyPostProcessingMissingParams:
    """_apply_post_processing returns ``(None, issues)`` when param_keys lack raw values."""

    @staticmethod
    def _minimal_schema_t() -> SchemaGraph:
        tmeta = TableMetadata(
            name="t",
            columns={
                "a": ColumnMetadata(name="a", data_type="varchar", role="categorical"),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"t": tmeta}, join_paths_multi={}, effective_structural_hash="h")

    def test_missing_raw_value_for_assigned_param_key_returns_none(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            raw_value=None,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="",
        )
        out, _issues = _apply_post_processing(intent, self._minimal_schema_t(), "question")
        assert out is None


class TestFormatRepairLoopEdgeCases:
    """Additional _format_repair_loop behaviour."""

    def test_malformed_json_triggers_repair_then_succeeds(self):
        good = json.dumps(
            {
                "tables": ["film"],
                "grain": "row_level",
                "select_cols": ["film.film_id"],
            }
        )
        with patch("aetherdialect._intent_process.llm_chat", return_value=good) as chat:
            intent, calls = _format_repair_loop("sys", "not json {", "q", max_retries=3)
        assert calls == 1
        assert chat.call_count == 1
        assert intent is not None

    def test_exhausted_retries_returns_last_parse_attempt(self):
        with patch("aetherdialect._intent_process.llm_chat", return_value="still not json") as chat:
            intent, calls = _format_repair_loop("sys", "{broken", "q", max_retries=2)
        assert calls == 2
        assert chat.call_count == 2
        assert intent is None


class TestBuildIntentParsePromptHintsAndEngineOps:
    """Edge cases for _build_intent_parse_prompt."""

    def test_prior_question_feedback_embedded_in_user_json(self):
        rows = [
            {
                "kind": "validation_failure",
                "summary": "hint a",
                "bucket": "OTHER",
                "effective_structural_hash": "h",
                "created_at": "t",
                "is_post_restart": "False",
            },
            {
                "kind": "validation_failure",
                "summary": "hint b",
                "bucket": "OTHER",
                "effective_structural_hash": "h",
                "created_at": "t2",
                "is_post_restart": "False",
            },
        ]
        _, user = _build_intent_parse_prompt(
            "q",
            "schema",
            ["t1"],
            prior_question_feedback=rows,
        )
        assert "prior_question_feedback" in user
        assert "hint a" in user

    def test_postgresql_includes_ilike_in_operator_reference(self):
        from aetherdialect._config import EngineConfig

        with patch.object(EngineConfig, "TYPE", "postgresql"):
            _, user = _build_intent_parse_prompt("q", "schema", ["t"])
        assert "ilike" in user


class TestClassifyUnionMergeCaseMore:
    """Extra classify_union_merge_case branches."""

    def test_equal(self):
        cols = [_col("t", "a")]
        intent = _runtime(["t"], cols)
        concrete = _concrete(["t"], cols)
        assert classify_union_merge_case(intent, concrete) is UnionSelectColumnDelta.EQUAL

    def test_intent_only_extra(self):
        intent = _runtime(["t"], [_col("t", "a"), _col("t", "b")])
        concrete = _concrete(["t"], [_col("t", "a")])
        assert classify_union_merge_case(intent, concrete) is UnionSelectColumnDelta.INTENT_ONLY_EXTRA


class TestStructuralBodyMatchesEdges:
    """More _structural_body_matches gates."""

    def test_false_when_grain_differs(self):
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        intent = replace(_runtime(["t"], [sc]), grain="grouped")
        concrete = _concrete(["t"], [sc])
        assert not _structural_body_matches(intent, concrete)

    def test_false_when_limit_differs(self):
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))
        intent = replace(_runtime(["t"], [sc]), limit=10)
        concrete = replace(_concrete(["t"], [sc]), limit=None)
        assert not _structural_body_matches(intent, concrete)


class TestSelectColDiffEdges:
    """Edge cases for select_col_diff."""

    def test_empty_lists(self):
        agg_match, diff = select_col_diff([], [])
        assert agg_match
        assert diff == 0


class TestFindTrustedTemplateMatchEdges:
    """More find_trusted_template_match behaviour."""

    def test_empty_history_no_match(self):
        sig = ConcreteIntent(
            intent_id="c",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=sig,
            intent_key="ik",
            tables_used=["t"],
            sql_param="S",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm",
            value_history=ValueHistory(questions=[], param_values=[], natural_language=[]),
            stats=TemplateStats(),
            trust_level=1,
        )
        assert find_trusted_template_match("any question", [tmpl]) is None

    def test_first_trusted_template_wins_when_both_match(self):
        def make_tpl(tid: str, q: str) -> Template:
            sig = ConcreteIntent(
                intent_id=tid,
                tables=["t"],
                grain="row_level",
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
            )
            return Template(
                id=tid,
                effective_structural_hash="h",
                intent_signature=sig,
                intent_key="ik",
                tables_used=["t"],
                sql_param="S",
                sql_fp="fp",
                shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
                colmap_sig="cm",
                value_history=ValueHistory(questions=[q], param_values=[{}], natural_language=[""]),
                stats=TemplateStats(),
                trust_level=1,
            )

        first = make_tpl("T_FIRST", "duplicate match text")
        second = make_tpl("T_SECOND", "duplicate match text")
        result = find_trusted_template_match("duplicate match text", [first, second])
        assert result is not None
        assert result.template.id == "T_FIRST"


class TestFindTrustedTemplateMatchUnionFamilyIntentKey:
    """Union-family ∩ intent-key narrowing for trusted template fuzzy reuse."""

    def test_find_trusted_template_match_uses_union_family_then_intent_key(self):
        qhist = "show widgets"
        sig1 = ConcreteIntent(
            intent_id="t1",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sig2 = ConcreteIntent(
            intent_id="t2",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )

        def make_tpl(tid: str, sig: ConcreteIntent, ik: str) -> Template:
            return Template(
                id=tid,
                effective_structural_hash="h",
                intent_signature=sig,
                intent_key=ik,
                tables_used=["orders"],
                sql_param="S",
                sql_fp="fp",
                shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
                colmap_sig="cm",
                value_history=ValueHistory(questions=[qhist], param_values=[{}], natural_language=[""]),
                stats=TemplateStats(),
                trust_level=1,
            )

        t1 = make_tpl("TAAA", sig1, "ik1")
        t2 = make_tpl("TZZZ", sig2, "ik2")

        rt2 = concrete_intent_to_runtime_skeleton(sig2)
        bk2 = body_similarity_key(rt2)
        jk2 = join_path_key_runtime(rt2)
        ik2 = intent_key(rt2)

        union_family_index = {
            bk2: ["TZZZ"],
            f"{bk2}|{jk2}": ["TZZZ"],
        }
        intent_key_index = {ik2: ["TZZZ"], "ik1": ["TAAA"]}

        r_all = find_trusted_template_match(qhist, [t2, t1])
        assert r_all is not None
        assert r_all.template.id == "TAAA"

        r_narrow = find_trusted_template_match(
            qhist,
            [t2, t1],
            union_family_index=union_family_index,
            intent_key_index=intent_key_index,
            candidate_intent=rt2,
        )
        assert r_narrow is not None
        assert r_narrow.template.id == "TZZZ"

    def test_find_trusted_template_match_falls_back_when_intersection_empty(self):
        qhist = "show widgets"
        sig1 = ConcreteIntent(
            intent_id="t1",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sig2 = ConcreteIntent(
            intent_id="t2",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.total"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )

        def make_tpl(tid: str, sig: ConcreteIntent, ik: str) -> Template:
            return Template(
                id=tid,
                effective_structural_hash="h",
                intent_signature=sig,
                intent_key=ik,
                tables_used=["orders"],
                sql_param="S",
                sql_fp="fp",
                shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
                colmap_sig="cm",
                value_history=ValueHistory(questions=[qhist], param_values=[{}], natural_language=[""]),
                stats=TemplateStats(),
                trust_level=1,
            )

        t1 = make_tpl("TAAA", sig1, "ik1")
        t2 = make_tpl("TZZZ", sig2, "ik2")
        rt2 = concrete_intent_to_runtime_skeleton(sig2)
        bk2 = body_similarity_key(rt2)
        jk2 = join_path_key_runtime(rt2)
        ik2 = intent_key(rt2)

        union_family_index = {bk2: ["TAAA"], f"{bk2}|{jk2}": ["TAAA"]}
        intent_key_index = {ik2: ["TZZZ"]}

        r = find_trusted_template_match(
            qhist,
            [t2, t1],
            union_family_index=union_family_index,
            intent_key_index=intent_key_index,
            candidate_intent=rt2,
        )
        assert r is not None
        assert r.template.id == "TAAA"


class TestClassifySchemaErrorEdge:
    """Boundary cases for _classify_schema_error."""

    def test_unknown_without_column_word_is_fallback(self):
        assert _classify_schema_error("Something unknown happened") == "schema_validation"


class TestIntentSimilarityThreeCtes:
    """intent_similarity weighting with three CTE steps."""

    def test_identical_triple_cte_still_perfect_score(self):
        sc = SelectCol(expr=NormalizedExpr.from_column("t.a"))

        def make_cte(name: str) -> RuntimeCteStep:
            return RuntimeCteStep(
                cte_name=name,
                tables=["t"],
                select_cols=[sc],
                group_by_cols=[],
                order_by_cols=[],
                filters_param=[],
                having_param=[],
                param_values={},
                output_columns=[],
            )

        ctes = [make_cte("c1"), make_cte("c2"), make_cte("c3")]
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=ctes,
            natural_language="x",
        )
        i2 = replace(i1, natural_language="y")
        assert intent_similarity(i1, i2) == pytest.approx(1.0)


class TestComputeFiltersSimilarityZeroJaccard:
    """bool_op branch only when Jaccard > 0."""

    def test_disjoint_signatures_ignore_bool_op_difference(self):
        fp1 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            bool_op="AND",
        )
        fp2 = FilterParam(
            left_expr=NormalizedExpr.from_column("t.b"),
            op="=",
            value_type="string",
            bool_op="OR",
        )
        assert _compute_filters_similarity([fp1], [fp2]) == 0.0


class TestMatchTemplateForUnionStableId:
    """match_template_for_union picks lexicographically smallest template id on equal rank."""

    def test_equal_rank_prefers_smaller_template_id(self):
        intent = _runtime(["t"], [_col("t", "a")])
        sig = _concrete(["t"], [_col("t", "a")])
        sig = replace(sig, chosen_join_candidate_id="J", chosen_join_path_signature=[])

        def tpl(tid: str) -> Template:
            return Template(
                id=tid,
                effective_structural_hash="h",
                intent_signature=sig,
                intent_key="k",
                tables_used=["t"],
                sql_param="S",
                sql_fp=f"fp_{tid}",
                shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
                colmap_sig="",
                value_history=ValueHistory(param_values=[], questions=[], natural_language=[]),
                stats=TemplateStats(),
                trust_level=1,
            )

        m = match_template_for_union(intent, {"T0002": tpl("T0002"), "T0001": tpl("T0001")})
        assert m is not None
        assert m[0].id == "T0001"


class TestInvokeIntentParseWithHints:
    """``_invoke_intent_parse_with_hints`` forwards failure hints into the parse prompt path."""

    def test_invoke_passes_in_turn_seed_to_full_parse(self, schema_graph: SchemaGraph):
        captured: dict[str, Any] = {}

        def _stub_full_parse(
            question: str,
            sg: SchemaGraph,
            max_retries: int = 3,
            **kwargs: Any,
        ) -> tuple[None, list[str], int]:
            captured.update(kwargs)
            return None, [], 1

        seed = [
            {
                "kind": "validation_failure",
                "summary": "join failed on scope",
                "bucket": "OTHER",
                "effective_structural_hash": "h",
                "created_at": "t",
                "is_post_restart": "False",
            }
        ]
        with patch(
            "aetherdialect._intent_process.full_intent_parse",
            side_effect=_stub_full_parse,
        ):
            _invoke_intent_parse_with_hints(
                "q norm",
                schema_graph,
                store={},
                in_turn_seed=seed,
            )
        assert captured.get("in_turn_seed") == seed


class TestCaseRegistryEmptyBranchGuard:
    """Detect empty ``case_registry`` CASE definitions for repair-loop revert."""

    def test_true_when_main_registry_has_no_branches(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[
                CaseRegistryStep(registry_id="c01", case_when=CaseWhenExpr(branches=[])),
            ],
        )
        assert _runtime_intent_case_registry_has_empty_branches(intent)

    def test_false_when_branch_present(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[
                CaseRegistryStep(
                    registry_id="c01",
                    case_when=CaseWhenExpr(branches=[CaseWhenBranch()]),
                ),
            ],
        )
        assert not _runtime_intent_case_registry_has_empty_branches(intent)


class TestApplyRuntimePostProcessingLite:
    """Deterministic post-processing lite helper."""

    def test_second_application_matches_first(self, schema_graph: SchemaGraph):
        rt = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            natural_language="",
        )
        first, issues1 = apply_runtime_post_processing_lite(rt, schema_graph, question_fallback="qf")
        assert first is not None
        assert not any((i.severity or "").lower() == "error" for i in issues1)
        second, issues2 = apply_runtime_post_processing_lite(first, schema_graph, question_fallback="qf")
        assert second is not None
        assert not any((i.severity or "").lower() == "error" for i in issues2)
        assert stable_json(first.to_dict()) == stable_json(second.to_dict())


class TestLogicalIntentNlRoundTrip:
    """Planner NL fields survive parse then serialise for the encoder."""

    def test_root_logical_round_trip(self) -> None:
        raw = {
            "tables": ["orders"],
            "select": "tier label from c0",
            "case": "when amount under 100 then low when amount at least 100 then high else unknown as c0",
        }
        li = logical_intent_from_parsed(raw)
        assert li.tables == ("orders",)
        assert li.select == "tier label from c0"
        assert "c0" in li.case
        assert _logical_intent_to_serialisable(li)["case"] == raw["case"]


class TestApplyPostProcessingIdempotence:
    """Second and third _apply_post_processing passes preserve harvested param_values."""

    @staticmethod
    def _film_schema() -> SchemaGraph:
        film = TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(
                    name="film_id",
                    data_type="integer",
                    role="identifier",
                    is_primary_key=True,
                ),
                "title": ColumnMetadata(name="title", data_type="varchar", role="categorical"),
            },
            primary_key=["film_id"],
            foreign_keys=[],
        )
        return SchemaGraph(tables={"film": film}, join_paths_multi={}, effective_structural_hash="h")

    def test_param_values_survive_repeated_post_processing(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.film_id"),
            op="=",
            value_type="integer",
            raw_value=1,
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="q",
        )
        schema = self._film_schema()
        out1, issues1 = _apply_post_processing(intent, schema, "q")
        assert out1 is not None
        assert not any(i.severity == "error" for i in issues1)
        assert out1.param_values
        first_key = next(iter(out1.param_values))
        assert out1.filters_param[0].raw_value is None
        out2, issues2 = _apply_post_processing(out1, schema, "q")
        assert out2 is not None
        assert not any(i.severity == "error" for i in issues2)
        assert out2.param_values == out1.param_values
        out3, issues3 = _apply_post_processing(out2, schema, "q")
        assert out3 is not None
        assert not any(i.severity == "error" for i in issues3)
        assert out3.param_values[first_key] == out1.param_values[first_key]


class TestAlignRuntimeTablesToPlanner:
    """Tests for _align_runtime_tables_to_planner."""

    def test_main_tables_replaced(self) -> None:
        from aetherdialect._contracts_base import LogicalIntent
        from aetherdialect._intent_process import _align_runtime_tables_to_planner

        logical = LogicalIntent(
            tables=("film", "inventory", "rental", "payment"),
            select="",
        )
        runtime = RuntimeIntent(
            tables=["film", "payment"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            natural_language="q",
        )
        out = _align_runtime_tables_to_planner(runtime, logical)
        assert out.tables == ["film", "inventory", "rental", "payment"]

    def test_cte_tables_aligned_by_name(self) -> None:
        from aetherdialect._contracts_base import CteIntent, LogicalIntent
        from aetherdialect._intent_process import _align_runtime_tables_to_planner

        logical = LogicalIntent(
            tables=("film",),
            select="",
            cte_steps=(
                CteIntent(
                    name="step_a",
                    tables=("a", "b", "bridge", "c"),
                ),
            ),
        )
        cte = RuntimeCteStep(cte_name="step_a", tables=["a", "c"])
        runtime = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = _align_runtime_tables_to_planner(runtime, logical)
        assert out.cte_steps[0].tables == ["a", "b", "bridge", "c"]

    def test_unmatched_cte_unchanged(self) -> None:
        from aetherdialect._contracts_base import LogicalIntent
        from aetherdialect._intent_process import _align_runtime_tables_to_planner

        logical = LogicalIntent(tables=("x",), select="")
        cte = RuntimeCteStep(cte_name="orphan", tables=["only"])
        runtime = RuntimeIntent(
            tables=["x"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("x.y"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            natural_language="q",
            cte_steps=[cte],
        )
        out = _align_runtime_tables_to_planner(runtime, logical)
        assert out.cte_steps[0].tables == ["only"]
