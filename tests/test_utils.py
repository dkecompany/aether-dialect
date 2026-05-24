"""Tests for utils module: tokenization, distance, hashing, shapes."""

from unittest.mock import patch

import pytest

from aetherdialect._config import (
    QUESTION_STARTS_AGG,
    QUESTION_STARTS_GROUP,
    QUESTION_STARTS_LIST,
)
from aetherdialect._contracts_base import (
    ColumnMetadata,
    LlmJsonExhausted,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
    concrete_intent_to_runtime_skeleton,
)
from aetherdialect._utils import (
    _describe_operation,
    _enforce_normalization_guard,
    _levenshtein_distance,
    _normalize_cte_steps,
    _normalize_cte_steps_for_key,
    _normalize_filters,
    _normalize_having_conditions,
    _pick_question_style,
    _tokenize,
    body_similarity_key,
    body_similarity_key_for_concrete,
    exact_question_match,
    extract_tables_from_sql,
    flatten_param_values,
    generate_bulk_anchors,
    generate_question,
    generate_question_from_sql,
    intent_key,
    match_question_against_template_history,
    question_token_fingerprint_from_raw,
    select_three_warmup_styles,
    sql_shape,
    template_instance_key_from_parts,
    validate_question,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance."""

    def test_identical_strings(self):
        """levenshtein_distance of identical strings is 0."""
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_string(self):
        """levenshtein_distance with empty string equals length of other."""
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_kitten_sitting(self):
        """levenshtein_distance of 'kitten' and 'sitting' is 3."""
        assert _levenshtein_distance("kitten", "sitting") == 3

    def test_symmetric(self):
        """levenshtein_distance is symmetric."""
        assert _levenshtein_distance("abc", "xyz") == _levenshtein_distance("xyz", "abc")

    def test_single_edit(self):
        """levenshtein_distance for single character difference is 1."""
        assert _levenshtein_distance("cat", "car") == 1

    def test_completely_different(self):
        """levenshtein_distance for completely different strings."""
        assert _levenshtein_distance("abc", "xyz") == 3


class TestTokenize:
    """Tests for tokenize."""

    def test_basic_tokenization(self):
        """Tokenize extracts alphanumeric tokens."""
        tokens = _tokenize("total orders by customer")
        assert "total" in tokens
        assert "order" in tokens
        assert "customer" in tokens

    def test_excludes_stopwords(self):
        """Tokenize excludes PolicyConfig.STOPWORDS."""
        tokens = _tokenize("what is the total amount")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "total" in tokens
        assert "amount" in tokens

    def test_returns_sorted(self):
        """Tokenize returns sorted tokens."""
        tokens = _tokenize("zebra apple mango")
        assert tokens == sorted(tokens)

    def test_empty_input(self):
        """Tokenize returns empty list for empty input."""
        assert _tokenize("") == []


class TestSqlShape:
    """Tests for sql_shape."""

    def test_detects_join(self):
        """sql_shape detects JOIN keywords."""
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.num_joins == 1

    def test_detects_group_by(self):
        """sql_shape detects GROUP BY clause."""
        sql = "SELECT status, COUNT(*) FROM orders GROUP BY status"
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.has_group_by is True

    def test_detects_agg(self):
        """sql_shape detects aggregation functions."""
        sql = "SELECT COUNT(*) FROM orders"
        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.has_agg is True

    def test_counts_filters(self):
        """sql_shape counts filters from intent."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.col"), op="=", value_type="string")
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        shape = sql_shape("SELECT * FROM t WHERE t.col = 'x'", intent, sqlglot_dialect="postgres")
        assert shape.num_filters == 1

    def test_detects_distinct(self):
        """sql_shape detects SELECT DISTINCT."""
        sql = "SELECT DISTINCT name FROM customers"
        intent = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.has_distinct is True


class TestIntentKey:
    """Tests for intent_key."""

    def test_deterministic(self, minimal_intent):
        """intent_key produces same hash for same intent."""
        k1 = intent_key(minimal_intent)
        k2 = intent_key(minimal_intent)
        assert k1 == k2

    def test_different_tables_different_key(self):
        """intent_key differs when tables differ."""
        i1 = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        i2 = RuntimeIntent(
            tables=["customers"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_same_structure_different_grain_same_key(self):
        """intent_key does not include grain in hash."""
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) == intent_key(i2)

    def test_returns_sha256_hex(self, minimal_intent):
        """intent_key returns 64-char hex string."""
        k = intent_key(minimal_intent)
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)

    def test_different_bool_op_different_key(self):
        """intent_key differs when filter bool_op differs."""
        fp_and = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            bool_op="AND",
        )
        fp_or = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            bool_op="OR",
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.col"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_and],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.col"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_or],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_different_filter_group_different_key(self):
        """intent_key differs when filter filter_group differs."""
        fp_none = FilterParam(left_expr=NormalizedExpr.from_column("t.col"), op="=", value_type="string")
        fp_grp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            filter_group=1,
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.col"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_none],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.col"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_grp],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_different_having_bool_op_different_key(self):
        """intent_key differs when having bool_op differs."""
        hp_and = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            bool_op="AND",
        )
        hp_or = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            bool_op="OR",
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_and],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_or],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_different_having_filter_group_different_key(self):
        """intent_key differs when having filter_group differs."""
        hp_none = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        hp_grp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            filter_group=1,
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_none],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_grp],
        )
        assert intent_key(i1) != intent_key(i2)


class TestNormalizeFilters:
    """Tests for normalize_filters."""

    def test_normalizes_filter_params(self):
        """normalize_filters converts FilterParam list."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.col"), op="=", value_type="string")
        result = _normalize_filters([fp])
        assert len(result) == 1
        assert result[0].op == "="

    def test_normalizes_dict_filters(self):
        """normalize_filters converts dict-based filters."""
        d = {"column": "t.col", "op": ">=", "value_type": "number"}
        result = _normalize_filters([d])
        assert len(result) == 1
        assert result[0].op == ">="

    def test_sorts_by_signature(self):
        """normalize_filters sorts by signature_key."""
        f1 = FilterParam(left_expr=NormalizedExpr.from_column("z.col"), op="=")
        f2 = FilterParam(left_expr=NormalizedExpr.from_column("a.col"), op="=")
        result = _normalize_filters([f1, f2])
        assert result[0].left_expr.primary_term == "a.col"

    def test_empty_input(self):
        """normalize_filters returns empty for empty input."""
        assert _normalize_filters([]) == []

    def test_preserves_bool_op(self):
        """normalize_filters preserves bool_op on FilterParam."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            bool_op="OR",
        )
        result = _normalize_filters([fp])
        assert result[0].bool_op == "OR"

    def test_preserves_filter_group(self):
        """normalize_filters preserves filter_group on FilterParam."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="=",
            value_type="string",
            filter_group=2,
        )
        result = _normalize_filters([fp])
        assert result[0].filter_group == 2

    def test_sorts_by_filter_group_first(self):
        """normalize_filters sorts by structural key across different filter_groups."""
        f1 = FilterParam(left_expr=NormalizedExpr.from_column("a.col"), op="=", filter_group=2)
        f2 = FilterParam(left_expr=NormalizedExpr.from_column("z.col"), op="=", filter_group=1)
        result = _normalize_filters([f1, f2])
        assert result[0].filter_group == 2
        assert result[1].filter_group == 1


class TestNormalizeHavingConditions:
    """Tests for normalize_having_conditions."""

    def test_normalizes_having_params(self):
        """normalize_having_conditions converts HavingParam list."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        result = _normalize_having_conditions([hp])
        assert len(result) == 1
        assert result[0].op == ">"

    def test_empty_input(self):
        """normalize_having_conditions returns empty for empty input."""
        assert _normalize_having_conditions([]) == []

    def test_preserves_bool_op(self):
        """normalize_having_conditions preserves bool_op on HavingParam."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            bool_op="OR",
        )
        result = _normalize_having_conditions([hp])
        assert result[0].bool_op == "OR"

    def test_preserves_filter_group(self):
        """normalize_having_conditions preserves filter_group on HavingParam."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
            filter_group=3,
        )
        result = _normalize_having_conditions([hp])
        assert result[0].filter_group == 3

    def test_sorts_by_filter_group_first(self):
        """normalize_having_conditions sorts by structural key; group order follows first occurrence of filter_group."""

        h1 = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.amount"), op=">", filter_group=2)
        h2 = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.id"), op=">", filter_group=1)
        result = _normalize_having_conditions([h1, h2])
        assert result[0].filter_group == 2
        assert result[1].filter_group == 1


class TestExtractTablesFromSql:
    """Tests for extract_tables_from_sql."""

    def test_extracts_from_clause(self):
        """extract_tables_from_sql finds tables in FROM/JOIN."""
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
        tables = extract_tables_from_sql(sql, ["orders", "customers", "products"], sqlglot_dialect="postgres")
        assert "orders" in tables
        assert "customers" in tables
        assert "products" not in tables

    def test_excludes_cte_names(self):
        """extract_tables_from_sql excludes CTE names."""
        sql = "WITH cte1 AS (SELECT * FROM orders) SELECT * FROM cte1"
        tables = extract_tables_from_sql(sql, ["orders", "cte1"], sqlglot_dialect="postgres")
        assert "orders" in tables

    def test_case_insensitive(self):
        """extract_tables_from_sql is case-insensitive."""
        sql = "SELECT * FROM ORDERS"
        tables = extract_tables_from_sql(sql, ["orders"], sqlglot_dialect="postgres")
        assert "orders" in tables


class TestExactQuestionMatch:
    """Tests for exact_question_match."""

    def test_identical_questions(self):
        """exact_question_match returns True for identical questions."""
        assert exact_question_match("How many customers?", "How many customers?") is True

    def test_fuzzy_within_distance(self):
        """exact_question_match accepts within max_distance."""
        assert exact_question_match("How many customer?", "How many customers?", max_distance=1) is True

    def test_reject_beyond_distance(self):
        """exact_question_match rejects beyond max_distance."""
        assert exact_question_match("Total revenue by city", "Average cost by region", max_distance=1) is False

    def test_different_token_count_rejects(self):
        """exact_question_match rejects different token counts."""
        assert exact_question_match("How many", "How many customers are there") is False

    def test_empty_string_rejects(self):
        """exact_question_match rejects empty string."""
        assert exact_question_match("", "hello") is False


class TestMatchQuestionAgainstTemplateHistory:
    """Tests for match_question_against_template_history."""

    @staticmethod
    def _tpl(tid: str, question: str, trust: int) -> Template:
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
            value_history=ValueHistory(
                questions=[question],
                param_values=[{}],
                natural_language=[""],
            ),
            stats=TemplateStats(),
            trust_level=trust,
        )

    def test_empty_templates_returns_none(self):
        assert match_question_against_template_history("any", []) is None

    def test_trust_zero_skipped(self):
        t = self._tpl("T0", "show orders", 0)
        assert match_question_against_template_history("show orders", [t]) is None

    def test_returns_match_with_exact_string_flag(self):
        t = self._tpl("T1", "show all orders", 1)
        m = match_question_against_template_history("show all orders", [t])
        assert m is not None
        assert m.template_id == "T1"
        assert m.history_index == 0
        assert m.is_exact_string_reuse is True

    def test_first_listed_template_wins_on_duplicate_stored_text(self):
        a = self._tpl("T_A", "same text here", 1)
        b = self._tpl("T_B", "same text here", 1)
        m = match_question_against_template_history("same text here", [a, b])
        assert m is not None
        assert m.template_id == "T_A"

    def test_fuzzy_typos_sets_exact_string_false_when_normals_differ(self):
        t = self._tpl("T1", "How many customers?", 1)
        m = match_question_against_template_history("How many customer?", [t], max_token_edit_distance=1)
        assert m is not None
        assert m.is_exact_string_reuse is False
        assert m.token_edit_sum == 0

    def test_question_token_index_pair_filter_overrides_list_order(self):
        """When the token index lists only one (template_id, history_index) pair, that row wins over earlier duplicates."""
        bad = self._tpl("T_BAD", "same text here", 1)
        good = self._tpl("T_GOOD", "same text here", 1)
        fp = question_token_fingerprint_from_raw("same text here")
        qtok = {fp: [["T_GOOD", "0"]]}
        m = match_question_against_template_history("same text here", [bad, good], question_token_index=qtok)
        assert m is not None
        assert m.template_id == "T_GOOD"
        m2 = match_question_against_template_history("same text here", [bad, good])
        assert m2 is not None
        assert m2.template_id == "T_BAD"


class TestNormalizeCteSteps:
    """Tests for normalize_cte_steps."""

    def test_dict_input_converted(self):
        """normalize_cte_steps converts dict to RuntimeCteStep."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": ["t1.id"],
                "group_by_cols": [],
                "output_columns": ["id"],
            }
        ]
        result = _normalize_cte_steps(steps)
        assert len(result) == 1
        assert isinstance(result[0], RuntimeCteStep)
        assert result[0].cte_name == "cte1"

    def test_runtime_cte_step_passthrough(self):
        """normalize_cte_steps passes through RuntimeCteStep objects."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
        )
        result = _normalize_cte_steps([cte])
        assert len(result) == 1
        assert result[0].cte_name == "cte1"

    def test_empty_list(self):
        """normalize_cte_steps returns empty list for empty input."""
        assert _normalize_cte_steps([]) == []

    def test_non_list_returns_empty(self):
        """normalize_cte_steps returns empty list for non-list input."""
        assert _normalize_cte_steps("not a list") == []

    def test_invalid_grain_defaults(self):
        """normalize_cte_steps defaults invalid grain to row_level."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "bad_grain",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": [],
            }
        ]
        result = _normalize_cte_steps(steps)
        assert result[0].grain == "row_level"

    def test_skips_no_cte_name(self):
        """normalize_cte_steps skips entries with no cte_name."""
        steps = [{"cte_name": "", "tables": ["t1"]}]
        result = _normalize_cte_steps(steps)
        assert len(result) == 0


class TestNormalizeCteStepsForKey:
    """Tests for normalize_cte_steps_for_key."""

    def test_returns_structural_dict(self):
        """normalize_cte_steps_for_key returns list of dicts."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            output_columns=["id"],
        )
        result = _normalize_cte_steps_for_key([cte])
        assert len(result) == 1
        assert result[0]["cte_name"] == "cte1"
        assert "select_cols" in result[0]
        assert "tables" in result[0]

    def test_empty_list(self):
        """normalize_cte_steps_for_key returns empty for empty input."""
        assert _normalize_cte_steps_for_key([]) == []

    def test_filter_bool_op_in_key(self):
        """normalize_cte_steps_for_key includes bool_op in filter key entries."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t1.status"),
            op="=",
            value_type="string",
            bool_op="OR",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
        )
        result = _normalize_cte_steps_for_key([cte])
        assert any("OR" in f for f in result[0]["filters_param"])

    def test_filter_group_in_key(self):
        """normalize_cte_steps_for_key includes filter_group in filter key entries."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t1.status"),
            op="=",
            value_type="string",
            filter_group=1,
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
        )
        result = _normalize_cte_steps_for_key([cte])
        assert any("1" in f for f in result[0]["filters_param"])


class TestDescribeOperation:
    """Tests for describe_operation."""

    def test_list_for_bare_columns(self):
        """describe_operation returns list for bare columns."""
        assert _describe_operation(["t.name", "t.id"]) == "list"

    def test_detects_count(self):
        """describe_operation detects COUNT."""
        assert _describe_operation(["COUNT(t.id)"]) == "count"

    def test_detects_sum(self):
        """describe_operation detects SUM."""
        assert _describe_operation(["SUM(t.amount)"]) == "sum"

    def test_detects_avg(self):
        """describe_operation detects AVG."""
        assert _describe_operation(["AVG(t.price)"]) == "avg"

    def test_detects_min(self):
        """describe_operation detects MIN."""
        assert _describe_operation(["MIN(t.price)"]) == "min"


class TestPickQuestionStyle:
    """Tests for pick_question_style."""

    def test_grouped_returns_group_starts(self):
        """pick_question_style returns group start for grouped query."""
        result = _pick_question_style(["t.name"], has_grouping=True)
        assert result in QUESTION_STARTS_GROUP

    def test_count_returns_how_many(self):
        """pick_question_style returns count start for COUNT."""
        result = _pick_question_style(["COUNT(t.id)"], has_grouping=False)
        assert result in ["How many", "Count", "What is the number of"]

    def test_sum_returns_total(self):
        """pick_question_style returns sum start for SUM."""
        result = _pick_question_style(["SUM(t.amount)"], has_grouping=False)
        assert result in ["What is the total", "Find the sum of", "Calculate the total"]

    def test_no_agg_returns_list_start(self):
        """pick_question_style returns list start for no aggregation."""
        result = _pick_question_style(["t.name", "t.id"], has_grouping=False)
        assert result in QUESTION_STARTS_LIST


class TestFlattenParamValues:
    """Tests for flatten_param_values."""

    def test_main_only(self):
        """flatten_param_values returns main param_values when no CTEs."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            param_values={"p1": "val1"},
        )
        result = flatten_param_values(intent)
        assert result == {"p1": "val1"}

    def test_cte_and_main_merged(self):
        """flatten_param_values merges CTE and main param_values."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            param_values={"p1": "cte_val"},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
            param_values={"p2": "main_val"},
        )
        result = flatten_param_values(intent)
        assert result["p1"] == "cte_val"
        assert result["p2"] == "main_val"

    def test_main_overrides_cte(self):
        """flatten_param_values gives main values priority over CTE."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            param_values={"p1": "cte_val"},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
            param_values={"p1": "main_val"},
        )
        result = flatten_param_values(intent)
        assert result["p1"] == "main_val"

    def test_empty_param_values(self):
        """flatten_param_values returns empty dict when no params."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = flatten_param_values(intent)
        assert result == {}

    def test_multiple_ctes(self):
        """flatten_param_values merges from multiple CTEs."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            param_values={"p1": "v1"},
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            tables=["t2"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            param_values={"p2": "v2"},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte1, cte2],
        )
        result = flatten_param_values(intent)
        assert result["p1"] == "v1"
        assert result["p2"] == "v2"

    def test_contains_op_normalizes_quoted_param_values_only(self):
        """flatten_param_values unwraps quoted values for contains filters only."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["film.special_features"])])
        fp = FilterParam(left_expr=left, op="contains", param_key="p1", raw_value=None)
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            param_values={"p1": '"Trailers"', "p2": "'keep'"},
        )
        out = flatten_param_values(intent)
        assert out["p1"] == "Trailers"
        assert out["p2"] == "'keep'"


class TestLevenshteinDistanceEdgeCases:
    """Edge-case tests for levenshtein_distance."""

    def test_both_empty(self):
        """levenshtein_distance of two empty strings is 0."""
        assert _levenshtein_distance("", "") == 0

    def test_single_char_insert(self):
        """levenshtein_distance for single char insertion is 1."""
        assert _levenshtein_distance("abc", "ab") == 1

    def test_unicode_strings(self):
        """levenshtein_distance handles unicode."""
        assert _levenshtein_distance("café", "cafe") == 1


class TestTokenizeEdgeCases:
    """Edge-case tests for tokenize."""

    def test_special_chars_ignored(self):
        """Tokenize ignores special characters."""
        tokens = _tokenize("hello! world?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_numeric_tokens(self):
        """Tokenize includes numeric tokens."""
        tokens = _tokenize("show top 10 results")
        assert "10" in tokens

    def test_underscore_tokens(self):
        """Tokenize preserves underscored tokens."""
        tokens = _tokenize("order_date is important")
        assert "order_date" in tokens


class TestSqlShapeEdgeCases:
    """Edge-case tests for sql_shape."""

    def test_no_join_no_group_no_agg(self):
        """sql_shape detects simple SELECT."""
        sql = "SELECT * FROM t"
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.num_joins == 0
        assert shape.has_group_by is False
        assert shape.has_agg is False

    def test_multiple_joins(self):
        """sql_shape counts multiple joins."""
        sql = "SELECT * FROM a JOIN b ON a.id = b.aid JOIN c ON b.id = c.bid"
        intent = RuntimeIntent(
            tables=["a", "b", "c"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        shape = sql_shape(sql, intent, sqlglot_dialect="postgres")
        assert shape.num_joins == 2

    def test_cte_filters_counted(self):
        """sql_shape counts CTE step filters."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t1.col"), op="=")
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        shape = sql_shape("SELECT * FROM t", intent, sqlglot_dialect="postgres")
        assert shape.num_filters == 1

    def test_cte_having_counted(self):
        """sql_shape counts CTE step HAVING params."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t1.id"),
            op=">",
            value_type="integer",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            having_param=[hp],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        shape = sql_shape("SELECT * FROM t", intent, sqlglot_dialect="postgres")
        assert shape.num_having == 1

    def test_expr_comparison_counted(self):
        """sql_shape counts expr comparison filters."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            right_expr=NormalizedExpr.from_column("t.b"),
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        shape = sql_shape("SELECT * FROM t WHERE a = b", intent, sqlglot_dialect="postgres")
        assert shape.num_filters == 1

    def test_expr_comparisons_from_cte(self):
        """sql_shape counts expr comparisons in CTE filters."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="column",
            right_expr=NormalizedExpr.from_column("t.b"),
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        shape = sql_shape("SELECT * FROM t", intent, sqlglot_dialect="postgres")
        assert shape.num_filters == 1

    def test_num_cte(self):
        """sql_shape counts CTE steps."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            tables=["t2"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte1, cte2],
        )
        shape = sql_shape("SELECT * FROM t", intent, sqlglot_dialect="postgres")
        assert shape.num_cte == 2


class TestNormalizeFiltersEdgeCases:
    """Edge-case tests for normalize_filters."""

    def test_skips_non_dict_non_filter(self):
        """normalize_filters skips invalid entries."""
        result = _normalize_filters([42, "bad", None])
        assert len(result) == 0

    def test_dict_with_right_expr(self):
        """normalize_filters handles dict with right_expr."""
        d = {
            "column": "t.a",
            "op": "=",
            "value_type": "column",
            "right_expr": {"groups": [{"terms": [{"column": "t.b"}]}]},
        }
        result = _normalize_filters([d])
        assert len(result) == 1
        assert result[0].right_expr is not None

    def test_dict_with_left_expr_dict(self):
        """normalize_filters handles dict with left_expr as dict."""
        d = {
            "left_expr": {"groups": [{"terms": [{"column": "t.col"}]}]},
            "op": ">=",
            "value_type": "number",
        }
        result = _normalize_filters([d])
        assert len(result) == 1
        assert result[0].op == ">="

    def test_dict_preserves_bool_op(self):
        """normalize_filters preserves bool_op from raw dict."""
        d = {"column": "t.col", "op": "=", "value_type": "string", "bool_op": "OR"}
        result = _normalize_filters([d])
        assert result[0].bool_op == "OR"

    def test_dict_preserves_filter_group(self):
        """normalize_filters preserves filter_group from raw dict."""
        d = {"column": "t.col", "op": "=", "value_type": "string", "filter_group": 2}
        result = _normalize_filters([d])
        assert result[0].filter_group == 2

    def test_dict_defaults_bool_op_to_and(self):
        """normalize_filters defaults bool_op to 'AND' when absent in dict."""
        d = {"column": "t.col", "op": "=", "value_type": "string"}
        result = _normalize_filters([d])
        assert result[0].bool_op == "AND"

    def test_dict_defaults_filter_group_to_none(self):
        """normalize_filters defaults filter_group to None when absent in dict."""
        d = {"column": "t.col", "op": "=", "value_type": "string"}
        result = _normalize_filters([d])
        assert result[0].filter_group is None


class TestNormalizeHavingEdgeCases:
    """Edge-case tests for normalize_having_conditions."""

    def test_dict_input(self):
        """normalize_having_conditions converts raw dict."""
        d = {"aggregation": "COUNT(t.id)", "op": ">", "value_type": "integer"}
        result = _normalize_having_conditions([d])
        assert len(result) == 1
        assert result[0].op == ">"

    def test_invalid_op_defaults(self):
        """normalize_having_conditions defaults invalid op to equals."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op="INVALID_OP",
            value_type="integer",
        )
        result = _normalize_having_conditions([hp])
        assert result[0].op == "="

    def test_skips_empty_primary_term(self):
        """normalize_having_conditions skips entries with empty primary_term."""
        d = {"aggregation": "", "op": ">", "value_type": "integer"}
        result = _normalize_having_conditions([d])
        assert len(result) == 0

    def test_dict_with_right_expr(self):
        """normalize_having_conditions handles dict with right_expr."""
        d = {
            "aggregation": "COUNT(t.id)",
            "op": ">",
            "value_type": "integer",
            "right_expr": {"groups": [{"terms": [{"column": "t.threshold"}]}]},
        }
        result = _normalize_having_conditions([d])
        assert len(result) == 1
        assert result[0].right_expr is not None

    def test_dict_preserves_bool_op(self):
        """normalize_having_conditions preserves bool_op from raw dict."""
        d = {
            "aggregation": "COUNT(t.id)",
            "op": ">",
            "value_type": "integer",
            "bool_op": "OR",
        }
        result = _normalize_having_conditions([d])
        assert result[0].bool_op == "OR"

    def test_dict_preserves_filter_group(self):
        """normalize_having_conditions preserves filter_group from raw dict."""
        d = {
            "aggregation": "COUNT(t.id)",
            "op": ">",
            "value_type": "integer",
            "filter_group": 3,
        }
        result = _normalize_having_conditions([d])
        assert result[0].filter_group == 3

    def test_dict_defaults_bool_op_to_and(self):
        """normalize_having_conditions defaults bool_op to 'AND' when absent in dict."""
        d = {"aggregation": "COUNT(t.id)", "op": ">", "value_type": "integer"}
        result = _normalize_having_conditions([d])
        assert result[0].bool_op == "AND"

    def test_dict_defaults_filter_group_to_none(self):
        """normalize_having_conditions defaults filter_group to None when absent in dict."""
        d = {"aggregation": "COUNT(t.id)", "op": ">", "value_type": "integer"}
        result = _normalize_having_conditions([d])
        assert result[0].filter_group is None


class TestExtractTablesEdgeCases:
    """Edge-case tests for extract_tables_from_sql."""

    def test_empty_known_tables_returns_empty(self):
        """extract_tables_from_sql returns empty when known_tables is empty."""
        result = extract_tables_from_sql("SELECT * FROM orders", [], sqlglot_dialect="postgres")
        assert result == []

    def test_empty_sql(self):
        """extract_tables_from_sql returns empty for empty SQL."""
        assert extract_tables_from_sql("", ["t"], sqlglot_dialect="postgres") == []

    def test_no_known_tables(self):
        """extract_tables_from_sql returns empty when no known tables match."""
        assert extract_tables_from_sql("SELECT * FROM orders", [], sqlglot_dialect="postgres") == []

    def test_with_cte(self):
        """extract_tables_from_sql identifies CTE and excludes it."""
        sql = "WITH summary AS (SELECT * FROM orders) SELECT * FROM summary JOIN customers ON 1=1"
        tables = extract_tables_from_sql(sql, ["orders", "customers", "summary"], sqlglot_dialect="postgres")
        assert "orders" in tables
        assert "customers" in tables
        assert "summary" not in tables

    def test_multi_cte_excludes_all_cte_names(self):
        """extract_tables_from_sql excludes all CTE names in multi-CTE queries."""
        sql = (
            "WITH cte1 AS (SELECT * FROM orders), cte2 AS (SELECT * FROM customers) SELECT * FROM cte1 JOIN cte2 ON 1=1"
        )
        tables = extract_tables_from_sql(sql, ["orders", "customers", "cte1", "cte2"], sqlglot_dialect="postgres")
        assert "orders" in tables
        assert "customers" in tables
        assert "cte1" not in tables
        assert "cte2" not in tables

    def test_multiple_tables(self):
        """extract_tables_from_sql finds multiple tables."""
        sql = "SELECT * FROM orders o JOIN customers c ON o.cid = c.id JOIN products p ON o.pid = p.id"
        tables = extract_tables_from_sql(sql, ["orders", "customers", "products"], sqlglot_dialect="postgres")
        assert len(tables) == 3


class TestExactQuestionMatchEdgeCases:
    """Edge-case tests for exact_question_match."""

    def test_both_empty(self):
        """exact_question_match rejects both empty."""
        assert exact_question_match("", "") is False

    def test_case_insensitive(self):
        """exact_question_match is case-insensitive."""
        assert exact_question_match("How Many Orders", "how many orders") is True

    def test_stopword_difference_ignored(self):
        """exact_question_match ignores stopwords in comparison."""
        assert exact_question_match("total orders", "total orders") is True


class TestNormalizeCteStepsEdgeCases:
    """Edge-case tests for normalize_cte_steps."""

    def test_dict_with_filters(self):
        """normalize_cte_steps converts dict with filter params."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": [],
                "filters_param": [
                    {
                        "left_expr": {"groups": [{"terms": [{"column": "t1.col"}]}]},
                        "op": "=",
                        "value_type": "string",
                    }
                ],
            }
        ]
        result = _normalize_cte_steps(steps)
        assert len(result) == 1
        assert len(result[0].filters_param) == 1

    def test_dict_with_having(self):
        """normalize_cte_steps converts dict with having params."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "grouped",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": [],
                "having_param": [
                    {
                        "left_expr": {"groups": [{"terms": [{"column": "COUNT(t1.id)"}]}]},
                        "op": ">",
                        "value_type": "integer",
                    }
                ],
            }
        ]
        result = _normalize_cte_steps(steps)
        assert len(result) == 1
        assert len(result[0].having_param) == 1

    def test_available_ctes_mutated(self):
        """normalize_cte_steps populates available_ctes dict."""
        available = {}
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": ["id", "name"],
            }
        ]
        _normalize_cte_steps(steps, available_ctes=available)
        assert "cte1" in available
        assert available["cte1"] == ["id", "name"]

    def test_output_column_metadata_generated(self):
        """normalize_cte_steps generates output column metadata for agg columns."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "grouped",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": ["count_id"],
            }
        ]
        result = _normalize_cte_steps(steps)
        ocm = result[0].output_column_metadata
        assert "count_id" in ocm
        assert ocm["count_id"].source == "aggregation"
        assert ocm["count_id"].agg_func == "count"

    def test_column_map_inferred(self):
        """normalize_cte_steps infers column_map from select_cols."""
        steps = [
            {
                "cte_name": "cte1",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": ["t1.name"],
                "group_by_cols": [],
                "output_columns": ["name"],
            }
        ]
        result = _normalize_cte_steps(steps)
        assert result[0].column_map.get("name") == "t1"


class TestNormalizeCteStepsForKeyEdgeCases:
    """Edge-case tests for normalize_cte_steps_for_key."""

    def test_with_filters_and_having(self):
        """normalize_cte_steps_for_key includes filter and having signatures."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.col"), op="=")
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
            having_param=[hp],
        )
        result = _normalize_cte_steps_for_key([cte])
        assert len(result) == 1
        assert len(result[0]["filters_param"]) == 1
        assert len(result[0]["having_param"]) == 1

    def test_with_order_by(self):
        """normalize_cte_steps_for_key includes order_by_cols signatures."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("t.id"), direction="ASC")
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            order_by_cols=[obc],
        )
        result = _normalize_cte_steps_for_key([cte])
        assert len(result[0]["order_by_cols"]) == 1


class TestDescribeOperationEdgeCases:
    """Edge-case tests for describe_operation."""

    def test_max(self):
        """describe_operation detects MAX."""
        assert _describe_operation(["MAX(t.price)"]) == "max"

    def test_mixed_agg_and_bare(self):
        """describe_operation returns first agg found."""
        assert _describe_operation(["t.name", "SUM(t.amount)"]) == "sum"

    def test_empty_list(self):
        """describe_operation returns list for empty input."""
        assert _describe_operation([]) == "list"

    def test_case_insensitive(self):
        """describe_operation is case-insensitive."""
        assert _describe_operation(["count(t.id)"]) == "count"


class TestPickQuestionStyleEdgeCases:
    """Edge-case tests for pick_question_style."""

    def test_avg_returns_average(self):
        """pick_question_style returns average start for AVG."""
        result = _pick_question_style(["AVG(t.price)"], has_grouping=False)
        assert result in [
            "What is the average",
            "Calculate the average",
            "Find the mean",
        ]

    def test_min_returns_minimum(self):
        """pick_question_style returns min start for MIN."""
        result = _pick_question_style(["MIN(t.price)"], has_grouping=False)
        assert "min" in result.lower()

    def test_max_returns_maximum(self):
        """pick_question_style returns max start for MAX."""
        result = _pick_question_style(["MAX(t.price)"], has_grouping=False)
        assert "max" in result.lower()

    def test_grouping_overrides_agg(self):
        """pick_question_style returns group start when has_grouping is True."""
        result = _pick_question_style(["COUNT(t.id)"], has_grouping=True)
        assert result in QUESTION_STARTS_GROUP


class TestIntentKeyEdgeCases:
    """Edge-case tests for intent_key."""

    def test_with_cte_steps(self):
        """intent_key includes CTE steps in hash."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_with_having(self):
        """intent_key includes HAVING conditions in hash."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            value_type="integer",
        )
        i1 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) != intent_key(i2)

    def test_order_independent_tables(self):
        """intent_key is order-independent for tables."""
        i1 = RuntimeIntent(
            tables=["a", "b"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        i2 = RuntimeIntent(
            tables=["b", "a"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) == intent_key(i2)

    def test_invalid_grain_defaults(self):
        """intent_key defaults invalid grain to row_level."""
        i1 = RuntimeIntent(
            tables=["t"],
            grain="bad_grain",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert intent_key(i1) == intent_key(i2)


class TestValidateQuestion:
    """Tests for validate_question."""

    @patch("aetherdialect._utils.llm_json")
    def test_valid_allowed_returns_true_and_corrected(self, mock_llm_json):
        """validate_question returns (True, 'allowed', corrected) when LLM says valid."""
        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "allowed",
            "corrected": "What is the total revenue?",
        }
        is_valid, query_type, corrected = validate_question("What is the total reveneu?")
        assert is_valid is True
        assert query_type == "allowed"
        assert corrected == "What is the total revenue?"

    @patch("aetherdialect._utils.llm_json")
    def test_invalid_returns_false_invalid_type(self, mock_llm_json):
        """validate_question returns (False, 'invalid', ...) when not valid."""
        mock_llm_json.return_value = {
            "valid_database_question": "no",
            "query_type": "unspecified",
            "corrected": "fixed typo",
        }
        is_valid, query_type, corrected = validate_question("random text")
        assert is_valid is False
        assert query_type == "invalid"
        assert corrected == "fixed typo"

    @patch("aetherdialect._utils.llm_json")
    def test_restricted_returns_false_restricted_type(self, mock_llm_json):
        """validate_question returns (False, 'restricted', ...) for restricted query."""
        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "restricted",
            "corrected": "List all users",
        }
        is_valid, query_type, corrected = validate_question("List all users")
        assert is_valid is False
        assert query_type == "restricted"

    @patch("aetherdialect._utils.llm_json")
    def test_restricted_wins_when_valid_flag_no(self, mock_llm_json):
        """restricted is reported even when valid_database_question is no (Group A precedence)."""

        mock_llm_json.return_value = {
            "valid_database_question": "no",
            "query_type": "restricted",
            "corrected": "drop table t",
        }
        is_valid, query_type, corrected = validate_question("drop table t")
        assert is_valid is False
        assert query_type == "restricted"
        assert corrected == "drop table t"

    @patch("aetherdialect._utils.llm_json")
    def test_cte_analytical_may_be_allowed_not_restricted(self, mock_llm_json):
        """Analytical questions that mention CTEs are classified allowed when the model agrees (CJ regression)."""

        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "allowed",
            "corrected": "Using a CTE show monthly revenue trends",
        }
        ok, qt, _ = validate_question("Using a CTE show monthly revenue trends")
        assert ok is True
        assert qt == "allowed"

    @patch("aetherdialect._utils.llm_json")
    def test_llm_json_exhausted_returns_invalid(self, mock_llm_json):
        """validate_question returns (False, 'invalid', original) when llm_json exhausts."""
        mock_llm_json.side_effect = LlmJsonExhausted(task="default", attempts=2)
        is_valid, query_type, corrected = validate_question("anything")
        assert is_valid is False
        assert query_type == "invalid"
        assert corrected == "anything"

    @patch("aetherdialect._utils.llm_json")
    def test_missing_corrected_uses_original_question(self, mock_llm_json):
        """validate_question uses original question when corrected is missing."""
        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "allowed",
            "corrected": "",
        }
        is_valid, _, corrected = validate_question("Original question")
        assert is_valid is True
        assert corrected == "Original question"


class TestNormalizationGuard:
    """Regression tests for :func:`aetherdialect._utils._enforce_normalization_guard`."""

    def test_tc001_aggregation_prefix_and_particles(self):
        """``count of`` opener plus grammatical particles must pass the Jaccard and introduced-token checks."""

        raw = "how many orders were placed in the last month for each customer"
        corrected = raw
        normalized = "count of orders placed in the last month for each customer"
        ok, code = _enforce_normalization_guard(corrected, normalized, raw_original=raw)
        assert ok is True
        assert code == "ok"


class TestGenerateQuestion:
    """Tests for generate_question."""

    @pytest.fixture
    def minimal_schema(self):
        """Minimal SchemaGraph with one table for generate_question tests."""
        t = TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="int", value_type="integer"),
                "amount": ColumnMetadata(name="amount", data_type="numeric", value_type="number"),
            },
            foreign_keys=[],
            primary_key=[],
        )
        return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"orders": t})

    @patch("aetherdialect._utils._pick_question_style", return_value="How many")
    @patch("aetherdialect._utils.llm_json")
    def test_returns_question_when_llm_returns_valid_json(self, mock_llm_json, _mock_style, minimal_schema):
        """generate_question returns question string when LLM returns valid JSON."""
        mock_llm_json.return_value = {"question": "How many orders are there?"}
        result = generate_question(
            tables=["orders"],
            select_terms=["COUNT(orders.id)"],
            filter_descriptions=[],
            group_by_terms=[],
            having_descriptions=[],
            schema=minimal_schema,
        )
        assert result == "How many orders are there?"

    @patch("aetherdialect._utils.llm_json")
    def test_returns_none_when_llm_returns_empty(self, mock_llm_json, minimal_schema):
        """generate_question returns None when response lacks a usable question."""
        mock_llm_json.return_value = {}
        result = generate_question(
            tables=["orders"],
            select_terms=["orders.id"],
            filter_descriptions=[],
            group_by_terms=[],
            having_descriptions=[],
            schema=minimal_schema,
        )
        assert result is None

    @patch("aetherdialect._utils.llm_json")
    def test_returns_none_when_question_key_missing(self, mock_llm_json, minimal_schema):
        """generate_question returns None when response has no 'question' key."""
        mock_llm_json.return_value = {"other_key": "value"}
        result = generate_question(
            tables=["orders"],
            select_terms=[],
            filter_descriptions=[],
            group_by_terms=[],
            having_descriptions=[],
            schema=minimal_schema,
        )
        assert result is None

    @patch("aetherdialect._utils.llm_json")
    def test_invokes_llm_with_system_and_user_prompt(self, mock_llm_json, minimal_schema):
        """generate_question calls llm_json with system and user prompt."""
        mock_llm_json.return_value = {"question": "Generated question."}
        generate_question(
            tables=["orders"],
            select_terms=["SUM(orders.amount)"],
            filter_descriptions=[],
            group_by_terms=[],
            having_descriptions=[],
            schema=minimal_schema,
        )
        assert mock_llm_json.call_count == 1
        assert len(mock_llm_json.call_args[0]) >= 2
        assert isinstance(mock_llm_json.call_args[0][1], str)


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_question_starts_group_not_empty(self):
        """QUESTION_STARTS_GROUP is non-empty."""
        assert len(QUESTION_STARTS_GROUP) > 0

    def test_question_starts_list_not_empty(self):
        """QUESTION_STARTS_LIST is non-empty."""
        assert len(QUESTION_STARTS_LIST) > 0

    def test_question_starts_agg_not_empty(self):
        """QUESTION_STARTS_AGG is non-empty."""
        assert len(QUESTION_STARTS_AGG) > 0


class TestBodySimilarityAndTemplateKeys:
    """Tests for body_similarity_key, body_similarity_key_for_concrete, template_instance_key_from_parts."""

    def test_body_similarity_key_matches_intent_key(self, minimal_intent):
        """body_similarity_key delegates to intent_key."""
        assert body_similarity_key(minimal_intent) == intent_key(minimal_intent)

    def test_body_similarity_key_for_concrete_matches_skeleton_intent_key(self):
        """ConcreteIntent fingerprint matches runtime skeleton hash."""
        concrete = ConcreteIntent(
            intent_id="i1",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sk = concrete_intent_to_runtime_skeleton(concrete)
        assert body_similarity_key_for_concrete(concrete) == intent_key(sk)

    def test_template_instance_key_from_parts_deterministic(self):
        """Same parts yield the same template instance key."""
        k1 = template_instance_key_from_parts("body", "join", "sql")
        k2 = template_instance_key_from_parts("body", "join", "sql")
        assert k1 == k2
        assert len(k1) == 64

    def test_template_instance_key_from_parts_order_sensitive(self):
        """Swapping join vs sql fingerprint changes the key."""
        a = template_instance_key_from_parts("b", "j1", "s1")
        b = template_instance_key_from_parts("b", "s1", "j1")
        assert a != b


class TestGenerateQuestionFromSql:
    """Tests for generate_question_from_sql with mocked LLM."""

    @pytest.fixture
    def empty_schema(self):
        """Schema with no tables (prompt still builds)."""
        return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})

    @patch("aetherdialect._utils.llm_json")
    def test_unrealistic_returns_empty_questions(self, mock_llm, empty_schema):
        """Unrealistic response omits generated questions."""
        mock_llm.return_value = {
            "is_realistic": False,
            "drop_reason": "nonsense",
            "questions": ["should", "be", "ignored"],
        }
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out == {
            "questions": [],
            "question": "",
            "is_realistic": False,
            "drop_reason": "nonsense",
            "drop_reason_category": "other",
        }

    @patch("aetherdialect._utils.llm_json")
    def test_unrealistic_default_drop_reason(self, mock_llm, empty_schema):
        """Missing drop_reason uses fallback string."""
        mock_llm.return_value = {"is_realistic": False, "drop_reason": None}
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out["drop_reason"] == "unrealistic"
        assert out["drop_reason_category"] == "other"

    @patch("aetherdialect._utils.llm_json")
    def test_is_realistic_string_coercion(self, mock_llm, empty_schema):
        """String truthiness coerces to bool before branching."""
        mock_llm.return_value = {
            "is_realistic": "yes",
            "questions": ["First phrase?"],
        }
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out is not None
        assert out["is_realistic"] is True
        assert out["question"] == "First phrase?"

    @patch("aetherdialect._utils.llm_json")
    def test_realistic_truncates_questions_list(self, mock_llm, empty_schema):
        """Long question lists truncate to WARMUP_QUESTIONS_MAX."""
        from aetherdialect._config import SeedWarmupConfig

        many = [f"Question number {i}?" for i in range(SeedWarmupConfig.WARMUP_QUESTIONS_MAX + 5)]
        mock_llm.return_value = {"is_realistic": True, "questions": many}
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out is not None
        assert len(out["questions"]) == SeedWarmupConfig.WARMUP_QUESTIONS_MAX

    @patch("aetherdialect._utils.llm_json")
    def test_realistic_dedupes_normalized_phrases(self, mock_llm, empty_schema):
        """Phrases that normalize identically are dropped."""
        mock_llm.return_value = {
            "is_realistic": True,
            "questions": ["Total sales?", "total sales?"],
        }
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out is not None
        assert len(out["questions"]) == 1

    @patch("aetherdialect._utils.llm_json")
    def test_realistic_falls_back_to_legacy_question_field(self, mock_llm, empty_schema):
        """Empty questions array uses legacy question string."""
        mock_llm.return_value = {
            "is_realistic": True,
            "questions": [],
            "question": "Legacy only?",
        }
        out = generate_question_from_sql("SELECT 1", empty_schema, [])
        assert out is not None
        assert out["questions"] == ["Legacy only?"]

    @patch("aetherdialect._utils.llm_json")
    def test_realistic_empty_after_parse_returns_none(self, mock_llm, empty_schema):
        """No usable phrases after filtering yields None."""
        mock_llm.return_value = {
            "is_realistic": True,
            "questions": ["   ", ""],
            "question": "",
        }
        assert generate_question_from_sql("SELECT 1", empty_schema, []) is None

    @patch("aetherdialect._utils.llm_json", side_effect=RuntimeError("boom"))
    def test_runtime_error_propagates(self, _mock_llm, empty_schema):
        """LLM transport failures propagate to the caller."""
        with pytest.raises(RuntimeError, match="boom"):
            generate_question_from_sql("SELECT 1", empty_schema, [])

    @patch("aetherdialect._utils.llm_json")
    def test_schema_table_descriptions_in_prompt(self, mock_llm, schema_graph):
        """Known tables contribute column lines to the user prompt."""
        mock_llm.return_value = {"is_realistic": True, "questions": ["ok?"]}
        generate_question_from_sql("SELECT * FROM orders", schema_graph, ["orders"])
        user_blob = mock_llm.call_args[0][1]
        assert "TABLE orders:" in user_blob
        assert "orders" in user_blob


class TestWarmupQuestionStyles:
    """Deterministic warmup style sampling and synthetic prompt wiring."""

    def test_select_three_stable(self):
        """Same inputs yield the same ordered triple."""
        a = select_three_warmup_styles(3, "intent-a")
        b = select_three_warmup_styles(3, "intent-a")
        assert a == b
        assert len(set(a)) == 3

    def test_select_three_varies_with_inputs(self):
        """Different seeds or intent ids can change the triple."""
        one = select_three_warmup_styles(0, "same")
        two = select_three_warmup_styles(1, "same")
        assert one != two

    @patch("aetherdialect._utils.llm_json")
    def test_warmup_triple_payload_includes_style_slots(self, mock_llm):
        """Synthetic warmup mode passes schema enrichment and style_slots."""
        mock_llm.return_value = {
            "is_realistic": True,
            "questions": ["First?", "Second?", "Third?"],
            "question": "First?",
        }
        empty_schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        triple = select_three_warmup_styles(7, "wf-intent")
        out = generate_question_from_sql(
            "SELECT 1",
            empty_schema,
            [],
            warmup_style_triple=triple,
        )
        assert out is not None
        assert out["questions"] == ["First?", "Second?", "Third?"]
        user_blob = mock_llm.call_args[0][1]
        assert "style_slots" in user_blob
        for s in triple:
            assert s in user_blob


class TestGenerateBulkAnchors:
    """Deterministic rule-based warmup anchors."""

    def test_non_empty_tuple_bounded_by_count(self):
        """Anchors respect count and dedupe."""

        from aetherdialect._contracts_core import RuntimeIntent

        ri = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        out = generate_bulk_anchors(ri, sg, 5)
        assert len(out) <= 5
        assert len(set(out)) == len(out)


class TestIntentKeyLimitAndOrder:
    """Additional intent_key / structural invariants."""

    def test_limit_does_not_change_intent_key(self):
        """limit is excluded from the fingerprint."""
        base = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            limit=None,
        )
        limited = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            limit=10,
        )
        assert intent_key(base) == intent_key(limited)

    def test_order_by_order_independent(self):
        """order_by_cols are sorted by signature in the key."""
        o1 = OrderByCol(expr=NormalizedExpr.from_column("t.a"), direction="ASC")
        o2 = OrderByCol(expr=NormalizedExpr.from_column("t.b"), direction="DESC")
        i1 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[o1, o2],
            filters_param=[],
        )
        i2 = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[o2, o1],
            filters_param=[],
        )
        assert intent_key(i1) == intent_key(i2)


class TestSqlShapeDistinctEdge:
    """SELECT DISTINCT detection edge cases."""

    def test_distinct_not_in_select_position_ignored(self):
        """Substring 'distinct' outside SELECT DISTINCT does not set the flag."""
        sql = "SELECT name FROM t WHERE note LIKE '%distinct%'"
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert sql_shape(sql, intent, sqlglot_dialect="postgres").has_distinct is False


class TestExtractTablesWordBoundary:
    """Word-boundary behavior for extract_tables_from_sql."""

    def test_order_does_not_match_orders(self):
        """Shorter table name must not match as prefix of a longer token."""
        sql = "SELECT * FROM orders"
        assert extract_tables_from_sql(sql, ["order", "orders"], sqlglot_dialect="postgres") == ["orders"]


class TestExactQuestionMatchBudget:
    """Per-token distance budget edge cases."""

    def test_max_distance_zero_rejects_single_char_typo(self):
        """Total distance must be zero when max_distance is 0."""
        assert exact_question_match("hello world", "hallo world", max_distance=0) is False

    def test_multi_token_distance_sums(self):
        """Budget is summed across all token pairs."""
        assert exact_question_match("cat dog", "car dog", max_distance=2) is True
        assert exact_question_match("cat dog", "car dot", max_distance=1) is False


class TestValidateQuestionEdgeCases:
    """Extra branches for validate_question."""

    @patch("aetherdialect._utils.llm_json")
    def test_corrected_none_uses_original(self, mock_llm_json):
        """Explicit null corrected falls back to original question."""
        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "allowed",
            "corrected": None,
        }
        ok, kind, corrected = validate_question("Keep me")
        assert ok is True
        assert corrected == "Keep me"

    @patch("aetherdialect._utils.llm_json")
    def test_query_type_case_insensitive(self, mock_llm_json):
        """query_type is normalised with lower()."""
        mock_llm_json.return_value = {
            "valid_database_question": "yes",
            "query_type": "ALLOWED",
            "corrected": "x",
        }
        ok, kind, _ = validate_question("x")
        assert ok is True
        assert kind == "allowed"


class TestGenerateQuestionPhrasing:
    """generate_question prefix enforcement and errors."""

    @pytest.fixture
    def minimal_schema(self):
        t = TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(name="id", data_type="int", value_type="integer"),
            },
            foreign_keys=[],
            primary_key=[],
        )
        return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"orders": t})

    @patch("aetherdialect._utils._pick_question_style", return_value="How many")
    @patch("aetherdialect._utils.llm_json")
    def test_phrasing_violation_returns_none(self, mock_llm_json, _mock_style, minimal_schema):
        """Question must start with template prefix from required style."""
        mock_llm_json.return_value = {"question": "What is the count of orders?"}
        assert (
            generate_question(
                tables=["orders"],
                select_terms=["COUNT(orders.id)"],
                filter_descriptions=[],
                group_by_terms=[],
                having_descriptions=[],
                schema=minimal_schema,
            )
            is None
        )

    @patch("aetherdialect._utils._pick_question_style", return_value="Prefix{ignored}")
    @patch("aetherdialect._utils.llm_json")
    def test_required_start_splits_on_brace(self, mock_llm_json, _mock_style, minimal_schema):
        """Brace suffix in style is stripped for startswith check."""
        mock_llm_json.return_value = {"question": "Prefix rest of question"}
        out = generate_question(
            tables=["orders"],
            select_terms=["orders.id"],
            filter_descriptions=[],
            group_by_terms=[],
            having_descriptions=[],
            schema=minimal_schema,
        )
        assert out == "Prefix rest of question"

    @patch("aetherdialect._utils.llm_json", side_effect=ValueError("bad json"))
    @patch("aetherdialect._utils._pick_question_style", return_value="What are")
    def test_llm_exception_propagates(self, _mock_style, _mock_llm, minimal_schema):
        """Exceptions from llm_json propagate to the caller."""
        with pytest.raises(ValueError, match="bad json"):
            generate_question(
                tables=["orders"],
                select_terms=["orders.id"],
                filter_descriptions=[],
                group_by_terms=[],
                having_descriptions=[],
                schema=minimal_schema,
            )


class TestFlattenParamValuesContainsEdges:
    """contains filter key collection across CTE vs main and right_expr."""

    def test_contains_in_cte_normalizes_param(self):
        """CTE-level contains filters participate in key collection."""
        left = NormalizedExpr(add_groups=[MulGroup(multiply=["t1.tags"])])
        fp = FilterParam(left_expr=left, op="contains", param_key="tags", raw_value=None)
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["t1"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
            filters_param=[fp],
            param_values={"tags": '"x"'},
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        assert flatten_param_values(intent)["tags"] == "x"

    def test_contains_with_right_expr_skips_normalization(self):
        """Param keys tied to contains + right_expr are not normalized."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.col"),
            op="contains",
            param_key="p1",
            right_expr=NormalizedExpr.from_column("t.other"),
            raw_value=None,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            param_values={"p1": '"keep"'},
        )
        assert flatten_param_values(intent)["p1"] == '"keep"'


class TestNormalizeFiltersHavingMoreEdges:
    """Branches in _normalize_filters / _normalize_having_conditions."""

    def test_filter_param_defaults_none_op(self):
        """FilterParam with missing op becomes equals."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.c"), op="", value_type="string")
        out = _normalize_filters([fp])
        assert len(out) == 1
        assert out[0].op == "="

    def test_having_dict_invalid_op_defaults_equals(self):
        """Dict having op outside VALID_HAVING_OPS maps to equals."""
        d = {"aggregation": "COUNT(t.id)", "op": "~~", "value_type": "integer"}
        out = _normalize_having_conditions([d])
        assert len(out) == 1
        assert out[0].op == "="

    def test_having_skips_non_dict_non_model(self):
        """Garbage entries are ignored."""
        assert _normalize_having_conditions([None, 7, "x"]) == []


class TestNormalizeCteStepsFilterClamp:
    """Invalid filter/having ops clamped inside CTE normalisation."""

    def test_invalid_filter_op_clamped_to_equals(self):
        steps = [
            {
                "cte_name": "c1",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": [],
                "filters_param": [
                    FilterParam(
                        left_expr=NormalizedExpr.from_column("t1.x"),
                        op="not_an_op",
                        value_type="string",
                    )
                ],
            }
        ]
        out = _normalize_cte_steps(steps)
        assert out[0].filters_param[0].op == "="

    def test_invalid_having_op_clamped_to_equals(self):
        steps = [
            {
                "cte_name": "c1",
                "tables": ["t1"],
                "grain": "grouped",
                "select_cols": [],
                "group_by_cols": [],
                "output_columns": [],
                "having_param": [
                    HavingParam(
                        left_expr=NormalizedExpr.from_agg("count", "t1.id"),
                        op="bogus",
                        value_type="integer",
                    )
                ],
            }
        ]
        out = _normalize_cte_steps(steps)
        assert out[0].having_param[0].op == "="

    def test_skips_unknown_step_type(self):
        """Non-dict, non-RuntimeCteStep steps are skipped."""
        cte = RuntimeCteStep(
            cte_name="ok",
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            output_columns=[],
        )
        out = _normalize_cte_steps([object(), cte, 99])
        assert len(out) == 1
        assert out[0].cte_name == "ok"

    def test_column_map_resolves_prior_cte_alias(self):
        """Qualified refs to earlier CTE names populate column_map."""
        steps = [
            {
                "cte_name": "first",
                "tables": ["t1"],
                "grain": "row_level",
                "select_cols": ["t1.id"],
                "group_by_cols": [],
                "output_columns": ["id"],
            },
            {
                "cte_name": "second",
                "tables": ["t2"],
                "grain": "row_level",
                "select_cols": ["first.id"],
                "group_by_cols": [],
                "output_columns": ["id"],
            },
        ]
        out = _normalize_cte_steps(steps)
        assert out[1].column_map.get("id") == "first"


class TestNormalizeCteStepsForKeyStringCols:
    """String select/order columns in key projection."""

    def test_string_select_and_order_by_in_key(self):
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["t1"],
            grain="row_level",
            select_cols=["t1.a", "t1.b"],
            group_by_cols=[],
            output_columns=[],
            order_by_cols=["t1.a"],
        )
        key = _normalize_cte_steps_for_key([cte])[0]
        assert key["select_cols"] == ["t1.a", "t1.b"]
        assert key["order_by_cols"] == ["t1.a"]
