"""Unit tests for aetherdialect._qsim_ops module."""

from aetherdialect._contracts_schema import QSimIntent, QSimSkeleton
from aetherdialect._qsim_ops import (
    _compute_skeleton_complexity_tier,
    _compute_table_set_richness,
    _extract_agg_info,
    _extract_tables_from_expr,
    _has_aggregation,
    _is_no_variance_skeleton,
    _normalize_qsim_intent,
    _parse_llm_response,
    _validate_skeleton_constraints,
)


def _skeleton(**overrides) -> QSimSkeleton:
    """Build a minimal QSimSkeleton with optional overrides."""
    defaults = dict(
        tables=["orders"],
        has_aggregation=False,
        num_filters=0,
        num_groupby=0,
        has_orderby=False,
        num_having=0,
        has_distinct=False,
        has_expr_comparison=False,
    )
    defaults.update(overrides)
    return QSimSkeleton(**defaults)


def _qsim_intent(**overrides) -> QSimIntent:
    """Build a minimal QSimIntent with optional overrides."""
    defaults = dict(
        intent_id="qi_001",
        tables=["orders"],
        grain="row_level",
        select_cols=["orders.order_id"],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        param_values={},
        question="",
        variant_idx=0,
        limit=None,
        distinct=False,
    )
    defaults.update(overrides)
    return QSimIntent(**defaults)


class TestIsNoVarianceSkeleton:
    """Tests for is_no_variance_skeleton."""

    def test_no_filters_no_having(self):
        """Zero filters and no having is no-variance."""
        skel = _skeleton(num_filters=0, num_having=0)
        assert _is_no_variance_skeleton(skel) is True

    def test_has_filters(self):
        """Any filters means it has variance."""
        skel = _skeleton(num_filters=2, num_having=0)
        assert _is_no_variance_skeleton(skel) is False

    def test_has_having(self):
        """Having alone means it has variance."""
        skel = _skeleton(num_filters=0, num_having=1)
        assert _is_no_variance_skeleton(skel) is False

    def test_both_filters_and_having(self):
        """Both filters and having means it has variance."""
        skel = _skeleton(num_filters=1, num_having=1)
        assert _is_no_variance_skeleton(skel) is False


class TestComputeSkeletonComplexityTier:
    """Tests for compute_skeleton_complexity_tier."""

    def test_tier_c_minimal(self):
        """Bare skeleton scores < 4 -> tier C."""
        skel = _skeleton()
        assert _compute_skeleton_complexity_tier(skel) == "C"

    def test_tier_c_just_orderby(self):
        """Only orderby gives score=1 -> tier C."""
        skel = _skeleton(has_orderby=True)
        assert _compute_skeleton_complexity_tier(skel) == "C"

    def test_tier_b_aggregation(self):
        """Aggregation alone gives score=3, not enough for B. Add 1 filter: 3+2=5 -> B."""
        skel = _skeleton(has_aggregation=True, num_filters=1)
        assert _compute_skeleton_complexity_tier(skel) == "B"

    def test_tier_b_boundary(self):
        """Score=4 -> tier B (distinct=2 + filter×1=2 -> 4)."""
        skel = _skeleton(has_distinct=True, num_filters=1)
        assert _compute_skeleton_complexity_tier(skel) == "B"

    def test_tier_a_high_complexity(self):
        """Aggregation + having + 2 filters + groupby -> A."""
        skel = _skeleton(
            has_aggregation=True,
            num_having=1,
            num_filters=2,
            num_groupby=1,
        )
        tier = _compute_skeleton_complexity_tier(skel)
        assert tier == "A"

    def test_tier_a_score_boundary(self):
        """Score = 8 exactly -> tier A. 2 filters(4) + aggregation(3) + orderby(1) = 8."""
        skel = _skeleton(has_aggregation=True, num_filters=2, has_orderby=True)
        assert _compute_skeleton_complexity_tier(skel) == "A"

    def test_all_features(self):
        """All features yield very high score -> tier A."""
        skel = _skeleton(
            has_aggregation=True,
            num_filters=3,
            num_groupby=2,
            num_having=1,
            has_orderby=True,
            has_distinct=True,
            has_expr_comparison=True,
        )
        assert _compute_skeleton_complexity_tier(skel) == "A"


class TestNormalizeQsimIntent:
    """Tests for normalize_qsim_intent."""

    def test_grain_correction_grouped_without_groupby(self, schema_graph):
        """Grouped grain with no group_by_cols corrects to row_level."""
        intent = _qsim_intent(grain="grouped", group_by_cols=[])
        result = _normalize_qsim_intent(intent, schema_graph)
        assert result.grain == "row_level"

    def test_grain_correction_agg_with_groupby(self, schema_graph):
        """Aggregation with group_by corrects grain to grouped."""
        intent = _qsim_intent(
            tables=["orders"],
            grain="row_level",
            select_cols=["COUNT(orders.order_id)", "orders.status"],
            group_by_cols=["orders.status"],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert result.grain == "grouped"

    def test_grain_correction_agg_without_groupby(self, schema_graph):
        """Aggregation without group_by corrects grain to scalar."""
        intent = _qsim_intent(
            tables=["orders"],
            grain="row_level",
            select_cols=["COUNT(orders.order_id)"],
            group_by_cols=[],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert result.grain == "scalar"

    def test_select_dedup(self, schema_graph):
        """Duplicate select_cols are deduplicated."""
        intent = _qsim_intent(
            select_cols=["orders.order_id", "orders.order_id", "orders.amount"],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert len(result.select_cols) == len(set(result.select_cols))

    def test_select_sorted(self, schema_graph):
        """Select cols are sorted alphabetically."""
        intent = _qsim_intent(
            select_cols=["orders.status", "orders.amount", "orders.order_id"],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert result.select_cols == sorted(result.select_cols)

    def test_intent_id_recomputed(self, schema_graph):
        """Intent ID is recomputed after normalization."""
        intent = _qsim_intent(intent_id="original_id")
        result = _normalize_qsim_intent(intent, schema_graph)
        assert result.intent_id != "original_id"

    def test_unused_table_pruned(self, schema_graph):
        """Tables not referenced in any clause may be pruned if remaining set is connected."""
        intent = _qsim_intent(
            tables=["orders", "customers"],
            select_cols=["orders.order_id"],
            group_by_cols=[],
            filters_param=[],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert "customers" not in result.tables

    def test_groupby_prefixed(self, schema_graph):
        """Bare group_by cols get table prefix."""
        intent = _qsim_intent(
            tables=["orders"],
            select_cols=["COUNT(orders.order_id)", "orders.status"],
            group_by_cols=["status"],
        )
        result = _normalize_qsim_intent(intent, schema_graph)
        assert all("." in g for g in result.group_by_cols)


class TestHasAggregation:
    """Tests for _has_aggregation."""

    def test_with_agg(self):
        """Detects aggregation pattern."""
        assert _has_aggregation(["COUNT(orders.order_id)"]) is True

    def test_without_agg(self):
        """Plain column has no aggregation."""
        assert _has_aggregation(["orders.order_id"]) is False

    def test_mixed(self):
        """Mixed list with one agg returns True."""
        assert _has_aggregation(["orders.status", "SUM(orders.amount)"]) is True

    def test_empty(self):
        """Empty list returns False."""
        assert _has_aggregation([]) is False


class TestExtractAggInfo:
    """Tests for _extract_agg_info."""

    def test_count(self):
        """Extracts count function."""
        result = _extract_agg_info("COUNT(orders.order_id)")
        assert result == ("count", "orders.order_id")

    def test_sum(self):
        """Extracts sum function."""
        result = _extract_agg_info("SUM(orders.amount)")
        assert result == ("sum", "orders.amount")

    def test_plain_column(self):
        """Plain column returns None."""
        assert _extract_agg_info("orders.status") is None

    def test_star(self):
        """COUNT(*) extracts correctly."""
        result = _extract_agg_info("COUNT(*)")
        assert result == ("count", "*")


class TestExtractTablesFromExpr:
    """Tests for _extract_tables_from_expr."""

    def test_single_table(self):
        """Single table.column reference."""
        assert _extract_tables_from_expr("orders.order_id") == {"orders"}

    def test_multiple_tables(self):
        """Multiple table references."""
        assert _extract_tables_from_expr("orders.amount + customers.balance") == {
            "orders",
            "customers",
        }

    def test_no_table(self):
        """Expression without table prefix."""
        assert _extract_tables_from_expr("42") == set()

    def test_empty_string(self):
        """Empty string returns empty set."""
        assert _extract_tables_from_expr("") == set()

    def test_aggregated(self):
        """Aggregated expression extracts inner table."""
        assert _extract_tables_from_expr("SUM(orders.amount)") == {"orders"}


class TestValidateSkeletonConstraints:
    """Tests for _validate_skeleton_constraints."""

    def test_valid_response(self):
        """Matching response passes validation."""
        skel = _skeleton(has_aggregation=True, num_filters=1, num_groupby=1)
        response = {
            "select_cols": ["COUNT(orders.order_id)", "orders.status"],
            "filters": [{"column": "orders.status", "op": "="}],
            "groupby_cols": ["orders.status"],
            "having": [],
            "distinct": False,
        }
        valid, violations = _validate_skeleton_constraints(response, skel)
        assert valid is True
        assert violations == []

    def test_missing_aggregation(self):
        """Missing required aggregation fails."""
        skel = _skeleton(has_aggregation=True)
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "having": [],
            "distinct": False,
        }
        valid, violations = _validate_skeleton_constraints(response, skel)
        assert valid is False
        assert any("aggregation" in v for v in violations)

    def test_unwanted_having(self):
        """Having present when forbidden fails."""
        skel = _skeleton(num_having=0)
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "having": [{"agg": "count", "op": ">"}],
            "distinct": False,
        }
        valid, violations = _validate_skeleton_constraints(response, skel)
        assert valid is False

    def test_distinct_mismatch(self):
        """Distinct mismatch fails."""
        skel = _skeleton(has_distinct=True)
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "having": [],
            "distinct": False,
        }
        valid, violations = _validate_skeleton_constraints(response, skel)
        assert valid is False
        assert any("distinct" in v for v in violations)


class TestParseLlmResponse:
    """Tests for parse_llm_response."""

    def test_valid_simple_select(self, schema_graph):
        """Valid simple select produces QSimIntent."""
        skel = _skeleton(tables=["orders"])
        column_roles = {
            f"{t}.{c}": cm.role for t in schema_graph.tables for c, cm in schema_graph.tables[t].columns.items()
        }
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "orderby_cols": [],
            "having": [],
        }
        result = _parse_llm_response(response, skel, schema_graph, column_roles)
        assert result is not None
        assert hasattr(result, "select_cols")

    def test_empty_select_rejected(self, schema_graph):
        """Empty select_cols returns None."""
        skel = _skeleton(tables=["orders"])
        column_roles = {}
        response = {
            "select_cols": [],
            "filters": [],
            "groupby_cols": [],
            "orderby_cols": [],
            "having": [],
        }
        result = _parse_llm_response(response, skel, schema_graph, column_roles)
        assert result is None

    def test_missing_agg_when_required(self, schema_graph):
        """Skeleton requiring aggregation rejects non-aggregated response."""
        skel = _skeleton(tables=["orders"], has_aggregation=True)
        column_roles = {}
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "orderby_cols": [],
            "having": [],
        }
        result = _parse_llm_response(response, skel, schema_graph, column_roles)
        assert result is None

    def test_missing_filters_when_required(self, schema_graph):
        """Skeleton requiring filters rejects response with none."""
        skel = _skeleton(tables=["orders"], num_filters=1)
        column_roles = {}
        response = {
            "select_cols": ["orders.order_id"],
            "filters": [],
            "groupby_cols": [],
            "orderby_cols": [],
            "having": [],
        }
        result = _parse_llm_response(response, skel, schema_graph, column_roles)
        assert result is None

    def test_missing_groupby_when_required(self, schema_graph):
        """Skeleton requiring groupby rejects response with none."""
        skel = _skeleton(tables=["orders"], num_groupby=1, has_aggregation=True)
        column_roles = {}
        response = {
            "select_cols": ["COUNT(orders.order_id)"],
            "filters": [],
            "groupby_cols": [],
            "orderby_cols": [],
            "having": [],
        }
        result = _parse_llm_response(response, skel, schema_graph, column_roles)
        assert result is None


class TestComputeTableSetRichness:
    """Tests for compute_table_set_richness."""

    def test_single_table(self, schema_graph):
        """Single table returns positive score."""
        column_roles = {
            f"{t}.{c}": cm.role for t in schema_graph.tables for c, cm in schema_graph.tables[t].columns.items()
        }
        score = _compute_table_set_richness(["orders"], schema_graph, column_roles)
        assert score > 0

    def test_multi_table_higher(self, schema_graph):
        """Multiple tables produce higher richness than single table."""
        column_roles = {
            f"{t}.{c}": cm.role for t in schema_graph.tables for c, cm in schema_graph.tables[t].columns.items()
        }
        single = _compute_table_set_richness(["orders"], schema_graph, column_roles)
        multi = _compute_table_set_richness(["orders", "customers"], schema_graph, column_roles)
        assert multi > single

    def test_empty_tables(self, schema_graph):
        """Empty table list returns zero."""
        score = _compute_table_set_richness([], schema_graph, {})
        assert score == 0
