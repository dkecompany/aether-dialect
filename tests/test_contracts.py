"""Tests for contracts_base and contracts_core dataclass contracts."""

import json
from dataclasses import FrozenInstanceError

import pytest

from aetherdialect._constants import ROLE_ALLOWED_AGGREGATIONS, VALID_AGGREGATION_FUNCTIONS
from aetherdialect._contracts_base import (
    ComplexityTier,
    ConfigError,
    EngineContext,
    ExprValue,
    FailureCategory,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    SensitivityClassification,
    WhereParam,
)
from aetherdialect._contracts_core import (
    ConcreteCteStep,
    ConcreteIntent,
    FeedbackCounts,
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeCteStep,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    SelectCol,
    Template,
    TemplateMatch,
    ValueHistory,
)
from aetherdialect._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    ColumnRole,
    CteOutputColumnMeta,
    ExpansionMetadata,
    FKEdge,
    IntentIssue,
    IntentValidationResult,
    QSimHaving,
    QSimIntent,
    QSimSkeleton,
    QSimSummary,
    QSimWhereParam,
    RetryFailureContext,
    SchemaGraph,
    SchemaLimits,
    SeedWarmupSummary,
    SkeletonPool,
    SQLShape,
    TableMetadata,
    TemplateStats,
    ValueDomain,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._utils import (
    data_type_to_value_type,
    is_date_type,
    is_numeric_type,
    is_string_type,
)


class TestExprValue:
    """Tests for ExprValue dataclass."""

    def test_from_dict_with_int(self):
        """ExprValue.from_dict handles bare int."""
        ev = ExprValue.from_dict(5)
        assert ev.value == 5.0
        assert ev.param_key == ""

    def test_from_dict_with_float(self):
        """ExprValue.from_dict handles bare float."""
        ev = ExprValue.from_dict(3.14)
        assert ev.value == 3.14
        assert ev.param_key == ""

    def test_from_dict_with_dict(self):
        """ExprValue.from_dict handles dict with value and param_key."""
        ev = ExprValue.from_dict({"value": 10.0, "param_key": "p1"})
        assert ev.value == 10.0
        assert ev.param_key == "p1"

    def test_round_trip(self):
        """ExprValue to_dict/from_dict round trip."""
        original = ExprValue(value=42.0, param_key="s1")
        rebuilt = ExprValue.from_dict(original.to_dict())
        assert rebuilt.value == original.value
        assert rebuilt.param_key == original.param_key

    def test_signature_key_is_val(self):
        """ExprValue.signature_key always returns 'val'."""
        assert ExprValue(value=1.0).signature_key == "val"
        assert ExprValue(value=99.0, param_key="p1").signature_key == "val"


class TestMulGroup:
    """Tests for MulGroup dataclass."""

    def test_post_init_sorts_multiply(self):
        """MulGroup.__post_init__ sorts multiply list."""
        g = MulGroup(multiply=["b.col", "a.col"])
        assert [m.column_ref for m in g.multiply] == ["a.col", "b.col"]

    def test_post_init_sorts_divide(self):
        """MulGroup.__post_init__ sorts divide list."""
        g = MulGroup(divide=["z.col", "a.col"])
        assert [d.column_ref for d in g.divide] == ["a.col", "z.col"]

    def test_post_init_lowercases_agg_func(self):
        """MulGroup.__post_init__ lowercases agg_func."""
        g = MulGroup(agg_func="SUM")
        assert g.agg_func == "sum"

    def test_post_init_swaps_scalar_inner_if_needed(self):
        """MulGroup.__post_init__ swaps scalar/inner_scalar alphabetically."""
        g = MulGroup(scalar_func="ROUND", inner_scalar_func="ABS")
        assert g.scalar_func == "abs"
        assert g.inner_scalar_func == "round"

    def test_post_init_no_swap_when_scalar_less_than_inner(self):
        """MulGroup.__post_init__ does not swap when scalar < inner alphabetically."""
        g = MulGroup(
            scalar_func="abs",
            inner_scalar_func="round",
            scalar_func_args=[],
            inner_scalar_func_args=[2],
        )
        assert g.scalar_func == "abs"
        assert g.inner_scalar_func == "round"
        assert g.inner_scalar_func_args == [2]

    def test_signature_key_includes_coeff(self):
        """MulGroup.signature_key starts with 'coeff'."""
        g = MulGroup(multiply=["t.col"], agg_func="sum")
        assert g.signature_key.startswith("coeff")
        assert "agg=sum" in g.signature_key
        assert "t.col" in g.signature_key

    def test_structural_key_excludes_coeff(self):
        """MulGroup.structural_key excludes coefficient prefix."""
        g = MulGroup(multiply=["t.col"], agg_func="sum")
        assert not g.structural_key.startswith("coeff")
        assert "agg=sum" in g.structural_key

    def test_signature_key_vs_structural_key(self):
        """Two MulGroups with different coefficients share structural_key but differ in signature_key."""
        g1 = MulGroup(coefficient=1.0, multiply=["t.col"])
        g2 = MulGroup(coefficient=2.0, multiply=["t.col"])
        assert g1.structural_key == g2.structural_key
        assert g1.signature_key == g2.signature_key

    def test_round_trip(self):
        """MulGroup to_dict/from_dict round trip."""
        original = MulGroup(coefficient=2.0, multiply=["a.x"], divide=["b.y"], agg_func="avg")
        rebuilt = MulGroup.from_dict(original.to_dict())
        assert rebuilt.coefficient == 2.0
        assert [m.column_ref for m in rebuilt.multiply] == ["a.x"]
        assert [d.column_ref for d in rebuilt.divide] == ["b.y"]
        assert rebuilt.agg_func == "avg"


class TestNormalizedExpr:
    """Tests for NormalizedExpr dataclass."""

    def test_from_column(self):
        """NormalizedExpr.from_column creates a leaf column reference."""
        expr = NormalizedExpr.from_column("orders.amount")
        assert expr.column_ref == "orders.amount"
        assert not expr.has_aggregation

    def test_from_agg(self):
        """NormalizedExpr.from_agg creates MulGroup with agg_func."""
        expr = NormalizedExpr.from_agg("count", "orders.order_id")
        assert len(expr.add_groups) == 1
        assert expr.add_groups[0].agg_func == "count"
        assert expr.add_groups[0].multiply[0].column_ref == "orders.order_id"
        assert expr.has_aggregation is True

    def test_has_aggregation_detects_agg_prefix(self):
        """NormalizedExpr.has_aggregation detects AGG() string prefix."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(orders.order_id)"])])
        assert expr.has_aggregation is True

    def test_has_aggregation_false_for_bare_column(self):
        """NormalizedExpr.has_aggregation returns False for bare column."""
        expr = NormalizedExpr.from_column("t.col")
        assert expr.has_aggregation is False

    def test_has_aggregation_raw_sql_set_function_or_window(self):
        assert NormalizedExpr(raw_sql="AVG(t.x) OVER ()").has_aggregation is True
        assert NormalizedExpr(raw_sql="SUM(a.b)").has_aggregation is True
        assert NormalizedExpr(raw_sql="LOWER(x.y)").has_aggregation is False

    def test_primary_column_strips_function(self):
        """NormalizedExpr.primary_column strips function wrappers."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["UPPER(customers.name)"])])
        assert expr.primary_column == "customers.name"

    def test_primary_column_unbalanced_paren_stops_strip(self):
        """Malformed calls fall back to raw_sql; primary_column has no embedded column."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=[NormalizedExpr(raw_sql="FOO(t.col")])])
        assert expr.primary_column == ""

    def test_primary_term_returns_raw(self):
        """NormalizedExpr.primary_term renders the first multiply leaf as SQL."""
        expr = NormalizedExpr(add_groups=[MulGroup(multiply=["UPPER(customers.name)"])])
        assert expr.primary_term == "UPPER(customers.name)"

    def test_round_trip(self):
        """NormalizedExpr to_dict/from_dict round trip."""
        original = NormalizedExpr(
            add_groups=[MulGroup(multiply=["t.x"], agg_func="sum")],
            sub_values=[ExprValue(value=1.0)],
        )
        rebuilt = NormalizedExpr.from_dict(original.to_dict())
        assert len(rebuilt.add_groups) == 1
        assert rebuilt.add_groups[0].agg_func == "sum"
        assert len(rebuilt.sub_values) == 1

    def test_signature_key_deterministic(self):
        """NormalizedExpr.signature_key is deterministic."""
        expr = NormalizedExpr.from_agg("sum", "t.col")
        assert expr.signature_key == expr.signature_key

    def test_post_init_sorts_groups(self):
        """NormalizedExpr.__post_init__ sorts add_groups by signature_key."""
        g1 = MulGroup(multiply=["z.col"])
        g2 = MulGroup(multiply=["a.col"])
        expr = NormalizedExpr(add_groups=[g1, g2])
        assert expr.add_groups[0].multiply[0].column_ref == "a.col"
        assert expr.add_groups[1].multiply[0].column_ref == "z.col"

    def test_has_column_reference_column(self):
        """NormalizedExpr.has_column_reference is True when add_groups has a column term."""
        expr = NormalizedExpr.from_column("t.c")
        assert expr.has_column_reference is True
        assert expr.is_literal_only is False

    def test_has_column_reference_agg(self):
        """NormalizedExpr.has_column_reference is True when agg_func is set at expr level."""
        expr = NormalizedExpr(agg_func="count")
        assert expr.has_column_reference is True
        assert expr.is_literal_only is False

    def test_has_column_reference_scalar(self):
        """NormalizedExpr.has_column_reference is True when scalar_func is set at expr level."""
        expr = NormalizedExpr(scalar_func="upper")
        assert expr.has_column_reference is True

    def test_has_column_reference_registry(self):
        """NormalizedExpr.has_column_reference is True when the column_ref is a registry id."""
        expr = NormalizedExpr(column_ref="w01")
        assert expr.has_column_reference is True

    def test_is_literal_only_empty(self):
        """NormalizedExpr with nothing set is literal-only (degenerate zero expression)."""
        expr = NormalizedExpr()
        assert expr.is_literal_only is True
        assert expr.has_column_reference is False

    def test_is_literal_only_with_values(self):
        """NormalizedExpr with just add_values is literal-only."""
        expr = NormalizedExpr(add_values=[ExprValue(value=5.0)])
        assert expr.is_literal_only is True


class TestWhereParam:
    """Tests for WhereParam dataclass."""

    def test_round_trip(self):
        """WhereParam to_dict/from_dict round trip."""
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="=",
            value_type="string",
            param_key="p1",
        )
        rebuilt = WhereParam.from_dict(fp.to_dict())
        assert rebuilt.op == "="
        assert rebuilt.value_type == "string"
        assert rebuilt.param_key == "p1"

    def test_post_init_normalizes_op(self):
        """WhereParam.__post_init__ lowercases and strips op."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="  LIKE  ")
        assert fp.op == "like"

    def test_post_init_both_columns_preserves_order(self):
        """WhereParam.__post_init__ leaves ordering untouched when both sides are column-bearing."""
        left = NormalizedExpr.from_column("z.col")
        right = NormalizedExpr.from_column("a.col")
        fp = WhereParam(left_expr=left, op=">", right_expr=right)
        assert fp.left_expr.primary_term == "z.col"
        assert fp.right_expr.primary_term == "a.col"
        assert fp.op == ">"

    def test_post_init_swaps_literal_left_column_right(self):
        """WhereParam.__post_init__ swaps literal-only left with column- bearing right and flips op."""
        left = NormalizedExpr(add_values=[ExprValue(value=5.0)])
        right = NormalizedExpr.from_column("orders.amount")
        fp = WhereParam(left_expr=left, op=">", right_expr=right)
        assert fp.left_expr.primary_term == "orders.amount"
        assert fp.op == "<"

    def test_post_init_both_literals_preserves_order(self):
        """WhereParam.__post_init__ leaves ordering untouched when both sides are literal-only."""
        left = NormalizedExpr(add_values=[ExprValue(value=1.0)])
        right = NormalizedExpr(add_values=[ExprValue(value=2.0)])
        fp = WhereParam(left_expr=left, op=">", right_expr=right)
        assert fp.op == ">"

    def test_from_dict_ignores_bool_op(self):
        """WhereParam.from_dict does not store bool_op on the leaf."""
        d = {"left_expr": "t.c", "op": "=", "bool_op": "OR"}
        fp = WhereParam.from_dict(d)
        assert not hasattr(fp, "bool_op")
        assert "bool_op" not in fp.to_dict()

    def test_from_dict_ignores_where_group(self):
        """WhereParam.from_dict does not store where_group on the leaf."""
        d = {"left_expr": "t.c", "op": "=", "where_group": 3}
        fp = WhereParam.from_dict(d)
        assert not hasattr(fp, "where_group")
        assert "where_group" not in fp.to_dict()

    def test_to_dict_omits_bool_op_and_where_group(self):
        """WhereParam.to_dict never emits bool_op or where_group."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        d = fp.to_dict()
        assert "bool_op" not in d
        assert "where_group" not in d

    def test_flat_where_list_hard_fails(self):
        """Flat where lists and where_param keys are rejected on deserialize."""
        with pytest.raises(ConfigError, match="flat list|where_param"):
            PredicateGroup.parse_where_field({"where": [{"left_expr": "t.a", "op": "="}]})
        with pytest.raises(ConfigError, match="where_param"):
            PredicateGroup.parse_where_field({"where_param": [{"left_expr": "t.a", "op": "="}]})


class TestHavingParam:
    """Tests for HavingParam dataclass."""

    def test_round_trip(self):
        """HavingParam to_dict/from_dict round trip."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.order_id"),
            op=">",
            value_type="integer",
            param_key="p2",
        )
        rebuilt = HavingParam.from_dict(hp.to_dict())
        assert rebuilt.op == ">"
        assert rebuilt.value_type == "integer"

    def test_default_value_type_is_number(self):
        """HavingParam defaults value_type to 'number'."""
        hp = HavingParam(left_expr=NormalizedExpr.from_agg("sum", "t.x"), op=">=")
        assert hp.value_type == "number"

    def test_from_dict_ignores_bool_op(self):
        """HavingParam.from_dict does not store bool_op on the leaf."""
        d = {"left_expr": "COUNT(t.id)", "op": ">", "bool_op": "OR"}
        hp = HavingParam.from_dict(d)
        assert not hasattr(hp, "bool_op")
        assert "bool_op" not in hp.to_dict()

    def test_from_dict_ignores_where_group(self):
        """HavingParam.from_dict does not store where_group on the leaf."""
        d = {"left_expr": "COUNT(t.id)", "op": ">", "where_group": 2}
        hp = HavingParam.from_dict(d)
        assert not hasattr(hp, "where_group")
        assert "where_group" not in hp.to_dict()

    def test_to_dict_omits_bool_op_and_where_group(self):
        """HavingParam.to_dict never emits bool_op or where_group."""
        hp = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.id"), op=">")
        d = hp.to_dict()
        assert "bool_op" not in d
        assert "where_group" not in d

    def test_flat_having_list_hard_fails(self):
        """Flat having lists and having_param keys are rejected on deserialize."""
        with pytest.raises(ConfigError, match="flat list|having_param"):
            PredicateGroup.parse_having_field({"having": [{"left_expr": "t.a", "op": "="}]})
        with pytest.raises(ConfigError, match="having_param"):
            PredicateGroup.parse_having_field({"having_param": [{"left_expr": "t.a", "op": "="}]})

    def test_post_init_swaps_literal_left_agg_right(self):
        """HavingParam.__post_init__ swaps literal-only left with aggregate right and flips op."""
        left = NormalizedExpr(add_values=[ExprValue(value=10.0)])
        right = NormalizedExpr.from_agg("count", "orders.order_id")
        hp = HavingParam(left_expr=left, op=">", right_expr=right)
        assert hp.left_expr.has_aggregation is True
        assert hp.op == "<"

    def test_post_init_both_aggregates_preserves_order(self):
        """HavingParam.__post_init__ leaves order intact when both sides are column-bearing."""
        left = NormalizedExpr.from_agg("count", "t.a")
        right = NormalizedExpr.from_agg("count", "t.b")
        hp = HavingParam(left_expr=left, op=">", right_expr=right)
        assert hp.op == ">"


class TestSelectCol:
    """Tests for SelectCol dataclass."""

    def test_is_aggregated(self):
        """SelectCol.is_aggregated delegates to expr.has_aggregation."""
        sc = SelectCol(expr=NormalizedExpr.from_agg("sum", "t.x"))
        assert sc.is_aggregated is True

    def test_not_aggregated(self):
        """SelectCol bare column is not aggregated."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.x"))
        assert sc.is_aggregated is False

    def test_round_trip(self):
        """SelectCol to_dict/from_dict round trip."""
        original = SelectCol(expr=NormalizedExpr.from_column("t.x"))
        rebuilt = SelectCol.from_dict(original.to_dict())
        assert rebuilt.expr.primary_term == "t.x"


class TestOrderByCol:
    """Tests for OrderByCol dataclass."""

    def test_direction_uppercased(self):
        """OrderByCol.__post_init__ uppercases direction."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("t.x"), direction="desc")
        assert obc.direction == "DESC"

    def test_round_trip(self):
        """OrderByCol to_dict/from_dict round trip."""
        original = OrderByCol(expr=NormalizedExpr.from_column("t.x"), direction="ASC")
        rebuilt = OrderByCol.from_dict(original.to_dict())
        assert rebuilt.direction == "ASC"


class TestCteRoundTrip:
    """Tests for CTE conversion functions."""

    def test_runtime_cte_to_concrete_preserves_fields(self):
        """runtime_cte_to_concrete preserves structural fields."""
        rt = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
            grain="grouped",
        )
        concrete = rt.to_concrete()
        assert isinstance(concrete, ConcreteCteStep)
        assert concrete.cte_name == "cte1"
        assert concrete.tables == ["orders"]
        assert concrete.grain == "grouped"

    def test_concrete_cte_to_runtime_drops_runtime_fields(self):
        """concrete_cte_to_runtime creates RuntimeCteStep with empty runtime fields."""
        concrete = ConcreteCteStep(cte_name="cte1", tables=["t1"])
        rt = concrete.to_runtime()
        assert isinstance(rt, RuntimeCteStep)
        assert rt.description == ""
        assert rt.param_values == {}

    def test_round_trip(self):
        """RuntimeCteStep -> ConcreteCteStep -> RuntimeCteStep preserves structure."""
        original = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id"))],
            grain="grouped",
            output_columns=["order_count"],
        )
        concrete = original.to_concrete()
        restored = concrete.to_runtime()
        assert restored.cte_name == original.cte_name
        assert restored.tables == original.tables
        assert restored.grain == original.grain
        assert len(restored.select_cols) == 1


class TestRuntimeIntentToConcreteIntent:
    """Tests for runtime_intent_to_concrete."""

    def test_basic_conversion(self, minimal_intent):
        """runtime_intent_to_concrete produces ConcreteIntent."""
        concrete = minimal_intent.to_concrete("id_123")
        assert isinstance(concrete, ConcreteIntent)
        assert concrete.intent_id == "id_123"
        assert concrete.tables == ["orders"]
        assert concrete.grain == "row_level"

    def test_drops_runtime_fields(self, grouped_intent):
        """ConcreteIntent omits natural_language; runtime param bag is stripped to empty."""
        concrete = grouped_intent.to_concrete("id_456")
        d = concrete.to_dict()
        assert d.get("param_values") == {}
        assert "natural_language" not in d

    def test_round_trip_via_dict(self, grouped_intent):
        """ConcreteIntent to_dict/from_dict round trip."""
        concrete = grouped_intent.to_concrete("id_789")
        rebuilt = ConcreteIntent.from_dict(concrete.to_dict())
        assert rebuilt.intent_id == "id_789"
        assert rebuilt.tables == concrete.tables


class TestConcreteIntentToRuntimeSkeleton:
    """Tests for concrete_intent_to_runtime_skeleton."""

    def test_clears_values(self, minimal_intent: RuntimeIntent) -> None:
        """Skeleton mirrors concrete shape with empty param_values."""
        concrete = minimal_intent.to_concrete("id_sk")
        skel = concrete.to_runtime_skeleton()
        assert skel.param_values == {}
        assert skel.tables == concrete.tables
        assert skel.grain == concrete.grain


class TestRuntimeIntent:
    """Tests for RuntimeIntent dataclass."""

    def test_expected_rows_scalar(self):
        """RuntimeIntent.expected_rows returns 'one' for scalar grain."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert intent.expected_rows == "one"

    def test_expected_rows_with_limit(self):
        """RuntimeIntent.expected_rows returns 'few' when limit is set."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            limit=10,
        )
        assert intent.expected_rows == "few"

    def test_expected_rows_many(self):
        """RuntimeIntent.expected_rows returns 'many' with no limit and non-scalar grain."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert intent.expected_rows == "many"

    def test_has_aggregation(self):
        """RuntimeIntent.has_aggregation detects aggregated select cols."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert intent.has_aggregation is True

    def test_round_trip(self, grouped_intent):
        """RuntimeIntent to_dict/from_dict round trip."""
        d = grouped_intent.to_dict()
        rebuilt = RuntimeIntent.from_dict(d)
        assert rebuilt.tables == grouped_intent.tables
        assert rebuilt.grain == grouped_intent.grain
        assert len(rebuilt.select_cols) == len(grouped_intent.select_cols)


class TestValueHistory:
    """Tests for ValueHistory dataclass."""

    def test_add_entry(self):
        """ValueHistory.add appends to all parallel lists."""
        vh = ValueHistory(param_values=[], questions=[], natural_language=[])
        vh.add({"p1": "x"}, "question one", "NL one")
        assert len(vh) == 1
        assert vh.questions[0] == "question one"

    def test_round_trip(self):
        """ValueHistory to_dict/from_dict round trip."""
        vh = ValueHistory(param_values=[{"p1": "a"}], questions=["q1"], natural_language=["nl1"])
        rebuilt = ValueHistory.from_dict(vh.to_dict())
        assert len(rebuilt) == 1


class TestFeedbackCounts:
    """Tests for FeedbackCounts dataclass."""

    def test_defaults(self):
        fc = FeedbackCounts()
        assert fc.accepts == 0
        assert fc.rejects == 0
        assert fc.last_path == 0

    def test_round_trip(self):
        fc = FeedbackCounts(accepts=2, rejects=1, last_path=4)
        rebuilt = FeedbackCounts.from_dict(fc.to_dict())
        assert rebuilt.accepts == 2
        assert rebuilt.rejects == 1
        assert rebuilt.last_path == 4


class TestQuestionFeedbackEntry:
    """Tests for QuestionFeedbackEntry."""

    def test_round_trip(self):
        """to_dict / from_dict preserves fields."""
        e = QuestionFeedbackEntry(
            summary="bad join",
            buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
            kind=FeedbackKind.VALIDATION_FAILURE,
            effective_structural_hash="abc",
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            is_post_restart=True,
        )
        assert QuestionFeedbackEntry.from_dict(e.to_dict()) == e

    def test_to_prompt_row_shape(self):
        """to_prompt_row exposes string values for prompts."""
        e = QuestionFeedbackEntry(
            summary="s",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="h",
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        row = e.to_prompt_row()
        assert row["kind"] == FeedbackKind.INTENT_REJECTED.value
        assert row["buckets"] == RejectionBucket.OTHER.value
        assert row["is_post_restart"] == "False"

    def test_frozen(self):
        """QuestionFeedbackEntry is immutable."""
        e = QuestionFeedbackEntry(
            summary="s",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="h",
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        with pytest.raises(FrozenInstanceError):
            e.summary = "x"


class TestFeedbackKind:
    """Tests for FeedbackKind enum."""

    def test_members(self):
        """Known discriminators are stable strings."""
        assert FeedbackKind.VALIDATION_FAILURE.value == "validation_failure"
        assert FeedbackKind.INTENT_REJECTED.value == "intent_rejected"


class TestQSimIntent:
    """Tests for QSimIntent dataclass."""

    def test_distinct_field_exists(self):
        """QSimIntent has distinct field."""
        qi = QSimIntent(
            intent_id="test",
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=[],
            having_param=[],
            distinct=True,
        )
        assert qi.distinct is True

    def test_round_trip(self):
        """QSimIntent to_dict/from_dict round trip preserves distinct."""
        qi = QSimIntent(
            intent_id="test",
            tables=["t"],
            grain="row_level",
            select_cols=["t.x"],
            group_by_cols=[],
            order_by_cols=[],
            where=[],
            having_param=[],
            distinct=True,
        )
        rebuilt = QSimIntent.from_dict(qi.to_dict())
        assert rebuilt.distinct is True
        assert rebuilt.select_cols == ["t.x"]


class TestColumnMetadaTypeFunctions:
    """Tests for type detection functions."""

    def test_is_numeric_type(self):
        """is_numeric_type detects numeric types."""
        assert is_numeric_type("integer") is True
        assert is_numeric_type("BIGINT") is True
        assert is_numeric_type("numeric(10,2)") is True
        assert is_numeric_type("varchar") is False

    def test_is_string_type(self):
        """is_string_type detects string types."""
        assert is_string_type("varchar") is True
        assert is_string_type("TEXT") is True
        assert is_string_type("integer") is False

    def test_is_string_type_empty_returns_false(self):
        """is_string_type returns False for empty string."""
        assert is_string_type("") is False

    def test_is_numeric_type_empty_returns_false(self):
        """is_numeric_type returns False for empty string."""
        assert is_numeric_type("") is False

    def test_is_date_type(self):
        """is_date_type detects temporal types."""
        assert is_date_type("timestamp") is True
        assert is_date_type("DATE") is True
        assert is_date_type("integer") is False

    def test_data_type_to_value_type_integer(self):
        """data_type_to_value_type maps integer types."""
        assert data_type_to_value_type("integer") == "integer"
        assert data_type_to_value_type("bigint") == "integer"
        assert data_type_to_value_type("serial") == "integer"

    def test_data_type_to_value_type_number(self):
        """data_type_to_value_type maps numeric types."""
        assert data_type_to_value_type("numeric") == "number"
        assert data_type_to_value_type("float") == "number"
        assert data_type_to_value_type("decimal(10,2)") == "number"

    def test_data_type_to_value_type_string(self):
        """data_type_to_value_type maps string types."""
        assert data_type_to_value_type("varchar") == "string"
        assert data_type_to_value_type("text") == "string"

    def test_data_type_to_value_type_date(self):
        """data_type_to_value_type maps temporal types."""
        assert data_type_to_value_type("timestamp") == "date"
        assert data_type_to_value_type("date") == "date"

    def test_data_type_to_value_type_boolean(self):
        """data_type_to_value_type maps boolean types."""
        assert data_type_to_value_type("boolean") == "boolean"
        assert data_type_to_value_type("bool") == "boolean"

    def test_data_type_to_value_type_fallback(self):
        """data_type_to_value_type falls back to unknown for unrecognized types."""
        assert data_type_to_value_type("xml") == "unknown"

    def test_data_type_to_value_type_empty_string(self):
        """data_type_to_value_type returns unknown for empty input."""
        assert data_type_to_value_type("") == "unknown"

    def test_data_type_to_value_type_whitespace_stripped(self):
        """data_type_to_value_type strips whitespace before lookup."""
        assert data_type_to_value_type("  integer  ") == "integer"

    def test_data_type_to_value_type_numeric_token_fallback(self):
        """data_type_to_value_type maps unknown type with numeric token to number."""
        assert data_type_to_value_type("myinteger") == "number"

    def test_is_date_type_interval(self):
        """is_date_type returns True for interval."""
        assert is_date_type("interval") is True


class TestColumnMetadataRoles:
    """Tests for ColumnMetadata role-based methods."""

    def test_get_valid_aggregations_identifier(self):
        """IDENTIFIER role only allows count."""
        cm = ColumnMetadata(
            name="id",
            data_type="integer",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
            row_count=100,
            valid_aggregations=["count"],
        )
        assert cm.get_valid_aggregations() == {"count"}

    def test_get_valid_aggregations_numeric_measure(self):
        """NUMERIC_MEASURE role allows all agg functions."""
        cm = ColumnMetadata(
            name="amount",
            data_type="numeric",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=50,
            row_count=100,
            valid_aggregations=["sum", "avg", "min", "max", "count"],
        )
        assert cm.get_valid_aggregations() == {"count", "sum", "avg", "min", "max"}

    def test_get_valid_aggregations_categorical(self):
        """CATEGORICAL role allows count, min, max but not sum/avg."""
        cm = ColumnMetadata(
            name="name",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=50,
            row_count=100,
            valid_aggregations=["count", "min", "max"],
        )
        aggs = cm.get_valid_aggregations()
        assert "count" in aggs
        assert "sum" not in aggs
        assert "avg" not in aggs

    def test_get_valid_aggregations_boolean(self):
        """BOOLEAN role allows count only."""
        cm = ColumnMetadata(
            name="active",
            data_type="boolean",
            role=ColumnRole.BOOLEAN.value,
            distinct_count=2,
            row_count=100,
            valid_aggregations=["count"],
        )
        aggs = cm.get_valid_aggregations()
        assert "count" in aggs
        assert "sum" not in aggs

    def test_get_valid_where_ops_pk(self):
        """Primary key column gets comparison ops."""
        cm = ColumnMetadata(
            name="id",
            data_type="integer",
            is_primary_key=True,
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
            row_count=100,
            valid_where_ops=[
                "=",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "between",
                "in",
                "not in",
                "is null",
                "is not null",
            ],
        )
        ops = cm.get_valid_where_ops()
        assert "=" in ops
        assert "between" in ops
        assert "is null" in ops

    def test_get_valid_where_ops_categorical(self):
        """Categorical string column gets LIKE ops."""
        cm = ColumnMetadata(
            name="name",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=50,
            row_count=100,
            valid_where_ops=[
                "=",
                "!=",
                "in",
                "not in",
                "like",
                "ilike",
                "not like",
                "not ilike",
                "is null",
                "is not null",
            ],
        )
        ops = cm.get_valid_where_ops()
        assert "like" in ops
        assert "ilike" in ops

    def test_is_filterable_audit_column(self):
        """AUDIT role columns may appear in predicates when not sensitivity-blocked."""
        cm = ColumnMetadata(
            name="created_at",
            data_type="timestamp",
            role=ColumnRole.AUDIT.value,
            distinct_count=100,
            row_count=100,
        )
        assert cm.is_filterable is True

    def test_is_groupable_categorical(self):
        """CATEGORICAL columns are groupable."""
        cm = ColumnMetadata(
            name="status",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=5,
            row_count=100,
        )
        assert cm.is_groupable is True

    def test_is_aggregatable_numeric_measure(self):
        """NUMERIC_MEASURE columns are aggregatable."""
        cm = ColumnMetadata(
            name="amount",
            data_type="numeric",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=50,
            row_count=100,
        )
        assert cm.is_aggregatable is True

    def test_is_aggregatable_categorical_false(self):
        """CATEGORICAL columns are not aggregatable."""
        cm = ColumnMetadata(
            name="name",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=50,
            row_count=100,
        )
        assert cm.is_aggregatable is False


class TestSQLShape:
    """Tests for SQLShape dataclass."""

    def test_defaults(self):
        """SQLShape defaults."""
        shape = SQLShape(num_joins=0, has_group_by=False, has_agg=False)
        assert shape.num_joins == 0
        assert shape.has_group_by is False
        assert shape.has_agg is False
        assert shape.has_distinct is False

    def test_round_trip(self):
        """SQLShape to_dict/from_dict round trip."""
        original = SQLShape(
            num_joins=2,
            has_group_by=True,
            has_agg=True,
            num_cte=1,
            num_where=3,
            has_distinct=True,
        )
        rebuilt = SQLShape.from_dict(original.to_dict())
        assert rebuilt.num_joins == 2
        assert rebuilt.has_group_by is True
        assert rebuilt.has_distinct is True


class TestCountDistinctNotValidAgg:
    """count_distinct is not a valid agg_func anywhere."""

    def test_no_count_distinct_in_valid_agg(self):
        """count_distinct is not in VALID_AGGREGATION_FUNCTIONS."""
        assert "count_distinct" not in VALID_AGGREGATION_FUNCTIONS

    def test_no_count_distinct_in_role_aggregations(self):
        """count_distinct is not in any ROLE_ALLOWED_AGGREGATIONS set."""
        for role, aggs in ROLE_ALLOWED_AGGREGATIONS.items():
            assert "count_distinct" not in aggs, f"count_distinct found in {role}"

    def test_no_distinct_field_on_select_col(self):
        """SelectCol does not have a 'distinct' attribute."""
        sc = SelectCol(expr=NormalizedExpr.from_column("t.x"))
        assert not hasattr(sc, "distinct")

    def test_no_distinct_field_on_runtime_intent(self):
        """RuntimeIntent does not have a 'distinct' attribute."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert not hasattr(intent, "distinct")

    def test_qsim_intent_has_distinct(self):
        """QSimIntent exposes a distinct field."""
        qi = QSimIntent(
            intent_id="t",
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=[],
            having_param=[],
        )
        assert hasattr(qi, "distinct")


class TestWhereParamSignatureKey:
    """Tests for WhereParam.signature_key."""

    def test_signature_key_deterministic(self):
        """WhereParam.signature_key is deterministic."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=", value_type="string")
        assert fp.signature_key == fp.signature_key

    def test_signature_key_includes_op(self):
        """WhereParam.signature_key includes the op."""
        fp = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op=">=", value_type="number")
        assert ">=" in fp.signature_key

    def test_different_ops_different_keys(self):
        """WhereParam with different ops have different signature_keys."""
        fp1 = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        fp2 = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="!=")
        assert fp1.signature_key != fp2.signature_key

    def test_signature_key_ignores_bool_op(self):
        """WhereParam.signature_key does not include bool_op from input dicts."""
        fp_plain = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        fp_with_bool = WhereParam.from_dict({"left_expr": "t.c", "op": "=", "bool_op": "OR"})
        assert fp_plain.signature_key == fp_with_bool.signature_key
        assert "bool_op" not in fp_plain.signature_key

    def test_signature_key_ignores_where_group(self):
        """WhereParam.signature_key does not include where_group from input dicts."""
        fp_plain = WhereParam(left_expr=NormalizedExpr.from_column("t.c"), op="=")
        fp_with_group = WhereParam.from_dict({"left_expr": "t.c", "op": "=", "where_group": 5})
        assert fp_plain.signature_key == fp_with_group.signature_key
        assert "where_group" not in fp_plain.signature_key


class TestHavingParamSignatureKey:
    """Tests for HavingParam.signature_key."""

    def test_signature_key_deterministic(self):
        """HavingParam.signature_key is deterministic."""
        hp = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.id"), op=">")
        assert hp.signature_key == hp.signature_key

    def test_signature_key_includes_op(self):
        """HavingParam.signature_key includes the op."""
        hp = HavingParam(left_expr=NormalizedExpr.from_agg("count", "t.id"), op=">=")
        assert ">=" in hp.signature_key


class TestSelectColSignatureKey:
    """Tests for SelectCol.signature_key."""

    def test_signature_key_delegates_to_expr(self):
        """SelectCol.signature_key is the expr's signature_key."""
        expr = NormalizedExpr.from_column("t.x")
        sc = SelectCol(expr=expr)
        assert sc.signature_key == expr.signature_key


class TestOrderByColExtended:
    """Additional OrderByCol tests."""

    def test_is_aggregated_true(self):
        """OrderByCol.is_aggregated True when expr has aggregation."""
        obc = OrderByCol(expr=NormalizedExpr.from_agg("sum", "t.x"), direction="DESC")
        assert obc.is_aggregated is True

    def test_is_aggregated_false(self):
        """OrderByCol.is_aggregated False for bare column."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("t.x"), direction="ASC")
        assert obc.is_aggregated is False

    def test_signature_key_includes_direction(self):
        """OrderByCol.signature_key includes direction."""
        obc = OrderByCol(expr=NormalizedExpr.from_column("t.x"), direction="DESC")
        assert "DESC" in obc.signature_key


class TestRuntimeCteStepExpectedRows:
    """Tests for RuntimeCteStep.expected_rows."""

    def test_scalar_returns_one(self):
        """RuntimeCteStep.expected_rows returns 'one' for scalar grain."""
        rt = RuntimeCteStep(cte_name="c1", tables=["t"], grain="scalar")
        assert rt.expected_rows == "one"

    def test_limit_returns_few(self):
        """RuntimeCteStep.expected_rows returns 'few' when limit is set."""
        rt = RuntimeCteStep(cte_name="c1", tables=["t"], grain="row_level", limit=10)
        assert rt.expected_rows == "few"

    def test_no_limit_returns_many(self):
        """RuntimeCteStep.expected_rows returns 'many' with no limit."""
        rt = RuntimeCteStep(cte_name="c1", tables=["t"], grain="grouped")
        assert rt.expected_rows == "many"


class TestQSimWhereParam:
    """Tests for QSimWhereParam."""

    def test_round_trip(self):
        """QSimWhereParam to_dict/from_dict round trip."""
        qf = QSimWhereParam(column="orders.status", op="=", value_type="categorical")
        rebuilt = QSimWhereParam.from_dict(qf.to_dict())
        assert rebuilt.column == "orders.status"
        assert rebuilt.op == "="

    def test_is_expr_comparison_false(self):
        """QSimWhereParam.is_expr_comparison False when no right_column."""
        qf = QSimWhereParam(column="t.c", op="=", value_type="categorical")
        assert qf.is_expr_comparison is False

    def test_is_expr_comparison_true(self):
        """QSimWhereParam.is_expr_comparison True when right_column set."""
        qf = QSimWhereParam(column="t.c1", op=">", value_type="numeric", right_column="t.c2")
        assert qf.is_expr_comparison is True

    def test_to_dict_excludes_empty_right_column(self):
        """QSimWhereParam.to_dict omits right_column when empty."""
        qf = QSimWhereParam(column="t.c", op="=", value_type="categorical")
        d = qf.to_dict()
        assert "right_column" not in d

    def test_to_dict_includes_right_column(self):
        """QSimWhereParam.to_dict includes right_column when set."""
        qf = QSimWhereParam(column="t.c1", op=">", value_type="numeric", right_column="t.c2")
        d = qf.to_dict()
        assert d["right_column"] == "t.c2"


class TestQSimHaving:
    """Tests for QSimHaving."""

    def test_round_trip(self):
        """QSimHaving to_dict/from_dict round trip."""
        qh = QSimHaving(expression="COUNT(orders.order_id)", op=">", value_type="number")
        rebuilt = QSimHaving.from_dict(qh.to_dict())
        assert rebuilt.expression == "COUNT(orders.order_id)"

    def test_is_expression_comparison_false(self):
        """QSimHaving.is_expression_comparison False when no right_expression."""
        qh = QSimHaving(expression="COUNT(t.id)", op=">", value_type="number")
        assert qh.is_expression_comparison is False

    def test_is_expression_comparison_true(self):
        """QSimHaving.is_expression_comparison True when right_expression set."""
        qh = QSimHaving(
            expression="SUM(t.a)",
            op=">",
            value_type="number",
            right_expression="SUM(t.b)",
        )
        assert qh.is_expression_comparison is True

    def test_to_dict_excludes_empty_right_expression(self):
        """QSimHaving.to_dict omits right_expression when empty."""
        qh = QSimHaving(expression="COUNT(t.id)", op=">", value_type="number")
        d = qh.to_dict()
        assert "right_expression" not in d


class TestSeedWarmupIntent:
    """Tests for SeedWarmupIntent."""

    def test_round_trip(self):
        """SeedWarmupIntent to_dict/from_dict round trip."""
        si = SeedWarmupIntent(
            intent_id="warm_1",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        rebuilt = SeedWarmupIntent.from_dict(si.to_dict())
        assert rebuilt.intent_id == "warm_1"
        assert rebuilt.tables == ["orders"]

    def test_distinct_select_index_round_trip(self):
        """``distinct_select_index`` survives ``to_dict`` / ``from_dict``."""
        si = SeedWarmupIntent(
            intent_id="warm_d",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            distinct_select_index=0,
        )
        rebuilt = SeedWarmupIntent.from_dict(si.to_dict())
        assert rebuilt.distinct_select_index == 0
        """SeedWarmupIntent.to_runtime_intent returns RuntimeIntent."""
        si = SeedWarmupIntent(
            intent_id="warm_1",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        ri = si.to_runtime_intent()
        assert isinstance(ri, RuntimeIntent)
        assert ri.tables == ["orders"]

    def test_expansion_metadata_round_trip(self):
        """SeedWarmupIntent preserves expansion_metadata through round trip."""
        em = ExpansionMetadata(operator="add_filter")
        si = SeedWarmupIntent(
            intent_id="warm_3",
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            expansion_metadata=em,
        )
        rebuilt = SeedWarmupIntent.from_dict(si.to_dict())
        assert rebuilt.expansion_metadata is not None
        assert rebuilt.expansion_metadata.operator == "add_filter"


class TestConcreteIntentStandalone:
    """Standalone tests for ConcreteIntent."""

    def test_from_dict_round_trip(self):
        """ConcreteIntent from_dict/to_dict round trip."""
        ci = ConcreteIntent(
            intent_id="ci_1",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        rebuilt = ConcreteIntent.from_dict(ci.to_dict())
        assert rebuilt.intent_id == "ci_1"
        assert rebuilt.tables == ["orders"]

    def test_has_join_path_fields(self):
        """ConcreteIntent stores join candidate id and path signature."""
        ci = ConcreteIntent(
            intent_id="ci_2",
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_candidate_id="jc_1",
            chosen_join_path_signature=["orders.customer_id=customers.customer_id"],
        )
        assert ci.chosen_join_candidate_id == "jc_1"
        assert len(ci.chosen_join_path_signature) == 1


class TestTemplate:
    """Tests for Template dataclass."""

    def test_round_trip(self, sample_template):
        """Template to_dict/from_dict round trip."""
        rebuilt = Template.from_dict(sample_template.to_dict())
        assert rebuilt.id == sample_template.id
        assert rebuilt.effective_structural_hash == sample_template.effective_structural_hash
        assert rebuilt.trust_level == sample_template.trust_level

    def test_display_alias_map_round_trip(self, sample_template):
        """Template ``display_alias_map`` survives to_dict/from_dict."""
        from dataclasses import replace

        tmpl = replace(sample_template, display_alias_map={"sig_a": "alias_a"})
        rebuilt = Template.from_dict(tmpl.to_dict())
        assert rebuilt.display_alias_map == {"sig_a": "alias_a"}

    def test_unknown_execution_sql_keys_ignored(self, sample_template):
        """Unknown ``execution_sql`` / ``spark_sql_param`` JSON keys are ignored."""
        d = sample_template.to_dict()
        d["execution_sql"] = "SELECT * FROM ignored_exec"
        d["spark_sql_param"] = "SELECT * FROM ignored_spark"
        rebuilt = Template.from_dict(d)
        assert rebuilt.sql_param == sample_template.sql_param

    def test_chosen_join_candidate_id_delegates(self, sample_template):
        """Template.chosen_join_candidate_id delegates to intent_signature."""
        assert sample_template.chosen_join_candidate_id == sample_template.intent_signature.chosen_join_candidate_id

    def test_chosen_join_path_signature_delegates(self, sample_template):
        """Template.chosen_join_path_signature delegates to intent_signature."""
        assert sample_template.chosen_join_path_signature == sample_template.intent_signature.chosen_join_path_signature


class TestClassifySeedWarmupComplexity:
    """Tests for :func:`classify_seed_warmup_intent_complexity`."""

    def test_single_table_no_agg_is_simple(self):
        """One bare dimension path maps to SIMPLE."""
        si = SeedWarmupIntent(
            intent_id="x",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        assert si.complexity_tier() == ComplexityTier.SIMPLE

    def test_two_tables_at_least_moderate(self):
        """Two referenced tables lift tier to MODERATE or above."""
        si = SeedWarmupIntent(
            intent_id="y",
            tables=["a", "b"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        assert si.complexity_tier() == ComplexityTier.MODERATE


class TestAnchorLatticeKeyForSeedIntent:
    """Tests for :func:`anchor_lattice_key_for_seed_intent` and :func:`anchor_lattice_signature`."""

    def test_distinct_intent_ids_share_cell_when_tier_and_novelty_match(self):
        em = ExpansionMetadata(operator="op", depth=0)
        a = SeedWarmupIntent(
            intent_id="a",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            source="synthetic",
            seed_index=3,
            expansion_metadata=em,
        )
        b = SeedWarmupIntent(
            intent_id="b",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            source="synthetic",
            seed_index=3,
            expansion_metadata=em,
        )
        ka = a.anchor_lattice_key()
        kb = b.anchor_lattice_key()
        assert ka == kb
        assert ka.signature("fp1") == kb.signature("fp1")
        assert ka.signature("fp1") != kb.signature("fp2")

    def test_expansion_depth_splits_cells(self):
        em_low = ExpansionMetadata(operator="op", depth=0)
        em_high = ExpansionMetadata(operator="op", depth=2)
        low = SeedWarmupIntent(
            intent_id="low",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            source="synthetic",
            seed_index=3,
            expansion_metadata=em_low,
        )
        high = SeedWarmupIntent(
            intent_id="high",
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            source="synthetic",
            seed_index=3,
            expansion_metadata=em_high,
        )
        assert low.anchor_lattice_key() != high.anchor_lattice_key()


class TestSeedWarmupResult:
    """Tests for SeedWarmupResult dataclass."""

    def test_to_dict(self):
        """SeedWarmupResult.to_dict serializes all fields."""
        ri = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        sr = SeedWarmupResult(intent=ri, question="test question", success=True)
        d = sr.to_dict()
        assert d["question"] == "test question"
        assert d["questions"] == []
        assert d["success"] is True
        assert "confidence" not in d

    def test_defaults(self):
        """SeedWarmupResult defaults."""
        ri = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        sr = SeedWarmupResult(intent=ri, question="q")
        assert sr.sql is None
        assert sr.success is False
        assert sr.questions == []
        assert sr.error is None
        assert sr.validation_issues == []


class TestTemplateMatch:
    """Tests for TemplateMatch dataclass."""

    def test_defaults(self):
        """TemplateMatch has correct defaults."""
        tm = TemplateMatch()
        assert tm.intent is None
        assert tm.best_template is None
        assert tm.similarity_score == 0.0
        assert tm.reuse_type == "none"


class TestFKEdge:
    """Tests for FKEdge dataclass."""

    def test_fields(self):
        """FKEdge stores src/dst table and column info."""
        fk = FKEdge(
            src_table="orders",
            src_cols=["customer_id"],
            dst_table="customers",
            dst_cols=["customer_id"],
        )
        assert fk.src_table == "orders"
        assert fk.dst_cols == ["customer_id"]


class TestValueDomain:
    """Tests for ValueDomain dataclass."""

    def test_defaults(self):
        """ValueDomain has correct defaults."""
        vd = ValueDomain()
        assert vd.values == []
        assert vd.min_val is None
        assert vd.max_val is None
        assert vd.data_type is None

    def test_with_values(self):
        """ValueDomain stores values."""
        vd = ValueDomain(values=["a", "b"], min_val="a", max_val="b", data_type="varchar")
        assert len(vd.values) == 2
        assert vd.data_type == "varchar"


class TestTableMetadata:
    """Tests for TableMetadata."""

    def test_column_names(self):
        """TableMetadata.column_names returns column keys."""
        tm = TableMetadata(
            name="t",
            columns={
                "c1": ColumnMetadata(name="c1", data_type="int", distinct_count=1, row_count=1),
                "c2": ColumnMetadata(name="c2", data_type="text", distinct_count=1, row_count=1),
            },
            primary_key=["c1"],
            foreign_keys=[],
        )
        assert set(tm.column_names) == {"c1", "c2"}

    def test_round_trip(self):
        """TableMetadata to_dict/from_dict round trip."""
        tm = TableMetadata(
            name="t",
            columns={"c1": ColumnMetadata(name="c1", data_type="int", distinct_count=1, row_count=1)},
            primary_key=["c1"],
            foreign_keys=[FKEdge(src_table="t", src_cols=["c1"], dst_table="u", dst_cols=["c1"])],
            role="FACT",
            row_count=10,
            description="test table",
        )
        rebuilt = TableMetadata.from_dict(tm.to_dict())
        assert rebuilt.name == "t"
        assert rebuilt.role == "FACT"
        assert rebuilt.row_count == 10
        assert len(rebuilt.foreign_keys) == 1


class TestSchemaGraphMethods:
    """Tests for SchemaGraph accessor methods."""

    def test_table_names(self, schema_graph):
        """SchemaGraph.table_names returns all table names."""
        assert set(schema_graph.table_names) == {"customers", "orders", "products"}

    def test_fk_edges(self, schema_graph):
        """SchemaGraph.fk_edges returns all FK edges from all tables."""
        edges = schema_graph.fk_edges
        assert len(edges) == 2

    def test_get_column_found(self, schema_graph):
        """SchemaGraph.get_column returns ColumnMetadata when found."""
        col = schema_graph.get_column("orders", "amount")
        assert col is not None
        assert col.name == "amount"

    def test_get_column_not_found(self, schema_graph):
        """SchemaGraph.get_column returns None for missing column."""
        assert schema_graph.get_column("orders", "nonexistent") is None

    def test_get_column_missing_table(self, schema_graph):
        """SchemaGraph.get_column returns None for missing table."""
        assert schema_graph.get_column("nonexistent", "col") is None

    def test_schema_literal_json(self, schema_graph):
        """SchemaGraph.schema_literal_json returns JSON with visible table keys."""
        raw = schema_graph.schema_literal_json
        payload = json.loads(raw)
        assert set(payload.keys()) >= {"customers", "orders", "products"}

    def test_schema_literal_json_unique_marker(self):
        """schema_literal_json marks single-column unique non-PK columns."""
        col = ColumnMetadata(
            name="email",
            data_type="varchar",
            is_unique=True,
            is_primary_key=False,
            distinct_count=50,
            distinct_ratio=0.9,
            null_ratio=0.0,
            role=ColumnRole.CATEGORICAL.value,
        )
        t = TableMetadata(name="users", columns={"email": col}, primary_key=[], foreign_keys=[])
        sg = SchemaGraph(tables={"users": t}, join_paths_multi={}, effective_structural_hash="h")
        payload = json.loads(sg.schema_literal_json)
        assert payload["users"]["columns"]["email"].get("unique") is True

    def test_schema_literal_json_join_paths_excluded(self):
        t1 = TableMetadata(
            name="orders",
            columns={"order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True)},
            primary_key=["order_id"],
            foreign_keys=[],
        )
        t2 = TableMetadata(
            name="customers",
            columns={"customer_id": ColumnMetadata(name="customer_id", data_type="integer", is_primary_key=True)},
            primary_key=["customer_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(
            tables={"orders": t1, "customers": t2},
            join_paths_multi={
                "orders": {
                    "customers": [
                        [
                            {
                                "src_table": "orders",
                                "src_col": "customer_id",
                                "dst_table": "customers",
                                "dst_col": "customer_id",
                            }
                        ]
                    ]
                },
                "customers": {
                    "orders": [
                        [
                            {
                                "src_table": "orders",
                                "src_col": "customer_id",
                                "dst_table": "customers",
                                "dst_col": "customer_id",
                            }
                        ]
                    ]
                },
            },
            effective_structural_hash="h",
        )
        raw = sg.schema_literal_json
        assert "join" not in raw.lower()
        payload = json.loads(raw)
        assert "orders" in payload and "customers" in payload

    def test_structural_schema_literal_json_filters_tables(self):
        t1 = TableMetadata(
            name="orders",
            columns={"order_id": ColumnMetadata(name="order_id", data_type="integer", is_primary_key=True)},
            primary_key=["order_id"],
            foreign_keys=[],
        )
        t2 = TableMetadata(
            name="customers",
            columns={"customer_id": ColumnMetadata(name="customer_id", data_type="integer", is_primary_key=True)},
            primary_key=["customer_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(
            tables={"orders": t1, "customers": t2},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        payload = json.loads(sg.structural_schema_literal_json(["orders"]))
        assert set(payload.keys()) == {"orders"}
        assert "description" not in payload["orders"]

    def test_structural_schema_literal_json_pk_only_when_true(self):
        t1 = TableMetadata(
            name="items",
            columns={
                "item_id": ColumnMetadata(name="item_id", data_type="integer", is_primary_key=True),
                "label": ColumnMetadata(
                    name="label",
                    data_type="text",
                    is_primary_key=False,
                    distinct_count=50,
                    distinct_ratio=0.9,
                    null_ratio=0.0,
                    is_canonical_duplicate=False,
                ),
            },
            primary_key=["item_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(tables={"items": t1}, join_paths_multi={}, effective_structural_hash="h")
        cols = json.loads(sg.structural_schema_literal_json(["items"]))["items"]["columns"]
        assert cols["item_id"].get("pk") is True
        assert "pk" not in cols["label"]

    def test_schema_literal_json_no_join_paths(self):
        t1 = TableMetadata(
            name="items",
            columns={"item_id": ColumnMetadata(name="item_id", data_type="integer", is_primary_key=True)},
            primary_key=["item_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(tables={"items": t1}, join_paths_multi={}, effective_structural_hash="h")
        raw = sg.schema_literal_json
        assert "join" not in raw.lower()

    def test_schema_literal_json_enriched_roles(self):
        """schema_literal_json includes table role, description, column role, and description."""
        t1 = TableMetadata(
            name="payment",
            role="fact",
            description="customer payments",
            columns={
                "payment_id": ColumnMetadata(
                    name="payment_id",
                    data_type="integer",
                    is_primary_key=True,
                    role="identifier",
                ),
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    role="numeric_measure",
                    description="revenue/income amount",
                    distinct_count=42,
                ),
            },
            primary_key=["payment_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(tables={"payment": t1}, join_paths_multi={}, effective_structural_hash="h")
        payload = json.loads(sg.schema_literal_json)["payment"]
        assert payload.get("role") == "fact"
        assert payload.get("description") == "customer payments"
        assert "role" not in payload["columns"]["payment_id"]
        assert payload["columns"]["amount"].get("role") == "numeric_measure"
        assert payload["columns"]["amount"].get("description") == "revenue/income amount"

    def test_round_trip(self, schema_graph):
        """SchemaGraph to_dict/from_dict round trip."""
        rebuilt = SchemaGraph.from_dict(schema_graph.to_dict())
        assert set(rebuilt.table_names) == set(schema_graph.table_names)
        assert rebuilt.effective_structural_hash == schema_graph.effective_structural_hash


class TestColumnMetadataRoundTrip:
    """Tests for ColumnMetadata serialization."""

    def test_round_trip(self):
        """ColumnMetadata to_dict/from_dict round trip."""
        cm = ColumnMetadata(
            name="amount",
            data_type="numeric",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=50,
            distinct_ratio=0.5,
            row_count=100,
        )
        rebuilt = ColumnMetadata.from_dict(cm.to_dict())
        assert rebuilt.name == "amount"
        assert rebuilt.role == ColumnRole.NUMERIC_MEASURE.value

    def test_round_trip_description(self):
        """ColumnMetadata preserves description through round trip."""
        cm = ColumnMetadata(
            name="amount",
            data_type="numeric",
            description="revenue/income amount",
        )
        rebuilt = ColumnMetadata.from_dict(cm.to_dict())
        assert rebuilt.description == "revenue/income amount"

    def test_round_trip_description_empty(self):
        """ColumnMetadata defaults description to empty string."""
        cm = ColumnMetadata(name="id", data_type="integer")
        rebuilt = ColumnMetadata.from_dict(cm.to_dict())
        assert rebuilt.description == ""

    def test_is_usable(self):
        """ColumnMetadata.is_usable True for non-AUDIT columns with sufficient variance."""
        cm = ColumnMetadata(
            name="c",
            data_type="int",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=5,
            row_count=10,
        )
        assert cm.is_usable is True

    def test_is_usable_low_distinct(self):
        """ColumnMetadata.is_usable False when distinct_count <= 1."""
        cm = ColumnMetadata(
            name="c",
            data_type="int",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=1,
            row_count=10,
        )
        assert cm.is_usable is False

    def test_is_usable_audit(self):
        """ColumnMetadata.is_usable for AUDIT follows statistical gates like other roles."""
        cm = ColumnMetadata(
            name="c",
            data_type="timestamp",
            role=ColumnRole.AUDIT.value,
            distinct_count=50,
            row_count=100,
        )
        assert cm.is_usable is True

    def test_is_usable_high_null_ratio(self):
        """ColumnMetadata.is_usable False when null_ratio reaches the policy threshold."""
        cm = ColumnMetadata(
            name="c",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=5,
            row_count=1000,
            null_ratio=0.99,
        )
        assert cm.is_usable is False

    def test_is_usable_pk_overrides_audit(self):
        """ColumnMetadata.is_usable True for PK columns even if other gates would hide them."""
        cm = ColumnMetadata(
            name="c",
            data_type="integer",
            role=ColumnRole.AUDIT.value,
            distinct_count=1,
            row_count=1,
            null_ratio=1.0,
            is_primary_key=True,
        )
        assert cm.is_usable is True

    def test_is_usable_fk_overrides_null_ratio(self):
        """ColumnMetadata.is_usable True for FK columns even when the column is mostly null."""
        cm = ColumnMetadata(
            name="c",
            data_type="integer",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=2,
            row_count=1000,
            null_ratio=1.0,
            is_foreign_key=True,
        )
        assert cm.is_usable is True

    def test_is_usable_sentinel_dominated_false(self):
        """ColumnMetadata.is_usable False when one value dominates the non-null distribution."""
        cm = ColumnMetadata(
            name="c",
            data_type="integer",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=3,
            row_count=1000,
            null_ratio=0.0,
            mode_frequency_ratio=0.99,
        )
        assert cm.is_usable is False

    def test_is_usable_sentinel_below_threshold_true(self):
        """ColumnMetadata.is_usable True when mode frequency stays below the sentinel threshold."""
        cm = ColumnMetadata(
            name="c",
            data_type="integer",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=3,
            row_count=1000,
            null_ratio=0.0,
            mode_frequency_ratio=0.50,
        )
        assert cm.is_usable is True

    def test_is_usable_sentinel_pk_override(self):
        """Primary keys remain usable even when sentinel-dominated."""
        cm = ColumnMetadata(
            name="c",
            data_type="integer",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=3,
            row_count=1000,
            mode_frequency_ratio=1.0,
            is_primary_key=True,
        )
        assert cm.is_usable is True

    def test_is_visible_default_true(self):
        """ColumnMetadata.is_visible mirrors is_usable when nothing is denied or sensitive."""
        cm = ColumnMetadata(name="c", data_type="text", distinct_count=10, row_count=100)
        assert cm.is_usable is True
        assert cm.is_visible is True

    def test_is_visible_denied_hides_pk(self):
        """is_denied hides even structural PK columns from LLM context."""
        cm = ColumnMetadata(
            name="id",
            data_type="integer",
            distinct_count=100,
            row_count=100,
            is_primary_key=True,
            is_denied=True,
        )
        assert cm.is_usable is True
        assert cm.is_visible is False

    def test_is_visible_restricted_stays_visible(self):
        """Restricted sensitivity remains in LLM context when usable."""
        cm = ColumnMetadata(
            name="email",
            data_type="text",
            distinct_count=50,
            row_count=100,
            sensitivity=SensitivityClassification.RESTRICTED,
        )
        assert cm.is_visible is True

    def test_is_visible_hidden(self):
        """Hidden sensitivity omits the column from LLM context."""
        cm = ColumnMetadata(
            name="ssn",
            data_type="text",
            distinct_count=50,
            row_count=100,
            sensitivity=SensitivityClassification.HIDDEN,
        )
        assert cm.is_visible is False

    def test_is_visible_audit_role_low_signal_hidden(self):
        """Audit columns follow the same statistical visibility gates as other roles."""
        cm = ColumnMetadata(
            name="created_at",
            data_type="timestamp",
            role=ColumnRole.AUDIT.value,
            distinct_count=1,
            row_count=100,
        )
        assert cm.is_visible is False

    def test_get_valid_having_ops(self):
        """ColumnMetadata.get_valid_having_ops returns list of ops."""
        cm = ColumnMetadata(
            name="c",
            data_type="numeric",
            role=ColumnRole.NUMERIC_MEASURE.value,
            distinct_count=50,
            row_count=100,
            valid_having_ops=["=", "!=", "<", "<=", ">", ">="],
        )
        ops = cm.get_valid_having_ops()
        assert ">" in ops
        assert "<" in ops

    def test_from_dict_fk_target_tuple(self):
        """ColumnMetadata.from_dict accepts fk_target as tuple."""
        d = {
            "name": "ref_id",
            "data_type": "int",
            "is_foreign_key": True,
            "fk_target": ("other", "id"),
        }
        cm = ColumnMetadata.from_dict(d)
        assert cm.fk_target == ("other", "id")

    def test_from_dict_fk_target_list_converted_to_tuple(self):
        """ColumnMetadata.from_dict converts fk_target list to tuple."""
        d = {
            "name": "ref_id",
            "data_type": "int",
            "is_foreign_key": True,
            "fk_target": ["other", "id"],
        }
        cm = ColumnMetadata.from_dict(d)
        assert cm.fk_target == ("other", "id")

    def test_post_init_sets_value_type_from_data_type_when_empty(self):
        """ColumnMetadata __post_init__ sets value_type from data_type when value_type empty."""
        cm = ColumnMetadata(name="x", data_type="varchar(100)", value_type="")
        assert cm.value_type == "string"

    def test_post_init_preserves_explicit_value_type(self):
        """ColumnMetadata __post_init__ does not overwrite explicit value_type."""
        cm = ColumnMetadata(name="x", data_type="varchar(100)", value_type="integer")
        assert cm.value_type == "integer"

    def test_is_filterable_restricted_sensitivity(self):
        """ColumnMetadata.is_filterable False when sensitivity is RESTRICTED."""
        cm = ColumnMetadata(
            name="user_password",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
            row_count=100,
            sensitivity=SensitivityClassification.RESTRICTED,
        )
        assert cm.is_filterable is False

    def test_is_filterable_hidden_sensitivity(self):
        """ColumnMetadata.is_filterable False when sensitivity is HIDDEN."""
        cm = ColumnMetadata(
            name="secret_token",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=10,
            row_count=100,
            sensitivity=SensitivityClassification.HIDDEN,
        )
        assert cm.is_filterable is False

    def test_is_filterable_override_true(self):
        """ColumnMetadata.is_filterable respects override True."""
        cm = ColumnMetadata(
            name="c",
            data_type="text",
            role=ColumnRole.FREE_TEXT.value,
            distinct_count=50,
            row_count=100,
            is_filterable_override=True,
        )
        assert cm.is_filterable is True

    def test_is_filterable_override_false(self):
        """ColumnMetadata.is_filterable respects override False."""
        cm = ColumnMetadata(
            name="c",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=50,
            row_count=100,
            is_filterable_override=False,
        )
        assert cm.is_filterable is False

    def test_get_valid_where_ops_uses_stored_when_set(self):
        """ColumnMetadata.get_valid_where_ops returns stored ops plus null when set."""
        cm = ColumnMetadata(
            name="c",
            data_type="varchar",
            valid_where_ops=["=", "!=", "in"],
        )
        ops = cm.get_valid_where_ops()
        assert "=" in ops
        assert "is null" in ops
        assert "is not null" in ops

    def test_get_valid_where_ops_foreign_key_includes_range(self):
        """ColumnMetadata.get_valid_where_ops for FK includes range operators."""
        cm = ColumnMetadata(
            name="ref_id",
            data_type="int",
            is_foreign_key=True,
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=10,
            valid_where_ops=[
                "=",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "between",
                "in",
                "not in",
                "is null",
                "is not null",
            ],
        )
        ops = cm.get_valid_where_ops()
        assert "between" in ops
        assert ">=" in ops

    def test_get_valid_where_ops_boolean_excludes_like(self):
        """ColumnMetadata.get_valid_where_ops for boolean excludes like."""
        cm = ColumnMetadata(
            name="active",
            data_type="bool",
            role=ColumnRole.BOOLEAN.value,
            distinct_count=2,
            valid_where_ops=["=", "!=", "in", "not in", "is null", "is not null"],
        )
        ops = cm.get_valid_where_ops()
        assert "like" not in ops
        assert "=" in ops

    def test_get_valid_where_ops_numeric_categorical_includes_between(self):
        """ColumnMetadata.get_valid_where_ops for numeric_categorical includes between."""
        cm = ColumnMetadata(
            name="code",
            data_type="int",
            role=ColumnRole.NUMERIC_CATEGORICAL.value,
            distinct_count=20,
            valid_where_ops=[
                "=",
                "!=",
                "in",
                "not in",
                "<",
                "<=",
                ">",
                ">=",
                "between",
                "is null",
                "is not null",
            ],
        )
        ops = cm.get_valid_where_ops()
        assert "between" in ops

    def test_get_valid_aggregations_uses_stored_when_set(self):
        """ColumnMetadata.get_valid_aggregations returns stored when set."""
        cm = ColumnMetadata(
            name="c",
            data_type="numeric",
            valid_aggregations=["COUNT", "SUM"],
        )
        assert cm.get_valid_aggregations() == {"count", "sum"}

    def test_get_valid_aggregations_identifier_only_count(self):
        """ColumnMetadata.get_valid_aggregations IDENTIFIER returns count only."""
        cm = ColumnMetadata(
            name="id",
            data_type="int",
            role=ColumnRole.IDENTIFIER.value,
            distinct_count=100,
            valid_aggregations=["count"],
        )
        assert cm.get_valid_aggregations() == {"count"}

    def test_get_valid_aggregations_audit_allows_count(self):
        """ColumnMetadata.get_valid_aggregations AUDIT allows count only."""
        cm = ColumnMetadata(
            name="updated_at",
            data_type="timestamp",
            role=ColumnRole.AUDIT.value,
        )
        assert cm.get_valid_aggregations() == {"count"}

    def test_get_valid_having_ops_uses_stored_when_set(self):
        """ColumnMetadata.get_valid_having_ops returns stored when set."""
        cm = ColumnMetadata(
            name="c",
            data_type="numeric",
            valid_having_ops=["=", ">"],
        )
        assert cm.get_valid_having_ops() == ["=", ">"]

    def test_get_valid_having_ops_empty_when_not_aggregatable(self):
        """ColumnMetadata.get_valid_having_ops empty when not aggregatable."""
        cm = ColumnMetadata(
            name="label",
            data_type="varchar",
            role=ColumnRole.CATEGORICAL.value,
            distinct_count=100,
            is_aggregatable_override=False,
        )
        assert cm.get_valid_having_ops() == []


class TestExpansionMetadata:
    """Tests for ExpansionMetadata."""

    def test_round_trip(self):
        """ExpansionMetadata to_dict/from_dict round trip."""
        em = ExpansionMetadata(operator="add_filter", parent_intent_id="p1")
        rebuilt = ExpansionMetadata.from_dict(em.to_dict())
        assert rebuilt.operator == "add_filter"
        assert rebuilt.parent_intent_id == "p1"

    def test_defaults(self):
        """ExpansionMetadata defaults parent_intent_id to None."""
        em = ExpansionMetadata(operator="op")
        assert em.parent_intent_id is None


class TestCteOutputColumnMeta:
    """Tests for CteOutputColumnMeta."""

    def test_round_trip(self):
        """CteOutputColumnMeta to_dict/from_dict round trip."""
        meta = CteOutputColumnMeta(
            source="orders.amount",
            agg_func="sum",
            role=ColumnRole.NUMERIC_MEASURE.value,
        )
        rebuilt = CteOutputColumnMeta.from_dict(meta.to_dict())
        assert rebuilt.source == "orders.amount"
        assert rebuilt.agg_func == "sum"

    def test_defaults(self):
        """CteOutputColumnMeta has correct defaults."""
        meta = CteOutputColumnMeta(source="t.c")
        assert meta.filterable is True
        assert meta.aggregatable is True
        assert meta.groupable is True


class TestRetryFailureContext:
    """Tests for RetryFailureContext."""

    def test_fields(self):
        """RetryFailureContext stores failure context."""
        ctx = RetryFailureContext(
            failure_type="missing_table",
            required_tables=["orders", "customers"],
            used_tables={"orders"},
            missing_tables={"customers"},
            attempt_number=1,
        )
        assert ctx.failure_type == "missing_table"
        assert ctx.missing_tables == {"customers"}


class TestIntentIssue:
    """Tests for IntentIssue."""

    def test_round_trip(self):
        """IntentIssue to_dict/from_dict round trip."""
        issue = IntentIssue(
            issue_id="I001",
            category=FailureCategory.WRONG_TABLES,
            severity="error",
            message="missing table",
            context={"table": "orders"},
        )
        rebuilt = IntentIssue.from_dict(issue.to_dict())
        assert rebuilt.issue_id == "I001"
        assert rebuilt.category == FailureCategory.WRONG_TABLES
        assert rebuilt.context == {"table": "orders"}


class TestIntentValidationResult:
    """Tests for IntentValidationResult."""

    def test_is_valid_no_issues(self):
        """IntentValidationResult is_valid True with no issues."""
        ivr = IntentValidationResult()
        assert ivr.is_valid is True

    def test_is_valid_false_with_error(self):
        """IntentValidationResult is_valid False with error issue."""
        issue = IntentIssue(
            issue_id="I001",
            category=FailureCategory.OTHER,
            severity="error",
            message="err",
        )
        ivr = IntentValidationResult(issues=[issue])
        assert ivr.is_valid is False

    def test_is_valid_true_with_warning_only(self):
        """IntentValidationResult is_valid True with only warnings."""
        issue = IntentIssue(
            issue_id="I002",
            category=FailureCategory.OTHER,
            severity="warning",
            message="warn",
        )
        ivr = IntentValidationResult(issues=[issue])
        assert ivr.is_valid is True

    def test_round_trip(self):
        """IntentValidationResult to_dict/from_dict round trip."""
        issue = IntentIssue(
            issue_id="I001",
            category=FailureCategory.OTHER,
            severity="error",
            message="err",
        )
        ivr = IntentValidationResult(issues=[issue])
        rebuilt = IntentValidationResult.from_dict(ivr.to_dict())
        assert len(rebuilt.issues) == 1
        assert rebuilt.issues[0].issue_id == "I001"

    def test_info_severity_issue_dropped_on_construction(self):
        """Info-severity issues are filtered out at construction."""
        issues = [
            IntentIssue(
                issue_id="I001",
                category=FailureCategory.OTHER,
                severity="error",
                message="err",
            ),
            IntentIssue(
                issue_id="I002",
                category=FailureCategory.OTHER,
                severity="warning",
                message="warn",
            ),
            IntentIssue(
                issue_id="I003",
                category=FailureCategory.OTHER,
                severity="info",
                message="note",
            ),
        ]
        ivr = IntentValidationResult(issues=issues)
        severities = sorted(i.severity for i in ivr.issues)
        assert severities == ["error", "warning"]
        assert all(i.issue_id != "I003" for i in ivr.issues)

    def test_info_severity_issue_dropped_on_from_dict(self):
        """from_dict round-trips drop info-severity issues via __post_init__."""
        raw = {
            "issues": [
                {
                    "issue_id": "I001",
                    "category": "c",
                    "severity": "error",
                    "message": "err",
                },
                {
                    "issue_id": "I003",
                    "category": "c",
                    "severity": "info",
                    "message": "note",
                },
            ]
        }
        ivr = IntentValidationResult.from_dict(raw)
        assert len(ivr.issues) == 1
        assert ivr.issues[0].severity == "error"

    def test_unknown_severity_dropped(self):
        """Any severity other than 'error' or 'warning' is dropped."""
        ivr = IntentValidationResult(
            issues=[
                IntentIssue(
                    issue_id="I001",
                    category=FailureCategory.OTHER,
                    severity="debug",
                    message="x",
                ),
                IntentIssue(
                    issue_id="I002",
                    category=FailureCategory.OTHER,
                    severity="",
                    message="y",
                ),
            ]
        )
        assert ivr.issues == []


class TestTemplateStats:
    """Tests for TemplateStats."""

    def test_defaults(self):
        """TemplateStats defaults to zero."""
        ts = TemplateStats()
        assert ts.accept == 0
        assert ts.reject == 0

    def test_round_trip(self):
        """TemplateStats to_dict/from_dict round trip."""
        ts = TemplateStats(accept=5, reject=2)
        rebuilt = TemplateStats.from_dict(ts.to_dict())
        assert rebuilt.accept == 5
        assert rebuilt.reject == 2


class TestQSimSkeleton:
    """Tests for QSimSkeleton."""

    def test_fields(self):
        """QSimSkeleton stores skeleton fields."""
        sk = QSimSkeleton(
            tables=["orders"],
            has_aggregation=True,
            num_where=2,
            num_groupby=1,
            has_orderby=True,
            num_having=0,
        )
        assert sk.tables == ["orders"]
        assert sk.has_aggregation is True
        assert sk.num_where == 2

    def test_defaults(self):
        """QSimSkeleton has defaults for optional fields."""
        sk = QSimSkeleton(
            tables=["t"],
            has_aggregation=False,
            num_where=0,
            num_groupby=0,
            has_orderby=False,
            num_having=0,
        )
        assert sk.has_distinct is False
        assert sk.has_expr_comparison is False


class TestQSimSummary:
    """Tests for QSimSummary."""

    def test_round_trip(self):
        """QSimSummary to_dict/from_dict round trip."""
        qs = QSimSummary(version=4, num_intents=20, num_questions=100, seed=42)
        rebuilt = QSimSummary.from_dict(qs.to_dict())
        assert rebuilt.version == 4
        assert rebuilt.seed == 42


class TestSeedWarmupSummary:
    """Tests for SeedWarmupSummary."""

    def test_fields(self):
        """SeedWarmupSummary stores run statistics."""
        ss = SeedWarmupSummary(version=1, total=10, success=8, failed=2, success_rate=0.8)
        assert ss.success_rate == 0.8
        assert ss.total == 10
        assert ss.unique_prompts == 0
        assert ss.deduped_prompts_count == 0
        assert ss.gold_prompts_count == 0


class TestSchemaLimits:
    """Tests for SchemaLimits."""

    def test_fields(self):
        """SchemaLimits stores computed limits."""
        sl = SchemaLimits(max_where_predicates=4, max_groupby=2, max_tables=3)
        assert sl.max_where_predicates == 4
        assert sl.max_tables == 3


class TestSkeletonPool:
    """Tests for SkeletonPool."""

    def test_empty_pool(self):
        """Empty SkeletonPool has zero-length keys."""
        pool = SkeletonPool(
            tier_a_by_table_set={},
            tier_b_by_table_set={},
            tier_c_by_table_set={},
            table_set_keys=[],
            tier_a_indices={},
            tier_b_indices={},
            tier_c_indices={},
        )
        assert pool.table_set_keys == []
        assert pool.current_table_idx == 0

    def test_populated_pool(self):
        """SkeletonPool stores tier data correctly."""
        skel = QSimSkeleton(
            tables=["orders"],
            has_aggregation=False,
            num_where=0,
            num_groupby=0,
            has_orderby=False,
            num_having=0,
            has_distinct=False,
            has_expr_comparison=False,
        )
        pool = SkeletonPool(
            tier_a_by_table_set={"orders": [skel]},
            tier_b_by_table_set={"orders": []},
            tier_c_by_table_set={"orders": []},
            table_set_keys=["orders"],
            tier_a_indices={"orders": 0},
            tier_b_indices={"orders": 0},
            tier_c_indices={"orders": 0},
        )
        assert len(pool.tier_a_by_table_set["orders"]) == 1
        assert pool.table_set_keys == ["orders"]

    def test_current_table_idx_default(self):
        """Default current_table_idx is 0."""
        pool = SkeletonPool(
            tier_a_by_table_set={},
            tier_b_by_table_set={},
            tier_c_by_table_set={},
            table_set_keys=[],
            tier_a_indices={},
            tier_b_indices={},
            tier_c_indices={},
        )
        assert pool.current_table_idx == 0


class TestWindowCaseRegistry:
    """Tests for window/case registry serialization on intents."""

    def test_runtime_intent_registry_round_trip_dict(self, minimal_intent: RuntimeIntent) -> None:
        """to_dict/from_dict preserves window_registry and the bare- registry-id select column."""
        ob = OrderByCol(expr=NormalizedExpr.from_column("orders.amount"), direction="ASC")
        wspec = WindowSpec(function="row_number", order_by=[ob])
        intent = RuntimeIntent(
            tables=minimal_intent.tables,
            grain=minimal_intent.grain,
            select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=wspec,
                )
            ],
            case_registry=[],
        )
        d = intent.to_dict()
        back = RuntimeIntent.from_dict(d)
        assert len(back.window_registry) == 1
        assert back.window_registry[0].registry_id == "w01"
        assert back.select_cols[0].expr.column_ref == "w01"
        assert back.select_cols[0].expr.registry_ref() == "w01"

    def test_duplicate_case_registry_id_emits_issue(self) -> None:
        """validate_scope_registries flags duplicate case ids."""
        from aetherdialect._validation_shape import validate_scope_registries

        cr = CaseRegistryStep(
            registry_id="c01",
            case_when=CaseWhenExpr(
                branches=[],
            ),
        )
        issues = validate_scope_registries(
            context="main",
            window_registry=[],
            case_registry=[cr, cr],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
        )
        assert any("duplicate case registry_id" in i.message for i in issues)

    def test_case_registry_from_dict_accepts_list_case_when(self) -> None:
        """``case_when`` may be a bare branch list (LLM shorthand)."""
        step = CaseRegistryStep.from_dict(
            {
                "registry_id": "c01",
                "case_when": [
                    {
                        "condition": {
                            "left_expr": "orders.amount",
                            "op": ">",
                            "value_type": "number",
                            "value": 0,
                        },
                        "result": "1",
                    }
                ],
            }
        )
        assert step.registry_id == "c01"
        assert len(step.case_when.branches) == 1


class TestCaseWhenExprConditionScope:
    """``CaseWhenExpr.condition_scope`` round-trips through to_dict/from_dict."""

    def test_default_scope_omitted_in_to_dict(self) -> None:
        cw = CaseWhenExpr(branches=[])
        d = cw.to_dict()
        assert "condition_scope" not in d

    def test_having_scope_round_trip(self) -> None:
        cw = CaseWhenExpr(branches=[], condition_scope="having")
        d = cw.to_dict()
        assert d.get("condition_scope") == "having"
        back = CaseWhenExpr.from_dict(d)
        assert back.condition_scope == "having"

    def test_unknown_scope_falls_back_to_where(self) -> None:
        back = CaseWhenExpr.from_dict({"branches": [], "condition_scope": "bogus"})
        assert back.condition_scope == "where"

    def test_signature_key_includes_scope(self) -> None:
        a = CaseWhenExpr(branches=[], condition_scope="where")
        b = CaseWhenExpr(branches=[], condition_scope="having")
        assert a.signature_key != b.signature_key


class TestCaseWhenStringResultExpr:
    """String ``result`` / ``else_result`` map to literals unless ``table.column`` shaped."""

    def test_branch_bare_label_is_string_literal(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": WhereParam.prompt_example_dict(),
                "result": "premium",
            }
        )
        assert br.result.string_literal == "premium"
        assert not br.result.column_ref

    def test_branch_qualified_ref_is_column(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": WhereParam.prompt_example_dict(),
                "result": "table.column",
            }
        )
        assert br.result.column_ref == "table.column"
        assert not br.result.string_literal

    def test_branch_dict_result_unchanged(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": WhereParam.prompt_example_dict(),
                "result": {"column_ref": "table.other_column"},
            }
        )
        assert br.result.column_ref == "table.other_column"

    def test_branch_literal_string_wins_over_result(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": WhereParam.prompt_example_dict(),
                "literal_string": "from_literal",
                "result": "ignored",
            }
        )
        assert br.result.string_literal == "from_literal"

    def test_else_bare_string_is_literal(self) -> None:
        cw = CaseWhenExpr.from_dict(
            {
                "branches": [],
                "else_result": "standard",
            }
        )
        assert cw.else_result is not None
        assert cw.else_result.string_literal == "standard"

    def test_else_qualified_string_is_column(self) -> None:
        cw = CaseWhenExpr.from_dict(
            {
                "branches": [],
                "else_result": "film.title",
            }
        )
        assert cw.else_result is not None
        assert cw.else_result.column_ref == "film.title"


class TestSchemaContextDenyParsing:
    """Tests for EngineContext deny_columns parsing (``table.column`` and ``*.column`` entries only)."""

    def test_qualified_entry_normalized(self):
        ctx = EngineContext(deny_columns=frozenset({"Contacts.Email"}))
        assert ctx.deny_columns == frozenset({"contacts.email"})
        assert ctx.qualified_denies() == frozenset({("contacts", "email")})
        assert ctx.glob_column_denies() == frozenset()

    def test_source_qualified_entry_normalized(self):
        ctx = EngineContext(deny_columns=frozenset({"alpha.contacts.email"}))
        assert ctx.deny_columns == frozenset({"contacts.email"})
        assert ctx.qualified_denies() == frozenset({("contacts", "email")})

    def test_glob_entry_normalized(self):
        ctx = EngineContext(deny_columns=frozenset({"*.Email"}))
        assert ctx.deny_columns == frozenset({"*.email"})
        assert ctx.qualified_denies() == frozenset()
        assert ctx.glob_column_denies() == frozenset({"email"})

    def test_mixed_qualified_and_glob(self):
        ctx = EngineContext(deny_columns=frozenset({"*.email", "contacts.ssn"}))
        assert ctx.qualified_denies() == frozenset({("contacts", "ssn")})
        assert ctx.glob_column_denies() == frozenset({"email"})

    def test_bare_token_rejected(self):
        with pytest.raises(ConfigError):
            EngineContext(deny_columns=frozenset({"email"}))

    def test_too_many_dots_rejected(self):
        with pytest.raises(ConfigError):
            EngineContext(deny_columns=frozenset({"a.b.c.d"}))

    def test_empty_entry_rejected(self):
        try:
            EngineContext(deny_columns=frozenset({""}))
        except (ConfigError, ValueError):
            return
        raise AssertionError("expected ConfigError")

    def test_allow_objects_collision_only_for_qualified(self):
        with pytest.raises(ConfigError):
            EngineContext(
                allow_objects=frozenset({"contacts"}),
                deny_columns=frozenset({"contacts.email"}),
            )
        ctx = EngineContext(allow_objects=frozenset({"contacts"}), deny_columns=frozenset({"*.email"}))
        assert "contacts" in ctx.allow_objects
        assert ctx.glob_column_denies() == frozenset({"email"})


class TestPipelineFeatureTags:
    def test_pipeline_feature_tags_include_qsim_and_expansion_only(self) -> None:
        from aetherdialect._contracts_core import PIPELINE_FEATURE_TAGS

        assert "multi_cte_chain" in PIPELINE_FEATURE_TAGS
        assert "in_list" in PIPELINE_FEATURE_TAGS
        assert "cte_wrap" in PIPELINE_FEATURE_TAGS

    def test_feasible_features_respects_date_columns(self) -> None:
        from aetherdialect._contracts_base import DatabaseFeatureCapability
        from aetherdialect._contracts_core import PipelineFeatureSpec

        cap_no_dates = DatabaseFeatureCapability(
            table_count=2,
            fk_edge_count=1,
            has_numeric_measures=True,
            has_date_columns=False,
            has_array_columns=False,
            has_categorical_columns=True,
            max_tables_on_any_join_path=2,
            max_fk_chain_depth=1,
            has_self_referential_fk=False,
            tables_supporting_self_join=frozenset(),
            has_window_capable_table_sets=True,
            aggregatable_columns_by_table={},
            date_columns_by_table={},
            array_columns_by_table={},
        )
        feasible = PipelineFeatureSpec.feasible_features_for_capability(cap_no_dates)
        assert "date_window_filter" not in feasible
        assert "in_list" in feasible

    def test_detect_intent_features_finds_having_and_distinct(self) -> None:
        from aetherdialect._contracts_core import SeedWarmupIntent

        intent = SeedWarmupIntent.from_dict(
            {
                "intent_id": "feat_test",
                "tables": ["t1"],
                "grain": "grouped",
                "select_cols": [{"expr": "count(*)", "is_aggregated": True}],
                "group_by_cols": ["t1.id"],
                "having": {
                    "op": "and",
                    "predicates": [
                        {
                            "left_expr": "count(*)",
                            "op": ">",
                            "value_type": "number",
                            "param_key": "p1",
                        }
                    ],
                },
                "distinct_select_index": 0,
            }
        )
        tags = intent.to_runtime_intent().detect_features()
        assert "having_aggregate_compare" in tags
        assert "distinct_select" in tags
