"""Combinatorial tests for filter_group / bool_op coercion, sort, and WHERE/HAVING rendering."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aetherdialect._contracts_base import SchemaGraph
from aetherdialect._contracts_core import (
    CaseWhenBranch,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._dialect import PostgresDialect
from aetherdialect._intent_expr import decompose_between_params
from aetherdialect._intent_process import _compute_filters_similarity
from aetherdialect._intent_repair import decompose_in_not_in_filters
from aetherdialect._intent_resolve import (
    _dedup_filters,
    coerce_filter_group_mode,
    normalize_filters_havings,
)
from aetherdialect._sql_gen import _build_deterministic_select_block
from aetherdialect._utils import _normalize_cte_steps
from aetherdialect._validation_execute import validate_semantics
from aetherdialect._validation_semantic import (
    validate_predicate_bool_op_filter_group_hints,
)


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _where_after_pipeline(filters: list[FilterParam]) -> str:
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=filters,
        having_param=[],
    )
    intent = decompose_between_params(intent)
    intent = decompose_in_not_in_filters(intent)
    intent = coerce_filter_group_mode(intent)
    intent = normalize_filters_havings(intent)
    sql = _build_deterministic_select_block(
        intent.select_cols,
        intent.tables,
        intent.group_by_cols,
        intent.order_by_cols,
        intent.filters_param,
        intent.having_param,
        intent.limit,
        intent.grain,
        _pg(),
    )
    for line in sql.split("\n"):
        if line.startswith("WHERE "):
            return line[6:].strip()
    return ""


def _having_after_pipeline(having: list[HavingParam]) -> str:
    intent = RuntimeIntent(
        tables=["t"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t.x"])], sub_groups=[])),
            SelectCol(
                expr=NormalizedExpr(
                    add_groups=[MulGroup(multiply=["t.id"], agg_func="count")],
                    sub_groups=[],
                )
            ),
        ],
        group_by_cols=[NormalizedExpr(add_groups=[MulGroup(multiply=["t.x"])], sub_groups=[])],
        order_by_cols=[],
        filters_param=[],
        having_param=having,
    )
    intent = decompose_between_params(intent)
    intent = decompose_in_not_in_filters(intent)
    intent = coerce_filter_group_mode(intent)
    intent = normalize_filters_havings(intent)
    sql = _build_deterministic_select_block(
        intent.select_cols,
        intent.tables,
        intent.group_by_cols,
        intent.order_by_cols,
        intent.filters_param,
        intent.having_param,
        intent.limit,
        intent.grain,
        _pg(),
    )
    for line in sql.split("\n"):
        if line.startswith("HAVING "):
            return line[7:].strip()
    return ""


def _unwrap_outer_parens(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


class TestBoolOpWhereMatrix:
    def test_empty(self) -> None:
        assert _where_after_pipeline([]) == ""

    def test_one_flat(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        assert _where_after_pipeline([fp]) == 'LOWER("t"."a") = :p1'

    def test_one_grouped(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
            filter_group=1,
        )
        assert _where_after_pipeline([fp]) == 'LOWER("t"."a") = :p1'

    def test_flat_and(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2 AND LOWER("t"."c") = :p3'

    def test_flat_mixed_forward(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                bool_op="AND",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                bool_op="OR",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                bool_op="AND",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.d"),
                op="=",
                param_key="p4",
                value_type="string",
                bool_op="AND",
            ),
        ]
        got = _unwrap_outer_parens(_where_after_pipeline(fs))
        assert got == ('LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2 OR LOWER("t"."c") = :p3 AND LOWER("t"."d") = :p4')

    def test_flat_or_backward_promote(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                bool_op="OR",
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2'

    def test_flat_or_backward_chain(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                bool_op="OR",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                bool_op="OR",
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2 OR LOWER("t"."c") = :p3'

    def test_two_disjuncts(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2'

    def test_two_in_one_group(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2'

    def test_four_or_of_and(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=2,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.d"),
                op="=",
                param_key="p4",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert (
            _where_after_pipeline(fs)
            == '(LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2) OR (LOWER("t"."c") = :p3 AND LOWER("t"."d") = :p4)'
        )

    def test_or_of_and_3disj(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=2,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.d"),
                op="=",
                param_key="p4",
                value_type="string",
                filter_group=3,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.e"),
                op="=",
                param_key="p5",
                value_type="string",
                filter_group=3,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.f"),
                op="=",
                param_key="p6",
                value_type="string",
                filter_group=3,
            ),
        ]
        assert _where_after_pipeline(fs) == (
            '(LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2) OR LOWER("t"."c") = :p3 OR '
            '(LOWER("t"."d") = :p4 AND LOWER("t"."e") = :p5 AND LOWER("t"."f") = :p6)'
        )

    def test_interleaved_groups(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.d"),
                op="=",
                param_key="p4",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert (
            _where_after_pipeline(fs)
            == '(LOWER("t"."a") = :p1 AND LOWER("t"."c") = :p3) OR (LOWER("t"."b") = :p2 AND LOWER("t"."d") = :p4)'
        )

    def test_single_group_many_rows(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=1,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 AND LOWER("t"."b") = :p2 AND LOWER("t"."c") = :p3'

    def test_mixed_mode_coerce(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2 OR LOWER("t"."c") = :p3'

    def test_fg0_used(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=0,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=1,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2'

    def test_fg_string_int(self) -> None:
        a = FilterParam.from_dict(
            {
                "left_expr": "t.a",
                "op": "=",
                "value_type": "string",
                "param_key": "p1",
                "filter_group": "1",
            }
        )
        b = FilterParam.from_dict(
            {
                "left_expr": "t.b",
                "op": "=",
                "value_type": "string",
                "param_key": "p2",
                "filter_group": "2",
            }
        )
        assert _where_after_pipeline([a, b]) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2'

    def test_fg_negative_clamps(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=-1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = :p1 OR LOWER("t"."b") = :p2'

    def test_between_flat(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.len"),
            op="between",
            param_key="p1",
            value_type="integer",
            raw_value=[10, 99],
        )
        w = _where_after_pipeline([fp])
        assert ">=" in w or "<=" in w
        assert "len" in w

    def test_between_in_group(self) -> None:
        fs = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                left_expr=NormalizedExpr.from_column("t.len"),
                op="between",
                param_key="p2",
                value_type="integer",
                raw_value=[1, 5],
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.d"),
                op="=",
                param_key="p3",
                value_type="string",
                filter_group=2,
            ),
        ]
        w = _where_after_pipeline(fs)
        assert w.startswith("(")
        assert " OR " in w
        assert "LOWER" in w or "t" in w

    def test_in_flat(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.rating"),
            op="in",
            param_key="p1",
            value_type="string",
            raw_value=["R", "PG"],
        )
        w = _where_after_pipeline([fp])
        assert "IN" in w or "OR" in w

    def test_in_grouped_native(self) -> None:
        fs = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("t.rating"),
                op="in",
                param_key="p1",
                value_type="string",
                raw_value=["R", "PG"],
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.len"),
                op=">",
                param_key="p2",
                value_type="integer",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.rating"),
                op="=",
                param_key="p3",
                value_type="string",
                raw_value="G",
                filter_group=2,
            ),
        ]
        w = _where_after_pipeline(fs)
        assert "IN (:p1)" in w
        assert " OR " in w

    def test_not_in_grouped_native(self) -> None:
        fs = [
            FilterParam(
                left_expr=NormalizedExpr.from_column("t.rating"),
                op="not in",
                param_key="p1",
                value_type="string",
                raw_value=["X"],
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
        ]
        w = _where_after_pipeline(fs)
        assert "NOT IN (:p1)" in w
        assert " OR " in w


class TestHavingOrOfAnd:
    def test_having_or_of_and(self) -> None:
        left_a = NormalizedExpr(add_groups=[MulGroup(multiply=["t.id"], agg_func="count")], sub_groups=[])
        left_b = NormalizedExpr(add_groups=[MulGroup(multiply=["t.amt"], agg_func="sum")], sub_groups=[])
        left_c = NormalizedExpr(add_groups=[MulGroup(multiply=["t.id"], agg_func="count")], sub_groups=[])
        left_d = NormalizedExpr(add_groups=[MulGroup(multiply=["t.amt"], agg_func="sum")], sub_groups=[])
        hs = [
            HavingParam(
                left_expr=left_a,
                op=">",
                param_key="h1",
                value_type="integer",
                filter_group=1,
            ),
            HavingParam(
                left_expr=left_b,
                op=">",
                param_key="h2",
                value_type="number",
                filter_group=1,
            ),
            HavingParam(
                left_expr=left_c,
                op="<",
                param_key="h3",
                value_type="integer",
                filter_group=2,
            ),
            HavingParam(
                left_expr=left_d,
                op="<",
                param_key="h4",
                value_type="number",
                filter_group=2,
            ),
        ]
        got = _having_after_pipeline(hs)
        assert got.startswith("(")
        assert " OR " in got


class TestCteCaseDedupQsimWarnings:
    def test_cte_preserves_grouping(self) -> None:
        cte = RuntimeCteStep(
            cte_name="c1",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            output_columns=["id"],
            filters_param=[
                FilterParam(
                    NormalizedExpr.from_column("t.a"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                    filter_group=1,
                    bool_op="OR",
                ),
                FilterParam(
                    NormalizedExpr.from_column("t.b"),
                    op="=",
                    param_key="p2",
                    value_type="string",
                    filter_group=2,
                ),
            ],
            having_param=[],
        )
        out = _normalize_cte_steps([cte])
        assert len(out) == 1
        f0, f1 = out[0].filters_param
        assert f0.filter_group == 1 and f1.filter_group == 2
        assert f0.bool_op == "OR"

    def test_case_branch_strips_grouping(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": {
                    "left_expr": "t.x",
                    "op": "=",
                    "value_type": "string",
                    "param_key": "p1",
                    "bool_op": "OR",
                    "filter_group": 7,
                },
                "literal_string": "yes",
            }
        )
        assert br.condition.bool_op == "AND"
        assert br.condition.filter_group is None

    def test_dedup_keeps_distinct_groups(self) -> None:
        fp = FilterParam(
            NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        a = replace(fp, filter_group=1)
        b = replace(fp, filter_group=2)
        out = _dedup_filters([a, b])
        assert len(out) == 2

    def test_dedup_collapses_within_group(self) -> None:
        fp = FilterParam(
            NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        a = replace(fp, filter_group=1, bool_op="AND")
        b = replace(fp, filter_group=1, bool_op="AND")
        out = _dedup_filters([a, b])
        assert len(out) == 1

    def test_qsim_grouped_match(self) -> None:
        f1 = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
        ]
        f2 = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p9",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p8",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert _compute_filters_similarity(f1, f2) == pytest.approx(1.0)

    def test_qsim_grouped_vs_old_template(self) -> None:
        template = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                bool_op="OR",
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                bool_op="AND",
            ),
        ]
        intent = [
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
            ),
            FilterParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
                filter_group=2,
            ),
        ]
        assert _compute_filters_similarity(template, intent) == pytest.approx(1.0)

    def test_warning_or_with_no_signal(self) -> None:
        issues = validate_predicate_bool_op_filter_group_hints(
            "Show rows where name is Alice or Bob",
            [
                FilterParam(
                    NormalizedExpr.from_column("t.name"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                ),
                FilterParam(
                    NormalizedExpr.from_column("t.name"),
                    op="=",
                    param_key="p2",
                    value_type="string",
                ),
            ],
            [],
        )
        ids = {i.issue_id for i in issues}
        assert "nl_or_without_predicate_signal" in ids

    def test_warning_both_signals(self) -> None:
        issues = validate_predicate_bool_op_filter_group_hints(
            "",
            [
                FilterParam(
                    NormalizedExpr.from_column("t.a"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                    filter_group=1,
                    bool_op="OR",
                )
            ],
            [],
        )
        assert any(i.issue_id.startswith("filter_both_bool_signals") for i in issues)


def test_validate_semantics_includes_bool_hints() -> None:
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[
            FilterParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
                filter_group=1,
                bool_op="OR",
            )
        ],
        having_param=[],
        natural_language="",
    )
    res = validate_semantics(intent, SchemaGraph(tables={}, join_paths_multi={}))
    ids = {i.issue_id for i in res.issues}
    assert "filter_both_bool_signals_0" in ids
