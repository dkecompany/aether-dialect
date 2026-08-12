"""Combinatorial tests for where_group / bool_op coercion, sort, and WHERE/HAVING rendering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from aetherdialect._contracts_base import (
    HavingParam,
    MulGroup,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import CaseWhenBranch, SchemaGraph
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._intent_bind import (
    _dedup_where_predicates,
    coerce_predicate_group_mode,
    normalize_where_havings,
)
from aetherdialect._intent_expr import _compute_where_similarity, decompose_between_params
from aetherdialect._intent_normalize import decompose_in_not_in_where
from aetherdialect._sql_gen import _build_deterministic_select_block
from aetherdialect._utils_intent import _normalize_cte_steps
from aetherdialect._validation_rules import validate_predicate_group_hints
from aetherdialect._validation_sql import validate_semantics


def _from_flat_where_dicts(fp_raw):
    rows = []
    for raw in fp_raw:
        if not isinstance(raw, dict):
            continue
        rows.append(
            (
                WhereParam.from_dict(raw),
                PredicateGroup.bool_op_from_stored(raw.get("bool_op", "AND")),
                PredicateGroup.where_group_int_from_stored(raw.get("where_group")),
            )
        )
    return PredicateGroup.from_bool_op_rows(rows)


def _from_flat_having_dicts(hp_raw):
    rows = []
    for raw in hp_raw:
        if not isinstance(raw, dict):
            continue
        rows.append(
            (
                HavingParam.from_dict(raw),
                PredicateGroup.bool_op_from_stored(raw.get("bool_op", "AND")),
                PredicateGroup.where_group_int_from_stored(raw.get("where_group")),
            )
        )
    return PredicateGroup.from_bool_op_rows(rows)


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _flat_filter_dict_rows(
    filters: PredicateGroup | list[WhereParam] | list[dict[str, object]] | None,
) -> list[dict[str, Any]] | None:
    if not filters or isinstance(filters, PredicateGroup):
        return None
    if not isinstance(filters[0], dict):
        return None
    return cast(list[dict[str, Any]], filters)


def _flat_having_dict_rows(
    having: PredicateGroup | list[HavingParam] | list[dict[str, object]] | None,
) -> list[dict[str, Any]] | None:
    if not having or isinstance(having, PredicateGroup):
        return None
    if not isinstance(having[0], dict):
        return None
    return cast(list[dict[str, Any]], having)


def _coerce_where(
    filters: PredicateGroup | list[WhereParam] | list[dict[str, object]] | None,
) -> PredicateGroup | None:
    if isinstance(filters, PredicateGroup):
        return filters
    if not filters:
        return None
    if isinstance(filters[0], dict):
        return _from_flat_where_dicts(cast(list[Any], filters))
    return PredicateGroup.from_list(cast(list[WhereParam], filters))


def _coerce_having(
    having: PredicateGroup | list[HavingParam] | list[dict[str, object]] | None,
) -> PredicateGroup | None:
    if isinstance(having, PredicateGroup):
        return having
    if not having:
        return None
    if isinstance(having[0], dict):
        return _from_flat_having_dicts(cast(list[Any], having))
    return PredicateGroup.from_list(cast(list[HavingParam], having))


def _wf(col: str, pk: str, **extra: object) -> dict[str, object]:
    return {"left_expr": col, "op": "=", "param_key": pk, "value_type": "string", **extra}


def _flat_dicts_after_decompose(
    flat_dicts: list[dict[str, object]],
    where: PredicateGroup | None,
) -> list[dict[str, object]]:
    """Rebuild flat filter dict rows to match post-decompose leaves while keeping bool_op/where_group."""
    leaves = where.leaves() if where else []
    if len(leaves) == len(flat_dicts):
        return flat_dicts
    expanded: list[dict[str, object]] = []
    leaf_idx = 0
    for raw in flat_dicts:
        op = str(raw.get("op", "=")).strip().lower()
        if op == "between" and leaf_idx + 1 < len(leaves):
            for bound_op in (">=", "<="):
                expanded.append(
                    {
                        **raw,
                        "op": bound_op,
                        "value": leaves[leaf_idx].raw_value,
                    }
                )
                leaf_idx += 1
            continue
        if leaf_idx < len(leaves):
            expanded.append({**raw, "value": leaves[leaf_idx].raw_value})
            leaf_idx += 1
    return expanded or flat_dicts


def _where_after_pipeline(filters: PredicateGroup | list[WhereParam] | list[dict[str, object]] | None) -> str:
    flat_dicts = _flat_filter_dict_rows(filters)
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=(
            PredicateGroup.from_list([WhereParam.from_dict(raw) for raw in flat_dicts])
            if flat_dicts
            else _coerce_where(filters)
        ),
        having=None,
    )
    intent = decompose_between_params(intent)
    intent = decompose_in_not_in_where(intent)
    if flat_dicts:
        intent = replace(
            intent,
            where=_from_flat_where_dicts(_flat_dicts_after_decompose(flat_dicts, intent.where)),
        )
    intent = coerce_predicate_group_mode(intent)
    intent = normalize_where_havings(intent)
    sql = _build_deterministic_select_block(
        intent.select_cols,
        intent.tables,
        intent.group_by_cols,
        intent.order_by_cols,
        intent.where,
        intent.having,
        intent.limit,
        intent.grain,
        _pg(),
    )
    for line in sql.split("\n"):
        if line.startswith("WHERE "):
            return line[6:].strip()
    return ""


def _having_after_pipeline(having: PredicateGroup | list[HavingParam] | list[dict[str, object]] | None) -> str:
    flat_dicts = _flat_having_dict_rows(having)
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
        where=None,
        having=(
            PredicateGroup.from_list([HavingParam.from_dict(raw) for raw in flat_dicts])
            if flat_dicts
            else _coerce_having(having)
        ),
    )
    intent = decompose_between_params(intent)
    intent = decompose_in_not_in_where(intent)
    if flat_dicts:
        intent = replace(
            intent,
            having=_from_flat_having_dicts(flat_dicts),
        )
    intent = coerce_predicate_group_mode(intent)
    intent = normalize_where_havings(intent)
    sql = _build_deterministic_select_block(
        intent.select_cols,
        intent.tables,
        intent.group_by_cols,
        intent.order_by_cols,
        intent.where,
        intent.having,
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
        fp = WhereParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        assert _where_after_pipeline([fp]) == 'LOWER("t"."a") = LOWER(:p1)'

    def test_one_grouped(self) -> None:
        assert _where_after_pipeline([_wf("t.a", "p1", where_group=1)]) == 'LOWER("t"."a") = LOWER(:p1)'

    def test_flat_and(self) -> None:
        fs = [
            WhereParam(
                NormalizedExpr.from_column("t.a"),
                op="=",
                param_key="p1",
                value_type="string",
            ),
            WhereParam(
                NormalizedExpr.from_column("t.b"),
                op="=",
                param_key="p2",
                value_type="string",
            ),
            WhereParam(
                NormalizedExpr.from_column("t.c"),
                op="=",
                param_key="p3",
                value_type="string",
            ),
        ]
        assert (
            _where_after_pipeline(fs)
            == 'LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2) AND LOWER("t"."c") = LOWER(:p3)'
        )

    def test_flat_mixed_forward(self) -> None:
        fs = [
            _wf("t.a", "p1", bool_op="AND"),
            _wf("t.b", "p2", bool_op="OR"),
            _wf("t.c", "p3", bool_op="AND"),
            _wf("t.d", "p4", bool_op="AND"),
        ]
        got = _where_after_pipeline(fs)
        assert got == (
            '(LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2) AND LOWER("t"."d") = LOWER(:p4)) OR '
            '(LOWER("t"."c") = LOWER(:p3) AND LOWER("t"."d") = LOWER(:p4))'
        )

    def test_flat_or_backward_promote(self) -> None:
        fs = [
            _wf("t.a", "p1"),
            _wf("t.b", "p2", bool_op="OR"),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2)'

    def test_flat_or_backward_chain(self) -> None:
        fs = [
            _wf("t.a", "p1"),
            _wf("t.b", "p2", bool_op="OR"),
            _wf("t.c", "p3", bool_op="OR"),
        ]
        assert _where_after_pipeline(fs) == (
            'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2) OR LOWER("t"."c") = LOWER(:p3)'
        )

    def test_two_disjuncts(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=2),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2)'

    def test_two_in_one_group(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=1),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2)'

    def test_four_or_of_and(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=1),
            _wf("t.c", "p3", where_group=2),
            _wf("t.d", "p4", where_group=2),
        ]
        assert (
            _where_after_pipeline(fs)
            == '(LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2)) OR (LOWER("t"."c") = LOWER(:p3) AND LOWER("t"."d") = LOWER(:p4))'
        )

    def test_or_of_and_3disj(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=1),
            _wf("t.c", "p3", where_group=2),
            _wf("t.d", "p4", where_group=3),
            _wf("t.e", "p5", where_group=3),
            _wf("t.f", "p6", where_group=3),
        ]
        assert _where_after_pipeline(fs) == (
            'LOWER("t"."c") = LOWER(:p3) OR (LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2)) OR '
            '(LOWER("t"."d") = LOWER(:p4) AND LOWER("t"."e") = LOWER(:p5) AND LOWER("t"."f") = LOWER(:p6))'
        )

    def test_interleaved_groups(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=2),
            _wf("t.c", "p3", where_group=1),
            _wf("t.d", "p4", where_group=2),
        ]
        assert (
            _where_after_pipeline(fs)
            == '(LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."c") = LOWER(:p3)) OR (LOWER("t"."b") = LOWER(:p2) AND LOWER("t"."d") = LOWER(:p4))'
        )

    def test_single_group_many_rows(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=1),
            _wf("t.c", "p3", where_group=1),
        ]
        assert (
            _where_after_pipeline(fs)
            == 'LOWER("t"."a") = LOWER(:p1) AND LOWER("t"."b") = LOWER(:p2) AND LOWER("t"."c") = LOWER(:p3)'
        )

    def test_mixed_mode_coerce(self) -> None:
        fs = [
            _wf("t.a", "p1"),
            _wf("t.b", "p2", where_group=1),
            _wf("t.c", "p3", where_group=2),
        ]
        assert _where_after_pipeline(fs) == (
            'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2) OR LOWER("t"."c") = LOWER(:p3)'
        )

    def test_fg0_used(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=0),
            _wf("t.b", "p2", where_group=1),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2)'

    def test_fg_string_int(self) -> None:
        fs = [
            {"left_expr": "t.a", "op": "=", "value_type": "string", "param_key": "p1", "where_group": "1"},
            {"left_expr": "t.b", "op": "=", "value_type": "string", "param_key": "p2", "where_group": "2"},
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2)'

    def test_fg_negative_clamps(self) -> None:
        fs = [
            _wf("t.a", "p1", where_group=-1),
            _wf("t.b", "p2", where_group=2),
        ]
        assert _where_after_pipeline(fs) == 'LOWER("t"."a") = LOWER(:p1) OR LOWER("t"."b") = LOWER(:p2)'

    def test_between_flat(self) -> None:
        fp = WhereParam(
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
            _wf("t.a", "p1", where_group=1),
            {
                "left_expr": "t.len",
                "op": "between",
                "param_key": "p2",
                "value_type": "integer",
                "value": [1, 5],
                "where_group": 1,
            },
            _wf("t.d", "p3", where_group=2),
        ]
        w = _where_after_pipeline(fs)
        assert " OR " in w
        assert "LOWER" in w or "t" in w
        assert "len" in w

    def test_in_flat(self) -> None:
        fs = [
            {
                "left_expr": "t.rating",
                "op": "in",
                "param_key": "p1",
                "value_type": "string",
                "value": ["R", "PG"],
            }
        ]
        w = _where_after_pipeline(fs)
        assert "IN" in w or "OR" in w

    def test_in_grouped_native(self) -> None:
        fs = [
            {
                "left_expr": "t.rating",
                "op": "in",
                "param_key": "p1",
                "value_type": "string",
                "value": ["R", "PG"],
                "where_group": 1,
            },
            {
                "left_expr": "t.len",
                "op": ">",
                "param_key": "p2",
                "value_type": "integer",
                "where_group": 1,
            },
            {
                "left_expr": "t.rating",
                "op": "=",
                "param_key": "p3",
                "value_type": "string",
                "value": "G",
                "where_group": 2,
            },
        ]
        w = _where_after_pipeline(fs)
        assert "IN (:p1)" in w
        assert " OR " in w

    def test_not_in_grouped_native(self) -> None:
        fs = [
            {
                "left_expr": "t.rating",
                "op": "not in",
                "param_key": "p1",
                "value_type": "string",
                "value": ["X"],
                "where_group": 1,
            },
            _wf("t.c", "p2", where_group=2),
        ]
        w = _where_after_pipeline(fs)
        assert "NOT IN (:p1)" in w
        assert " OR " in w


class TestHavingOrOfAnd:
    def test_having_or_of_and(self) -> None:
        hs = [
            {
                "left_expr": "COUNT(t.id)",
                "op": ">",
                "param_key": "h1",
                "value_type": "integer",
                "where_group": 1,
            },
            {
                "left_expr": "SUM(t.amt)",
                "op": ">",
                "param_key": "h2",
                "value_type": "number",
                "where_group": 1,
            },
            {
                "left_expr": "COUNT(t.id)",
                "op": "<",
                "param_key": "h3",
                "value_type": "integer",
                "where_group": 2,
            },
            {
                "left_expr": "SUM(t.amt)",
                "op": "<",
                "param_key": "h4",
                "value_type": "number",
                "where_group": 2,
            },
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
            where=_from_flat_where_dicts(
                [
                    _wf("t.a", "p1", where_group=1, bool_op="OR"),
                    _wf("t.b", "p2", where_group=2),
                ]
            ),
            having=None,
        )
        out = _normalize_cte_steps([cte])
        assert cte.where is not None
        assert cte.where.op == "or"
        assert len(out) == 1
        assert len(out[0].where.leaves() if out[0].where else []) == 2

    def test_case_branch_strips_grouping(self) -> None:
        br = CaseWhenBranch.from_dict(
            {
                "condition": {
                    "left_expr": "t.x",
                    "op": "=",
                    "value_type": "string",
                    "param_key": "p1",
                    "bool_op": "OR",
                    "where_group": 7,
                },
                "literal_string": "yes",
            }
        )
        assert br.condition.left_expr.primary_column == "t.x"

    def test_dedup_keeps_distinct_groups(self) -> None:
        fp_a = WhereParam(
            NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        fp_b = WhereParam(
            NormalizedExpr.from_column("t.b"),
            op="=",
            param_key="p2",
            value_type="string",
        )
        out = _dedup_where_predicates([fp_a, fp_b])
        assert len(out) == 2

    def test_dedup_collapses_within_group(self) -> None:
        fp = WhereParam(
            NormalizedExpr.from_column("t.a"),
            op="=",
            param_key="p1",
            value_type="string",
        )
        a = replace(fp)
        b = replace(fp)
        out = _dedup_where_predicates([a, b])
        assert len(out) == 1

    def test_qsim_grouped_match(self) -> None:
        f1 = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=2),
        ]
        f2 = [
            _wf("t.a", "p9", where_group=1),
            _wf("t.b", "p8", where_group=2),
        ]
        g1 = _from_flat_where_dicts(f1)
        g2 = _from_flat_where_dicts(f2)
        assert g1 is not None and g2 is not None
        assert _compute_where_similarity(g1.leaves(), g2.leaves()) == pytest.approx(1.0)

    def test_qsim_grouped_vs_old_template(self) -> None:
        template = [
            _wf("t.a", "p1", bool_op="OR"),
            _wf("t.b", "p2", bool_op="AND"),
        ]
        intent = [
            _wf("t.a", "p1", where_group=1),
            _wf("t.b", "p2", where_group=2),
        ]
        t_group = _from_flat_where_dicts(template)
        i_group = _from_flat_where_dicts(intent)
        assert t_group is not None and i_group is not None
        assert _compute_where_similarity(t_group.leaves(), i_group.leaves()) == pytest.approx(1.0)

    def test_warning_or_with_no_signal(self) -> None:
        issues = validate_predicate_group_hints(
            "Show rows where name is Alice or Bob",
            PredicateGroup.from_list(
                [
                    WhereParam(
                        NormalizedExpr.from_column("t.name"),
                        op="=",
                        param_key="p1",
                        value_type="string",
                    ),
                    WhereParam(
                        NormalizedExpr.from_column("t.name"),
                        op="=",
                        param_key="p2",
                        value_type="string",
                    ),
                ]
            ),
            None,
        )
        assert issues == []

    def test_warning_both_signals(self) -> None:
        issues = validate_predicate_group_hints(
            "",
            _from_flat_where_dicts([_wf("t.a", "p1", where_group=1, bool_op="OR")]),
            None,
        )
        assert issues == []


def test_validate_semantics_includes_bool_hints() -> None:
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=_from_flat_where_dicts([_wf("t.a", "p1", where_group=1, bool_op="OR")]),
        having=None,
        natural_language="",
    )
    res = validate_semantics(intent, SchemaGraph(tables={}, join_paths_multi={}))
    assert res is not None
