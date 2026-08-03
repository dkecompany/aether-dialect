"""Tests for PredicateGroup composition and legacy migration."""

from __future__ import annotations

import pytest

from aetherdialect._constants import MAX_PREDICATE_NESTING_DEPTH
from aetherdialect._contracts_base import (
    HavingParam,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
    coerce_predicate_group,
    map_predicate_group,
    merge_predicate_groups,
    normalize_predicate_group_cnf,
    normalize_predicate_group_dnf,
    parse_having_field,
    parse_where_field,
    partition_predicate_group,
    predicate_group_from_legacy_flat_where_dicts,
    predicate_group_from_list,
    predicate_leaves,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._dialect import get_dialect
from aetherdialect._sql_gen import render_predicate_clause, render_predicate_group_sql


def _fp(col: str, op: str = "=", value: str = "x") -> WhereParam:
    return WhereParam(left_expr=NormalizedExpr.from_column(col), op=op, raw_value=value, param_key="p1")


@pytest.mark.fast
def test_predicate_group_round_trip_dict() -> None:
    group = PredicateGroup(op="or", groups=(PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),))
    payload = group.to_dict()
    restored = PredicateGroup.from_dict(payload)
    assert restored is not None
    assert restored.op == "or"
    assert len(restored.leaves()) == 2


@pytest.mark.fast
def test_legacy_filters_param_migrates_to_where() -> None:
    legacy = [
        {"left_expr": "t.a", "op": "=", "value": 1, "where_group": 1},
        {"left_expr": "t.b", "op": "=", "value": 2, "where_group": 1},
        {"left_expr": "t.c", "op": "=", "value": 3, "where_group": 2},
    ]
    group = predicate_group_from_legacy_flat_where_dicts(legacy)
    assert group is not None
    assert group.op == "or"
    assert len(group.groups) == 2
    assert [p.left_expr.primary_column for p in group.leaves()] == ["t.a", "t.b", "t.c"]


@pytest.mark.fast
def test_runtime_intent_accepts_where_field() -> None:
    intent = RuntimeIntent.from_dict(
        {
            "tables": ["t"],
            "select_cols": [{"expr": "t.id"}],
            "where": {
                "op": "and",
                "predicates": [{"left_expr": "t.id", "op": "=", "value_type": "integer", "value": 1}],
            },
        }
    )
    assert intent.where is not None
    assert len(intent.where.leaves()) == 1


@pytest.mark.fast
def test_partition_predicate_group_splits_leaves() -> None:
    group = PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b")))
    kept, dropped = partition_predicate_group(group, lambda p: p.left_expr.primary_column == "t.a")
    assert kept is not None and dropped is not None
    assert [p.left_expr.primary_column for p in kept.leaves()] == ["t.a"]
    assert [p.left_expr.primary_column for p in dropped.leaves()] == ["t.b"]


@pytest.mark.fast
def test_render_predicate_group_sql_parentheses() -> None:
    group = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a", value="1"), _fp("t.b", value="2"))),
            PredicateGroup(op="and", predicates=(_fp("t.c", value="3"),)),
        ),
    )

    dialect = get_dialect("duckdb")

    def render_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect)

    sql = render_predicate_group_sql(group, render_leaf)
    assert "OR" in sql
    assert '"t"."a"' in sql and '"t"."c"' in sql


@pytest.mark.fast
def test_coerce_predicate_group_enforces_depth() -> None:
    def _deep_or_tree(depth: int) -> PredicateGroup:
        if depth <= 1:
            return PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b")))
        left = _deep_or_tree(depth - 1)
        right = PredicateGroup(op="and", predicates=(_fp("t.c"),))
        return PredicateGroup(op="or", groups=(left, right))

    nested = _deep_or_tree(MAX_PREDICATE_NESTING_DEPTH + 1)
    assert nested.depth() > MAX_PREDICATE_NESTING_DEPTH
    coerced = coerce_predicate_group(nested)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH


@pytest.mark.fast
def test_merge_predicate_groups_flattens_same_op() -> None:
    left = predicate_group_from_list([_fp("t.a")])
    right = predicate_group_from_list([_fp("t.b")])
    merged = merge_predicate_groups("and", [left, right])
    assert merged is not None
    assert [p.left_expr.primary_column for p in merged.leaves()] == ["t.a", "t.b"]


@pytest.mark.fast
def test_map_predicate_group_transforms_leaves() -> None:
    group = predicate_group_from_list([_fp("t.a")])
    mapped = map_predicate_group(group, lambda p: WhereParam(left_expr=p.left_expr, op="!=", raw_value=p.raw_value))
    assert mapped is not None
    assert mapped.leaves()[0].op == "!="


@pytest.mark.fast
def test_parse_where_and_having_fields() -> None:
    assert parse_where_field({"where": None}) is None
    having = parse_having_field({"having_param": [{"left_expr": "COUNT(*)", "op": ">", "value": 1}]})
    assert having is not None
    assert isinstance(having.leaves()[0], HavingParam)


@pytest.mark.fast
def test_predicate_leaves_empty() -> None:
    assert predicate_leaves(None) == []
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    assert predicate_leaves(intent.where) == []


@pytest.mark.fast
def test_normalize_predicate_group_cnf_nested_and_of_or() -> None:
    """CNF normalizes A AND (B OR C) into AND-of-OR clauses."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b"), _fp("t.c"))),),
    )
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert normalized.depth() <= MAX_PREDICATE_NESTING_DEPTH
    assert len(normalized.groups) == 2
    assert all(child.op == "or" for child in normalized.groups)
    assert normalized.groups[0].leaves()[0].left_expr.primary_column == "t.a"
    assert [p.left_expr.primary_column for p in normalized.groups[1].leaves()] == ["t.b", "t.c"]


@pytest.mark.fast
def test_normalize_predicate_group_cnf_expands_and_children() -> None:
    """CNF expands nested AND children into single-predicate OR clauses."""
    group = PredicateGroup(
        op="and",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),
            PredicateGroup(op="and", predicates=(_fp("t.c"), _fp("t.d"))),
        ),
    )
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert len(normalized.groups) == 4
    assert all(child.op == "or" and len(child.predicates) == 1 for child in normalized.groups)
    assert [child.leaves()[0].left_expr.primary_column for child in normalized.groups] == [
        "t.a",
        "t.b",
        "t.c",
        "t.d",
    ]


@pytest.mark.fast
def test_coerce_predicate_group_prefers_cnf_when_shallower() -> None:
    """coerce_predicate_group picks the shallower of DNF and CNF normal forms."""
    group = PredicateGroup(
        op="and",
        groups=(
            PredicateGroup(
                op="or",
                groups=(
                    PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),
                    PredicateGroup(op="and", predicates=(_fp("t.c"), _fp("t.d"))),
                ),
            ),
        ),
    )
    assert group.depth() == 3
    coerced = coerce_predicate_group(group)
    assert coerced is not None
    assert coerced.op == "or"
    assert coerced.depth() == 2
    assert len(coerced.groups) == 2
    assert all(child.op == "and" for child in coerced.groups)


@pytest.mark.fast
def test_coerce_predicate_group_and_of_or_within_depth() -> None:
    """coerce_predicate_group keeps AND-of-OR trees within nesting depth."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b"), _fp("t.c"))),),
    )
    coerced = coerce_predicate_group(group)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH
    assert coerced.op == "and"
    assert len(coerced.groups) == 2
    assert all(child.op == "or" for child in coerced.groups)


@pytest.mark.fast
def test_normalize_predicate_group_cnf_and_of_or() -> None:
    """CNF preserves top-level AND with OR child groups."""
    group = PredicateGroup(
        op="and",
        groups=(
            PredicateGroup(op="or", predicates=(_fp("t.a"), _fp("t.b"))),
            PredicateGroup(op="or", predicates=(_fp("t.c"), _fp("t.d"))),
        ),
    )
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert len(normalized.groups) == 2
    assert all(child.op == "or" for child in normalized.groups)
    assert [p.left_expr.primary_column for p in normalized.groups[0].leaves()] == ["t.a", "t.b"]
    assert [p.left_expr.primary_column for p in normalized.groups[1].leaves()] == ["t.c", "t.d"]


@pytest.mark.fast
def test_normalize_predicate_group_cnf_lifts_leaf_predicates_into_or() -> None:
    """CNF wraps AND-level leaf predicates in single-predicate OR groups."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b"), _fp("t.c"))),),
    )
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert len(normalized.groups) == 2
    assert normalized.groups[0].op == "or"
    assert normalized.groups[0].leaves()[0].left_expr.primary_column == "t.a"
    assert [p.left_expr.primary_column for p in normalized.groups[1].leaves()] == ["t.b", "t.c"]


@pytest.mark.fast
def test_normalize_predicate_group_cnf_flattens_single_and_child() -> None:
    """CNF collapses a single-child AND wrapper."""
    inner = PredicateGroup(op="or", predicates=(_fp("t.a"), _fp("t.b")))
    group = PredicateGroup(op="and", groups=(inner,))
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "or"
    assert [p.left_expr.primary_column for p in normalized.leaves()] == ["t.a", "t.b"]


@pytest.mark.fast
def test_normalize_predicate_group_cnf_empty_returns_none() -> None:
    assert normalize_predicate_group_cnf(None) is None


@pytest.mark.fast
def test_normalize_predicate_group_dnf_or_of_and() -> None:
    """DNF preserves top-level OR with AND child groups."""
    group = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),
            PredicateGroup(op="and", predicates=(_fp("t.c"), _fp("t.d"))),
        ),
    )
    normalized = normalize_predicate_group_dnf(group)
    assert normalized is not None
    assert normalized.op == "or"
    assert len(normalized.groups) == 2
    assert all(child.op == "and" for child in normalized.groups)
    assert [p.left_expr.primary_column for p in normalized.groups[0].leaves()] == ["t.a", "t.b"]


@pytest.mark.fast
def test_render_predicate_group_sql_after_cnf_and_of_or() -> None:
    """CNF AND-of-OR shape renders with OR precedence inside AND clauses."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a", value="1"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b", value="2"), _fp("t.c", value="3"))),),
    )
    normalized = normalize_predicate_group_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert all(child.op == "or" for child in normalized.groups)

    dialect = get_dialect("duckdb")

    def render_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect)

    sql = render_predicate_group_sql(normalized, render_leaf)
    upper = sql.upper()
    assert " OR " in upper
    assert " AND " in upper
    assert '"t"."a"' in sql
    assert '"t"."b"' in sql and '"t"."c"' in sql


@pytest.mark.fast
def test_predicate_nesting_depth_four_raises_warning() -> None:
    """Depth-four predicate trees are rejected by schema validation."""
    from aetherdialect._validation_schema import validate_predicate_nesting_depth

    def _deep_or_tree(depth: int) -> PredicateGroup:
        if depth <= 1:
            return PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b")))
        left = _deep_or_tree(depth - 1)
        right = PredicateGroup(op="and", predicates=(_fp("t.c"),))
        return PredicateGroup(op="or", groups=(left, right))

    nested = _deep_or_tree(MAX_PREDICATE_NESTING_DEPTH + 1)
    issues = validate_predicate_nesting_depth(nested, None)
    assert any("nesting exceeds" in issue.message for issue in issues)
    assert any(issue.severity == "error" for issue in issues)
    """coerce_predicate_group normalizes nested OR-of-AND trees."""
    group = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a"), _fp("t.b"))),
            PredicateGroup(
                op="and",
                predicates=(_fp("t.c"),),
            ),
        ),
    )
    coerced = coerce_predicate_group(group)
    assert coerced is not None
    assert coerced.op == "or"
    assert len(coerced.groups) == 2
    assert coerced.groups[0].op == "and"
