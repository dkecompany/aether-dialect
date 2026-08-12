"""Tests for PredicateGroup composition and nested where/having trees."""

from __future__ import annotations

import pytest

from aetherdialect._constants import MAX_PREDICATE_NESTING_DEPTH
from aetherdialect._contracts_base import (
    ConfigError,
    HavingParam,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import render_predicate_clause, render_predicate_group_sql


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
def test_from_dict_accepts_nested_groups_inside_predicates() -> None:
    """Compose often nests AND groups under ``predicates``; parse them as groups."""
    restored = PredicateGroup.from_dict(
        {
            "op": "or",
            "predicates": [
                {
                    "op": "and",
                    "predicates": [
                        {"left_expr": "t.rating", "op": "=", "value": "A", "value_type": "string"},
                        {"left_expr": "t.length", "op": ">", "value": 10, "value_type": "integer"},
                    ],
                    "groups": [],
                },
                {
                    "op": "and",
                    "predicates": [
                        {"left_expr": "t.rating", "op": "=", "value": "B", "value_type": "string"},
                        {"left_expr": "t.length", "op": ">", "value": 20, "value_type": "integer"},
                    ],
                    "groups": [],
                },
            ],
            "groups": [],
        }
    )
    assert restored is not None
    assert restored.op == "or"
    assert len(restored.groups) == 2
    assert len(restored.predicates) == 0
    assert len(restored.leaves()) == 4
    assert [p.raw_value for p in restored.leaves()] == ["A", 10, "B", 20]


@pytest.mark.fast
def test_flat_where_dicts_build_predicate_group() -> None:
    flat_where = [
        {"left_expr": "t.a", "op": "=", "value": 1, "where_group": 1},
        {"left_expr": "t.b", "op": "=", "value": 2, "where_group": 1},
        {"left_expr": "t.c", "op": "=", "value": 3, "where_group": 2},
    ]
    group = _from_flat_where_dicts(flat_where)
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
    kept, dropped = PredicateGroup.partition(group, lambda p: p.left_expr.primary_column == "t.a")
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

    dialect = DialectRegistry.get("duckdb")

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
    coerced = PredicateGroup.coerce(nested)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH


@pytest.mark.fast
def test_merge_predicate_groups_flattens_same_op() -> None:
    left = PredicateGroup.from_list([_fp("t.a")])
    right = PredicateGroup.from_list([_fp("t.b")])
    merged = PredicateGroup.merge("and", [left, right])
    assert merged is not None
    assert [p.left_expr.primary_column for p in merged.leaves()] == ["t.a", "t.b"]


@pytest.mark.fast
def test_map_predicate_group_transforms_leaves() -> None:
    group = PredicateGroup.from_list([_fp("t.a")])
    mapped = PredicateGroup.map(group, lambda p: WhereParam(left_expr=p.left_expr, op="!=", raw_value=p.raw_value))
    assert mapped is not None
    assert mapped.leaves()[0].op == "!="


@pytest.mark.fast
def test_parse_where_and_having_fields() -> None:
    assert PredicateGroup.parse_where_field({"where": None}) is None
    with pytest.raises(ConfigError, match="having_param is not accepted"):
        PredicateGroup.parse_having_field({"having_param": [{"left_expr": "COUNT(*)", "op": ">", "value": 1}]})
    having = PredicateGroup.parse_having_field(
        {
            "having": {
                "op": "and",
                "predicates": [{"left_expr": "COUNT(*)", "op": ">", "value": 1, "value_type": "integer"}],
                "groups": [],
            }
        }
    )
    assert having is not None
    assert isinstance(having.leaves()[0], HavingParam)


@pytest.mark.fast
def test_predicate_leaves_empty() -> None:
    assert PredicateGroup.predicate_leaves(None) == []
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    assert PredicateGroup.predicate_leaves(intent.where) == []


@pytest.mark.fast
def test_normalize_predicate_group_cnf_nested_and_of_or() -> None:
    """CNF normalizes A AND (B OR C) into AND-of-OR clauses."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b"), _fp("t.c"))),),
    )
    normalized = PredicateGroup.normalize_cnf(group)
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
    normalized = PredicateGroup.normalize_cnf(group)
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
    """PredicateGroup.coerce picks the shallower of DNF and CNF normal forms."""
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
    coerced = PredicateGroup.coerce(group)
    assert coerced is not None
    assert coerced.op == "or"
    assert coerced.depth() == 2
    assert len(coerced.groups) == 2
    assert all(child.op == "and" for child in coerced.groups)


@pytest.mark.fast
def test_coerce_predicate_group_and_of_or_within_depth() -> None:
    """PredicateGroup.coerce keeps AND-of-OR trees within nesting depth."""
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b"), _fp("t.c"))),),
    )
    coerced = PredicateGroup.coerce(group)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH
    assert coerced.op == "and"
    assert {p.left_expr.primary_column for p in coerced.leaves()} == {"t.a", "t.b", "t.c"}
    rendered_ops = {coerced.op} | {child.op for child in coerced.groups}
    assert "or" in rendered_ops or any(p.left_expr.primary_column in {"t.b", "t.c"} for p in coerced.predicates)


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
    normalized = PredicateGroup.normalize_cnf(group)
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
    normalized = PredicateGroup.normalize_cnf(group)
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
    normalized = PredicateGroup.normalize_cnf(group)
    assert normalized is not None
    assert normalized.op == "or"
    assert [p.left_expr.primary_column for p in normalized.leaves()] == ["t.a", "t.b"]


@pytest.mark.fast
def test_normalize_predicate_group_cnf_empty_returns_none() -> None:
    assert PredicateGroup.normalize_cnf(None) is None


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
    normalized = PredicateGroup.normalize_dnf(group)
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
    normalized = PredicateGroup.normalize_cnf(group)
    assert normalized is not None
    assert normalized.op == "and"
    assert all(child.op == "or" for child in normalized.groups)

    dialect = DialectRegistry.get("duckdb")

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
    from aetherdialect._validation_sql import validate_predicate_nesting_depth

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
    """PredicateGroup.coerce normalizes nested OR-of-AND trees."""
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
    coerced = PredicateGroup.coerce(group)
    assert coerced is not None
    assert coerced.op == "or"
    assert {p.left_expr.primary_column for p in coerced.leaves()} == {"t.a", "t.b", "t.c"}
    assert coerced.leaf_count() == 3


@pytest.mark.fast
def test_flatten_associative_and() -> None:
    nested = PredicateGroup(
        op="and",
        groups=(
            PredicateGroup(op="and", predicates=(_fp("t.a", value="1"), _fp("t.b", value="2"))),
            PredicateGroup(op="and", predicates=(_fp("t.c", value="3"),)),
        ),
    )
    flat = PredicateGroup.flatten(nested)
    assert flat is not None
    assert flat.op == "and"
    assert not flat.groups
    assert [p.left_expr.primary_column for p in flat.predicates] == ["t.a", "t.b", "t.c"]


@pytest.mark.fast
def test_normalize_dnf_distributes_and_over_or() -> None:
    group = PredicateGroup(
        op="and",
        predicates=(_fp("t.a", value="1"),),
        groups=(PredicateGroup(op="or", predicates=(_fp("t.b", value="2"), _fp("t.c", value="3"))),),
    )
    dnf = PredicateGroup.normalize_dnf(group)
    assert dnf is not None
    assert dnf.op == "or"
    assert len(dnf.groups) == 2
    assert all(child.op == "and" for child in dnf.groups)
    leaf_sets = [{p.left_expr.primary_column for p in child.leaves()} for child in dnf.groups]
    assert {"t.a", "t.b"} in leaf_sets
    assert {"t.a", "t.c"} in leaf_sets


@pytest.mark.fast
def test_normalize_cnf_distributes_or_over_and() -> None:
    group = PredicateGroup(
        op="or",
        predicates=(_fp("t.a", value="1"),),
        groups=(PredicateGroup(op="and", predicates=(_fp("t.b", value="2"), _fp("t.c", value="3"))),),
    )
    cnf = PredicateGroup.normalize_cnf(group)
    assert cnf is not None
    assert cnf.op == "and"
    assert len(cnf.groups) == 2
    assert all(child.op == "or" for child in cnf.groups)


@pytest.mark.fast
def test_absorb_or_drops_stronger_conjunction() -> None:
    a = _fp("t.a", value="1")
    b = _fp("t.b", value="2")
    group = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(op="and", predicates=(a,)),
            PredicateGroup(op="and", predicates=(a, b)),
        ),
    )
    absorbed = PredicateGroup.absorb(group)
    assert absorbed is not None
    assert absorbed.leaf_count() == 1
    assert absorbed.leaves()[0].left_expr.primary_column == "t.a"


@pytest.mark.fast
def test_absorb_and_drops_weaker_disjunction() -> None:
    a = _fp("t.a", value="1")
    b = _fp("t.b", value="2")
    group = PredicateGroup(
        op="and",
        groups=(
            PredicateGroup(op="or", predicates=(a,)),
            PredicateGroup(op="or", predicates=(a, b)),
        ),
    )
    absorbed = PredicateGroup.absorb(group)
    assert absorbed is not None
    assert absorbed.leaf_count() == 1
    assert absorbed.leaves()[0].left_expr.primary_column == "t.a"


@pytest.mark.fast
def test_idempotent_duplicate_leaves_under_and() -> None:
    group = PredicateGroup(op="and", predicates=(_fp("t.a", value="1"), _fp("t.a", value="1")))
    coerced = PredicateGroup.coerce(group)
    assert coerced is not None
    assert coerced.leaf_count() == 1


@pytest.mark.fast
def test_or_keeps_same_column_different_values() -> None:
    group = PredicateGroup(
        op="or",
        predicates=(
            WhereParam(left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="A", value_type="string"),
            WhereParam(left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="B", value_type="string"),
        ),
    )
    assert group.predicates[0].identity_key != group.predicates[1].identity_key
    coerced = PredicateGroup.coerce(group)
    assert coerced is not None
    assert coerced.leaf_count() == 2


@pytest.mark.fast
def test_rebuild_from_leaves_preserves_or_on_count_mismatch() -> None:
    original = PredicateGroup(
        op="or",
        predicates=(
            WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op=">", raw_value=1, value_type="number"),
            WhereParam(left_expr=NormalizedExpr.from_column("t.b"), op="<", raw_value=2, value_type="number"),
        ),
    )
    rebuilt = PredicateGroup.rebuild_from_leaves(
        original,
        [WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op=">", raw_value=9, value_type="number")],
    )
    assert rebuilt is not None
    assert rebuilt.op == "or"
    assert rebuilt.leaf_count() == 1


@pytest.mark.fast
def test_contradiction_sentinel() -> None:
    bad = PredicateGroup.contradiction()
    assert bad.is_contradiction()
    assert not PredicateGroup(op="and", predicates=(_fp("t.a"),)).is_contradiction()


@pytest.mark.fast
def test_simplify_equality_clash_under_and() -> None:
    from aetherdialect._intent_normalize import simplify_predicate_semantics

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="and",
            predicates=(
                WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", raw_value="1", value_type="string"),
                WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", raw_value="2", value_type="string"),
            ),
        ),
    )
    out = simplify_predicate_semantics(intent)
    assert out.where is not None
    assert out.where.is_contradiction()


@pytest.mark.fast
def test_simplify_range_intersection_tightens_bounds() -> None:
    from aetherdialect._intent_normalize import simplify_predicate_semantics

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.n"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="and",
            predicates=(
                WhereParam(left_expr=NormalizedExpr.from_column("t.n"), op=">", raw_value=3, value_type="number"),
                WhereParam(left_expr=NormalizedExpr.from_column("t.n"), op=">", raw_value=5, value_type="number"),
                WhereParam(left_expr=NormalizedExpr.from_column("t.n"), op="<", raw_value=10, value_type="number"),
            ),
        ),
    )
    out = simplify_predicate_semantics(intent)
    assert out.where is not None
    leaves = out.where.leaves()
    assert len(leaves) == 2
    by_op = {leaf.op: leaf.raw_value for leaf in leaves}
    assert by_op[">"] == 5
    assert by_op["<"] == 10


@pytest.mark.fast
def test_simplify_empty_range_is_contradiction() -> None:
    from aetherdialect._intent_normalize import simplify_predicate_semantics

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.n"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="and",
            predicates=(
                WhereParam(left_expr=NormalizedExpr.from_column("t.n"), op=">", raw_value=10, value_type="number"),
                WhereParam(left_expr=NormalizedExpr.from_column("t.n"), op="<", raw_value=5, value_type="number"),
            ),
        ),
    )
    out = simplify_predicate_semantics(intent)
    assert out.where is not None
    assert out.where.is_contradiction()


@pytest.mark.fast
def test_simplify_or_equalities_to_in() -> None:
    from aetherdialect._intent_normalize import simplify_predicate_semantics

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.rating"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="A", value_type="string"
                ),
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="B", value_type="string"
                ),
            ),
        ),
    )
    out = simplify_predicate_semantics(intent)
    assert out.where is not None
    assert out.where.leaf_count() == 1
    leaf = out.where.leaves()[0]
    assert leaf.op == "in"
    assert leaf.raw_value == ["A", "B"]


@pytest.mark.fast
def test_simplify_null_vs_equality_clash() -> None:
    from aetherdialect._intent_normalize import simplify_predicate_semantics

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="and",
            predicates=(
                WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="is null", value_type="null"),
                WhereParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", raw_value="x", value_type="string"),
            ),
        ),
    )
    out = simplify_predicate_semantics(intent)
    assert out.where is not None
    assert out.where.is_contradiction()


@pytest.mark.fast
def test_dedup_value_aware_keeps_distinct_or_equalities() -> None:
    from aetherdialect._intent_bind import normalize_where_havings

    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.rating"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup(
            op="or",
            predicates=(
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="A", value_type="string"
                ),
                WhereParam(
                    left_expr=NormalizedExpr.from_column("t.rating"), op="=", raw_value="B", value_type="string"
                ),
            ),
        ),
    )
    out = normalize_where_havings(intent)
    assert out.where is not None
    assert out.where.leaf_count() == 2


@pytest.mark.fast
def test_simplify_exprs_preserves_nested_or_and() -> None:
    from aetherdialect._intent_bind import simplify_exprs

    where = PredicateGroup(
        op="or",
        groups=(
            PredicateGroup(
                op="and",
                predicates=(
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.rating"),
                        op="=",
                        raw_value="A",
                        value_type="string",
                    ),
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.n"),
                        op="<",
                        raw_value=10,
                        value_type="integer",
                    ),
                ),
            ),
            PredicateGroup(
                op="and",
                predicates=(
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.rating"),
                        op="=",
                        raw_value="B",
                        value_type="string",
                    ),
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t.n"),
                        op=">",
                        raw_value=2,
                        value_type="number",
                    ),
                ),
            ),
        ),
    )
    intent = RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.rating"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
    )
    out = simplify_exprs(intent)
    assert out.where is not None
    assert out.where.op == "or"
    assert len(out.where.groups) == 2
    assert out.where.leaf_count() == 4
