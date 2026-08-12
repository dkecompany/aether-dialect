"""Fast tests for sandbox failure remediation gates."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ConfigError, HavingParam, NormalizedExpr, PredicateGroup
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import WindowRegistryStep, WindowSpec
from aetherdialect._intent_expr import parse_expr_string
from aetherdialect._intent_normalize import prune_unreferenced_registries
from aetherdialect._validation_rules import validate_having_operator_is_numeric
from aetherdialect._validation_shape import validate_scope_registries

pytestmark = pytest.mark.fast

NormalizedExpr.register_parse_expr_string(parse_expr_string)


def test_from_stored_json_parses_count_call() -> None:
    expr = NormalizedExpr.from_stored_json("COUNT(item.item_id)")
    assert expr.agg_func == "count" or any(g.agg_func == "count" for g in expr.add_groups)
    assert expr.column_ref != "COUNT(item.item_id)"


def test_predicate_group_wraps_flat_having_leaf_dict() -> None:
    group = PredicateGroup.from_dict(
        {"left_expr": "COUNT(item.item_id)", "op": ">", "value": 5, "value_type": "integer"},
        having=True,
    )
    assert group is not None
    assert len(group.predicates) == 1
    leaf = group.predicates[0]
    assert isinstance(leaf, HavingParam)
    assert leaf.op == ">"
    assert leaf.left_expr.has_aggregation


def test_predicate_group_from_stored_list_having() -> None:
    with pytest.raises(ConfigError, match="nested objects"):
        PredicateGroup.from_stored(
            [{"left_expr": "COUNT(*)", "op": ">=", "value": 2, "value_type": "integer"}],
            having=True,
        )


def test_having_in_op_allowed() -> None:
    hp = HavingParam(
        left_expr=NormalizedExpr.from_agg("count", "t.a"),
        op="in",
        value_type="integer",
        raw_value=[1, 2],
    )
    assert validate_having_operator_is_numeric([hp]) == []


def test_having_ilike_rejected() -> None:
    hp = HavingParam(
        left_expr=NormalizedExpr.from_agg("count", "t.a"),
        op="ilike",
        value_type="string",
    )
    issues = validate_having_operator_is_numeric([hp])
    assert len(issues) == 1


def test_prune_unreferenced_window_registry() -> None:
    intent = RuntimeIntent(
        tables=["item"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("item.item_id"))],
        window_registry=[
            WindowRegistryStep(registry_id="w01", window_spec=WindowSpec(function="row_number")),
        ],
        natural_language="list items",
    )
    out = prune_unreferenced_registries(intent)
    assert out.window_registry == []


def test_dangling_window_ref_hard_reject() -> None:
    intent = RuntimeIntent(
        tables=["item"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("w01"))],
        window_registry=[],
        natural_language="rank items",
    )
    issues = validate_scope_registries(
        context="main",
        window_registry=list(intent.window_registry or []),
        case_registry=list(intent.case_registry or []),
        select_cols=list(intent.select_cols or []),
        group_by_cols=list(intent.group_by_cols or []),
        order_by_cols=list(intent.order_by_cols or []),
    )
    assert any(i.issue_id.startswith("registry_dangling_window") for i in issues)
