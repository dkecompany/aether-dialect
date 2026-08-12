"""Federation member depends_on DAG and empty-select key projection."""

from __future__ import annotations

import pytest

from aetherdialect import FederationRuntimeError
from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._federation_plan import (
    _drop_member_dependency_edges_to_dag,
    _intent_exprs_local_to_tables,
    _member_stage_dependencies,
)
from tests.federation_helpers import build_two_member_federation


@pytest.mark.fast
def test_opposing_member_deps_collapse_to_dag() -> None:
    """Mutual member depends_on edges are reduced to a DAG."""
    deps = {
        "a": {"member_b"},
        "b": {"member_a"},
    }
    result = _drop_member_dependency_edges_to_dag(deps)
    assert len(result) == 1
    sole = next(iter(result))
    assert sole in {"a", "b"}
    assert result[sole] == (f"member_{'b' if sole == 'a' else 'a'}",)


@pytest.mark.fast
def test_member_stage_dependencies_are_manifest_oriented_dag() -> None:
    """depends_on comes from manifest join orientation and forms a DAG."""
    fed = build_two_member_federation()
    deps = _member_stage_dependencies(fed.manifest, {"a", "b"})
    flat = {(src, dep) for src, stages in deps.items() for dep in stages}
    assert not ({("a", "member_b"), ("b", "member_a")} <= flat)


@pytest.mark.fast
def test_empty_member_select_projects_combine_keys() -> None:
    """Filter-only member intents project local combine keys instead of staying empty."""
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.status"),
                    op="=",
                    raw_value="open",
                )
            ]
        ),
    )
    out = _intent_exprs_local_to_tables(
        intent,
        {"left_t"},
        multi_source=True,
        combine_key_cols=["left_t.id"],
    )
    assert [sc.expr.primary_column for sc in out.select_cols] == ["left_t.id"]


@pytest.mark.fast
def test_empty_member_select_without_keys_still_errors() -> None:
    """Without combine keys, an empty multi-source select still fails closed."""
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("right_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(FederationRuntimeError, match="at least one select column"):
        _intent_exprs_local_to_tables(
            intent,
            {"left_t"},
            multi_source=True,
            combine_key_cols=[],
        )
