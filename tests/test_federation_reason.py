"""Unsupported-operator ineligible reasons must name the lacking federation member."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._federation import (
    federation_ineligible_answerable_hint,
    federation_ineligible_reason_code,
    parse_federation_manifest,
    _federation_unsupported_operator_reason,
)
from aetherdialect._intent_process import NormalizedExpr


@pytest.mark.fast
def test_unsupported_contains_names_lacking_member_and_resolves_member_capability_hint() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_reason_l34",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "csv", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    intent = RuntimeIntent(
        tables=["ta"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("ta.tags"),
                    op="contains",
                    value_type="string",
                    raw_value="x",
                ),
            ]
        ),
    )
    reason = _federation_unsupported_operator_reason(intent, manifest)
    assert reason is not None
    assert "member capability" in reason
    assert "'b'" in reason or "csv" in reason
    assert "contains" in reason
    assert federation_ineligible_reason_code(reason) == "member_capability"
    hint = federation_ineligible_answerable_hint(reason)
    assert hint is not None
    assert "member" in hint.lower()


@pytest.mark.fast
def test_unsupported_having_operator_names_lacking_member() -> None:
    from aetherdialect._contracts_base import HavingParam

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_reason_l34_having",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "csv", "role": "owner"},
            ],
            "table_namespace": {"ta": "a"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    intent = RuntimeIntent(
        tables=["ta"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=predicate_group_from_list(
            [
                HavingParam(
                    left_expr=NormalizedExpr.from_agg("count", "ta.id"),
                    op="contains",
                    value_type="string",
                    raw_value="1",
                ),
            ]
        ),
    )
    reason = _federation_unsupported_operator_reason(intent, manifest)
    assert reason is not None
    assert "member capability" in reason
    assert "'a'" in reason or "'b'" in reason
    assert federation_ineligible_reason_code(reason) == "member_capability"
