"""Tests for mapping suggestion getters and dry-run turn plan preview."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import FederationMappingSuggestion, PlanPreviewResult
from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SourceStep
from aetherdialect._pipeline import build_plan_preview_from_intent
from tests.test_aether_federation_public_surface import _fed
from tests.test_aetherdialect import _make_aether_stub


def _intent(*, tables: tuple[str, ...], join_sig: list[str] | None = None) -> RuntimeIntent:
    return RuntimeIntent(
        tables=list(tables),
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=list(join_sig or []),
    )


@pytest.mark.fast
def test_mapping_suggestions_returns_cached_values() -> None:
    fed = _fed()
    suggestions = (
        FederationMappingSuggestion(
            logical="customer_id",
            members=("a", "b"),
            kind="column",
            score=0.92,
            role="join_key",
        ),
    )
    fed._federation_mapping_suggestions = suggestions
    assert fed.mapping_suggestions() == suggestions


@pytest.mark.fast
def test_preview_plan_returns_tables_and_join_path_without_execution() -> None:
    engine = _make_aether_stub()
    intent = _intent(tables=("orders", "customers"), join_sig=["orders.customer_id=customers.id"])
    with patch(
        "aetherdialect._pipeline.invoke_intent_parse_with_hints",
        return_value=(intent, [], 1, None),
    ):
        preview = engine.preview_plan("count orders by customer")
    assert isinstance(preview, PlanPreviewResult)
    assert preview.tables == ("customers", "orders")
    assert preview.join_path == ("orders.customer_id=customers.id",)
    assert preview.federates is False
    assert preview.member_source_ids == ()


@pytest.mark.fast
def test_preview_plan_marks_federation_when_applicable() -> None:
    fed = _fed()
    intent = _intent(tables=("left_t", "right_t"), join_sig=["left_t.id=right_t.id"])
    federated_plan = FederatedPlan(
        steps=(
            SourceStep(source_id="a", sub_intent=_intent(tables=("left_t",))),
            SourceStep(source_id="b", sub_intent=_intent(tables=("right_t",))),
        ),
        combine=(),
    )
    with (
        patch(
            "aetherdialect._pipeline.invoke_intent_parse_with_hints",
            return_value=(intent, [], 1, None),
        ),
        patch(
            "aetherdialect._pipeline.plan_federated_intent",
            return_value=federated_plan,
        ),
    ):
        preview = fed.preview_plan("join left and right")
    assert preview.federates is True
    assert preview.member_source_ids == ("a", "b")
    assert preview.tables == ("left_t", "right_t")


@pytest.mark.fast
def test_build_plan_preview_surfaces_ineligible_reason() -> None:
    intent = _intent(tables=("left_t", "right_t"))
    plan = FederatedPlan(steps=(), ineligible_reason="cross-source join path is not declared for referenced sources")
    preview = build_plan_preview_from_intent("join tables", intent, federated_plan=plan)
    assert preview.federates is False
    assert preview.ineligible_reason == "cross-source join path is not declared for referenced sources"
