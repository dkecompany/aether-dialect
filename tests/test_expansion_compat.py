"""Unit tests for expansion operator compatibility gating."""

from __future__ import annotations

from aetherdialect._constants import (
    DISTINCT_ADD,
    EMI_MUTATE,
    GROUPBY_REMOVE,
    HAVING_VALUE_ADD,
    SELECT_CASE_LABEL_ADD,
    WHERE_ADD,
    WHERE_OR_GROUP,
    WINDOW_LAG_ADD,
    WINDOW_RANK_ADD,
)
from aetherdialect._contracts_core import SeedWarmupIntent
from aetherdialect._expansion_ops import expansion_compatible


def _base_intent(**kwargs: object) -> SeedWarmupIntent:
    data = {
        "intent_id": "test_intent",
        "tables": ["t1"],
        "grain": "row_level",
        "select_cols": [],
        "group_by_cols": [],
        "where": [],
    }
    data.update(kwargs)
    return SeedWarmupIntent.from_dict(data)


class TestExpansionCompatible:
    def test_distinct_blocks_window_add(self) -> None:
        parent = _base_intent(
            expansion_metadata={
                "operator": DISTINCT_ADD,
                "depth": 1,
                "expansion_path": [DISTINCT_ADD],
            },
        )
        assert expansion_compatible(parent, WINDOW_RANK_ADD) is False

    def test_window_blocks_second_window(self) -> None:
        parent = _base_intent(
            expansion_metadata={
                "operator": WINDOW_RANK_ADD,
                "depth": 1,
                "expansion_path": [WINDOW_RANK_ADD],
            },
        )
        assert expansion_compatible(parent, WINDOW_LAG_ADD) is False

    def test_emi_blocks_or_group(self) -> None:
        parent = _base_intent(
            expansion_metadata={
                "operator": EMI_MUTATE,
                "depth": 1,
                "expansion_path": [EMI_MUTATE],
            },
        )
        assert expansion_compatible(parent, WHERE_OR_GROUP) is False

    def test_filter_add_allowed_on_fresh_intent(self) -> None:
        parent = _base_intent()
        assert expansion_compatible(parent, WHERE_ADD) is True

    def test_having_blocks_groupby_remove(self) -> None:
        parent = _base_intent(
            expansion_metadata={
                "operator": HAVING_VALUE_ADD,
                "depth": 1,
                "expansion_path": [HAVING_VALUE_ADD],
            },
        )
        assert expansion_compatible(parent, GROUPBY_REMOVE) is False

    def test_case_add_blocked_when_case_registry_present(self) -> None:
        parent = _base_intent(case_registry=[{"registry_id": "c01", "case_when": {"branches": []}}])
        assert expansion_compatible(parent, SELECT_CASE_LABEL_ADD) is False
