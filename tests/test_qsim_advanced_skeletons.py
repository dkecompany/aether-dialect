"""Unit tests for QSim advanced skeleton slots and compliance helpers."""

from __future__ import annotations

from dataclasses import replace

from aetherdialect._contracts_base import DatabaseFeatureCapability
from aetherdialect._contracts_schema import QSimFilter, QSimIntent, QSimSkeleton
from aetherdialect._qsim_ops import (
    _qsim_advanced_slot_detected,
    _skeleton_suitable_for_advanced,
    append_advanced_skeleton_variants,
)


def _minimal_cap(**kwargs: object) -> DatabaseFeatureCapability:
    data = {
        "table_count": 3,
        "fk_edge_count": 2,
        "has_numeric_measures": True,
        "has_date_columns": True,
        "has_array_columns": False,
        "has_categorical_columns": True,
        "max_tables_on_any_join_path": 3,
        "max_fk_chain_depth": 2,
        "has_self_referential_fk": False,
        "tables_supporting_self_join": frozenset(),
        "has_window_capable_table_sets": True,
        "aggregatable_columns_by_table": {},
        "date_columns_by_table": {},
        "array_columns_by_table": {},
    }
    data.update(kwargs)
    return DatabaseFeatureCapability(**data)


class TestAdvancedSkeletonVariants:
    def test_append_adds_feasible_slots(self) -> None:
        base = QSimSkeleton(
            tables=["t1"],
            has_aggregation=False,
            num_filters=1,
            num_groupby=0,
            has_orderby=False,
            num_having=0,
            has_distinct=False,
        )
        cap = _minimal_cap()
        extended = append_advanced_skeleton_variants([base], cap)
        slots = {s.advanced_slot for s in extended if s.advanced_slot}
        assert "distinct_select" in slots or "date_window_filter" in slots
        assert len(extended) > len([base])

    def test_suitable_for_distinct_requires_non_agg(self) -> None:
        sk = QSimSkeleton(
            tables=["t1"],
            has_aggregation=True,
            num_filters=0,
            num_groupby=1,
            has_orderby=False,
            num_having=0,
        )
        assert _skeleton_suitable_for_advanced(sk, "distinct_select") is False
        sk2 = replace(sk, has_aggregation=False, num_groupby=0, num_filters=1)
        assert _skeleton_suitable_for_advanced(sk2, "distinct_select") is True


class TestAdvancedSlotDetection:
    def test_detects_distinct(self) -> None:
        intent = QSimIntent(
            intent_id="x",
            tables=["t1"],
            grain="row_level",
            select_cols=["t1.a"],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            distinct=True,
        )
        assert _qsim_advanced_slot_detected(intent, "distinct_select") is True

    def test_detects_date_window_filter(self) -> None:
        intent = QSimIntent(
            intent_id="x",
            tables=["t1"],
            grain="row_level",
            select_cols=["t1.a"],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                QSimFilter(column="t1.created_at", op=">=", value_type="temporal"),
            ],
            having_param=[],
        )
        assert _qsim_advanced_slot_detected(intent, "date_window_filter") is True
