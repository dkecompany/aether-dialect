"""Tests for federation semi-join reduction and plan-template matching."""

from __future__ import annotations

import pandas as pd

from aetherdialect._contracts_base import FederationPlanTemplate, where_leaves
from aetherdialect._contracts_core import JoinSpec, RuntimeIntent, SourceStep
from aetherdialect._federation import (
    distinct_semijoin_keys,
    federation_plan_combine_hash,
    federation_plan_matches_template,
    federation_plan_step_fingerprints,
    inject_semijoin_where,
    order_federation_execution_steps,
    semijoin_key_columns,
)
from aetherdialect._utils import intent_key


def test_distinct_semijoin_keys_respects_cap() -> None:
    frame = pd.DataFrame({"id": list(range(5))})
    assert distinct_semijoin_keys(frame, "id", cap=10) == [0, 1, 2, 3, 4]
    assert distinct_semijoin_keys(frame, "id", cap=3) is None


def test_inject_semijoin_where_appends_in_clause() -> None:
    intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    updated = inject_semijoin_where(intent, "id", [1, 2])
    leaves = where_leaves(updated.where) or []
    assert len(leaves) == 1
    assert leaves[0].op == "in"


def test_inject_semijoin_where_empty_keys_uses_sentinel() -> None:
    intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    updated = inject_semijoin_where(intent, "id", [], value_type="integer")
    fp = (where_leaves(updated.where) or [])[0]
    assert fp.op == "in"
    assert fp.value_type == "integer"
    assert fp.param_key == "p1"
    assert fp.raw_value is None
    assert updated.param_values == {"p1": ["__AETHERDIALECT_EMPTY_SEMIJOIN__"]}


def test_column_where_value_type_prefers_integer_for_int_columns() -> None:
    from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
    from aetherdialect._federation import column_where_value_type
    from aetherdialect._schema_graph import recompute_join_paths_multi

    tables = {
        "t": TableMetadata(
            name="t",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
    }
    schema = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    assert column_where_value_type(schema, "t", "id") == "integer"


def test_order_steps_prefers_limited_intent() -> None:
    limited = SourceStep(
        source_id="a",
        sub_intent=RuntimeIntent(
            tables=["t_a"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            limit=5,
        ),
    )
    broad = SourceStep(
        source_id="b",
        sub_intent=RuntimeIntent(
            tables=["t_b"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
    )
    from aetherdialect._contracts_core import FederatedPlan

    plan = FederatedPlan(steps=(broad, limited))
    ordered = order_federation_execution_steps(plan)
    assert ordered[0].source_id == "a"


def test_semijoin_key_columns_oriented() -> None:
    from aetherdialect._contracts_core import FederatedPlan

    plan = FederatedPlan(
        steps=(),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="lid",
                right_key="rid",
                logical_key="id",
                kind="inner",
            ),
        ),
    )
    assert semijoin_key_columns(plan, "a", "b") == ("lid", "rid")
    assert semijoin_key_columns(plan, "b", "a") == ("rid", "lid")


def test_plan_template_fingerprint_match() -> None:
    from aetherdialect._contracts_core import FederatedPlan

    step_a = SourceStep(
        source_id="a",
        sub_intent=RuntimeIntent(
            tables=["t_a"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
    )
    step_b = SourceStep(
        source_id="b",
        sub_intent=RuntimeIntent(
            tables=["t_b"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
    )
    plan = FederatedPlan(
        steps=(step_a, step_b),
        combine=(
            JoinSpec(
                left_source="a",
                right_source="b",
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="inner",
            ),
        ),
    )
    fps = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key)
    template = FederationPlanTemplate(
        plan_id="p1",
        composite_schema_graph_id="sg1",
        intent_key="p1",
        step_fingerprints=fps,
        combine_hash=federation_plan_combine_hash(plan),
    )
    assert federation_plan_matches_template(plan, template, step_fingerprints=fps)


def test_plan_template_fingerprint_includes_grain() -> None:
    from aetherdialect._contracts_core import FederatedPlan

    many_intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    scalar_intent = RuntimeIntent(
        tables=["t_a"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan_many = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=many_intent),),
    )
    plan_scalar = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=scalar_intent),),
    )
    fps_many = federation_plan_step_fingerprints(plan_many, intent_key_fn=intent_key)
    fps_scalar = federation_plan_step_fingerprints(plan_scalar, intent_key_fn=intent_key)
    assert fps_many != fps_scalar
    template = FederationPlanTemplate(
        plan_id="p1",
        composite_schema_graph_id="sg1",
        intent_key="p1",
        step_fingerprints=fps_many,
        combine_hash=federation_plan_combine_hash(plan_many),
    )
    assert federation_plan_matches_template(plan_many, template, step_fingerprints=fps_many)
    assert not federation_plan_matches_template(plan_scalar, template, step_fingerprints=fps_scalar)
