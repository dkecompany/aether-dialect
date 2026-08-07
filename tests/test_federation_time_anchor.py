"""Tests for federated temporal anchor binding across member SQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aetherdialect._contracts_base import (
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import AnchoredTemporalBind, FederationExecutionContext, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._core_utils import pop_federation_execution_context, push_federation_execution_context
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation import (
    compose_composite_graph,
    federation_plan_step_fingerprints,
    parse_federation_mappings,
    plan_federated_intent,
    resolve_anchored_temporal_bind,
)
from aetherdialect._intent_process import NormalizedExpr, intent_key
from aetherdialect._sql_gen import build_deterministic_sql
from tests.federation_helpers import enriched_manifest, federation_member_graph, stamp_union_disjointness_profiling

ANCHOR = datetime(2026, 1, 15, 12, 30, 0, tzinfo=UTC)


def _events_column() -> dict[str, ColumnMetadata]:
    return {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
        "created_at": ColumnMetadata(name="created_at", data_type="timestamp", sensitivity="none"),
    }


def _three_member_union_bundle() -> tuple[dict[str, object], object, object, object]:
    members = {
        "a": federation_member_graph("events_a", source_id="a", columns=_events_column()),
        "b": federation_member_graph("events_b", source_id="b", columns=_events_column()),
        "c": federation_member_graph("events_c", source_id="c", columns=_events_column()),
    }
    for source_id, table_name, key_col, samples in (
        ("a", "events_a", "id", ("a1", "a2")),
        ("b", "events_b", "id", ("b1", "b2")),
        ("c", "events_c", "id", ("c1", "c2")),
        ("a", "events_a", "created_at", ("ca1", "ca2")),
        ("b", "events_b", "created_at", ("cb1", "cb2")),
        ("c", "events_c", "created_at", ("cc1", "cc2")),
    ):
        stamp_union_disjointness_profiling(
            members[source_id].tables[table_name],
            key_col=key_col,
            overlap_sample=samples,
        )
    manifest = enriched_manifest(
        members,
        {
            "federation_id": "fed_time_anchor",
            "cross_source_joins": [],
        },
        member_graphs=members,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.1",
            "logical_tables": [
                {
                    "logical": "events",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "events_a", "columns": {"id": "id", "created_at": "created_at"}},
                        {"source": "b", "table": "events_b", "columns": {"id": "id", "created_at": "created_at"}},
                        {"source": "c", "table": "events_c", "columns": {"id": "id", "created_at": "created_at"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    return members, manifest, mappings, composite


def _date_window_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["events"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("events.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("events.created_at"),
                    op=">=",
                    value_type="date_window",
                    raw_value={"unit": "day", "amount": 30},
                )
            ]
        ),
    )


@pytest.mark.fast
def test_all_member_statements_share_one_anchor() -> None:
    _, manifest, mappings, composite = _three_member_union_bundle()
    intent = _date_window_intent()
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 3

    bind = resolve_anchored_temporal_bind(intent, anchor=ANCHOR)
    assert isinstance(bind, AnchoredTemporalBind)
    fed_ctx = FederationExecutionContext(plan_id="fed-time-anchor", temporal_bind=bind)
    token = push_federation_execution_context(fed_ctx)
    dialect = DialectRegistry.get("duckdb")
    try:
        member_sql = [
            build_deterministic_sql(step.sub_intent, schema=composite, dialect=dialect) for step in plan.steps
        ]
    finally:
        pop_federation_execution_context(token)

    anchor_marker = "2026-01-15"
    assert len(member_sql) == 3
    assert all(anchor_marker in sql for sql in member_sql)
    for sql in member_sql:
        assert "CURRENT_DATE" not in sql.upper()


@pytest.mark.fast
def test_anchor_participates_in_plan_fingerprint() -> None:
    _, manifest, mappings, composite = _three_member_union_bundle()
    intent = _date_window_intent()
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    bind_a = resolve_anchored_temporal_bind(intent, anchor=ANCHOR)
    bind_b = resolve_anchored_temporal_bind(intent, anchor=ANCHOR + timedelta(days=1))
    fps_a = federation_plan_step_fingerprints(
        plan,
        intent_key_fn=intent_key,
        manifest=manifest,
        temporal_bind=bind_a,
    )
    fps_b = federation_plan_step_fingerprints(
        plan,
        intent_key_fn=intent_key,
        manifest=manifest,
        temporal_bind=bind_b,
    )
    assert fps_a != fps_b
    fps_a_repeat = federation_plan_step_fingerprints(
        plan,
        intent_key_fn=intent_key,
        manifest=manifest,
        temporal_bind=bind_a,
    )
    assert fps_a == fps_a_repeat


@pytest.mark.fast
def test_single_engine_generation_unchanged_without_anchor() -> None:
    members, _, _, _ = _three_member_union_bundle()
    intent = RuntimeIntent(
        tables=["events_a"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("events_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("events_a.created_at"),
                    op=">=",
                    value_type="date_window",
                    raw_value={"unit": "day", "amount": 30},
                )
            ]
        ),
    )
    dialect = DialectRegistry.get("duckdb")
    sql = build_deterministic_sql(intent, schema=members["a"], dialect=dialect)
    assert "CURRENT_DATE" in sql.upper() or "DATE(" in sql.upper()
