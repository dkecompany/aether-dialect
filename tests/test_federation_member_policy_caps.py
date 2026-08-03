"""Per-member federation policy cap resolution."""

from __future__ import annotations

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._federation import (
    compose_composite_graph,
    federation_member_resolved_limits,
    federation_plan_step_fingerprints,
    parse_federation_manifest,
    plan_federated_intent,
    resolve_member_limits_for_source,
)
from aetherdialect._schema_graph import ColumnMetadata, SchemaGraph, TableMetadata, recompute_join_paths_multi
from aetherdialect._utils import intent_key


def _graph(table: str, *, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _manifest_with_limits() -> dict:
    return {
        "federation_id": "fed_caps",
        "sources": [
            {
                "source_id": "a",
                "engine": "duckdb",
                "role": "owner",
                "limits": {
                    "row_cap": 1000,
                    "timeout_ms": 5000,
                    "max_query_cost_bytes": 42.0,
                },
            },
            {"source_id": "b", "engine": "duckdb", "role": "owner"},
        ],
        "table_namespace": {"left_t": "a", "right_t": "b"},
        "cross_source_joins": [
            {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
        ],
        "coordinator": {"default_source_row_cap": 2000, "default_source_timeout_ms": 9000},
    }


@pytest.mark.fast
def test_member_row_cap_prefers_source_limit_over_coordinator_default() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    resolved = resolve_member_limits_for_source(manifest, "a")
    assert resolved.row_cap == 1000
    assert resolved.timeout_ms == 5000


@pytest.mark.fast
def test_member_row_cap_falls_back_to_coordinator_default() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    resolved = resolve_member_limits_for_source(manifest, "b")
    assert resolved.row_cap == 2000
    assert resolved.timeout_ms == 9000


@pytest.mark.fast
def test_member_cost_cap_falls_back_to_policy_config_when_unset() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    resolved = resolve_member_limits_for_source(manifest, "b")
    assert resolved.max_query_cost_rows == PolicyConfig.MAX_QUERY_COST_ROWS
    assert resolved.max_query_cost_bytes == PolicyConfig.MAX_QUERY_COST_BYTES
    assert resolved.profile_timeout_ms == PolicyConfig.PROFILE_TIMEOUT_MS


@pytest.mark.fast
def test_member_cost_cap_uses_source_override_when_declared() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    resolved = resolve_member_limits_for_source(manifest, "a")
    assert resolved.max_query_cost_bytes == 42.0


@pytest.mark.fast
def test_plan_records_resolved_member_limits_for_each_step() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    limits = federation_member_resolved_limits(plan, manifest)
    by_source = {item.source_id: item for item in limits}
    assert by_source["a"].row_cap == 1000
    assert by_source["b"].row_cap == 2000


@pytest.mark.fast
def test_plan_fingerprints_embed_resolved_limits() -> None:
    manifest = parse_federation_manifest(_manifest_with_limits(), include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", source_id="a"), "b": _graph("right_t", source_id="b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    fingerprints = federation_plan_step_fingerprints(plan, intent_key_fn=intent_key, manifest=manifest)
    limits = federation_member_resolved_limits(plan, manifest)
    assert limits[0].row_cap == 1000
    assert '"row_cap":1000' in fingerprints[0][1]
    assert '"max_query_cost_bytes":42.0' in fingerprints[0][1]
