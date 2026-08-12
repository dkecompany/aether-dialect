"""Tests for timezone-aware timestamp handling in the federation coordinator."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH
from aetherdialect._contracts_base import (
    FederationDeclarationError,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    _coordinator_duckdb_type_to_pyarrow,
    _coordinator_relation_column_types_from_names,
    probe_federation_member_connections,
)
from aetherdialect._federation_manifest import (
    binding_from_member_engine,
    emit_federation_member_timezone_mismatch_diagnostics,
    federation_manifest_document,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import enriched_manifest, federation_member_graph, stamp_union_disjointness_profiling


def _graph(
    table: str,
    *,
    source_id: str,
    created_at_type: str,
) -> object:
    tables = {
        table: TableMetadata(
            name=table,
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "created_at": ColumnMetadata(name="created_at", data_type=created_at_type, sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


@pytest.mark.fast
def test_aware_column_keeps_offset_through_transfer() -> None:
    schema = _graph("events", source_id="a", created_at_type="timestamptz")
    column_types = _coordinator_relation_column_types_from_names(
        ("created_at",),
        "a",
        schema=schema,
        plan=None,
    )
    assert column_types == [("created_at", "TIMESTAMP WITH TIME ZONE")]
    arrow_type = _coordinator_duckdb_type_to_pyarrow("TIMESTAMP WITH TIME ZONE")
    assert str(arrow_type) == "timestamp[us, tz=UTC]"


@pytest.mark.fast
def test_mixed_awareness_union_refused() -> None:
    members = {
        "a": federation_member_graph(
            "events_a",
            source_id="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "created_at": ColumnMetadata(name="created_at", data_type="timestamptz", sensitivity="none"),
            },
        ),
        "b": federation_member_graph(
            "events_b",
            source_id="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "created_at": ColumnMetadata(name="created_at", data_type="timestamp", sensitivity="none"),
            },
        ),
    }
    for source_id, table_name, key_col, samples in (
        ("a", "events_a", "id", ("a1", "a2")),
        ("b", "events_b", "id", ("b1", "b2")),
        ("a", "events_a", "created_at", ("ca1", "ca2")),
        ("b", "events_b", "created_at", ("cb1", "cb2")),
    ):
        stamp_union_disjointness_profiling(
            members[source_id].tables[table_name],
            key_col=key_col,
            overlap_sample=samples,
        )
    manifest = enriched_manifest(
        members,
        {
            "federation_id": "fed_tz_mixed",
            "cross_source_joins": [],
        },
        member_graphs=members,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "events",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "events_a", "columns": {"id": "id", "created_at": "created_at"}},
                        {"source": "b", "table": "events_b", "columns": {"id": "id", "created_at": "created_at"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        compose_composite_graph(members, manifest, mappings)
    message = str(exc_info.value)
    assert "created_at" in message
    assert "timestamptz" in message.lower() or "time zone" in message.lower()
    assert "timestamp" in message.lower()


def _mock_db_member(source_id: str, *, timezone_value: str | None) -> MagicMock:
    engine = MagicMock()
    engine.dialect = "postgresql"
    engine._named_connection = source_id
    engine._connection = source_id
    engine._context_name = "master"
    engine._schema_role = "owner"
    mock_conn = MagicMock()
    engine._execution_engine.connect.return_value.__enter__.return_value = mock_conn

    def fake_execute(stmt: object) -> MagicMock:
        sql = str(stmt)
        result = MagicMock()
        if "current_setting" in sql.lower() or "time zone" in sql.lower() or "timezone" in sql.lower():
            result.fetchone.return_value = (timezone_value,) if timezone_value is not None else (None,)
            result.scalar.return_value = timezone_value
        return result

    mock_conn.execute.side_effect = fake_execute
    return engine


@pytest.mark.fast
def test_member_timezone_recorded_in_roster() -> None:
    members = {
        "east": _mock_db_member("east", timezone_value="America/New_York"),
        "west": _mock_db_member("west", timezone_value="America/Los_Angeles"),
    }
    probe_federation_member_connections(members)
    assert members["east"]._session_timezone == "America/New_York"
    assert members["west"]._session_timezone == "America/Los_Angeles"

    bindings = tuple(binding_from_member_engine(engine) for _, engine in sorted(members.items()))
    manifest = replace(
        parse_federation_manifest({"federation_id": "fed_tz_roster", "cross_source_joins": []}),
        sources=bindings,
    )
    tz_by_source = {binding.source_id: binding.session_timezone for binding in manifest.sources}
    assert tz_by_source["east"] == "America/New_York"
    assert tz_by_source["west"] == "America/Los_Angeles"

    doc = federation_manifest_document(manifest, include_derived=True)
    serialized = {entry["source_id"]: entry.get("session_timezone") for entry in doc["sources"]}
    assert serialized["east"] == "America/New_York"
    assert serialized["west"] == "America/Los_Angeles"


@pytest.mark.fast
def test_mismatched_zones_emit_diagnostic() -> None:
    members = {
        "a": federation_member_graph(
            "events_a",
            source_id="a",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "created_at": ColumnMetadata(name="created_at", data_type="timestamp", sensitivity="none"),
            },
        ),
        "b": federation_member_graph(
            "events_b",
            source_id="b",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "created_at": ColumnMetadata(name="created_at", data_type="timestamp", sensitivity="none"),
            },
        ),
    }
    for source_id, table_name, key_col, samples in (
        ("a", "events_a", "id", ("a1", "a2")),
        ("b", "events_b", "id", ("b1", "b2")),
        ("a", "events_a", "created_at", ("ca1", "ca2")),
        ("b", "events_b", "created_at", ("cb1", "cb2")),
    ):
        stamp_union_disjointness_profiling(
            members[source_id].tables[table_name],
            key_col=key_col,
            overlap_sample=samples,
        )
    manifest = enriched_manifest(
        members,
        {
            "federation_id": "fed_tz_mismatch",
            "cross_source_joins": [],
        },
        member_graphs=members,
    )
    manifest = replace(
        manifest,
        sources=tuple(
            replace(binding, session_timezone=tz)
            for binding, tz in zip(
                manifest.sources,
                ("America/New_York", "America/Los_Angeles"),
                strict=True,
            )
        ),
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "events",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "events_a", "columns": {"id": "id", "created_at": "created_at"}},
                        {"source": "b", "table": "events_b", "columns": {"id": "id", "created_at": "created_at"}},
                    ],
                },
            ],
            "logical_columns": [],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    intent = RuntimeIntent(
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
    plan = plan_federated_intent(intent, composite, manifest, mappings)
    assert plan.ineligible_reason is None

    with patch("aetherdialect._federation_manifest.notify") as notify_mock:
        emit_federation_member_timezone_mismatch_diagnostics(manifest, plan, schema=composite)

    emitted_codes = [call.kwargs.get("code") for call in notify_mock.call_args_list]
    assert DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH in emitted_codes
    mismatch_call = next(
        call
        for call in notify_mock.call_args_list
        if call.kwargs.get("code") == DIAGNOSTIC_CODE_FEDERATION_MEMBER_TIMEZONE_MISMATCH
    )
    details = dict(mismatch_call.kwargs.get("details") or ())
    assert details.get("logical_column") == "created_at"
