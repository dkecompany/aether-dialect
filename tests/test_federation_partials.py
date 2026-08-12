"""Regression tests for federation compose, cancel, and partial-failure paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants_runtime import (
    FEDERATION_COMPOSITE_RECONCILIATION_NOTE,
    REPHRASE_HINT_MESSAGES,
)
from aetherdialect._contracts_base import FederationConfigError, FederationDeclarationError, PredicateGroup, WhereParam
from aetherdialect._contracts_core import (
    AnchoredTemporalBind,
    FederationExecutionContext,
    GenerationPath,
    RephraseHint,
    RuntimeIntent,
    SourceStep,
    SqlGenerationOutcome,
)
from aetherdialect._contracts_schema import ColumnMetadata, FederationSourceBinding, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import (
    compare_replica_member_parity,
    compose_composite_graph,
    member_schema_slice,
)
from aetherdialect._federation_execute import (
    federation_member_execution_batches,
    federation_member_parallelism_cap,
)
from aetherdialect._federation_manifest import (
    build_federation_manifest_from_members,
    cached_or_suggest_cross_source_mappings,
    federation_prompt_fields_for_schema,
    federation_scaled_join_candidate_cap,
    federation_scaled_join_path_tie_cap,
    parse_federation_declaration,
    parse_federation_manifest,
    parse_federation_mappings,
    resolve_anchored_temporal_bind,
    schema_spans_multiple_sources,
    validate_federation_source_slug_uniqueness,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import (
    federation_turn_cancelled,
    pop_federation_execution_context,
    push_federation_execution_context,
)
from tests.federation_helpers import stamp_sandbox_payment_union_profiling


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
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


def _composite_graph() -> SchemaGraph:
    tables = {
        "left_t": TableMetadata(
            name="left_t",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
            member_source_ids=["a"],
        ),
        "right_t": TableMetadata(
            name="right_t",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id="b",
            member_source_ids=["b"],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


def test_sandbox_declaration_includes_union_replica_and_joins() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "data"
    manifest, mappings = parse_federation_declaration(
        json.loads((root / "federation_declaration.json").read_text(encoding="utf-8")),
    )
    assert manifest.federation_id == "sandbox_rental_shop"
    assert len(manifest.cross_source_joins) == 4
    payment_union = next(row for row in mappings.logical_tables if row.logical == "payment")
    customer_replica = next(row for row in mappings.logical_tables if row.logical == "customer")
    staff_replica = next(row for row in mappings.logical_tables if row.logical == "staff")
    assert payment_union.semantics == "union"
    assert customer_replica.authoritative_source == "storefront"
    assert staff_replica.authoritative_source == "storefront"
    crm_staff = next(member for member in staff_replica.members if member.source == "crm")
    assert set(crm_staff.columns) == {"staff_id", "first_name", "last_name", "store_id"}


def test_sandbox_replica_mappings_compose() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "data"
    manifest, mappings = parse_federation_declaration(
        json.loads((root / "federation_declaration.json").read_text(encoding="utf-8")),
    )
    customer_cols = {
        "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
        "first_name": ColumnMetadata(name="first_name", data_type="text", sensitivity="none"),
        "last_name": ColumnMetadata(name="last_name", data_type="text", sensitivity="none"),
        "address_id": ColumnMetadata(name="address_id", data_type="integer", sensitivity="none"),
        "create_date": ColumnMetadata(name="create_date", data_type="timestamp", sensitivity="none"),
        "email": ColumnMetadata(name="email", data_type="text", sensitivity="none"),
        "loyalty_tier": ColumnMetadata(name="loyalty_tier", data_type="text", sensitivity="none"),
        "store_id": ColumnMetadata(name="store_id", data_type="integer", sensitivity="none"),
    }
    crm_customer_cols = {
        "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
        "first_name": ColumnMetadata(name="first_name", data_type="text", sensitivity="none"),
        "last_name": ColumnMetadata(name="last_name", data_type="text", sensitivity="none"),
        "address_id": ColumnMetadata(name="address_id", data_type="integer", sensitivity="none"),
        "create_date": ColumnMetadata(name="create_date", data_type="timestamp", sensitivity="none"),
        "email_addr": ColumnMetadata(name="email_addr", data_type="text", sensitivity="none"),
        "loyalty_tier": ColumnMetadata(name="loyalty_tier", data_type="text", sensitivity="none"),
        "store_id": ColumnMetadata(name="store_id", data_type="integer", sensitivity="none"),
    }
    staff_cols = {
        "staff_id": ColumnMetadata(name="staff_id", data_type="integer", sensitivity="none"),
        "first_name": ColumnMetadata(name="first_name", data_type="text", sensitivity="none"),
        "last_name": ColumnMetadata(name="last_name", data_type="text", sensitivity="none"),
        "store_id": ColumnMetadata(name="store_id", data_type="integer", sensitivity="none"),
    }
    city_cols = {
        "city_id": ColumnMetadata(name="city_id", data_type="integer", sensitivity="none"),
        "city": ColumnMetadata(name="city", data_type="text", sensitivity="none"),
        "country_id": ColumnMetadata(name="country_id", data_type="integer", sensitivity="none"),
        "last_update": ColumnMetadata(name="last_update", data_type="timestamp", sensitivity="none"),
    }
    country_cols = {
        "country_id": ColumnMetadata(name="country_id", data_type="integer", sensitivity="none"),
        "country": ColumnMetadata(name="country", data_type="text", sensitivity="none"),
        "last_update": ColumnMetadata(name="last_update", data_type="timestamp", sensitivity="none"),
    }
    members = {
        "storefront": SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns=dict(customer_cols),
                    primary_key=["customer_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
                "staff": TableMetadata(
                    name="staff",
                    columns=dict(staff_cols),
                    primary_key=["staff_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
                "city": TableMetadata(
                    name="city",
                    columns=dict(city_cols),
                    primary_key=["city_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
                "country": TableMetadata(
                    name="country",
                    columns=dict(country_cols),
                    primary_key=["country_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none"),
                        "inventory_id": ColumnMetadata(name="inventory_id", data_type="integer", sensitivity="none"),
                        "last_update": ColumnMetadata(name="last_update", data_type="timestamp", sensitivity="none"),
                    },
                    primary_key=["rental_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
                "payment": TableMetadata(
                    name="payment",
                    columns={
                        "payment_id": ColumnMetadata(name="payment_id", data_type="integer", sensitivity="none"),
                        "rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none"),
                        "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
                        "payment_date": ColumnMetadata(name="payment_date", data_type="timestamp", sensitivity="none"),
                    },
                    primary_key=["payment_id"],
                    foreign_keys=[],
                    source_id="storefront",
                ),
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
        "catalog": SchemaGraph(
            tables={
                "inventory": TableMetadata(
                    name="inventory",
                    columns={
                        "inventory_id": ColumnMetadata(name="inventory_id", data_type="integer", sensitivity="none"),
                        "last_update": ColumnMetadata(name="last_update", data_type="timestamp", sensitivity="none"),
                    },
                    primary_key=["inventory_id"],
                    foreign_keys=[],
                    source_id="catalog",
                ),
                "item": TableMetadata(
                    name="item",
                    columns={"item_id": ColumnMetadata(name="item_id", data_type="integer", sensitivity="none")},
                    primary_key=["item_id"],
                    foreign_keys=[],
                    source_id="catalog",
                ),
                "city": TableMetadata(
                    name="city",
                    columns=dict(city_cols),
                    primary_key=["city_id"],
                    foreign_keys=[],
                    source_id="catalog",
                ),
                "country": TableMetadata(
                    name="country",
                    columns=dict(country_cols),
                    primary_key=["country_id"],
                    foreign_keys=[],
                    source_id="catalog",
                ),
                "payment": TableMetadata(
                    name="payment",
                    columns={
                        "payment_id": ColumnMetadata(name="payment_id", data_type="integer", sensitivity="none"),
                        "rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none"),
                        "amount": ColumnMetadata(name="amount", data_type="numeric", sensitivity="none"),
                        "payment_date": ColumnMetadata(name="payment_date", data_type="timestamp", sensitivity="none"),
                    },
                    primary_key=["payment_id"],
                    foreign_keys=[],
                    source_id="catalog",
                ),
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
        "logistics": SchemaGraph(
            tables={
                "purchase_line": TableMetadata(
                    name="purchase_line",
                    columns={"item_id": ColumnMetadata(name="item_id", data_type="integer", sensitivity="none")},
                    primary_key=["item_id"],
                    foreign_keys=[],
                    source_id="logistics",
                ),
                "delivery": TableMetadata(
                    name="delivery",
                    columns={"rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none")},
                    primary_key=["rental_id"],
                    foreign_keys=[],
                    source_id="logistics",
                ),
                "receipts": TableMetadata(
                    name="receipts",
                    columns={
                        "rcpt_id": ColumnMetadata(name="rcpt_id", data_type="integer", sensitivity="none"),
                        "rent_id": ColumnMetadata(name="rent_id", data_type="integer", sensitivity="none"),
                        "amt": ColumnMetadata(name="amt", data_type="numeric", sensitivity="none"),
                        "dt": ColumnMetadata(name="dt", data_type="timestamp", sensitivity="none"),
                    },
                    primary_key=["rcpt_id"],
                    foreign_keys=[],
                    source_id="logistics",
                ),
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
        "crm": SchemaGraph(
            tables={
                "promotion_redemption": TableMetadata(
                    name="promotion_redemption",
                    columns={"rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none")},
                    primary_key=["rental_id"],
                    foreign_keys=[],
                    source_id="crm",
                ),
                "customer": TableMetadata(
                    name="customer",
                    columns=dict(crm_customer_cols),
                    primary_key=["customer_id"],
                    foreign_keys=[],
                    source_id="crm",
                ),
                "staff": TableMetadata(
                    name="staff",
                    columns={key: staff_cols[key] for key in ("staff_id", "first_name", "last_name", "store_id")},
                    primary_key=["staff_id"],
                    foreign_keys=[],
                    source_id="crm",
                ),
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
    }
    stamp_sandbox_payment_union_profiling(members)
    composite = compose_composite_graph(members, manifest, mappings)
    assert "customer" in composite.tables
    assert set(composite.tables["customer"].member_source_ids) == {"storefront", "crm"}
    staff = composite.tables["staff"]
    assert set(staff.member_source_ids) == {"storefront", "crm"}
    assert staff.source_id == "storefront"


def test_replica_mapping_requires_authoritative_source() -> None:
    with pytest.raises(FederationConfigError, match="authoritative_source"):
        parse_federation_mappings(
            {
                "version": "0.2.3",
                "logical_tables": [
                    {
                        "logical": "entity",
                        "semantics": "replica",
                        "members": [
                            {"source": "a", "table": "entity_a", "columns": {"id": "id"}},
                            {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                        ],
                    }
                ],
            }
        )


def test_all_file_member_federation_rejected_at_build() -> None:
    members = {
        "files_a": MagicMock(
            dialect="csv",
            _schema_role="owner",
            _context_name="master",
            _connection="files_a",
            _named_connection="files_a",
        ),
        "files_b": MagicMock(
            dialect="csv",
            _schema_role="owner",
            _context_name="master",
            _connection="files_b",
            _named_connection="files_b",
        ),
    }
    declaration = parse_federation_manifest({"federation_id": "fed_files", "cross_source_joins": []})
    with pytest.raises(FederationDeclarationError, match="all file engines"):
        build_federation_manifest_from_members(
            members,
            declaration=declaration,
            member_graphs={
                "files_a": _graph("upload_a", source_id="files_a"),
                "files_b": _graph("upload_b", source_id="files_b"),
            },
        )


def test_federation_prompt_fields_for_composite_schema() -> None:
    assert federation_prompt_fields_for_schema(_composite_graph()) == {}
    assert schema_spans_multiple_sources(_graph("solo", source_id="solo")) is False


def test_rephrase_hints_cover_federation_outcomes() -> None:
    assert RephraseHint.FEDERATION_INELIGIBLE.value in REPHRASE_HINT_MESSAGES
    assert RephraseHint.FEDERATION_PARTIAL_FAILURE.value in REPHRASE_HINT_MESSAGES


def test_compose_scales_join_path_tie_cap_by_member_count() -> None:
    assert federation_scaled_join_path_tie_cap(4) >= federation_scaled_join_path_tie_cap(1) * 4


def test_compose_and_member_slice_use_storage_ceiling_at_build_time() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cap_asym",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"a": _graph("t_a", source_id="a"), "b": _graph("t_b", source_id="b")}
    with patch("aetherdialect._federation_compose.recompute_join_paths_multi", return_value={}) as mock_recompute:
        composite = compose_composite_graph(members, manifest)
        composite_call = mock_recompute.call_args
        assert composite_call is not None
        assert composite_call.kwargs.get("tie_cap") is None
        mock_recompute.reset_mock()
        member_schema_slice(composite, "a", manifest=manifest)
        slice_call = mock_recompute.call_args
        assert slice_call is not None
        assert slice_call.kwargs.get("tie_cap") is None
    assert federation_scaled_join_path_tie_cap(1) == PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP
    assert federation_scaled_join_candidate_cap(1) == PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP
    assert federation_scaled_join_candidate_cap(2) == PolicyConfig.JOIN_CANDIDATE_CROSS_PRODUCT_CAP * 2


def test_compose_composite_graph_stores_paths_without_build_time_tie_cap() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_scale",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
                {"source_id": "c", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_c": "c"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "a": _graph("t_a", source_id="a"),
        "b": _graph("t_b", source_id="b"),
        "c": _graph("t_c", source_id="c"),
    }
    composite = compose_composite_graph(members, manifest)
    assert composite.join_paths_multi
    assert federation_scaled_join_path_tie_cap(3) == PolicyConfig.JOIN_SHORTEST_PATH_TIE_CAP * 3


def test_gate_kwargs_load_named_context_from_member_tree(tmp_path) -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_gate",
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "context": "restricted", "role": "owner"},
                {"source_id": "beta", "engine": "duckdb", "context": "master", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha", "entity_b": "beta"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    member_dir = tmp_path / "aetherdialect" / "conn_alpha"
    member_dir.mkdir(parents=True)
    context_payload = {
        "version": "0.2.3",
        "allow_objects": ["entity_a"],
        "deny_objects": [],
        "deny_columns": [],
        "allow_columns": ["entity_a.id"],
    }
    (member_dir / "schema_context.restricted.json").write_text(json.dumps(context_payload), encoding="utf-8")
    owner = MagicMock()
    owner._artifacts_root = tmp_path
    owner._runtime_config = MagicMock(engine_context=MagicMock())
    owner._federation_source_runtimes = {"alpha": MagicMock(artifacts_dir=str(member_dir))}
    gates = MainExecutionOps.federation_gate_kwargs_by_source(owner, None, manifest)
    assert "entity_a" in gates["alpha"]["schema_context"].allow_objects


def test_federation_stores_honor_session_space_name() -> None:
    owner = MagicMock()
    owner._federation_source_runtimes = {"a": MagicMock(artifacts_dir="/tmp/member_a")}
    graphs = {"a": _graph("left_t", source_id="a")}
    with pytest.MonkeyPatch.context() as mp:
        captured: dict[str, str] = {}

        def _fake_load(*args, **kwargs):
            captured["space_name"] = kwargs.get("space_name", "")
            return MagicMock()

        mp.setattr("aetherdialect._templates_ops.TemplateOps.load_template_store", _fake_load)
        MainExecutionOps.federation_stores_by_source(owner, graphs, space_name="sales")
        assert captured["space_name"] == "sales"


def test_session_space_name_prefers_choice_port() -> None:
    port = MagicMock()
    port.space_name = "finance"
    owner = MagicMock()
    owner._context_name = "master"
    assert MainExecutionOps.session_space_name_for_federation(owner, port) == "finance"


def test_resolve_anchored_temporal_bind_for_date_window() -> None:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr={"column_ref": "orders.created_at"},
                    op=">=",
                    value_type="date_window",
                    raw_value={"unit": "day", "amount": 30},
                )
            ]
        ),
    )
    bind = resolve_anchored_temporal_bind(intent)
    assert isinstance(bind, AnchoredTemporalBind)
    assert bind.anchor_iso


def test_resolve_anchored_temporal_bind_absent_without_temporal_refs() -> None:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    assert resolve_anchored_temporal_bind(intent) is None


def test_mapping_suggestions_cache_reuses_member_hash(tmp_path) -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cache",
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "role": "owner"},
                {"source_id": "beta", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha", "entity_b": "beta"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "alpha": _graph("entity_a", source_id="alpha"),
        "beta": _graph("entity_b", source_id="beta"),
    }
    fed_dir = tmp_path / "fed_cache"
    fed_dir.mkdir()
    first = cached_or_suggest_cross_source_mappings(
        members,
        manifest,
        str(fed_dir),
    )
    second = cached_or_suggest_cross_source_mappings(
        members,
        manifest,
        str(fed_dir),
    )
    assert first == second


def test_composite_reconciliation_note_constant_present() -> None:
    assert "canonical wording" in FEDERATION_COMPOSITE_RECONCILIATION_NOTE.lower()


def test_missing_member_store_raises_instead_of_composite_fallback() -> None:
    owner = MagicMock()
    owner._federation_source_runtimes = {}
    graphs = {"a": _graph("left_t", source_id="a")}
    with pytest.raises(FederationConfigError, match="member store missing"):
        MainExecutionOps.federation_stores_by_source(owner, graphs)


def test_compare_replica_member_parity_detects_column_drift() -> None:
    members = {
        "a": SchemaGraph(
            tables={
                "entity_a": TableMetadata(
                    name="entity_a",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                        "b": ColumnMetadata(name="b", data_type="integer", sensitivity="none"),
                    },
                    primary_key=["id"],
                    foreign_keys=[],
                    source_id="a",
                )
            },
            join_paths_multi=recompute_join_paths_multi({}),
        ),
        "b": _graph("entity_b", source_id="b"),
    }
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "entity",
                    "semantics": "replica",
                    "authoritative_source": "a",
                    "members": [
                        {"source": "a", "table": "entity_a", "columns": {"id": "id", "b": "b"}},
                        {"source": "b", "table": "entity_b", "columns": {"id": "id"}},
                    ],
                }
            ],
        }
    )
    drift = compare_replica_member_parity(mappings, members)
    assert drift and "drift" in drift[0]


def test_federation_member_parallelism_cap_respects_manifest() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_parallel",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "a", "entity_b": "b"},
            "cross_source_joins": [],
            "coordinator": {"max_parallel_members": 2},
        },
        include_derived_roster=True,
    )
    assert federation_member_parallelism_cap(manifest, 5) == 2
    assert federation_member_parallelism_cap(None, 3) == 3


def test_execute_suspend_context_snapshots_turn_policy() -> None:
    snap = MagicMock()
    intent = RuntimeIntent(
        tables=["orders"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    gen_out = SqlGenerationOutcome("", True, GenerationPath.INTENT_DIRECT_MATCH, None)
    ctx = MainExecutionOps._sql_execute_suspend_context(
        snap,
        "SELECT 1",
        None,
        gen_out,
        None,
        False,
        intent,
    )
    policy = MainExecutionOps.snapshot_turn_policy()
    assert ctx.turn_policy is not None
    assert ctx.turn_policy.max_compose_repairs == policy.max_compose_repairs


def test_validate_federation_source_slug_uniqueness_raises_on_collision() -> None:
    sources = (
        FederationSourceBinding(source_id="a", engine="duckdb", connection="shared"),
        FederationSourceBinding(source_id="b", engine="duckdb", connection="shared"),
    )
    with pytest.raises(FederationConfigError, match="same connection slug"):
        validate_federation_source_slug_uniqueness(sources)


def test_federation_member_execution_batches_groups_shared_connections() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_batch",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "connection": "shared", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "connection": "shared", "role": "owner"},
                {"source_id": "c", "engine": "duckdb", "connection": "other", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_c": "c"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    steps = (
        SourceStep(
            source_id="a",
            sub_intent=RuntimeIntent(
                tables=["t_a"],
                grain="row_level",
                select_cols=[],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
        ),
        SourceStep(
            source_id="b",
            sub_intent=RuntimeIntent(
                tables=["t_b"],
                grain="row_level",
                select_cols=[],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
        ),
        SourceStep(
            source_id="c",
            sub_intent=RuntimeIntent(
                tables=["t_c"],
                grain="row_level",
                select_cols=[],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
        ),
    )
    batches = federation_member_execution_batches(steps, manifest)
    assert len(batches) == 2
    first_ids = {step.source_id for step in batches[0]}
    assert first_ids.isdisjoint({step.source_id for step in batches[1]})


@pytest.mark.fast
def test_cancel() -> None:
    """Same-thread ContextVar cancel path used by coordinator partial- failure handling."""
    ctx = FederationExecutionContext(plan_id="plan-1")
    token = push_federation_execution_context(ctx)
    try:
        assert not federation_turn_cancelled()
        ctx.cancel()
        assert federation_turn_cancelled()
    finally:
        pop_federation_execution_context(token)
