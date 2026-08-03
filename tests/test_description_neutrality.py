"""I12: model-facing descriptions must not leak federation or physical vocabulary."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES
from aetherdialect._contracts_base import ConfigError, DescriptionOwner, LogicalIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    collect_federation_description_forbidden_tokens,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
    raise_if_descriptions_name_federation_sources,
)
from aetherdialect._intent_process import _build_intent_compose_prompt
from aetherdialect._schema_catalog import (
    _enrich_fk_column_descriptions,
    description_neutrality_violations,
    sanitize_description_text,
    sanitize_schema_graph_descriptions,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import federation_member_graph


@pytest.mark.fast
def test_sanitize_description_strips_source_id_token() -> None:
    forbidden = frozenset({"storefront"})
    cleaned = sanitize_description_text("Payments recorded in storefront ledger", forbidden)
    assert "storefront" not in cleaned.lower()
    assert cleaned


@pytest.mark.fast
def test_sanitize_description_strips_original_name_token() -> None:
    forbidden = frozenset({"Customer Account"})
    cleaned = sanitize_description_text("Row from Customer Account table", forbidden)
    assert "customer account" not in cleaned.lower()


@pytest.mark.fast
def test_description_neutrality_violations_detect_physical_alias() -> None:
    hits = description_neutrality_violations(
        "Imported from legacy_payment staging area",
        frozenset({"legacy_payment"}),
    )
    assert hits == ["legacy_payment"]


@pytest.mark.fast
def test_compose_composite_sanitizes_member_descriptions_with_source_id() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_desc",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment": "storefront", "inventory": "catalog"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "storefront": federation_member_graph(
            "payment",
            source_id="storefront",
            columns={
                "payment_id": ColumnMetadata(
                    name="payment_id",
                    data_type="integer",
                    sensitivity="none",
                    description="storefront payment rows",
                ),
            },
            id_type="integer",
        ),
        "catalog": federation_member_graph("inventory", source_id="catalog"),
    }
    members["storefront"].tables["payment"].primary_key = ["payment_id"]
    members["storefront"].tables["payment"].description = "storefront payment ledger"
    with pytest.raises(ConfigError, match="must not name a source or member"):
        compose_composite_graph(members, manifest)


@pytest.mark.fast
def test_raise_if_descriptions_name_federation_sources_rejects_member_text() -> None:
    graph = federation_member_graph("payment", source_id="storefront")
    graph.tables["payment"].description = "storefront ledger"
    with pytest.raises(ConfigError, match="must not name a source or member"):
        raise_if_descriptions_name_federation_sources(
            {"storefront": graph},
            ["storefront", "catalog"],
        )


@pytest.mark.fast
def test_fk_enrich_description_sanitized_with_forbidden_tokens() -> None:
    from aetherdialect._federation import _resolve_composite_table_names

    legacy_customer = TableMetadata(
        name="legacy_customer",
        columns={
            "customer_id": ColumnMetadata(name="customer_id", data_type="integer", sensitivity="none"),
            "name": ColumnMetadata(name="name", data_type="varchar", sensitivity="none"),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
    )
    order = TableMetadata(
        name="order",
        columns={
            "order_id": ColumnMetadata(name="order_id", data_type="integer", sensitivity="none"),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                sensitivity="none",
                is_foreign_key=True,
                fk_target=("legacy_customer", "customer_id"),
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[],
    )
    member_graph = SchemaGraph(
        tables={"legacy_customer": legacy_customer, "order": order},
        join_paths_multi={},
    )
    _enrich_fk_column_descriptions(member_graph)
    raw_desc = member_graph.tables["order"].columns["customer_id"].description or ""
    assert "legacy_customer" in raw_desc.lower()
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_fk_alias",
            "sources": [{"source_id": "src", "engine": "duckdb", "role": "owner"}],
            "aliases": {"customer": {"source": "src", "table": "legacy_customer"}},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"src": member_graph}
    composite_names = _resolve_composite_table_names(members, manifest)
    tokens = collect_federation_description_forbidden_tokens(
        members,
        manifest,
        parse_federation_mappings({"version": 2}),
        composite_names,
    )
    sanitize_schema_graph_descriptions(member_graph, tokens)
    cleaned = member_graph.tables["order"].columns["customer_id"].description or ""
    assert "legacy_customer" not in cleaned.lower()


@pytest.mark.fast
def test_federation_compose_capabilities_avoid_union_vocabulary() -> None:
    joined = "\n".join(FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES).lower()
    for phrase in ("logical union", "replica", "logical table mappings", "federation"):
        assert phrase not in joined
    assert any("do not emit sql union" in cap for cap in joined.split("\n"))


@pytest.mark.fast
def test_compose_prompt_payload_descriptions_avoid_source_id_tokens() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_prompt",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "role": "owner"},
                {"source_id": "catalog", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment": "storefront", "inventory": "catalog"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "storefront": federation_member_graph("payment", source_id="storefront"),
        "catalog": federation_member_graph("inventory", source_id="catalog"),
    }
    members["storefront"].tables["payment"].description = "Payment records"
    composite = compose_composite_graph(members, manifest)
    prompt = json.loads(
        _build_intent_compose_prompt(
            LogicalIntent(tables=("payment",), select="count rows"),
            composite.schema_payload_compose(["payment"]),
            schema_graph=composite,
        ),
    )
    joined = json.dumps(prompt).lower()
    assert "storefront" not in joined


@pytest.mark.fast
def test_collect_federation_description_forbidden_tokens_includes_aliases() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_alias",
            "sources": [{"source_id": "src", "engine": "duckdb", "role": "owner"}],
            "aliases": {"payment": {"source": "src", "table": "legacy_payment"}},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "src": federation_member_graph(
            "legacy_payment",
            source_id="src",
            columns={
                "payment_id": ColumnMetadata(name="payment_id", data_type="integer", sensitivity="none"),
            },
        ),
    }
    from aetherdialect._federation import _resolve_composite_table_names

    composite_names = _resolve_composite_table_names(members, manifest)
    tokens = collect_federation_description_forbidden_tokens(
        members,
        manifest,
        parse_federation_mappings({"version": 2}),
        composite_names,
    )
    assert "legacy_payment" in tokens
    assert "src" in tokens


@pytest.mark.fast
def test_sanitize_schema_graph_descriptions_clears_owner_when_empty() -> None:
    table = TableMetadata(
        name="payment",
        columns={
            "payment_id": ColumnMetadata(
                name="payment_id",
                data_type="integer",
                sensitivity="none",
                description="storefront",
                description_owner=DescriptionOwner.CATALOG,
            ),
        },
        primary_key=["payment_id"],
        foreign_keys=[],
        description="storefront",
        description_owner=DescriptionOwner.NOTES,
    )
    graph = SchemaGraph(
        tables={"payment": table},
        join_paths_multi=recompute_join_paths_multi({"payment": table}),
    )
    sanitize_schema_graph_descriptions(graph, frozenset({"storefront"}))
    assert graph.tables["payment"].description == ""
    assert graph.tables["payment"].description_owner is None
    assert graph.tables["payment"].columns["payment_id"].description == ""
    assert graph.tables["payment"].columns["payment_id"].description_owner is None
