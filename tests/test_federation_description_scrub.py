"""Merged federation descriptions and enum labels must not reach composite prompts unscrubbed."""

from __future__ import annotations

import json

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, LogicalIntent
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._intent_loop import _build_intent_compose_prompt
from tests.federation_helpers import federation_member_graph


@pytest.mark.fast
def test_compose_scrubs_table_description_naming_member_identifier() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_desc_scrub",
            "sources": [{"source_id": "storefront", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"payment": "storefront"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "storefront": federation_member_graph("payment", source_id="storefront"),
    }
    members["storefront"].tables["payment"].description = "storefront payment ledger"
    composite = compose_composite_graph(members, manifest)
    assert "storefront" not in (composite.tables["payment"].description or "").lower()


@pytest.mark.fast
def test_compose_prompt_enum_labels_omit_member_identifiers() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_enum_scrub",
            "sources": [{"source_id": "storefront", "engine": "duckdb", "role": "owner"}],
            "table_namespace": {"payment": "storefront"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {
        "storefront": federation_member_graph(
            "payment",
            source_id="storefront",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "status": ColumnMetadata(name="status", data_type="varchar", sensitivity="none"),
            },
        ),
    }
    members["storefront"].enum_values = {"status": ["storefront_active", "storefront_pending"]}
    composite = compose_composite_graph(members, manifest)
    payload = json.loads(composite.schema_payload_compose(["payment"]))
    joined = json.dumps(payload).lower()
    assert "storefront" not in joined
    enum_types = payload.get("enum_types") or {}
    assert enum_types
    for labels in enum_types.values():
        for label in labels:
            assert "storefront" not in str(label).lower()


@pytest.mark.fast
def test_compose_prompt_table_descriptions_omit_member_identifiers() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_prompt_desc",
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
    members["catalog"].tables["inventory"].description = "Stock records"
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
    assert "catalog" not in joined
