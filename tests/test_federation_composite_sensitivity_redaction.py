"""Composite federation graphs redact profile samples on hidden columns."""

from __future__ import annotations

from aetherdialect._contracts_base import SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import stamp_union_disjointness_profiling


def _payment_graph(
    source_id: str,
    *,
    table_name: str,
    email_sensitivity: str,
    email_samples: list[str],
) -> SchemaGraph:
    table = TableMetadata(
        name=table_name,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "email": ColumnMetadata(
                name="email",
                data_type="text",
                sensitivity=email_sensitivity,
                value_type="string",
                frequent_values=list(email_samples),
                value_overlap_sample=list(email_samples),
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table.name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


_MANIFEST = {
    "federation_id": "fed_union_pay",
    "cross_source_joins": [],
}


def test_hidden_member_sensitivity_strips_profile_samples_on_composite() -> None:
    members = {
        "a": _payment_graph(
            "a",
            table_name="payment_a",
            email_sensitivity="none",
            email_samples=["open@example.com"],
        ),
        "b": _payment_graph(
            "b",
            table_name="payment_b",
            email_sensitivity="hidden",
            email_samples=[],
        ),
    }
    stamp_union_disjointness_profiling(members["a"].tables["payment_a"], key_col="id", overlap_sample=("a1", "a2"))
    stamp_union_disjointness_profiling(members["b"].tables["payment_b"], key_col="id", overlap_sample=("b1", "b2"))
    stamp_union_disjointness_profiling(members["a"].tables["payment_a"], key_col="email", overlap_sample=("ea1",))
    stamp_union_disjointness_profiling(members["b"].tables["payment_b"], key_col="email", overlap_sample=("eb1",))
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_columns": [],
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {
                            "source": "a",
                            "table": "payment_a",
                            "columns": {"id": "id", "email": "email"},
                        },
                        {
                            "source": "b",
                            "table": "payment_b",
                            "columns": {"id": "id", "email": "email"},
                        },
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    email = composite.tables["payment"].columns["email"]
    assert email.sensitivity == SensitivityClassification.HIDDEN
    assert email.frequent_values == []
    assert email.value_overlap_sample == []
    assert email.min_val is None
    assert email.max_val is None
