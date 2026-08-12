"""Tests for federation decomposition repair batch: stats, freshness, roles, spaces."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from aetherdialect import OwnerOnlyOperationError
from aetherdialect._contracts_base import FederationConfigError, FederationDeclarationError
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FederationMappings,
    FederationPlanTemplate,
    FKEdge,
    InferenceTag,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._federation_compose import (
    _merge_enum_values,
    assert_federation_sql_history_warmup_allowed,
    compose_composite_graph,
    composite_schema_payload_counts,
    federation_composite_semantic_edges_hash,
)
from aetherdialect._federation_execute import (
    credit_federation_plan_accept,
    load_federation_composite_graph,
    mappings_replay_matches,
    persist_federation_tree,
    save_federation_plan_template,
)
from aetherdialect._federation_manifest import (
    federation_artifact_paths,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._federation_plan import qsim_intent_eligible_on_federation
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.federation_helpers import stamp_union_disjointness_profiling


def _table(
    name: str,
    *,
    source_id: str = "",
    columns: dict[str, ColumnMetadata] | None = None,
    foreign_keys: list[FKEdge] | None = None,
    enum_values: dict[str, list[str]] | None = None,
    profiling_hash: str = "",
    notes_sha256: str = "",
) -> SchemaGraph:
    table = TableMetadata(
        name=name,
        columns=columns
        or {
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=foreign_keys or [],
        source_id=source_id,
    )
    tables = {name: table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{name}",
        effective_structural_hash=f"eff_{source_id}_{name}",
        enum_values=enum_values,
        profiling_hash=profiling_hash,
        notes_sha256=notes_sha256,
    )


_PAYMENT_MANIFEST = {
    "federation_id": "fed_stats",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"payment_a": "a", "payment_b": "b"},
    "cross_source_joins": [],
}


def test_merge_enum_values_keys_by_member_source() -> None:
    members = {
        "a": _table("payment", source_id="a", enum_values={"status": ["open", "closed"]}),
        "b": _table("payment", source_id="b", enum_values={"status": ["pending"]}),
    }
    merged = _merge_enum_values(members)
    assert merged is not None
    assert merged["a::status"] == ["open", "closed"]
    assert merged["b::status"] == ["pending"]


def test_collapse_unions_column_statistics_from_all_members() -> None:
    manifest = parse_federation_manifest(_PAYMENT_MANIFEST, include_derived_roster=True)
    members = {
        "a": _table(
            "payment",
            source_id="a",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    sensitivity="none",
                    frequent_values=["1"],
                    distinct_count=10,
                ),
            },
        ),
        "b": _table(
            "payment",
            source_id="b",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    sensitivity="none",
                    frequent_values=["2"],
                    distinct_count=7,
                ),
            },
        ),
    }
    stamp_union_disjointness_profiling(members["a"].tables["payment"], overlap_sample=("1", "2"))
    stamp_union_disjointness_profiling(members["b"].tables["payment"], overlap_sample=("3", "4"))
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment", "columns": {"id": "id"}},
                    ],
                },
            ],
        },
    )
    composite = compose_composite_graph(members, manifest, mappings)
    col = composite.tables["payment"].columns["id"]
    assert col.distinct_count == 0
    assert set(col.frequent_values) == {"1", "2"}


def test_load_composite_graph_rejects_stale_fingerprint() -> None:
    manifest = parse_federation_manifest(_PAYMENT_MANIFEST, include_derived_roster=True)
    mappings = FederationMappings(version="0.2.3")
    members = {
        "a": _table("payment", source_id="a", profiling_hash="p_a", notes_sha256="n_a"),
        "b": _table("payment", source_id="b", profiling_hash="p_b", notes_sha256="n_b"),
    }
    composite = compose_composite_graph(members, manifest, mappings)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=manifest,
            mappings=mappings,
            composite=composite,
            member_graphs=members,
        )
        assert load_federation_composite_graph(tmp) is not None
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["effective_structural_hash"] = "stale_hash"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        assert load_federation_composite_graph(tmp) is None
        members["a"] = _table("payment", source_id="a", profiling_hash="changed", notes_sha256="n_a")
        assert not mappings_replay_matches(tmp, members, manifest, mappings)


def test_cross_source_edge_visible_in_compose_payload() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_rel",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    members = {"a": _table("left_t", source_id="a"), "b": _table("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    payload = json.loads(composite.schema_payload_compose(["left_t", "right_t"]))
    assert payload["left_t"]["columns"]["id"]["fk"] == "right_t.id"
    paths = composite.join_paths_multi.get("left_t", {}).get("right_t", [])
    assert paths == []


def test_parse_manifest_rejects_non_identifier_source_id() -> None:
    bad = dict(_PAYMENT_MANIFEST)
    bad["sources"] = [
        {"source_id": "my-source", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ]
    bad["table_namespace"] = {"payment_a": "my-source", "payment_b": "b"}
    with pytest.raises(FederationDeclarationError, match="identifier-safe"):
        parse_federation_manifest(bad, include_derived_roster=True)


def test_composite_schema_payload_counts() -> None:
    manifest = parse_federation_manifest(_PAYMENT_MANIFEST, include_derived_roster=True)
    members = {
        "a": _table("payment", source_id="a", enum_values={"status": ["a", "b"]}),
        "b": _table("payment", source_id="b"),
    }
    composite = compose_composite_graph(members, manifest)
    counts = composite_schema_payload_counts(composite)
    assert counts["tables"] == 2
    assert counts["columns"] == 2
    assert counts["enum_types"] == 1
    assert counts["enum_labels"] == 2


def test_sql_history_warmup_refused_on_federation() -> None:
    engine = MagicMock()
    engine._is_aether_federation = True
    with pytest.raises(FederationConfigError, match="SQL history"):
        assert_federation_sql_history_warmup_allowed(engine)


def test_consumer_cannot_save_federation_plan_template() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        template = FederationPlanTemplate(
            plan_id="plan1",
            composite_schema_graph_id="cg1",
            intent_key="ik1",
            step_fingerprints=(("a", "fp1"),),
            combine_hash="hash1",
            question="q",
        )
        with pytest.raises(OwnerOnlyOperationError, match="save_federation_plan_template"):
            save_federation_plan_template(tmp, template, schema_role="consumer")
        with pytest.raises(OwnerOnlyOperationError, match="credit_federation_plan_accept"):
            credit_federation_plan_accept(tmp, "plan1", "q", schema_role="consumer")


def test_qsim_intent_requires_decomposable_plan() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_qsim",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"a": _table("left_t", source_id="a"), "b": _table("right_t", source_id="b")}
    composite = compose_composite_graph(members, manifest)
    assert not qsim_intent_eligible_on_federation(["left_t", "right_t"], composite, manifest)


def test_semantic_edges_hash_stable() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_hash",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _table("left_t", source_id="a"), "b": _table("right_t", source_id="b")},
        manifest,
    )
    first = federation_composite_semantic_edges_hash(composite)
    second = federation_composite_semantic_edges_hash(composite)
    assert first == second
    assert any(edge.inference_tag == InferenceTag.CROSS_SOURCE for edge in composite.tables["left_t"].foreign_keys)
