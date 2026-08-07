"""Tests for deterministic cross-source mapping suggestions."""

from __future__ import annotations

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _cross_source_column_suggestion_score,
    _mapping_suggestion_cutoff,
    parse_federation_manifest,
    parse_federation_mappings,
    resolve_source_column_table,
    suggest_cross_source_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, source_id: str, email_sample: tuple[str, ...] = ()) -> SchemaGraph:
    table_meta = TableMetadata(
        name=table,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "email": ColumnMetadata(
                name="email",
                data_type="text",
                sensitivity="none",
                value_overlap_sample=email_sample,
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table: table_meta}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="test-profiled",
    )


_MANIFEST = {
    "federation_id": "fed_suggest",
    "sources": [
        {"source_id": "alpha", "engine": "duckdb", "role": "owner"},
        {"source_id": "beta", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"entity_a": "alpha", "entity_b": "beta"},
    "cross_source_joins": [],
}


def test_suggest_cross_source_mappings_finds_email_overlap() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _graph("entity_a", "alpha", ("a@x.com", "b@x.com")),
        "beta": _graph("entity_b", "beta", ("a@x.com", "c@x.com")),
    }
    suggestions = suggest_cross_source_mappings(members, manifest)
    assert suggestions
    top = suggestions[0]
    assert top.logical == "email"
    assert top.kind == "column"
    assert top.score > 0.5
    assert "entity_a.email" in top.members
    assert "entity_b.email" in top.members


def test_suggest_cross_source_mappings_groups_three_members() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_suggest_three",
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "role": "owner"},
                {"source_id": "beta", "engine": "duckdb", "role": "owner"},
                {"source_id": "gamma", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {
                "entity_a": "alpha",
                "entity_b": "beta",
                "entity_c": "gamma",
            },
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    shared = ("a@x.com", "b@x.com")
    members = {
        "alpha": _graph("entity_a", "alpha", shared),
        "beta": _graph("entity_b", "beta", shared),
        "gamma": _graph("entity_c", "gamma", shared),
    }
    suggestions = suggest_cross_source_mappings(members, manifest)
    email_suggestions = [row for row in suggestions if row.logical == "email"]
    assert email_suggestions
    members_set = set(email_suggestions[0].members)
    assert len(members_set) == 3
    assert members_set == {"entity_a.email", "entity_b.email", "entity_c.email"}


def test_resolve_source_column_table_prefers_declared_table() -> None:
    shared_email = ColumnMetadata(
        name="email",
        data_type="text",
        sensitivity="none",
        value_overlap_sample=("a@x.com",),
    )
    alpha_tables = {
        "orders": TableMetadata(
            name="orders",
            columns={"email": shared_email},
            primary_key=["email"],
            foreign_keys=[],
            source_id="alpha",
        ),
        "users": TableMetadata(
            name="users",
            columns={"email": shared_email},
            primary_key=["email"],
            foreign_keys=[],
            source_id="alpha",
        ),
    }
    graph = SchemaGraph(
        tables=alpha_tables,
        join_paths_multi=recompute_join_paths_multi(alpha_tables),
        profiling_hash="test-profiled",
    )
    assert resolve_source_column_table(graph, "alpha", "email") is None
    assert resolve_source_column_table(graph, "alpha", "users.email") == "users"
    assert resolve_source_column_table(graph, "alpha", "email", declared_table="users") == "users"


@pytest.mark.fast
def test_resolve_source_column_table_resolves_three_part_ref_with_manifest() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    graph = _graph("entity_a", "alpha", ("a@x.com",))
    assert (
        resolve_source_column_table(
            graph,
            "alpha",
            "alpha.entity_a.email",
            manifest=manifest,
        )
        == "entity_a"
    )


def test_suggest_cross_source_mappings_skips_when_mappings_present() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    members = {
        "alpha": _graph("entity_a", "alpha", ("a@x.com",)),
        "beta": _graph("entity_b", "beta", ("a@x.com",)),
    }
    mappings = parse_federation_mappings(
        {
            "logical_columns": [
                {
                    "logical": "email",
                    "unify_in_graph": True,
                    "members": ["entity_a.email", "entity_b.email"],
                }
            ]
        }
    )
    suggestions = suggest_cross_source_mappings(members, manifest, existing_mappings=mappings)
    assert suggestions == ()


def test_suggest_cross_source_mappings_rejects_sensitive_columns() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    alpha_tables = {
        "entity_a": TableMetadata(
            name="entity_a",
            columns={
                "ssn": ColumnMetadata(
                    name="ssn",
                    data_type="text",
                    sensitivity="restricted",
                    value_overlap_sample=("111-22-3333",),
                ),
            },
            primary_key=["ssn"],
            foreign_keys=[],
            source_id="alpha",
        )
    }
    beta_tables = {
        "entity_b": TableMetadata(
            name="entity_b",
            columns={
                "ssn": ColumnMetadata(
                    name="ssn",
                    data_type="text",
                    sensitivity="none",
                    value_overlap_sample=("111-22-3333",),
                ),
            },
            primary_key=["ssn"],
            foreign_keys=[],
            source_id="beta",
        )
    }
    members = {
        "alpha": SchemaGraph(
            tables=alpha_tables,
            join_paths_multi=recompute_join_paths_multi(alpha_tables),
            profiling_hash="test-profiled",
        ),
        "beta": SchemaGraph(
            tables=beta_tables,
            join_paths_multi=recompute_join_paths_multi(beta_tables),
            profiling_hash="test-profiled",
        ),
    }
    suggestions = suggest_cross_source_mappings(members, manifest)
    assert not suggestions


def test_build_federation_source_runtimes_binds_per_source_dialects() -> None:
    from unittest.mock import MagicMock

    from aetherdialect._dialect import DialectRegistry
    from aetherdialect._federation import build_federation_manifest_from_members
    from aetherdialect._main_execution import MainExecutionOps
    from tests.conftest import duckdb_engine_identity

    members = {
        "alpha": MagicMock(
            dialect="duckdb",
            _connection="alpha",
            _context_name="master",
            _schema_role="owner",
            _schema_graph=_graph("entity_a", "alpha"),
        ),
        "beta": MagicMock(
            dialect="duckdb",
            _connection="beta",
            _context_name="master",
            _schema_role="owner",
            _schema_graph=_graph("entity_b", "beta"),
        ),
    }
    manifest = build_federation_manifest_from_members(
        members,
        declaration=parse_federation_manifest(_MANIFEST, include_derived_roster=True),
    )
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        manifest, None, default, default_identity=duckdb_engine_identity()
    )
    assert set(runtimes) == {"alpha", "beta"}
    for runtime in runtimes.values():
        assert runtime.dialect is not None
        assert runtime.engine == "duckdb"


def test_mapping_suggestion_cross_source_cutoff_is_more_permissive() -> None:
    assert (
        PolicyConfig.FEDERATION_MAPPING_SUGGESTION_CROSS_SOURCE_CUTOFF
        < PolicyConfig.FEDERATION_MAPPING_SUGGESTION_WITHIN_SOURCE_CUTOFF
    )


def test_mapping_suggestion_overlap_only_pair_uses_source_specific_cutoff() -> None:
    left = ColumnMetadata(
        name="a",
        data_type="text",
        sensitivity="none",
        value_overlap_sample=("one", "two", "three", "four", "five"),
    )
    right = ColumnMetadata(
        name="b",
        data_type="text",
        sensitivity="none",
        value_overlap_sample=("one", "two", "three", "four", "other"),
    )
    score = _cross_source_column_suggestion_score(left, right, "a", "b")
    assert score >= _mapping_suggestion_cutoff(same_source=False)
    assert score < _mapping_suggestion_cutoff(same_source=True)
