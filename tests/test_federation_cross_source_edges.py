"""Tests for federated cross-source edge materialization and key-type clique validation."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError, InferenceTag
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _materialize_cross_source_edges,
    _types_compatible,
    compose_composite_graph,
    parse_federation_manifest,
    parse_federation_mappings,
    validate_cross_source_keys_on_graph,
)
from aetherdialect._schema_graph import recompute_join_paths_multi, table_pair_has_structural_fk


def _member_graph(
    table: str,
    source_id: str,
    *,
    data_type: str = "integer",
    value_type: str | None = None,
) -> SchemaGraph:
    col_kwargs: dict[str, object] = {
        "name": "id",
        "data_type": data_type,
        "sensitivity": "none",
    }
    if value_type is not None:
        col_kwargs["value_type"] = value_type
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(**col_kwargs)},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


def _n_member_fixture(n: int) -> tuple[object, object, dict[str, SchemaGraph]]:
    source_ids = [chr(ord("a") + i) for i in range(n)]
    table_names = [f"t_{sid}" for sid in source_ids]
    manifest = parse_federation_manifest(
        {
            "federation_id": f"fed_clique_{n}",
            "sources": [{"source_id": sid, "engine": "duckdb", "role": "owner"} for sid in source_ids],
            "table_namespace": dict(zip(table_names, source_ids, strict=True)),
            "cross_source_joins": [],
            "coordinator": {"row_cap": 1000},
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": 1,
            "logical_columns": [
                {
                    "logical": "shared_id",
                    "role": "join_key",
                    "unify_in_graph": True,
                    "members": [f"{tbl}.id" for tbl in table_names],
                },
            ],
        },
    )
    members = {sid: _member_graph(tbl, sid) for sid, tbl in zip(source_ids, table_names, strict=True)}
    return manifest, mappings, members


def _validation_schema(
    value_types_by_table: dict[str, str],
    *,
    data_types_by_table: dict[str, str] | None = None,
) -> SchemaGraph:
    """Build a composite-shaped graph with per-table logical join-key types."""
    tables: dict[str, TableMetadata] = {}
    for table, value_type in value_types_by_table.items():
        data_type = (data_types_by_table or {}).get(table, "integer")
        tables[table] = TableMetadata(
            name=table,
            columns={
                "shared_id": ColumnMetadata(
                    name="shared_id",
                    data_type=data_type,
                    value_type=value_type,
                    sensitivity="none",
                ),
            },
            primary_key=["shared_id"],
            foreign_keys=[],
            source_id=table.removeprefix("t_"),
        )
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_validate_keys",
        effective_structural_hash="eff_validate_keys",
    )


def _logical_join_key_edges(edges: list) -> list[tuple[str, str, str, str]]:
    return sorted(
        (
            edge.src_table,
            edge.src_cols[0],
            edge.dst_table,
            edge.dst_cols[0],
        )
        for edge in edges
        if edge.inference_tag == InferenceTag.CROSS_SOURCE
    )


@pytest.mark.fast
def test_logical_join_key_two_members_yields_one_edge() -> None:
    manifest, mappings, _members = _n_member_fixture(2)
    edges = _materialize_cross_source_edges(manifest, mappings)
    pairs = _logical_join_key_edges(edges)
    assert len(pairs) == 1
    src_tbl, src_col, dst_tbl, dst_col = pairs[0]
    assert src_tbl != dst_tbl
    assert {src_col, dst_col} == {"shared_id"}
    assert pairs == sorted(pairs)
    assert len(set(pairs)) == 1


@pytest.mark.fast
def test_logical_join_key_three_members_yields_all_pairs() -> None:
    manifest, mappings, _members = _n_member_fixture(3)
    edges = _materialize_cross_source_edges(manifest, mappings)
    pairs = _logical_join_key_edges(edges)
    assert len(pairs) == 3
    tables = {tbl for pair in pairs for tbl in (pair[0], pair[2])}
    assert tables == {"t_a", "t_b", "t_c"}
    undirected = {frozenset({p[0], p[2]}) for p in pairs}
    assert undirected == {
        frozenset({"t_a", "t_b"}),
        frozenset({"t_a", "t_c"}),
        frozenset({"t_b", "t_c"}),
    }


@pytest.mark.fast
def test_logical_join_key_four_members_yields_six_pairs() -> None:
    manifest, mappings, _members = _n_member_fixture(4)
    edges = _materialize_cross_source_edges(manifest, mappings)
    pairs = _logical_join_key_edges(edges)
    assert len(pairs) == 6
    undirected = {frozenset({p[0], p[2]}) for p in pairs}
    expected = {
        frozenset(pair)
        for pair in (
            ("t_a", "t_b"),
            ("t_a", "t_c"),
            ("t_a", "t_d"),
            ("t_b", "t_c"),
            ("t_b", "t_d"),
            ("t_c", "t_d"),
        )
    }
    assert undirected == expected


@pytest.mark.fast
def test_logical_join_key_edges_are_deterministic() -> None:
    manifest, mappings, _members = _n_member_fixture(4)
    first = _logical_join_key_edges(_materialize_cross_source_edges(manifest, mappings))
    second = _logical_join_key_edges(_materialize_cross_source_edges(manifest, mappings))
    assert first == second


@pytest.mark.fast
def test_logical_join_key_clique_makes_peripheral_members_joinable() -> None:
    manifest, mappings, members = _n_member_fixture(3)
    composite = compose_composite_graph(members, manifest, mappings)
    assert table_pair_has_structural_fk(composite, "t_b", "t_c")
    assert table_pair_has_structural_fk(composite, "t_a", "t_b")
    assert table_pair_has_structural_fk(composite, "t_a", "t_c")


@pytest.mark.fast
def test_validate_keys_rejects_nontransitive_peripheral_pair() -> None:
    """A↔B and A↔C compatible while B↔C is not must fail (star would miss B↔C)."""
    manifest, mappings, _members = _n_member_fixture(3)
    schema = _validation_schema(
        {
            "t_a": "numeric",
            "t_b": "int",
            "t_c": "float",
        },
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "logical column 'shared_id'" in message
    assert "t_b.shared_id" in message
    assert "t_c.shared_id" in message
    assert "(int)" in message
    assert "(float)" in message
    assert "t_a.shared_id" not in message


@pytest.mark.fast
def test_validate_keys_four_members_names_peripheral_incompatible_pair() -> None:
    manifest, mappings, _members = _n_member_fixture(4)
    schema = _validation_schema(
        {
            "t_a": "numeric",
            "t_b": "int",
            "t_c": "int",
            "t_d": "float",
        },
    )
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "logical column 'shared_id'" in message
    assert "t_b.shared_id" in message
    assert "t_d.shared_id" in message
    assert "(int)" in message
    assert "(float)" in message


@pytest.mark.fast
def test_validate_keys_reported_pair_is_deterministic() -> None:
    manifest, mappings, _members = _n_member_fixture(4)
    schema = _validation_schema(
        {
            "t_a": "numeric",
            "t_b": "int",
            "t_c": "int",
            "t_d": "float",
        },
    )
    messages: list[str] = []
    for _ in range(5):
        with pytest.raises(FederationDeclarationError) as exc_info:
            validate_cross_source_keys_on_graph(schema, manifest, mappings)
        messages.append(str(exc_info.value))
    assert len(set(messages)) == 1
    assert "t_b.shared_id" in messages[0]
    assert "t_d.shared_id" in messages[0]


@pytest.mark.fast
def test_validate_keys_compatible_three_members_pass() -> None:
    manifest, mappings, members = _n_member_fixture(3)
    compose_composite_graph(members, manifest, mappings)
    schema = _validation_schema({"t_a": "integer", "t_b": "integer", "t_c": "integer"})
    validate_cross_source_keys_on_graph(schema, manifest, mappings)


@pytest.mark.fast
def test_validate_keys_compatible_four_members_pass() -> None:
    manifest, mappings, members = _n_member_fixture(4)
    compose_composite_graph(members, manifest, mappings)
    schema = _validation_schema(
        {"t_a": "integer", "t_b": "integer", "t_c": "integer", "t_d": "integer"},
    )
    validate_cross_source_keys_on_graph(schema, manifest, mappings)


@pytest.mark.fast
def test_validate_keys_rejects_missing_value_type() -> None:
    manifest, mappings, _members = _n_member_fixture(2)
    schema = _validation_schema(
        {"t_a": "integer", "t_b": ""},
        data_types_by_table={"t_a": "integer", "t_b": ""},
    )
    with pytest.raises(
        FederationDeclarationError,
        match=r"type could not be determined for 't_b\.shared_id'",
    ):
        validate_cross_source_keys_on_graph(schema, manifest, mappings)


@pytest.mark.fast
def test_types_compatible_rejects_empty_or_missing() -> None:
    assert _types_compatible("", "integer") is False
    assert _types_compatible("integer", "") is False
    assert _types_compatible("  ", "varchar") is False
    assert _types_compatible("integer", "bigint") is True
    assert _types_compatible("varchar", "text") is True
    assert _types_compatible("integer", "date") is False


def _two_member_join_manifest() -> tuple[object, object]:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_join_key_unique",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {"left": "t_a.join_id", "right": "t_b.join_id", "kind": "inner", "logical_key": "join_id"},
            ],
            "coordinator": {"row_cap": 1000},
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings({"version": 1, "logical_columns": []})
    return manifest, mappings


def _join_key_schema(
    *,
    right_row_count: int = 10,
    right_distinct_count: int = 10,
    right_is_unique: bool = False,
) -> SchemaGraph:
    tables = {
        "t_a": TableMetadata(
            name="t_a",
            columns={
                "join_id": ColumnMetadata(
                    name="join_id",
                    data_type="integer",
                    value_type="integer",
                    sensitivity="none",
                    is_primary_key=True,
                ),
            },
            primary_key=["join_id"],
            foreign_keys=[],
            source_id="a",
            row_count=10,
        ),
        "t_b": TableMetadata(
            name="t_b",
            columns={
                "join_id": ColumnMetadata(
                    name="join_id",
                    data_type="integer",
                    value_type="integer",
                    sensitivity="none",
                    is_unique=right_is_unique,
                    row_count=right_row_count,
                    distinct_count=right_distinct_count,
                ),
            },
            primary_key=[],
            foreign_keys=[],
            source_id="b",
            row_count=right_row_count,
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.fast
def test_cross_source_join_non_unique_right_key_refuses_at_declaration() -> None:
    manifest, mappings = _two_member_join_manifest()
    schema = _join_key_schema(right_row_count=10, right_distinct_count=4)
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "t_b.join_id" in message
    assert "t_a.join_id" in message
    assert "not unique" in message


@pytest.mark.fast
def test_cross_source_join_unprofilable_key_refuses_at_declaration() -> None:
    manifest, mappings = _two_member_join_manifest()
    schema = _join_key_schema(right_row_count=0, right_distinct_count=0)
    with pytest.raises(FederationDeclarationError) as exc_info:
        validate_cross_source_keys_on_graph(schema, manifest, mappings)
    message = str(exc_info.value)
    assert "t_b.join_id" in message
    assert "t_a.join_id" in message
    assert "could not be established" in message
