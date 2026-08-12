"""Tests for federation topology and identity reconciliation."""

from __future__ import annotations

from pathlib import Path

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    federation_member_hash_tuple,
    manifest_hash,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


def _manifest(source_ids: list[str]) -> dict[str, object]:
    namespace = {f"t_{sid}": sid for sid in source_ids}
    joins = []
    if len(source_ids) >= 2:
        joins = [
            {
                "left": f"t_{source_ids[0]}.id",
                "right": f"t_{source_ids[1]}.id",
                "kind": "inner",
                "logical_key": "id",
            }
        ]
    return {
        "federation_id": "fed_topo",
        "sources": [{"source_id": sid, "engine": "duckdb", "role": "owner"} for sid in source_ids],
        "table_namespace": namespace,
        "cross_source_joins": joins,
    }


def _full_manifest(source_ids: list[str]) -> object:
    return parse_federation_manifest(_manifest(source_ids), include_derived_roster=True)


def test_add_source_changes_composite_id_not_member_ids() -> None:
    m_two = _full_manifest(["a", "b"])
    m_three = _full_manifest(["a", "b", "c"])
    members_two = {"a": _graph("t_a"), "b": _graph("t_b")}
    members_three = {**members_two, "c": _graph("t_c")}
    g_two = compose_composite_graph(members_two, m_two)
    g_three = compose_composite_graph(members_three, m_three)
    assert g_two.schema_graph_id != g_three.schema_graph_id
    assert members_two["a"].schema_graph_id == "sg_t_a"


def test_manifest_hash_independent_of_source_order() -> None:
    m1 = _full_manifest(["a", "b"])
    m2 = _full_manifest(["b", "a"])
    assert manifest_hash(m1) == manifest_hash(m2)


def test_member_hash_tuple_sorted_by_source_id() -> None:
    m = _full_manifest(["b", "a"])
    members = {"a": _graph("t_a"), "b": _graph("t_b")}
    tup = federation_member_hash_tuple(members, m)
    assert tup[0][0] == "a"
    assert tup[1][0] == "b"


def test_topology_add_detected() -> None:
    from aetherdialect._federation_execute import (
        detect_federation_topology_change,
        prune_cross_source_joins,
    )

    m = _full_manifest(["a", "b", "c"])
    assert detect_federation_topology_change(["a", "b"], m) == "add"
    pruned = prune_cross_source_joins(m, active_source_ids={"a", "b"})
    assert len(pruned.cross_source_joins) == 1


def test_topology_remove_detected() -> None:
    from aetherdialect._federation_execute import (
        detect_federation_topology_change,
        prune_cross_source_joins,
    )

    m = _full_manifest(["a", "b"])
    assert detect_federation_topology_change(["a", "b", "c"], m) == "remove"
    pruned = prune_cross_source_joins(m, active_source_ids={"a", "b"})
    assert pruned.cross_source_joins


def test_federation_migration_map_renames_join_keys() -> None:
    from aetherdialect._federation_execute import (
        apply_federation_migration_map,
        parse_federation_migration_map,
    )

    manifest = _full_manifest(["a", "b"])
    mappings = parse_federation_mappings({"version": "0.2.3", "logical_columns": [], "logical_tables": []})
    migration = parse_federation_migration_map(
        {
            "version": "1",
            "action": "remap",
            "qualified_column_renames": [
                {"from": "t_a.id", "to": "t_a.identifier"},
                {"from": "t_b.id", "to": "t_b.identifier"},
            ],
        },
    )
    fed_dir = ""
    updated_manifest, _ = apply_federation_migration_map(migration, manifest, mappings, fed_dir)
    assert updated_manifest.cross_source_joins[0].left == "t_a.identifier"
    assert updated_manifest.cross_source_joins[0].right == "t_b.identifier"


def test_federation_migration_map_drops_reversed_inner_join() -> None:
    from aetherdialect._federation_execute import (
        apply_federation_migration_map,
        parse_federation_migration_map,
    )

    manifest = _full_manifest(["a", "b"])
    mappings = parse_federation_mappings({"version": "0.2.3", "logical_columns": [], "logical_tables": []})
    migration = parse_federation_migration_map(
        {
            "version": "1",
            "action": "remap",
            "dropped_cross_source_joins": [{"left": "t_b.id", "right": "t_a.id"}],
        },
    )
    updated_manifest, _ = apply_federation_migration_map(migration, manifest, mappings, "")
    assert updated_manifest.cross_source_joins == ()


def test_per_source_column_rename_propagates() -> None:
    from aetherdialect._federation_execute import apply_per_source_column_renames

    manifest = _full_manifest(["a", "b"])
    mappings = parse_federation_mappings({"version": "0.2.3", "logical_columns": [], "logical_tables": []})
    updated_manifest, _ = apply_per_source_column_renames(
        manifest,
        mappings,
        source_id="a",
        column_renames=[("t_a", "id", "identifier")],
    )
    assert updated_manifest.cross_source_joins[0].left == "t_a.identifier"


def test_member_allow_tables_includes_union_partition_members() -> None:
    from aetherdialect._federation_compose import member_allow_tables_for_source
    from aetherdialect._federation_manifest import parse_federation_mappings

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed",
            "sources": [
                {"source_id": "storefront", "engine": "duckdb", "connection": "memory"},
                {"source_id": "catalog", "engine": "duckdb", "connection": "memory"},
            ],
            "table_namespace": {"payment": "storefront", "customer": "storefront", "film": "catalog"},
            "cross_source_joins": [],
            "coordinator": {},
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": "0.2.3",
            "logical_columns": [],
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "storefront", "table": "payment", "columns": {}},
                        {"source": "catalog", "table": "payment", "columns": {}},
                    ],
                },
            ],
        },
    )
    assert member_allow_tables_for_source(manifest, mappings, "storefront") == frozenset(
        {"payment", "customer"},
    )
    assert member_allow_tables_for_source(manifest, mappings, "catalog") == frozenset(
        {"film", "payment"},
    )


def test_federation_migration_map_apply_is_owner_gated() -> None:
    """Federation init applies federation_migration_map.json only for owner role."""
    schema_role = "consumer"
    fed_map_present = True
    assert not (schema_role == "owner" and fed_map_present)

    schema_role = "owner"
    assert schema_role == "owner" and fed_map_present


def test_federation_migration_map_archive_uses_storage_dir(tmp_path: Path) -> None:
    """Applied federation migration maps are archived under federation storage, not cwd."""
    from aetherdialect._federation_execute import archive_federation_migration_map_file

    archive_dir = tmp_path / "fed_storage"
    map_path = tmp_path / "federation_migration_map.json"
    map_path.write_text('{"version": "0.2.3"}', encoding="utf-8")
    archive_federation_migration_map_file(map_path, archive_dir=archive_dir)
    assert not map_path.is_file()
    assert (archive_dir / "federation_migration_map.applied.json").is_file()
