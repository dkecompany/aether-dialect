"""Shared helpers for federation unit tests."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FederationManifest,
    FederationMappings,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    build_federation_manifest_from_members,
    federation_declaration_document,
    parse_federation_manifest,
)
from aetherdialect._schema_graph import recompute_join_paths_multi

TWO_MEMBER_CROSS_JOIN_MANIFEST_DECL: dict[str, Any] = {
    "federation_id": "fed_two_member",
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@dataclass(frozen=True)
class TwoMemberFederation:
    """Minimal two-member federation used across federation unit tests."""

    manifest: FederationManifest
    member_graphs: dict[str, SchemaGraph]
    composite: SchemaGraph
    left_source: str = "a"
    right_source: str = "b"
    left_table: str = "left_t"
    right_table: str = "right_t"


def union_disjointness_key_column(
    *,
    name: str = "id",
    data_type: str = "integer",
    overlap_sample: tuple[str, ...] | list[str],
    row_count: int | None = None,
    **kwargs: Any,
) -> ColumnMetadata:
    """Build a union key column with disjointness profiling required for union collapse."""
    sample = list(overlap_sample)
    rc = row_count if row_count is not None else max(len(sample), 1)
    return ColumnMetadata(
        name=name,
        data_type=data_type,
        sensitivity="none",
        is_primary_key=kwargs.pop("is_primary_key", True),
        value_overlap_sample=sample,
        row_count=rc,
        distinct_count=rc,
        **kwargs,
    )


def stamp_union_disjointness_profiling(
    table: TableMetadata,
    *,
    key_col: str = "id",
    overlap_sample: tuple[str, ...] | list[str],
    row_count: int | None = None,
) -> None:
    """Stamp union key disjointness profiling onto an existing member table."""
    sample = list(overlap_sample)
    rc = row_count if row_count is not None else max(int(table.row_count or 0), len(sample), 1)
    table.row_count = rc
    col = table.columns.get(key_col)
    if col is None:
        table.columns[key_col] = union_disjointness_key_column(
            name=key_col,
            overlap_sample=sample,
            row_count=rc,
        )
        return
    col = copy.deepcopy(col)
    table.columns[key_col] = col
    col.value_overlap_sample = sample
    col.row_count = rc
    col.distinct_count = max(int(col.distinct_count or 0), len(sample))


def stamp_sandbox_payment_union_profiling(members: Mapping[str, SchemaGraph]) -> None:
    """Stamp disjointness profiling for the sandbox payment union across three members."""
    for source_id, table_name, key_col, samples in (
        ("storefront", "payment", "payment_id", ("sf_p1", "sf_p2")),
        ("catalog", "payment", "payment_id", ("cat_p1", "cat_p2")),
        ("logistics", "receipts", "rcpt_id", ("log_p1", "log_p2")),
    ):
        table = members[source_id].tables[table_name]
        table.columns = copy.deepcopy(table.columns)
        stamp_union_disjointness_profiling(table, key_col=key_col, overlap_sample=samples)


def union_member_graph_pair(
    left_table: str,
    right_table: str,
    *,
    left_source: str = "a",
    right_source: str = "b",
    left_samples: tuple[str, ...] = ("1", "2"),
    right_samples: tuple[str, ...] = ("3", "4"),
) -> dict[str, SchemaGraph]:
    """Build two one-table member graphs with union disjointness profiling on ``id``."""
    left_tbl = federation_member_table(left_table, source_id=left_source)
    right_tbl = federation_member_table(right_table, source_id=right_source)
    stamp_union_disjointness_profiling(left_tbl, overlap_sample=left_samples)
    stamp_union_disjointness_profiling(right_tbl, overlap_sample=right_samples)
    return {
        left_source: federation_member_graph(
            left_table,
            source_id=left_source,
            columns=left_tbl.columns,
        ),
        right_source: federation_member_graph(
            right_table,
            source_id=right_source,
            columns=right_tbl.columns,
        ),
    }


def federation_member_table(
    name: str,
    *,
    source_id: str,
    columns: Mapping[str, ColumnMetadata] | None = None,
    id_type: str = "integer",
) -> TableMetadata:
    """Build a single-column-or-custom federation member table."""
    cols = (
        dict(columns)
        if columns is not None
        else {
            "id": ColumnMetadata(name="id", data_type=id_type, sensitivity="none"),
        }
    )
    return TableMetadata(
        name=name,
        columns=cols,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def federation_member_graph(
    table: str,
    *,
    source_id: str,
    columns: Mapping[str, ColumnMetadata] | None = None,
    id_type: str = "integer",
    ddl_probe_hash: str = "",
) -> SchemaGraph:
    """Build a one-table member schema graph for federation tests."""
    tables = {
        table: federation_member_table(
            table,
            source_id=source_id,
            columns=columns,
            id_type=id_type,
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
        ddl_probe_hash=ddl_probe_hash,
    )


def two_member_cross_join_manifest(
    *,
    federation_id: str = "fed_two_member",
    left_table: str = "left_t",
    right_table: str = "right_t",
    left_source: str = "a",
    right_source: str = "b",
) -> FederationManifest:
    """Parse the standard two-member cross-join manifest declaration."""
    member_graphs = {
        left_source: federation_member_graph(left_table, source_id=left_source),
        right_source: federation_member_graph(right_table, source_id=right_source),
    }
    return enriched_manifest(
        member_graphs,
        {
            "federation_id": federation_id,
            "cross_source_joins": [
                {
                    "left": f"{left_table}.id",
                    "right": f"{right_table}.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        member_graphs=member_graphs,
    )


def build_two_member_federation(
    *,
    federation_id: str = "fed_two_member",
    left_table: str = "left_t",
    right_table: str = "right_t",
    left_source: str = "a",
    right_source: str = "b",
    left_ddl_probe_hash: str = "",
    right_ddl_probe_hash: str = "",
) -> TwoMemberFederation:
    """Compose the canonical two-member federation bundle used in unit tests."""
    member_graphs = {
        left_source: federation_member_graph(
            left_table,
            source_id=left_source,
            ddl_probe_hash=left_ddl_probe_hash,
        ),
        right_source: federation_member_graph(
            right_table,
            source_id=right_source,
            ddl_probe_hash=right_ddl_probe_hash,
        ),
    }
    manifest = enriched_manifest(
        member_graphs,
        {
            "federation_id": federation_id,
            "cross_source_joins": [
                {
                    "left": f"{left_table}.id",
                    "right": f"{right_table}.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        member_graphs=member_graphs,
    )
    composite = compose_composite_graph(member_graphs, manifest)
    return TwoMemberFederation(
        manifest=manifest,
        member_graphs=member_graphs,
        composite=composite,
        left_source=left_source,
        right_source=right_source,
        left_table=left_table,
        right_table=right_table,
    )


def write_federation_manifest_file(
    directory: str | Path,
    payload: Mapping[str, Any],
    *,
    filename: str = "federation_manifest.json",
) -> Path:
    """Write an authored federation manifest JSON file and return its path."""
    path = Path(directory) / filename
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_federation_mappings_file(
    directory: str | Path,
    payload: Mapping[str, Any],
    *,
    filename: str = "federation_mappings.json",
) -> Path:
    """Write a federation mappings JSON file and return its path."""
    path = Path(directory) / filename
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_federation_declaration_file(
    directory: str | Path,
    manifest_payload: Mapping[str, Any],
    mappings_payload: Mapping[str, Any] | None = None,
    *,
    filename: str = "federation_declaration.json",
) -> Path:
    """Write a unified federation declaration JSON file and return its path."""
    from aetherdialect._constants import FEDERATION_MAPPINGS_VERSION
    from aetherdialect._contracts_schema import FederationMappings
    from aetherdialect._federation_manifest import parse_federation_mappings

    manifest = parse_federation_manifest(manifest_payload)
    mappings = (
        parse_federation_mappings(mappings_payload)
        if mappings_payload is not None
        else FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    )
    path = Path(directory) / filename
    path.write_text(
        json.dumps(federation_declaration_document(manifest, mappings), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def enriched_manifest(
    members: Mapping[str, Any],
    declaration: Mapping[str, Any] | FederationManifest,
    *,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    mappings: FederationMappings | None = None,
) -> FederationManifest:
    """Parse a declaration and merge member-derived roster fields for compose tests."""
    if isinstance(declaration, FederationManifest):
        parsed = declaration
    else:
        decl = dict(declaration)
        decl.pop("sources", None)
        decl.pop("table_namespace", None)
        parsed = parse_federation_manifest(decl)
    graphs = dict(member_graphs) if member_graphs is not None else None
    if isinstance(declaration, FederationManifest) and declaration.sources:
        return declaration
    if graphs is not None and members and all(isinstance(value, SchemaGraph) for value in members.values()):
        from aetherdialect._federation_manifest import manifest_with_derived_roster

        return manifest_with_derived_roster(parsed, member_graphs=graphs, mappings=mappings)
    return build_federation_manifest_from_members(
        members,
        declaration=parsed,
        member_graphs=graphs,
        mappings=mappings,
    )


def hash_directory_tree(root: Path) -> str:
    """Return a stable digest of every file under *root*."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def template_store_partition_count(artifacts_dir: str, graph: SchemaGraph) -> int:
    """Return the number of template partitions persisted for *graph*."""
    from aetherdialect._templates_ops import TemplateOps

    store = TemplateOps.load_template_store(str(graph.schema_graph_id), graph, artifacts_dir=artifacts_dir)
    return len(store.partition_map)


def seed_member_template_stores(
    artifacts_root: str,
    manifest: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph],
) -> None:
    """Persist empty on-disk template stores for every federation member."""
    from aetherdialect._federation_execute import federation_source_artifacts_dir
    from aetherdialect._templates_ops import TemplateOps

    for binding in manifest.sources:
        graph = member_graphs[binding.source_id]
        artifacts_dir = federation_source_artifacts_dir(
            artifacts_root,
            binding,
            federation_id=str(manifest.federation_id or "") or None,
        )
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        store = TemplateOps.empty_template_store_for_space(str(graph.schema_graph_id), artifacts_dir=artifacts_dir)
        TemplateOps.save_template_store(store)


def build_staged_two_member_prepare_outcome(
    fed: TwoMemberFederation,
) -> tuple[Any, SchemaGraph, FederationManifest]:
    """Build a staged two-member prepare outcome for federation execution tests."""
    from aetherdialect._contracts_core import (
        FederatedPlan,
        FederatedPreparedStep,
        FederatedPrepareOutcome,
        FederatedStage,
        JoinSpec,
        RuntimeIntent,
        SourceStep,
    )
    from aetherdialect._federation_execute import federation_plan_combine_hash

    intent_a = RuntimeIntent(
        tables=[fed.left_table],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_b = RuntimeIntent(
        tables=[fed.right_table],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    step_a = SourceStep(source_id=fed.left_source, sub_intent=intent_a)
    step_b = SourceStep(source_id=fed.right_source, sub_intent=intent_b)
    plan = FederatedPlan(
        steps=(step_a, step_b),
        combine=JoinSpec(
            left_source=fed.left_source,
            right_source=fed.right_source,
            left_key="id",
            right_key="id",
            logical_key="id",
            kind="inner",
        ),
        stages=(
            FederatedStage(
                stage_id="member_a",
                kind="member",
                source_ids=(fed.left_source,),
            ),
            FederatedStage(
                stage_id="member_b",
                kind="member",
                source_ids=(fed.right_source,),
                depends_on=("member_a",),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=(fed.left_source, fed.right_source),
                depends_on=("member_a", "member_b"),
            ),
        ),
        scope_sources=frozenset({fed.left_source, fed.right_source}),
    )
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql=(f"SELECT a.id FROM {fed.left_table} a JOIN {fed.right_table} b ON a.id = b.id"),
        glue_sql="SELECT * FROM src_a INNER JOIN src_b USING (id)",
        steps=(
            FederatedPreparedStep(
                source_id=fed.left_source,
                sub_intent=intent_a,
                sql=f"SELECT id FROM {fed.left_table}",
            ),
            FederatedPreparedStep(
                source_id=fed.right_source,
                sub_intent=intent_b,
                sql=f"SELECT id FROM {fed.right_table}",
            ),
        ),
        composite_schema_graph_id=str(fed.composite.schema_graph_id),
        combine_hash=federation_plan_combine_hash(plan),
    )
    return prepared, fed.composite, fed.manifest
