"""Physical key defects must refuse rather than render partial ON clauses."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import (
    EngineContext,
    FederationCoordinatorConfig,
    FederationCrossSourceJoin,
    FederationDeclarationError,
    SchemaInvariantError,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect
from aetherdialect._federation import FederationManifest, validate_cross_source_keys_on_graph
from aetherdialect._schema_graph import refuse_incompatible_catalog_foreign_keys
from aetherdialect._schema_overrides import build_schema_graph_with_diff
from aetherdialect._sql_gen import JoinColumnCountMismatchError, _join_edges_from_signature

pytestmark = pytest.mark.usefixtures("stub_schema_llm_classifier")


class _FullBuildStubDialect(Dialect):
    name = "stub"

    def __init__(self, reflected_sg: SchemaGraph) -> None:
        super().__init__(MagicMock())
        self._reflected = reflected_sg

    def compute_ddl_probe(self, engine_context: EngineContext) -> str:
        return "probe-1"

    def reflect_schema_graph(
        self,
        *,
        include: Any = "tables",
        allow_objects: Any = None,
        deny_objects: Any = None,
        sql_file: Any = None,
    ) -> SchemaGraph:
        return copy.deepcopy(self._reflected)

    def profile_schema(self, sg: SchemaGraph) -> None:
        pass


@pytest.mark.fast
def test_join_signature_with_unequal_column_counts_raises() -> None:
    sig = ["orders.id,line_no->customers.id"]
    with pytest.raises(JoinColumnCountMismatchError):
        _join_edges_from_signature(sig, ["catalog_fk"], "orders", None)


@pytest.mark.fast
def test_catalog_fk_type_refusal_runs_before_join_path_computation(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = TableMetadata(
        name="parent",
        columns={"id": ColumnMetadata(name="id", data_type="varchar", value_type="string", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    child = TableMetadata(
        name="child",
        columns={"pid": ColumnMetadata(name="pid", data_type="integer", value_type="integer")},
        primary_key=[],
        foreign_keys=[
            FKEdge(src_table="child", src_cols=["pid"], dst_table="parent", dst_cols=["id"]),
        ],
    )
    schema = SchemaGraph(
        tables={"parent": parent, "child": child},
        join_paths_multi={},
        effective_structural_hash="fk-type",
    )
    paths_computed: list[bool] = []

    def _track_recompute(_tables: object) -> dict[str, object]:
        paths_computed.append(True)
        return {}

    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(tmp_path / "schema_graph.json.gz"))
    monkeypatch.setattr(
        "aetherdialect._schema_overrides.recompute_join_paths_multi",
        _track_recompute,
    )
    dialect = _FullBuildStubDialect(schema)
    with pytest.raises(SchemaInvariantError, match="incompatible value types"):
        build_schema_graph_with_diff(dialect, EngineContext())
    assert paths_computed == []


@pytest.mark.fast
def test_catalog_foreign_key_with_incompatible_types_refuses_at_graph_build() -> None:
    parent = TableMetadata(
        name="parent",
        columns={"id": ColumnMetadata(name="id", data_type="varchar", value_type="string", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    child = TableMetadata(
        name="child",
        columns={"pid": ColumnMetadata(name="pid", data_type="integer", value_type="integer")},
        primary_key=[],
        foreign_keys=[
            FKEdge(src_table="child", src_cols=["pid"], dst_table="parent", dst_cols=["id"]),
        ],
    )
    schema = SchemaGraph(
        tables={"parent": parent, "child": child},
        join_paths_multi={},
        effective_structural_hash="fk-type",
    )
    with pytest.raises(SchemaInvariantError, match="incompatible value types"):
        refuse_incompatible_catalog_foreign_keys(schema)


@pytest.mark.fast
def test_composite_cross_source_join_declaration_refuses_by_name() -> None:
    schema = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={
                    "a": ColumnMetadata(name="a", data_type="integer", value_type="integer", is_primary_key=True),
                    "b": ColumnMetadata(name="b", data_type="integer", value_type="integer", is_primary_key=True),
                },
                primary_key=["a", "b"],
                foreign_keys=[],
                source_id="left",
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", is_primary_key=True)
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="right",
            ),
        },
        join_paths_multi={},
        effective_structural_hash="composite-x",
    )
    manifest = FederationManifest(
        federation_id="fed",
        sources=(),
        table_namespace={"left_t": "left", "right_t": "right"},
        cross_source_joins=(
            FederationCrossSourceJoin(left="left_t.a", right="right_t.id", kind="inner", logical_key="a"),
        ),
        coordinator=FederationCoordinatorConfig(),
        aliases=(),
    )
    with pytest.raises(FederationDeclarationError, match="composite primary key"):
        validate_cross_source_keys_on_graph(schema, manifest)
