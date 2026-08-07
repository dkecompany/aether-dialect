"""Regression tests for single-source federation SQL routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation import (
    build_federation_manifest_from_members,
    compose_composite_graph,
    federation_plan_is_degenerate,
    parse_federation_manifest,
    plan_federated_intent,
    resolve_federated_member_schema,
    source_ids_for_intent,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import generate_and_validate_sql, prepare_federated_sql_plan
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps
from tests.conftest import duckdb_engine_identity
from tests.federation_helpers import enriched_manifest


def _graph(table: str, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_single",
    "cross_source_joins": [
        {
            "left": "rental.id",
            "right": "film.id",
            "kind": "inner",
            "logical_key": "id",
        },
    ],
}


def _member_graphs() -> dict[str, SchemaGraph]:
    return {
        "storefront": _graph("rental", "storefront"),
        "catalog": _graph("film", "catalog"),
    }


def _composed_manifest():
    member_graphs = _member_graphs()
    return enriched_manifest(member_graphs, _MANIFEST, member_graphs=member_graphs)


def _runtime_manifest() -> object:
    member_graphs = _member_graphs()
    members = {
        "storefront": MagicMock(
            dialect="duckdb",
            _connection="storefront",
            _context_name="master",
            _schema_role="owner",
            _schema_graph=member_graphs["storefront"],
        ),
        "catalog": MagicMock(
            dialect="duckdb",
            _connection="catalog",
            _context_name="master",
            _schema_role="owner",
            _schema_graph=member_graphs["catalog"],
        ),
    }
    return build_federation_manifest_from_members(
        members,
        declaration=parse_federation_manifest(_MANIFEST, include_derived_roster=True),
        member_graphs=member_graphs,
    )


def test_build_federation_source_runtimes_bind_catalog_schema() -> None:
    manifest = _runtime_manifest()
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        manifest, None, default, default_identity=duckdb_engine_identity()
    )
    assert runtimes["storefront"].dialect.schema_name() == "main"
    assert runtimes["catalog"].dialect.schema_name() == "catalog"


def test_catalog_dialect_qualifies_film_to_catalog_schema() -> None:
    duckdb = pytest.importorskip("duckdb")
    manifest = _runtime_manifest()
    default = DialectRegistry.get("duckdb")
    connection = duckdb.connect(":memory:")
    connection.execute("ATTACH ':memory:' AS catalog")
    connection.execute("CREATE TABLE catalog.film (film_id INTEGER)")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        manifest,
        None,
        default,
        default_identity=duckdb_engine_identity(),
        native_connection=connection,
    )
    catalog_dialect = runtimes["catalog"].dialect
    qualified = catalog_dialect.finalize_render(
        "SELECT film_id FROM film",
        {},
    )
    assert "catalog" in qualified.lower()
    assert "main" not in qualified.lower().replace("catalog", "")


def test_single_source_catalog_intent_uses_catalog_schema() -> None:
    manifest = _composed_manifest()
    member_graphs = _member_graphs()
    composite = compose_composite_graph(member_graphs, manifest)
    intent = RuntimeIntent(
        tables=["film"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 1
    assert plan.steps[0].source_id == "catalog"
    assert federation_plan_is_degenerate(plan)
    assert source_ids_for_intent(intent, composite, None, manifest) == frozenset({"catalog"})
    catalog_schema = resolve_federated_member_schema(
        "catalog", composite, manifest=manifest, member_graphs=member_graphs
    )
    assert "film" in catalog_schema.tables


def test_degenerate_federated_prepare_matches_direct_member_sql() -> None:
    manifest = _composed_manifest()
    member_graphs = _member_graphs()
    composite = compose_composite_graph(member_graphs, manifest)
    intent = RuntimeIntent(
        tables=["film"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert federation_plan_is_degenerate(plan)
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        _runtime_manifest(), None, default, default_identity=duckdb_engine_identity()
    )
    store = TemplateOps.empty_template_store(composite.schema_graph_id)
    with patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    ):

        class _Owner:
            _federation_member_graphs = member_graphs
            _federation_dialects = {sid: runtime.dialect for sid, runtime in runtimes.items()}

        owner = _Owner()
        single_source = MainExecutionOps._federation_single_source_sql_context(
            owner,
            intent,
            composite,
            manifest,
            None,
            default,
        )
        assert single_source is not None
        source_dialect, member_schema = single_source
        direct = generate_and_validate_sql(
            "list films",
            intent,
            member_schema,
            {},
            {},
            source_dialect,
            store,
        )
        fed = prepare_federated_sql_plan(
            "list films",
            plan,
            composite,
            dialect=default,
            dialects_by_source={sid: runtime.dialect for sid, runtime in runtimes.items()},
            join_candidates={},
            cmap={},
            store=store,
            source_runtimes=runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
        )
    assert direct.success
    assert fed.success
    assert len(fed.steps) == 1
    # Byte-identical SQL: federated degenerate path must not rewrite member SQL.
    assert direct.sql == fed.display_sql
    assert fed.steps[0].sql == direct.sql


def test_federated_step_sql_context_prefers_runtime_dialect() -> None:
    from aetherdialect._federation import plan_federated_intent
    from aetherdialect._pipeline import _federated_step_sql_context

    manifest = _composed_manifest()
    composite = compose_composite_graph(_member_graphs(), manifest)
    intent = RuntimeIntent(
        tables=["rental", "film"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert len(plan.steps) == 2
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        _runtime_manifest(), None, default, default_identity=duckdb_engine_identity()
    )
    catalog_step = next(step for step in plan.steps if step.source_id == "catalog")
    dialect, _sub_schema = _federated_step_sql_context(
        catalog_step,
        composite,
        dialect=default,
        dialects_by_source=None,
        source_runtimes=runtimes,
        manifest=manifest,
    )
    assert dialect.schema_name() == "catalog"


def test_catalog_dialect_rewrites_stale_main_schema_qualifiers() -> None:
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        _runtime_manifest(), None, default, default_identity=duckdb_engine_identity()
    )
    catalog_dialect = runtimes["catalog"].dialect
    qualified = catalog_dialect.finalize_render(
        'SELECT film_id FROM main."film"',
        {},
    )
    assert "catalog" in qualified.lower()
    assert 'main."film"' not in qualified.lower()
