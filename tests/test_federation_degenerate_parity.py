"""Degenerate single-member federation must match standalone SQL and document learning differences."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import MulGroup, NormalizedExpr, OrderByCol, WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeCteStep, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    CteOutputColumnMeta,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._dialect import get_dialect, get_dialect_class
from aetherdialect._federation import federation_plan_is_degenerate, parse_federation_manifest, plan_federated_intent
from aetherdialect._main_execution import _build_federation_source_runtimes, _federation_single_source_sql_context
from aetherdialect._pipeline import generate_and_validate_sql, prepare_federated_sql_plan
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql
from aetherdialect._templates import empty_template_store
from tests.conftest import duckdb_engine_identity
from tests.test_federation_single_source import _composed_manifest, _member_graphs, _runtime_manifest

_MEMBER_ENGINES = (
    "duckdb",
    "sqlite",
    "postgresql",
    "mysql",
    "mariadb",
    "sqlserver",
    "snowflake",
    "bigquery",
    "redshift",
    "databricks",
)

_MANIFEST = {
    "federation_id": "fed_degenerate_parity",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b", "parent": "a", "child": "a"},
    "cross_source_joins": [],
}


def _member_graph(table: str, source_id: str, *, extra_tables: dict[str, TableMetadata] | None = None) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    if extra_tables:
        tables.update(extra_tables)
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="test-profiled",
    )


def _dialect_for_engine(engine: str):
    dialect_cls = get_dialect_class(engine)
    dialect = dialect_cls.__new__(dialect_cls)
    if engine == "databricks":
        dialect.config = SimpleNamespace(CATALOG="test_catalog", SCHEMA="test_schema")
    return dialect


def _composite_and_plan(intent: RuntimeIntent):
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    child = TableMetadata(
        name="child",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "parent_id": ColumnMetadata(
                name="parent_id",
                data_type="integer",
                sensitivity="none",
                fk_target=("parent", "id"),
            ),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
    )
    parent = TableMetadata(
        name="parent",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="a",
    )
    member_schema = _member_graph(
        "left_t",
        "a",
        extra_tables={"parent": parent, "child": child},
    )
    composite = SchemaGraph(
        tables={
            **member_schema.tables,
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi=recompute_join_paths_multi(member_schema.tables),
    )
    plan = plan_federated_intent(intent, composite, manifest, member_graphs={"a": member_schema})
    assert federation_plan_is_degenerate(plan)
    return member_schema, plan


def _intent_for_shape(shape: str) -> RuntimeIntent:
    if shape == "parameterized":
        return RuntimeIntent(
            tables=["left_t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=predicate_group_from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("left_t.id"),
                        op=">",
                        value_type="integer",
                        param_key="p1",
                        raw_value=0,
                    ),
                ]
            ),
            param_values={"p1": 0},
        )
    if shape == "cte":
        cte = RuntimeCteStep(
            cte_name="agg_cte",
            tables=["left_t"],
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["left_t.id"])])),
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["left_t.id"], agg_func="count")],
                    )
                ),
            ],
            output_columns=["id", "row_count"],
            group_by_cols=[NormalizedExpr(add_groups=[MulGroup(multiply=["left_t.id"])])],
            order_by_cols=[],
            where=None,
        )
        return RuntimeIntent(
            tables=["left_t"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["agg_cte.row_count"])])),
                SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["left_t.id"])])),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[cte],
        )
    if shape == "window":
        return RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            window_registry=[
                WindowRegistryStep(
                    registry_id="w01",
                    window_spec=WindowSpec(
                        function="row_number",
                        order_by=[OrderByCol(expr=NormalizedExpr.from_column("left_t.id"))],
                    ),
                )
            ],
        )
    if shape == "probe":
        ocm = {
            "parent_id": CteOutputColumnMeta(
                source="passthrough",
                lineage_phys_table="child",
                lineage_phys_column="parent_id",
                lineage_fk_to_table="parent",
                lineage_fk_to_column="id",
            )
        }
        probe = RuntimeCteStep(
            cte_name="probe",
            emission="semi_join",
            tables=["child"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
            output_columns=["parent_id"],
            output_column_metadata=ocm,
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        return RuntimeIntent(
            tables=["parent", "probe"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[probe],
        )
    raise AssertionError(f"unknown shape: {shape}")


@pytest.mark.fast
@pytest.mark.parametrize("engine", _MEMBER_ENGINES)
@pytest.mark.parametrize("shape", ("parameterized", "cte", "window", "probe"))
def test_degenerate_render_parity_matches_direct_for_shape(engine: str, shape: str) -> None:
    """Degenerate federated SQL rendering must be byte-identical to standalone for advanced shapes."""
    intent = _intent_for_shape(shape)
    member_schema, plan = _composite_and_plan(intent)
    dialect = _dialect_for_engine(engine)
    direct_sql = build_deterministic_sql(intent, schema=member_schema, dialect=dialect)
    federated_sql = build_deterministic_sql(plan.steps[0].sub_intent, schema=member_schema, dialect=dialect)
    assert federated_sql == direct_sql


@pytest.mark.fast
@pytest.mark.parametrize("engine", _MEMBER_ENGINES)
def test_degenerate_prepare_matches_direct_member_sql(engine: str) -> None:
    """Degenerate federated prepare must emit the same SQL as standalone across engines."""
    manifest = _composed_manifest()
    member_graphs = _member_graphs()
    composite = SchemaGraph(
        tables=member_graphs["storefront"].tables | member_graphs["catalog"].tables,
        join_paths_multi=recompute_join_paths_multi(
            member_graphs["storefront"].tables | member_graphs["catalog"].tables
        ),
    )
    intent = RuntimeIntent(
        tables=["rental"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("rental.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert federation_plan_is_degenerate(plan)
    default = get_dialect(engine) if engine == "duckdb" else _dialect_for_engine(engine)
    runtimes = _build_federation_source_runtimes(
        _runtime_manifest(),
        None,
        default,
        default_identity=duckdb_engine_identity(),
    )
    store = empty_template_store(composite.schema_graph_id)
    with patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    ):

        class _Owner:
            _federation_member_graphs = member_graphs
            _federation_dialects = {sid: runtime.dialect for sid, runtime in runtimes.items()}

        owner = _Owner()
        single_source = _federation_single_source_sql_context(
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
            "list rentals",
            intent,
            member_schema,
            {},
            {},
            source_dialect,
            store,
        )
        fed = prepare_federated_sql_plan(
            "list rentals",
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
    assert direct.success and fed.success
    assert direct.sql == fed.display_sql
    assert fed.steps[0].sql == direct.sql


@pytest.mark.fast
def test_degenerate_prepare_uses_unscoped_template_learning_kwargs() -> None:
    """Degenerate prepare passes member_source_id=None and persist_template_learning=False."""
    manifest = _composed_manifest()
    member_graphs = _member_graphs()
    composite = SchemaGraph(
        tables=member_graphs["storefront"].tables | member_graphs["catalog"].tables,
        join_paths_multi=recompute_join_paths_multi(
            member_graphs["storefront"].tables | member_graphs["catalog"].tables
        ),
    )
    intent = RuntimeIntent(
        tables=["rental"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("rental.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert federation_plan_is_degenerate(plan)
    default = get_dialect("duckdb")
    runtimes = _build_federation_source_runtimes(
        _runtime_manifest(),
        None,
        default,
        default_identity=duckdb_engine_identity(),
    )
    store = empty_template_store(composite.schema_graph_id)
    captured: list[dict[str, object]] = []

    def _capture_gen(*args, **kwargs):
        captured.append(dict(kwargs))
        return generate_and_validate_sql(*args, **kwargs)

    with patch(
        "aetherdialect._pipeline._run_sql_validation_cascade",
        return_value=(True, "", None, []),
    ):
        with patch("aetherdialect._pipeline.generate_and_validate_sql", side_effect=_capture_gen):
            prepare_federated_sql_plan(
                "list rentals",
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
    assert captured
    assert captured[0]["persist_template_learning"] is False
    assert captured[0]["member_source_id"] is None
