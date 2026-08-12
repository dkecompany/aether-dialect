"""Tests for federation coordinator execution."""

from __future__ import annotations

import pandas as pd
import pytest

from aetherdialect._contracts_base import FederationInvariantError, FederationRuntimeError
from aetherdialect._contracts_core import FederatedPrepareOutcome, RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    execute_federation_coordinator,
    federation_plan_combine_hash,
    federation_plan_step_fingerprints,
    revalidate_prepared_federation_plan,
)
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import (
    plan_federated_intent,
    render_federation_glue,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline_execute import execute_federated_prepare
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils_intent import intent_key
from tests.conftest import duckdb_engine_identity


def _graph(table: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_exec",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


def test_render_join_coerces_key_types() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "CAST" in glue.upper()
    assert "BIGINT" in glue.upper()


@pytest.mark.fast
def test_render_and_execute_join(two_member_federation) -> None:
    fed = two_member_federation
    intent = RuntimeIntent(
        tables=[fed.left_table, fed.right_table],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, fed.composite, fed.manifest)
    assert plan.combine is not None
    glue = render_federation_glue(
        plan,
        {fed.left_source: "src_a", fed.right_source: "src_b"},
    )
    assert "JOIN" in glue.upper()
    frames = {
        fed.left_source: pd.DataFrame({"id": [1, 2]}),
        fed.right_source: pd.DataFrame({"id": [2, 3]}),
    }
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1


def test_render_glue_raises_when_join_frame_missing() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    try:
        render_federation_glue(plan, {"a": "src_a"})
        raised = False
    except FederationRuntimeError:
        raised = True
    assert raised


def test_coordinator_raises_on_row_cap_exceeded() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    frames = {
        "a": pd.DataFrame({"id": [1, 2, 3]}),
        "b": pd.DataFrame({"id": [2]}),
    }
    try:
        execute_federation_coordinator(frames, plan, row_cap=2)
        raised = False
    except FederationRuntimeError as exc:
        raised = True
        assert "row cap exceeded" in str(exc).lower()
    assert raised


def test_render_glue_honors_declared_join_kind_on_later_hops() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_left",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
                {"source_id": "c", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_c": "c"},
            "cross_source_joins": [
                {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
                {"left": "t_b.id", "right": "t_c.id", "kind": "left", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("t_a"), "b": _graph("t_b"), "c": _graph("t_c")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b", "t_c"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b", "c": "src_c"})
    assert " LEFT JOIN " in glue.upper()


def _prepared_outcome(plan, composite: SchemaGraph) -> FederatedPrepareOutcome:
    return FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="",
        composite_schema_graph_id=str(composite.schema_graph_id or ""),
        combine_hash=federation_plan_combine_hash(plan),
        step_fingerprints=federation_plan_step_fingerprints(plan, intent_key_fn=intent_key),
    )


def test_revalidate_prepared_plan_rejects_composite_drift() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    prepared = _prepared_outcome(plan, composite)
    drifted = SchemaGraph(
        tables=composite.tables,
        join_paths_multi=composite.join_paths_multi,
        schema_graph_id="sg_drifted",
    )
    with pytest.raises(FederationInvariantError, match="composite schema graph changed"):
        revalidate_prepared_federation_plan(prepared, drifted)


def test_execute_federated_prepare_revalidates_before_execution() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    prepared = _prepared_outcome(plan, composite)
    drifted = SchemaGraph(
        tables=composite.tables,
        join_paths_multi=composite.join_paths_multi,
        schema_graph_id="sg_drifted",
    )
    with pytest.raises(FederationInvariantError, match="composite schema graph changed"):
        execute_federated_prepare(
            prepared,
            drifted,
            dialect=object(),
            dialects_by_source=None,
        )


def test_federation_member_probe_qualification_uses_attached_schema() -> None:
    duckdb = pytest.importorskip("duckdb")
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_probe",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "connection": "memory", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "connection": "ext_schema", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    connection = duckdb.connect(":memory:")
    connection.execute("ATTACH ':memory:' AS ext_schema")
    connection.execute("CREATE TABLE ext_schema.right_t (id INTEGER)")
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        manifest,
        None,
        default,
        default_identity=duckdb_engine_identity(),
        native_connection=connection,
    )
    qualified = runtimes["b"].dialect.qualified_table_ref("right_t")
    assert "ext_schema" in qualified.lower()
    assert "main" not in qualified.lower().replace("ext_schema", "")


@pytest.mark.fast
def test_member_row_cap_raises_typed_limit_key() -> None:
    import pandas as pd

    from aetherdialect._contracts_base import FederationCapExceededError
    from aetherdialect._federation_execute import _enforce_federation_row_cap

    with pytest.raises(FederationCapExceededError) as excinfo:
        _enforce_federation_row_cap(pd.DataFrame({"id": [1, 2]}), 1, source_id="a")
    assert excinfo.value.limit_key == "row_cap"
    assert excinfo.value.source_id == "a"


@pytest.mark.fast
def test_member_execute_passes_source_timeout() -> None:
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from aetherdialect._contracts_core import FederatedPlan, SourceStep
    from aetherdialect._pipeline_execute import _execute_federation_source_step

    manifest = parse_federation_manifest(
        {
            **_MANIFEST,
            "sources": [
                {
                    "source_id": "a",
                    "engine": "duckdb",
                    "role": "owner",
                    "limits": {"timeout_ms": 12_345},
                },
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph({"a": _graph("left_t"), "b": _graph("right_t")}, manifest)
    step = SourceStep(
        source_id="a",
        sub_intent=RuntimeIntent(
            tables=["left_t"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
    )
    prepared_by_source = {
        "a": type("Prep", (), {"sub_intent": step.sub_intent, "sql": "SELECT 1", "structural_defaults": {}})(),
    }
    with patch("aetherdialect._pipeline_execute.execute_guarded_sql") as exec_mock:
        exec_mock.return_value = [{"id": 1}]
        with patch("aetherdialect._pipeline_execute.build_result_dataframe", return_value=pd.DataFrame({"id": [1]})):
            mock_dialect = MagicMock()
            mock_dialect.finalize_render.return_value = "SELECT 1"
            _execute_federation_source_step(
                step,
                prepared_by_source=prepared_by_source,
                composite_schema=composite,
                dialect_map={"a": mock_dialect},
                dialect=mock_dialect,
                manifest=manifest,
                executed={},
                plan=FederatedPlan(steps=(step,)),
                semijoin_cap=50_000,
                q_norm="",
                join_candidates=None,
                cmap=None,
                store=None,
                gate_kwargs={},
            )
    assert exec_mock.call_args.kwargs["timeout_ms"] == 12_345
