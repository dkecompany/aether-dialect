"""Tests for federation decomposition, ordering, and combine projection improvements."""

from __future__ import annotations

from unittest.mock import patch

from aetherdialect._contracts_base import FederationMappings, MulGroup, NormalizedExpr, WhereParam
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedStage,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._contracts_schema import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
    WindowRegistryStep,
    WindowSpec,
)
from aetherdialect._federation import (
    _build_source_sub_intent,
    compose_composite_graph,
    derive_execution_order_from_stages,
    federation_table_set,
    order_federation_execution_steps,
    parse_federation_manifest,
    plan_federated_intent,
    plan_federated_stages,
    render_federation_glue,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
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
    "federation_id": "fed_improve",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"t_a": "a", "t_b": "b"},
    "cross_source_joins": [
        {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
    ],
}


def test_sub_intent_partitions_cte_steps_by_source() -> None:
    members = {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")}
    manifest = enriched_manifest(members, _MANIFEST, member_graphs=members)
    composite = compose_composite_graph(members, manifest)
    cte_a = RuntimeCteStep(
        cte_name="cte_a",
        tables=["t_a"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    cte_b = RuntimeCteStep(
        cte_name="cte_b",
        tables=["t_b"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_b.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b", "cte_a", "cte_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte_a, cte_b],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    by_source = {step.source_id: step for step in plan.steps}
    assert [cte.cte_name for cte in by_source["a"].sub_intent.cte_steps or []] == ["cte_a"]
    assert [cte.cte_name for cte in by_source["b"].sub_intent.cte_steps or []] == ["cte_b"]


def test_sub_intent_filters_unreferenced_window_registry() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[
            WindowRegistryStep(
                registry_id="w01",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[NormalizedExpr.from_column("t_a.id")],
                ),
            ),
            WindowRegistryStep(
                registry_id="w02",
                window_spec=WindowSpec(
                    function="row_number",
                    partition_by=[NormalizedExpr.from_column("t_b.id")],
                ),
            ),
        ],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    by_source = {step.source_id: step for step in plan.steps}
    assert by_source["a"].sub_intent.window_registry == []
    assert by_source["b"].sub_intent.window_registry == []


def test_sub_intent_keeps_referenced_case_registry_only() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    branch = CaseWhenBranch(
        condition=WhereParam(
            left_expr=NormalizedExpr.from_column("t_a.id"),
            op=">",
            value_type="number",
            raw_value=0,
        ),
        result=NormalizedExpr(add_values=[]),
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr(column_ref="c01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        case_registry=[
            CaseRegistryStep(
                registry_id="c01",
                case_when=CaseWhenExpr(branches=[branch]),
            ),
            CaseRegistryStep(
                registry_id="c02",
                case_when=CaseWhenExpr(
                    branches=[
                        CaseWhenBranch(
                            condition=WhereParam(
                                left_expr=NormalizedExpr.from_column("t_b.id"),
                                op=">",
                                value_type="number",
                                raw_value=0,
                            ),
                            result=NormalizedExpr(add_values=[]),
                        ),
                    ],
                ),
            ),
        ],
    )
    table_set = federation_table_set(intent, composite, manifest)
    mappings = FederationMappings(version=2)
    members = {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")}
    step_a = _build_source_sub_intent(
        intent,
        "a",
        set(table_set.tables),
        dict(table_set.source_by_table),
        mappings,
        composite,
        manifest,
        multi_source=True,
        member_schema=members["a"],
    )
    step_b = _build_source_sub_intent(
        intent,
        "b",
        set(table_set.tables),
        dict(table_set.source_by_table),
        mappings,
        composite,
        manifest,
        multi_source=True,
        member_schema=members["b"],
    )
    assert step_a is not None
    assert step_b is not None
    assert [step.registry_id for step in step_a.sub_intent.case_registry or []] == ["c01"]
    assert step_b.sub_intent.case_registry == []


def test_derive_execution_order_respects_stage_depends_on() -> None:
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["t_a"], grain="many", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
                ),
            ),
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["t_b"], grain="many", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
                ),
            ),
        ),
        stages=(
            FederatedStage(stage_id="member_a", kind="member", source_ids=("a",)),
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",), depends_on=("member_a",)),
            FederatedStage(
                stage_id="coordinator", kind="coordinator", source_ids=("a", "b"), depends_on=("member_a", "member_b")
            ),
        ),
    )
    assert derive_execution_order_from_stages(plan) == ("a", "b")


def test_order_federation_execution_steps_uses_stage_dependencies() -> None:
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["t_b"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                    limit=1,
                ),
            ),
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["t_a"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
        stages=(
            FederatedStage(stage_id="member_a", kind="member", source_ids=("a",)),
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",), depends_on=("member_a",)),
        ),
    )
    ordered = order_federation_execution_steps(plan)
    assert [step.source_id for step in ordered] == ["a", "b"]


def test_plan_federated_stages_emits_member_and_coordinator() -> None:
    stages = plan_federated_stages({"a", "b"}, ())
    assert len(stages) == 3
    assert stages[0].kind == "member"
    assert stages[-1].kind == "coordinator"
    assert stages[-1].stage_id == "coordinator"
    assert stages[-1].depends_on == ("member_a", "member_b")


def test_plan_federated_stages_scalar_coordinator_stage_id() -> None:
    from aetherdialect._contracts_core import ResidualSpec, SelectCol
    from aetherdialect._intent_process import NormalizedExpr

    residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id")),),
        group_by_cols=(),
        order_by_cols=(),
    )
    stages = plan_federated_stages(
        {"a", "b"},
        (),
        intent=RuntimeIntent(
            tables=["t_a", "t_b"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        ),
        residual=residual,
    )
    assert stages[-1].stage_id == "coordinator_scalar"


def test_member_prepare_skips_expand_shared_pk_tables_for_refs() -> None:
    from aetherdialect._pipeline import generate_and_validate_sql
    from aetherdialect._templates import empty_template_store

    schema = _graph("t_a", "a")
    intent = RuntimeIntent(
        tables=["t_a"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    store = empty_template_store(schema.schema_graph_id)
    with patch("aetherdialect._pipeline.expand_shared_pk_tables_for_refs") as expand_mock:
        expand_mock.side_effect = lambda value, _schema: value
        with patch("aetherdialect._pipeline.build_deterministic_sql", return_value="SELECT id FROM t_a"):
            with patch(
                "aetherdialect._pipeline._run_sql_validation_cascade",
                return_value=(True, "", None, []),
            ):
                generate_and_validate_sql(
                    "q",
                    intent,
                    schema,
                    {},
                    {},
                    None,
                    store,
                    member_source_id="a",
                )
    expand_mock.assert_not_called()


def test_scalar_cross_source_avg_plans_residual() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("avg", "t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.residual is not None
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "NULLIF" in glue.upper()
    assert "SUM" in glue.upper()
    assert "COUNT" in glue.upper()


def test_scalar_cross_source_count_distinct_marks_ineligible() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[
            SelectCol(
                expr=NormalizedExpr(
                    add_groups=[
                        MulGroup(
                            multiply=[NormalizedExpr.from_column("t_a.id")],
                            agg_func="count",
                            distinct=True,
                        ),
                    ],
                ),
            ),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "cross-source aggregate not supported: count(distinct t_a.id)"


def test_scalar_cross_source_count_is_eligible() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    assert len(plan.steps) == 2
    assert plan.residual is not None


def test_render_federation_glue_uses_explicit_columns_when_projected() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("t_a.id")),
            SelectCol(expr=NormalizedExpr.from_column("t_b.id")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    glue = render_federation_glue(plan, {"a": "src_a", "b": "src_b"}, schema=composite)
    assert "SELECT *" not in glue
    assert 'l."id"' in glue
    assert 'r."id"' in glue
