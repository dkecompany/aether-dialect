"""Deep integration tests for staged federation execution and combine graph shape."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationInvariantError, FederationMappings
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedStage,
    JoinSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata, WindowRegistryStep, WindowSpec
from aetherdialect._federation import (
    _build_combine_join_tree,
    _collect_member_reducing_edges,
    _member_window_rows_are_final,
    _render_coordinator_spanning_cte_sql,
    _render_federation_combine_sql,
    _window_requires_coordinator,
    apply_projected_keys_to_intent,
    derive_execution_order_from_stages,
    derive_federation_stages_in_order,
    federation_plan_is_degenerate,
    order_federation_execution_steps,
    parse_federation_manifest,
    plan_federated_stages,
)
from aetherdialect._intent_expr import NormalizedExpr, SelectCol
from aetherdialect._schema_graph import recompute_join_paths_multi


@pytest.mark.fast
def test_build_combine_join_tree_star_topology() -> None:
    join_specs = (
        JoinSpec("a", "d", "id", "id", "id", "inner"),
        JoinSpec("b", "d", "id", "id", "id", "inner"),
    )
    tree = _build_combine_join_tree(join_specs, {"a", "b", "d"})
    assert tree.source_id == "d"
    assert len(tree.children) == 2


@pytest.mark.fast
def test_render_combine_sql_uses_graph_not_list_fold() -> None:
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="d",
                sub_intent=RuntimeIntent(
                    tables=["td"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
        combine=(
            JoinSpec("a", "d", "id", "id", "id", "inner"),
            JoinSpec("b", "d", "id", "id", "id", "inner"),
        ),
    )
    sql = _render_federation_combine_sql(plan, {"a": "src_a", "b": "src_b", "d": "src_d"})
    assert "JOIN" in sql.upper()
    assert "src_d" in sql
    assert "src_a" in sql


@pytest.mark.fast
def test_plan_stages_include_reducing_edges_and_spanning_cte() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cte",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[],
    )
    sources = {"a", "b"}
    source_by_table = dict(manifest.table_namespace)
    stages = plan_federated_stages(
        sources,
        (),
        intent=intent,
        source_by_table=source_by_table,
        manifest=manifest,
    )
    reducing = _collect_member_reducing_edges(manifest, FederationMappings(version=2), sources, intent, source_by_table)
    assert reducing
    member_stages = [stage for stage in stages if stage.kind == "member"]
    assert any(stage.reducing_edges for stage in member_stages)
    ordered = derive_federation_stages_in_order(
        FederatedPlan(steps=(), stages=stages),
    )
    assert ordered[-1].kind == "coordinator"


@pytest.mark.fast
def test_apply_projected_keys_remaps_distinct_index() -> None:
    col_a = SelectCol(expr=NormalizedExpr.from_column("ta.id"))
    col_b = SelectCol(expr=NormalizedExpr.from_column("tb.id"))
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[col_a, col_b],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        distinct_select_index=1,
    )
    updated = apply_projected_keys_to_intent(intent, ("ta.extra",))
    assert updated.distinct_select_index == 1


@pytest.mark.fast
def test_coordinator_spanning_cte_composes_join_graph() -> None:
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
                projected_keys=("id",),
            ),
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["tb"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
                projected_keys=("id",),
            ),
        ),
        combine=(JoinSpec("a", "b", "id", "id", "id", "inner"),),
        stages=(
            FederatedStage(
                stage_id="coordinator_cte",
                kind="cte",
                source_ids=("a", "b"),
                spanning_cte_names=("span_cte",),
            ),
        ),
    )
    sql = _render_coordinator_spanning_cte_sql(plan, {"a": "src_a", "b": "src_b"})
    assert "WITH span_cte AS" in sql
    assert "SELECT * FROM (" not in sql
    assert "_fed_span_base" not in sql
    assert "JOIN" in sql.upper()
    assert "src_a" in sql
    assert "src_b" in sql
    assert 'SELECT "id" FROM span_cte' in sql


@pytest.mark.fast
def test_spanning_cte_three_member_stage_order() -> None:
    source_by_table = {"ta": "a", "tb": "b", "tc": "c"}
    cte = RuntimeCteStep(
        cte_name="span_cte",
        tables=["ta", "tb"],
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb", "tc"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    stages = plan_federated_stages(
        {"a", "b", "c"},
        (),
        intent=intent,
        source_by_table=source_by_table,
    )
    ordered = derive_federation_stages_in_order(FederatedPlan(steps=(), stages=stages))
    kinds = [stage.kind for stage in ordered]
    assert kinds[:3] == ["member", "member", "member"]
    assert "cte" in kinds
    assert kinds.index("cte") < kinds.index("coordinator")
    cte_stage = next(stage for stage in ordered if stage.kind == "cte")
    assert cte_stage.source_ids == ("a", "b")
    assert cte_stage.depends_on == ("member_a", "member_b")
    coordinator = next(stage for stage in ordered if stage.kind == "coordinator")
    assert "coordinator_cte" in coordinator.depends_on
    assert "member_c" in coordinator.depends_on


@pytest.mark.fast
def test_diamond_depends_on_executes_in_valid_order() -> None:
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["tb"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="c",
                sub_intent=RuntimeIntent(
                    tables=["tc"],
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
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",)),
            FederatedStage(
                stage_id="member_c",
                kind="member",
                source_ids=("c",),
                depends_on=("member_a", "member_b"),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a", "b", "c"),
                depends_on=("member_a", "member_b", "member_c"),
            ),
        ),
    )
    member_order = derive_execution_order_from_stages(plan)
    assert member_order.index("a") < member_order.index("c")
    assert member_order.index("b") < member_order.index("c")
    full = derive_federation_stages_in_order(plan)
    ids = [stage.stage_id for stage in full]
    assert ids.index("member_a") < ids.index("member_c")
    assert ids.index("member_b") < ids.index("member_c")
    assert ids.index("member_c") < ids.index("coordinator")


@pytest.mark.fast
def test_stage_dependency_cycle_raises_typed_error() -> None:
    plan = FederatedPlan(
        steps=(),
        stages=(
            FederatedStage(
                stage_id="member_a",
                kind="member",
                source_ids=("a",),
                depends_on=("member_b",),
            ),
            FederatedStage(
                stage_id="member_b",
                kind="member",
                source_ids=("b",),
                depends_on=("member_a",),
            ),
        ),
    )
    with pytest.raises(FederationInvariantError, match="cycle"):
        derive_execution_order_from_stages(plan)
    with pytest.raises(FederationInvariantError, match="cycle"):
        derive_federation_stages_in_order(plan)


@pytest.mark.fast
def test_window_not_pushed_when_join_changes_multiplicity() -> None:
    source_by_table = {"ta": "a", "tb": "b"}
    combine = (JoinSpec("a", "b", "id", "id", "id", "inner"),)
    entry = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("ta.id")],
            order_by=[],
        ),
    )
    schema = SchemaGraph(
        tables={
            "ta": TableMetadata(
                name="ta",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "tb": TableMetadata(
                name="tb",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_win",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": source_by_table,
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    assert (
        _member_window_rows_are_final(
            "a",
            entry,
            source_by_table=source_by_table,
            manifest=manifest,
            schema=schema,
            combine=combine,
        )
        is False
    )
    assert _window_requires_coordinator(
        entry,
        source_by_table=source_by_table,
        manifest=manifest,
        schema=schema,
        combine=combine,
    )


@pytest.mark.fast
def test_window_pushed_when_rows_provably_final() -> None:
    source_by_table = {"ta": "a", "tb": "b"}
    combine = (JoinSpec("a", "b", "id", "a_id", "id", "left"),)
    entry = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(
            function="row_number",
            partition_by=[NormalizedExpr.from_column("ta.id")],
            order_by=[],
        ),
    )
    schema = SchemaGraph(
        tables={
            "ta": TableMetadata(
                name="ta",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "tb": TableMetadata(
                name="tb",
                columns={
                    "a_id": ColumnMetadata(
                        name="a_id",
                        data_type="integer",
                        sensitivity="none",
                        is_unique=True,
                    ),
                },
                primary_key=[],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_win_final",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": source_by_table,
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.a_id", "kind": "left", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    assert (
        _member_window_rows_are_final(
            "a",
            entry,
            source_by_table=source_by_table,
            manifest=manifest,
            schema=schema,
            combine=combine,
        )
        is True
    )
    assert not _window_requires_coordinator(
        entry,
        source_by_table=source_by_table,
        manifest=manifest,
        schema=schema,
        combine=combine,
    )


@pytest.mark.fast
def test_degenerate_single_member_plan_has_no_coordinator_stage() -> None:
    intent = RuntimeIntent(
        tables=["ta"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(
        steps=(SourceStep(source_id="a", sub_intent=intent, projected_keys=()),),
        combine=None,
        residual=None,
        stages=(),
        scope_sources=frozenset({"a"}),
    )
    assert federation_plan_is_degenerate(plan)
    assert plan_federated_stages({"a"}, plan.steps, intent=intent, source_by_table={"ta": "a"}) == ()


@pytest.mark.fast
def test_stage_order_deterministic_across_builds() -> None:
    source_by_table = {"ta": "a", "tb": "b"}
    cte = RuntimeCteStep(
        cte_name="span_cte",
        tables=["ta", "tb"],
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[cte],
    )
    first = plan_federated_stages({"b", "a"}, (), intent=intent, source_by_table=source_by_table)
    second = plan_federated_stages({"a", "b"}, (), intent=intent, source_by_table=source_by_table)
    assert first == second
    assert [stage.stage_id for stage in first] == [stage.stage_id for stage in second]


@pytest.mark.fast
def test_order_steps_uses_schema_selectivity() -> None:
    tables = {
        "ta": TableMetadata(
            name="ta",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    sensitivity="none",
                    distinct_ratio=0.1,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
        "tb": TableMetadata(
            name="tb",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    sensitivity="none",
                    distinct_ratio=0.9,
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="b",
        ),
    }
    schema = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["tb"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
                projected_keys=("id",),
            ),
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
                projected_keys=("id",),
            ),
        ),
        stages=(
            FederatedStage(
                stage_id="member_a",
                kind="member",
                source_ids=("a",),
            ),
            FederatedStage(
                stage_id="member_b",
                kind="member",
                source_ids=("b",),
            ),
        ),
    )
    ordered = order_federation_execution_steps(plan, schema=schema)
    assert ordered[0].source_id == "a"


@pytest.mark.fast
def test_federation_stage_execution_waves_respect_dependencies() -> None:
    from aetherdialect._federation import federation_execution_wave_member_steps, federation_stage_execution_waves

    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["tb"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
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
            FederatedStage(
                stage_id="member_b",
                kind="member",
                source_ids=("b",),
                depends_on=("member_a",),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a", "b"),
                depends_on=("member_a", "member_b"),
            ),
        ),
    )
    ordered = order_federation_execution_steps(plan)
    waves = federation_stage_execution_waves(plan, ordered)
    member_waves = federation_execution_wave_member_steps(waves)
    assert len(waves) == 3
    assert [wave.stage.kind for wave in waves] == ["member", "member", "coordinator"]
    assert waves[-1].stage.stage_id == "coordinator"
    assert not waves[-1].member_steps
    assert [step.source_id for step in member_waves] == ["a", "b"]
    assert [step.source_id for step in member_waves] == [step.source_id for step in ordered]


@pytest.mark.fast
def test_execution_waves_cover_spanning_cte_and_coordinator_stages() -> None:
    from aetherdialect._federation import federation_execution_wave_member_steps, federation_stage_execution_waves

    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id="a",
                sub_intent=RuntimeIntent(
                    tables=["ta"],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id="b",
                sub_intent=RuntimeIntent(
                    tables=["tb"],
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
            FederatedStage(stage_id="member_b", kind="member", source_ids=("b",)),
            FederatedStage(
                stage_id="coordinator_cte",
                kind="cte",
                source_ids=("a", "b"),
                depends_on=("member_a", "member_b"),
                spanning_cte_names=("span_cte",),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a", "b"),
                depends_on=("coordinator_cte",),
            ),
        ),
    )
    ordered = order_federation_execution_steps(plan)
    waves = federation_stage_execution_waves(plan, ordered)
    assert len(waves) == 3
    assert [wave.stage.kind for wave in waves] == ["member", "cte", "coordinator"]
    assert waves[1].stage.stage_id == "coordinator_cte"
    assert waves[2].stage.stage_id == "coordinator"
    assert not waves[1].member_steps
    assert not waves[2].member_steps
    flattened = federation_execution_wave_member_steps(waves)
    assert [step.source_id for step in flattened] == [step.source_id for step in ordered]


@pytest.mark.fast
def test_execution_waves_cycle_raises_invariant_error() -> None:
    from aetherdialect._federation import federation_stage_execution_waves

    plan = FederatedPlan(
        steps=(),
        stages=(
            FederatedStage(
                stage_id="member_a",
                kind="member",
                source_ids=("a",),
                depends_on=("coordinator",),
            ),
            FederatedStage(
                stage_id="coordinator",
                kind="coordinator",
                source_ids=("a",),
                depends_on=("member_a",),
            ),
        ),
    )
    with pytest.raises(FederationInvariantError, match="cycle"):
        federation_stage_execution_waves(plan, ())
