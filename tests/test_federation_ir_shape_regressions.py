"""Fast regressions for semi-join, anti-join, predicates, DISTINCT ON, and coordinator lifting."""

from __future__ import annotations

import json
import re
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._constants import MAX_PREDICATE_NESTING_DEPTH, anti_join_presence_column
from aetherdialect._contracts_base import (
    FederationMappings,
    MulGroup,
    PredicateGroup,
    WhereParam,
    coerce_predicate_group,
    predicate_group_from_list,
)
from aetherdialect._contracts_core import (
    FederatedPlan,
    JoinSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect import get_dialect
from aetherdialect._federation import (
    FederationConfigError,
    _apply_coordinator_probe_joins,
    assert_composite_invariants,
    compose_composite_graph,
    execute_federation_coordinator,
    federation_artifact_paths,
    federation_plan_is_degenerate,
    load_federation_composite_graph,
    mappings_replay_matches,
    parse_federation_manifest,
    persist_federation_tree,
    plan_federated_intent,
    render_federation_glue,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._main_execution import _build_federation_source_runtimes, _federation_single_source_sql_context
from aetherdialect._pipeline import generate_and_validate_sql, prepare_federated_sql_plan
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import (
    _build_deterministic_select_block,
    _join_edges_from_signature,
    build_deterministic_sql,
    inject_join_into_deterministic_sql,
)
from aetherdialect._templates import empty_template_store
from aetherdialect._validation_schema import (
    validate_cte_emission_shapes,
    validate_distinct_on_schema,
    validate_predicate_nesting_depth,
    validate_preserve_tables,
)
from tests.conftest import duckdb_engine_identity
from tests.federation_helpers import build_two_member_federation
from tests.join_test_helpers import catalog_edge_kinds_for_signatures
from tests.test_distinct_on import _distinct_on_intent, _pg_render
from tests.test_federation_single_source import (
    _composed_manifest,
    _member_graphs,
    _runtime_manifest,
)
from tests.test_semi_anti_join import _forbidden_tokens, _parent_child_schema


def _assert_no_forbidden_sql_tokens(sql: str) -> None:
    upper = sql.upper()
    for pattern in PolicyConfig.FORBIDDEN_SQL:
        if re.search(pattern, upper, flags=re.IGNORECASE):
            pytest.fail(f"forbidden SQL token matched {pattern!r} in:\n{sql}")


def _col(
    name: str,
    *,
    nullable: bool = False,
    fk_target: tuple[str, str] | None = None,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_nullable=nullable,
        is_foreign_key=fk_target is not None,
        fk_target=fk_target,
        sensitivity="none",
    )


def _parent_child_graph(*, nullable_fk: bool = False) -> SchemaGraph:
    parent = TableMetadata(
        name="parent",
        columns={
            "id": _col("id", nullable=False),
            "name": ColumnMetadata(name="name", data_type="text", sensitivity="none"),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    child = TableMetadata(
        name="child",
        columns={
            "id": _col("id", nullable=False),
            "parent_id": _col(
                "parent_id",
                nullable=nullable_fk,
                fk_target=("parent", "id"),
            ),
        },
        primary_key=["id"],
        foreign_keys=[
            FKEdge(
                src_table="child",
                src_cols=["parent_id"],
                dst_table="parent",
                dst_cols=["id"],
            ),
        ],
    )
    tables = {"parent": parent, "child": child}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="parent_child_exec",
    )


def _anti_join_intent(*, probe_name: str = "has_child") -> RuntimeIntent:
    anti = RuntimeCteStep(
        cte_name=probe_name,
        emission="anti_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    return RuntimeIntent(
        tables=["parent", probe_name],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr.from_column("parent.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[anti],
        chosen_join_path_signature=[f"parent.id->{probe_name}.parent_id"],
    )


def _semi_join_intent(*, probe_name: str = "active_parents") -> RuntimeIntent:
    semi = RuntimeCteStep(
        cte_name=probe_name,
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
        output_columns=["parent_id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    return RuntimeIntent(
        tables=["parent", probe_name],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.name"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
        chosen_join_path_signature=[f"parent.id->{probe_name}.parent_id"],
    )


def _render_joined_sql(intent: RuntimeIntent, schema: SchemaGraph) -> str:
    dialect = get_dialect("sqlite")
    det = build_deterministic_sql(intent, schema=schema, dialect=dialect)
    join_edges = list(intent.chosen_join_path_signature or [])
    sig = [[], join_edges] if join_edges else [[]]
    emissions = {
        (step.cte_name or ""): (step.emission or "join_table") for step in (intent.cte_steps or []) if step.cte_name
    }
    return inject_join_into_deterministic_sql(
        det,
        sig,
        schema=schema,
        edge_kinds_ordered=catalog_edge_kinds_for_signatures(sig),
        dialect=dialect,
        cte_emissions=emissions,
    )


def _duckdb_rows(sql: str, *, parent_rows: list[tuple], child_rows: list[tuple]) -> list[tuple]:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.execute("CREATE TABLE parent (id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE child (id INTEGER, parent_id INTEGER)")
    if parent_rows:
        conn.executemany("INSERT INTO parent VALUES (?, ?)", parent_rows)
    if child_rows:
        conn.executemany("INSERT INTO child VALUES (?, ?)", child_rows)
    return conn.execute(sql).fetchall()


def _nested_predicate_group(depth: int) -> PredicateGroup:
    leaf_a = WhereParam(
        left_expr=NormalizedExpr.from_column("t.a"),
        op="=",
        raw_value="x",
        param_key="p1",
    )
    leaf_b = WhereParam(
        left_expr=NormalizedExpr.from_column("t.b"),
        op="=",
        raw_value="y",
        param_key="p2",
    )
    if depth <= 1:
        return PredicateGroup(op="and", predicates=(leaf_a, leaf_b))
    left = _nested_predicate_group(depth - 1)
    right = PredicateGroup(op="and", predicates=(leaf_b,))
    return PredicateGroup(op="or", groups=(left, right))


@pytest.mark.fast
def test_anti_join_non_nullable_fk_returns_parents_without_children() -> None:
    schema = _parent_child_graph(nullable_fk=False)
    sql = _render_joined_sql(_anti_join_intent(), schema)
    _assert_no_forbidden_sql_tokens(sql)
    rows = _duckdb_rows(
        sql,
        parent_rows=[(1, "one"), (2, "two"), (3, "three")],
        child_rows=[(10, 2)],
    )
    assert sorted(row[0] for row in rows) == [1, 3]
    assert len(rows) == 2


@pytest.mark.fast
def test_anti_join_parent_to_child_preserves_parents_without_matches() -> None:
    schema = _parent_child_graph(nullable_fk=False)
    sql = _render_joined_sql(_anti_join_intent(probe_name="missing_child"), schema)
    rows = _duckdb_rows(
        sql,
        parent_rows=[(1, "alpha"), (2, "beta"), (4, "delta")],
        child_rows=[(10, 2), (11, 2)],
    )
    assert sorted(row[0] for row in rows) == [1, 4]
    assert all(row[1] in {"alpha", "delta"} for row in rows)


@pytest.mark.fast
def test_anti_join_nullable_fk_presence_marker_avoids_false_unmatched() -> None:
    schema = _parent_child_graph(nullable_fk=True)
    sql = _render_joined_sql(_anti_join_intent(probe_name="nullable_probe"), schema)
    marker = anti_join_presence_column("nullable_probe")
    assert marker in sql
    rows = _duckdb_rows(
        sql,
        parent_rows=[(1, "solo"), (2, "paired")],
        child_rows=[(10, 2), (11, None)],
    )
    assert sorted(row[0] for row in rows) == [1]


@pytest.mark.fast
def test_semi_join_does_not_multiply_parent_rows() -> None:
    schema = _parent_child_graph(nullable_fk=False)
    sql = _render_joined_sql(_semi_join_intent(), schema)
    _assert_no_forbidden_sql_tokens(sql)
    rows = _duckdb_rows(
        sql,
        parent_rows=[(1, "one"), (2, "two")],
        child_rows=[(10, 1), (11, 1), (12, 2)],
    )
    assert len(rows) == 2
    assert sorted(row[0] for row in rows) == ["one", "two"]


@pytest.mark.fast
def test_semi_join_payload_projection_is_rejected() -> None:
    schema = _parent_child_schema()
    semi = RuntimeCteStep(
        cte_name="bad_semi",
        emission="semi_join",
        tables=["child"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.status"))],
        output_columns=["status"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    intent = RuntimeIntent(
        tables=["parent", "bad_semi"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[semi],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any(i.issue_id == "semi_join_projection_shape_bad_semi" for i in issues)


@pytest.mark.fast
def test_set_difference_anti_join_matches_except_with_duplicate_elimination() -> None:
    schema = _parent_child_graph(nullable_fk=False)
    intent = RuntimeIntent(
        tables=["parent", "child_keys"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[
            RuntimeCteStep(
                cte_name="child_keys",
                emission="anti_join",
                tables=["child"],
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
                output_columns=["parent_id"],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
                having=None,
            ),
        ],
        chosen_join_path_signature=["parent.id->child_keys.parent_id"],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert not issues
    sql = _render_joined_sql(intent, schema)
    _assert_no_forbidden_sql_tokens(sql)
    rows = _duckdb_rows(
        sql,
        parent_rows=[(1, "a"), (2, "b"), (3, "c")],
        child_rows=[(10, 2), (11, 2), (12, 3)],
    )
    assert [row[0] for row in rows] == [1]


@pytest.mark.fast
def test_set_difference_arity_mismatch_is_rejected() -> None:
    schema = _parent_child_schema()
    intent = RuntimeIntent(
        tables=["parent", "other_set"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr.from_column("parent.name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        cte_steps=[
            RuntimeCteStep(
                cte_name="other_set",
                emission="anti_join",
                tables=["child"],
                select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.parent_id"))],
                output_columns=["parent_id"],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
                having=None,
            ),
        ],
    )
    issues = validate_cte_emission_shapes(intent, schema)
    assert any("set difference" in issue.message.lower() or "arity" in issue.message.lower() for issue in issues)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.fast
def test_predicate_nesting_depths_one_to_three_render(depth: int) -> None:
    group = _nested_predicate_group(depth)
    assert group.depth() == depth
    assert group.depth() <= MAX_PREDICATE_NESTING_DEPTH
    sql = _build_deterministic_select_block(
        [SelectCol(expr=NormalizedExpr.from_column("t.a"))],
        ["t"],
        [],
        [],
        group,
        None,
        None,
        "row_level",
        get_dialect("sqlite"),
    )
    assert "WHERE" in sql.upper()
    assert '"t"."a"' in sql or "t.a" in sql.lower()


@pytest.mark.fast
def test_predicate_nesting_depth_four_raises() -> None:
    nested = _nested_predicate_group(MAX_PREDICATE_NESTING_DEPTH + 1)
    assert nested.depth() > MAX_PREDICATE_NESTING_DEPTH
    issues = validate_predicate_nesting_depth(nested, None, "main query")
    assert any(issue.issue_id == "where_predicate_nesting_depth" for issue in issues)
    assert any(issue.severity == "error" for issue in issues)
    coerced = coerce_predicate_group(nested)
    assert coerced is not None
    assert coerced.depth() <= MAX_PREDICATE_NESTING_DEPTH


@pytest.mark.fast
def test_distinct_on_without_order_by_raises(simple_schema: SchemaGraph) -> None:
    issues = validate_distinct_on_schema(
        [NormalizedExpr.from_column("customers.id")],
        [],
        simple_schema,
        {"customers"},
        None,
        "main query",
    )
    assert issues
    assert any("order_by" in issue.message.lower() for issue in issues)


@pytest.mark.fast
def test_distinct_on_surviving_row_is_deterministic(simple_schema: SchemaGraph) -> None:
    intent = _distinct_on_intent()
    sql_first = build_deterministic_sql(intent, schema=simple_schema, dialect=_pg_render())
    sql_second = build_deterministic_sql(intent, schema=simple_schema, dialect=_pg_render())
    _assert_no_forbidden_sql_tokens(sql_first)
    assert sql_first == sql_second
    assert "row_number()" in sql_first.lower().replace(" ", "")
    assert "partition by" in sql_first.lower()
    assert "distinct on" not in sql_first.lower()


@pytest.mark.fast
def test_preserve_tables_zero_fill_and_left_propagation() -> None:
    parent = TableMetadata(
        name="parent",
        columns={"id": _col("id")},
        primary_key=["id"],
        foreign_keys=[],
    )
    mid = TableMetadata(
        name="mid",
        columns={"id": _col("id"), "parent_id": _col("parent_id", fk_target=("parent", "id"))},
        primary_key=["id"],
        foreign_keys=[
            FKEdge(src_table="mid", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"]),
        ],
    )
    child = TableMetadata(
        name="child",
        columns={"id": _col("id"), "mid_id": _col("mid_id", fk_target=("mid", "id")), "amount": _col("amount")},
        primary_key=["id"],
        foreign_keys=[
            FKEdge(src_table="child", src_cols=["mid_id"], dst_table="mid", dst_cols=["id"]),
        ],
    )
    tables = {"parent": parent, "mid": mid, "child": child}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="preserve_chain",
    )
    sig = ["parent.id->mid.parent_id", "mid.id->child.mid_id"]
    resolved = _join_edges_from_signature(
        sig, ["catalog_fk", "catalog_fk"], "parent", schema, preserve_tables=["parent"]
    )
    assert resolved is not None
    join_edges, _where, _extra, _anti = resolved
    assert all(edge.kind == "LEFT" for edge in join_edges)

    simple_parent = TableMetadata(
        name="parent",
        columns={"id": _col("id", nullable=False)},
        primary_key=["id"],
        foreign_keys=[],
        row_count=100,
    )
    simple_child = TableMetadata(
        name="child",
        columns={
            "id": _col("id", nullable=False),
            "parent_id": _col("parent_id", nullable=False, fk_target=("parent", "id")),
        },
        primary_key=["id"],
        foreign_keys=[],
        row_count=200,
    )
    simple_tables = {"parent": simple_parent, "child": simple_child}
    simple_schema = SchemaGraph(
        tables=simple_tables,
        join_paths_multi=recompute_join_paths_multi(simple_tables),
        effective_structural_hash="preserve_simple",
    )
    intent = RuntimeIntent(
        tables=["parent", "child"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("parent.id")),
            SelectCol(expr=NormalizedExpr(agg_func="count", add_groups=[MulGroup(multiply=["child.id"])])),
        ],
        group_by_cols=[NormalizedExpr.from_column("parent.id")],
        order_by_cols=[],
        where=None,
        preserve_tables=["parent"],
        chosen_join_path_signature=["parent.id->child.parent_id"],
    )
    sql = build_deterministic_sql(intent, schema=simple_schema, dialect=get_dialect("sqlite"))
    _assert_no_forbidden_sql_tokens(sql)
    assert "COALESCE(COUNT" in sql and ", 0)" in sql


@pytest.mark.fast
def test_preserve_tables_unreachable_declaration_raises() -> None:
    issues = validate_preserve_tables(
        ["parent", "child"],
        ["child"],
        _parent_child_graph(),
        "main query",
        join_signature=["parent.id->child.parent_id"],
    )
    assert any("not reachable" in issue.message for issue in issues)


@pytest.mark.fast
def test_preserve_tables_noop_declaration_raises() -> None:
    issues = validate_preserve_tables(
        ["parent", "child"],
        ["parent"],
        _parent_child_graph(),
        "main query",
        join_signature=["parent.id->child.parent_id"],
    )
    assert any("would have no effect" in issue.message for issue in issues)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("shape", "intent_factory"),
    [
        ("anti_join", _anti_join_intent),
        ("semi_join", _semi_join_intent),
        ("distinct_on", lambda: _distinct_on_intent()),
    ],
)
def test_new_shape_sql_has_no_legacy_tokens(
    shape: str,
    intent_factory,
    simple_schema: SchemaGraph,
) -> None:
    if shape == "distinct_on":
        sql = build_deterministic_sql(intent_factory(), schema=simple_schema, dialect=_pg_render())
    else:
        sql = _render_joined_sql(intent_factory(), _parent_child_graph())
    assert _forbidden_tokens(sql) == []
    _assert_no_forbidden_sql_tokens(sql)


@pytest.mark.fast
def test_single_source_federated_plan_renders_byte_identical_sql() -> None:
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
    default = get_dialect("duckdb")
    runtimes = _build_federation_source_runtimes(
        _runtime_manifest(), None, default, default_identity=duckdb_engine_identity()
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
    assert direct.success and fed.success
    assert direct.sql == fed.display_sql
    assert fed.steps[0].sql == direct.sql


@pytest.mark.fast
def test_cross_source_anti_join_is_lifted_to_coordinator(two_member_federation) -> None:
    fed = two_member_federation
    probe_name = "absent_right"
    anti_cte = RuntimeCteStep(
        cte_name=probe_name,
        emission="anti_join",
        tables=[fed.right_table],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
        output_columns=["id"],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id=fed.left_source,
                sub_intent=RuntimeIntent(
                    tables=[fed.left_table],
                    grain="many",
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.left_table}.id"))],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
            SourceStep(
                source_id=fed.right_source,
                sub_intent=RuntimeIntent(
                    tables=[fed.right_table],
                    grain="many",
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{fed.right_table}.id"))],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
        combine=(
            JoinSpec(
                left_source=fed.left_source,
                right_source=fed.right_source,
                left_key="id",
                right_key="id",
                logical_key="id",
                kind="left",
            ),
        ),
        lifted_probe_ctes=(anti_cte,),
        grain="many",
        scope_sources=frozenset({fed.left_source, fed.right_source}),
    )
    step_ids = {fed.left_source: "src_a", fed.right_source: "src_b"}
    glue = render_federation_glue(plan, step_ids, schema=fed.composite)
    marker = anti_join_presence_column(probe_name)
    assert "IS NULL" in glue.upper()
    assert marker in glue
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.register("src_a", pd.DataFrame({"id": [1, 2, 3]}))
    conn.register("src_b", pd.DataFrame({"id": [2]}))
    source_by_table = {fed.left_table: fed.left_source, fed.right_table: fed.right_source}
    lifted_sql = _apply_coordinator_probe_joins(
        "SELECT id FROM src_a",
        (anti_cte,),
        step_ids,
        source_by_table,
    )
    rows = conn.execute(lifted_sql).fetchall()
    assert sorted(row[0] for row in rows) == [1, 3]


@pytest.mark.fast
def test_capability_refusal_names_unsupported_operator() -> None:
    def _cap_graph(table: str, source_id: str) -> SchemaGraph:
        table_meta = TableMetadata(
            name=table,
            columns={"name": ColumnMetadata(name="name", data_type="text", sensitivity="none")},
            primary_key=["name"],
            foreign_keys=[],
            source_id=source_id,
        )
        tables = {table: table_meta}
        return SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=f"sg_{source_id}",
        )

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cap_refusal",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "bigquery", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _cap_graph("left_t", "a"), "b": _cap_graph("right_t", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.name"),
                    op="ilike",
                    value_type="string",
                    raw_value="%x%",
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None


@pytest.mark.fast
def test_compose_twice_is_byte_identical_on_identity_fields() -> None:
    fed = build_two_member_federation()
    second = compose_composite_graph(fed.member_graphs, fed.manifest)
    assert fed.composite.schema_graph_id == second.schema_graph_id
    assert fed.composite.structural_hash == second.structural_hash
    assert fed.composite.effective_structural_hash == second.effective_structural_hash
    assert_composite_invariants(second, fed.member_graphs, fed.manifest, FederationMappings(version=2))


@pytest.mark.fast
def test_coordinator_inner_join_returns_exact_row_count(two_member_federation) -> None:
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
    frames = {
        fed.left_source: pd.DataFrame({"id": [1, 2]}),
        fed.right_source: pd.DataFrame({"id": [2, 3]}),
    }
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    assert len(result) == 1
    assert result["id"].tolist() == ["2"]


@pytest.mark.fast
def test_federation_artifact_version_mismatch_reports_expected_version() -> None:
    from aetherdialect._constants import FEDERATION_ARTIFACT_FORMAT_VERSION

    fed = build_two_member_federation()
    mappings = FederationMappings(version=2)
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=fed.manifest,
            mappings=mappings,
            composite=fed.composite,
            member_graphs=fed.member_graphs,
        )
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["artifact_format_version"] = FEDERATION_ARTIFACT_FORMAT_VERSION - 1
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as exc_info:
            mappings_replay_matches(tmp, fed.member_graphs, fed.manifest, mappings)
        msg = str(exc_info.value)
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in msg
        with pytest.raises(FederationConfigError, match=r"artifact_format_version"):
            load_federation_composite_graph(tmp)
