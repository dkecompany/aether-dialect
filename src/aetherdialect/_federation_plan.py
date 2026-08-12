"""Federated plan stages, glue/residual SQL, sub-intents, and combine resolution."""

from __future__ import annotations

import copy
import importlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from ._config import PolicyConfig
from ._constants import (
    DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_DECIMAL_FALLBACK,
    DIAGNOSTIC_CODE_FEDERATION_TIMESTAMP_NORMALISED,
    DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED,
    FEDERATION_AVERAGE_SCALE_HEADROOM,
    FEDERATION_COMBINE_SEMI_KIND,
    FEDERATION_COORDINATOR_DECIMAL_FALLBACK,
    FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION,
    FEDERATION_COORDINATOR_DUCKDB_TYPE_MAP,
    FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS,
    FEDERATION_MAPPINGS_VERSION,
    FEDERATION_QUALIFIED_COLUMN_REF_RE,
    FEDERATION_QUALIFIED_THREE_PART_REF_RE,
    RENDERED_SELECT_ALIAS_RE,
    VALID_GRAINS,
)
from ._constants_runtime import (
    ASK_PHASE_I,
)
from ._contracts_base import (
    CteEmissionKind,
    EngineContext,
    FederationDeclarationError,
    FederationInvariantError,
    FederationJoinFanOutError,
    FederationRuntimeError,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    OrderByNullPlacement,
    PredicateGroup,
    SensitivityClassification,
    SpaceContext,
    WhereParam,
)
from ._contracts_core import (
    AnchoredTemporalBind,
    FederatedPlan,
    FederatedStage,
    FederationMemberResolvedLimits,
    FederationReducingEdge,
    FederationTableSet,
    JoinSpec,
    ResidualSpec,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    UnionSpec,
)
from ._contracts_schema import (
    ColumnMetadata,
    FederationCrossSourceJoin,
    FederationManifest,
    FederationMappings,
    FederationSourceLimits,
    InferenceTag,
    SchemaGraph,
    TableMetadata,
)
from ._dialect import (
    Dialect,
    DialectRegistry,
)
from ._federation_compose import (
    column_data_type_is_timezone_aware,
    intent_column_member_coverage_ineligible_reason,
    intent_table_sources,
    member_schema_slice,
    select_replica_member_source,
    source_ids_for_intent,
    source_join_key_is_unique,
    spanning_cte_decomposition_ineligible_reason,
    spanning_cte_names,
    split_qualified_column,
    table_source_index,
)
from ._federation_manifest import (
    assign_cte_sources,
    cross_source_probe_cte_ineligible_reason,
    cross_source_probe_cte_steps,
    cte_probe_join_keys,
    distinct_on_spans_sources,
    federation_ir_capability_reason,
    federation_unsupported_operator_reason,
    intent_registry_kw,
    manifest_with_derived_roster,
    predicate_clause_label,
    predicate_group_spans_sources,
    predicate_param_sources,
    resolve_anchored_temporal_bind,
    sources_for_refs,
    sources_for_table,
    validate_federation_cross_source_join_kind,
)
from ._intent_expr import extract_columns_from_expr
from ._intent_normalize import (
    apply_column_replacer_to_intent,
    build_column_term_replacer,
    build_table_term_replacer,
    collect_referenced_tables,
    expand_shared_pk_tables_for_refs,
    reconcile_tables,
    where_scope_registries_to_referenced,
)
from ._schema_graph import (
    assert_consumer_intent_in_scope,
    recompute_join_paths_multi,
)
from ._schema_reflect import resolve_federation_qualified_ref
from ._sql_gen import (
    anti_join_presence_column,
    get_join_choice_from_llm,
    maybe_pin_order_expr_collation,
    render_expr_sql,
    render_order_by_sql,
    render_predicate_clause,
    render_predicate_group_sql,
    render_select_col_sql,
    wrap_core_sql_with_distinct_on,
)
from ._utils import (
    debug,
    effective_profile_timeout_ms,
    emit_ask_phase,
    notify,
    parse_numeric_type_arguments,
)

_WINDOW_FINALITY_CTX: SimpleNamespace | None = None


def qsim_intent_eligible_on_federation(
    tables: Sequence[str], schema: SchemaGraph, manifest: FederationManifest, mappings: FederationMappings | None = None
) -> bool:
    """Return True when a QSim table set is answerable via federation decomposition."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    intent = RuntimeIntent(
        tables=list(tables), grain="row_level", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
    )
    sources = source_ids_for_intent(intent, schema, mappings, manifest)
    if len(sources) <= 1:
        return True
    plan = plan_federated_intent(intent, schema, manifest, mappings)
    return plan.ineligible_reason is None


def plan_federated_intent(
    intent: RuntimeIntent,
    schema: SchemaGraph,
    manifest: FederationManifest,
    mappings: FederationMappings | None = None,
    *,
    space: SpaceContext | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
    dialects_by_source: Mapping[str, Any] | None = None,
) -> FederatedPlan:
    """Decompose a validated intent into per-source steps and a combine specification."""
    emit_ask_phase(ASK_PHASE_I)
    debug(f"[{ASK_PHASE_I}] decompose tables={list(intent.tables or ())}")
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    manifest = manifest_with_derived_roster(manifest, member_graphs=member_graphs, composite=schema)
    table_set = federation_table_set(intent, schema, manifest, mappings)
    tables = set(table_set.tables)
    if space is not None:
        if space.tables:
            tables &= set(space.tables)
        if space.deny_objects:
            tables -= set(space.deny_objects)
        if tables and (space.columns or space.deny_columns):
            scope_ctx = EngineContext(
                allow_objects=frozenset(space.tables) if space.tables else frozenset(),
                deny_objects=space.deny_objects,
                allow_columns=space.columns,
                deny_columns=space.deny_columns,
            )
            if not assert_consumer_intent_in_scope(intent, scope_ctx, schema, frozenset(tables)):
                return FederatedPlan(steps=(), ineligible_reason="intent references columns outside the active space")
    if not tables:
        return FederatedPlan(steps=(), ineligible_reason="no tables referenced")
    capability_reason = federation_ir_capability_reason(intent, schema.database_feature_capability, schema=schema)
    if capability_reason:
        return FederatedPlan(steps=(), ineligible_reason=capability_reason)
    capability_reason = federation_unsupported_operator_reason(intent, manifest, dialects_by_source=dialects_by_source)
    if capability_reason:
        return FederatedPlan(steps=(), ineligible_reason=capability_reason)
    source_by_table = dict(table_set.source_by_table)
    sources: set[str] = set()
    for table in tables:
        sources.update(_planning_sources_for_table(table, manifest, mappings, source_by_table, schema))
    multi_source = len(sources) > 1
    if multi_source:
        raw_sql_reason = _unattributable_raw_sql_reason(intent)
        if raw_sql_reason:
            return FederatedPlan(steps=(), ineligible_reason=raw_sql_reason)
        coverage_reason = intent_column_member_coverage_ineligible_reason(intent, schema)
        if coverage_reason:
            return FederatedPlan(steps=(), ineligible_reason=coverage_reason)
        clause_reason = _federation_clause_ineligible_reason(intent, manifest, mappings, source_by_table, schema=schema)
        if clause_reason:
            return FederatedPlan(steps=(), ineligible_reason=clause_reason)
        spanning_cte_reason = spanning_cte_decomposition_ineligible_reason(intent, source_by_table)
        if spanning_cte_reason:
            return FederatedPlan(steps=(), ineligible_reason=spanning_cte_reason)
        agg_reason = _cross_source_aggregate_ineligible_reason(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        if agg_reason:
            return FederatedPlan(steps=(), ineligible_reason=agg_reason)
    steps: list[SourceStep] = []
    union_specs = _union_specs_for_intent(tables, mappings, source_by_table)
    combine: tuple[JoinSpec, ...] | None = _join_specs_for_sources(
        manifest, mappings, frozenset(sources), schema=schema, scope_tables=frozenset(tables)
    )
    if len(sources) > 1 and combine is None and not union_specs:
        return FederatedPlan(
            steps=(), ineligible_reason="cross-source join path is not declared for referenced sources"
        )
    global _WINDOW_FINALITY_CTX
    _WINDOW_FINALITY_CTX = SimpleNamespace(
        manifest=manifest, schema=schema, combine=combine, source_by_table=source_by_table
    )
    try:
        for source_id in sorted(sources):
            member_schema = _member_schema_for_sub_intent_repair(
                source_id, schema, manifest=manifest, member_graphs=member_graphs
            )
            sub = _build_source_sub_intent(
                intent,
                source_id,
                tables,
                source_by_table,
                mappings,
                schema,
                manifest,
                multi_source=multi_source,
                member_schema=member_schema,
                chosen_specs=combine or (),
                space=space,
            )
            if sub is not None:
                steps.append(sub)
    finally:
        _WINDOW_FINALITY_CTX = None
    if multi_source and len(steps) < len(sources):
        built = {step.source_id for step in steps}
        dropped = tuple(sorted(sources - built))
        raise FederationInvariantError(
            "federation plan dropped member(s) "
            f"{list(dropped)} that scope discovery found "
            f"(scope sources={sorted(sources)})"
        )
    residual = _residual_spec_for_intent(
        intent, source_by_table, manifest, mappings, schema=schema, scope_tables=tables, combine=combine
    )
    stages = plan_federated_stages(
        sources,
        tuple(steps),
        intent=intent,
        source_by_table=source_by_table,
        manifest=manifest,
        mappings=mappings,
        residual=residual,
        schema=schema,
        combine=combine,
    )
    grain = intent.grain or "row_level"
    if grain not in VALID_GRAINS:
        grain = "row_level"
    lifted_probe_ctes = cross_source_probe_cte_steps(intent, source_by_table)
    plan = FederatedPlan(
        steps=tuple(steps),
        union_specs=tuple(union_specs),
        combine=combine,
        residual=residual,
        stages=stages,
        grain=grain,
        scope_sources=frozenset(sources),
        lifted_probe_ctes=lifted_probe_ctes,
    )
    if multi_source:
        plan = rewrite_federated_residual_aggregate_fan_out(plan, schema, manifest, mappings)
        validate_federation_coordinator_column_types(plan, schema, manifest=manifest)
        validate_federation_scalar_grain_member_frames(plan)
        emit_federation_rounding_mode_mixed_diagnostics(manifest, plan, intent, schema=schema)
    return plan


def federation_plan_is_degenerate(plan: FederatedPlan) -> bool:
    """Return True when *plan* is a single-member graph with no coordinator combine work."""
    if plan.ineligible_reason or len(plan.steps) != 1:
        return False
    if plan.residual is not None:
        return False
    if effective_union_specs(plan):
        return False
    if plan.combine:
        return False
    if len(plan.scope_sources) > 1:
        return False
    return True


def resolve_federated_member_schema(
    source_id: str,
    composite_schema: SchemaGraph,
    *,
    manifest: FederationManifest | None = None,
    member_graphs: Mapping[str, SchemaGraph] | None = None,
) -> SchemaGraph:
    """Return the member schema slice, preferring loaded member graphs when present."""
    if member_graphs is not None and source_id in member_graphs:
        return member_graphs[source_id]
    member_graph = member_graphs.get(source_id) if member_graphs is not None else None
    return member_schema_slice(composite_schema, source_id, manifest=manifest, member_graph=member_graph)


def plan_federated_stages(
    sources: set[str],
    steps: tuple[SourceStep, ...],
    *,
    intent: RuntimeIntent | None = None,
    source_by_table: Mapping[str, str] | None = None,
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    residual: ResidualSpec | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> tuple[FederatedStage, ...]:
    """Build a staged execution graph with member stages, optional spanning CTE, and coordinator."""
    if len(sources) <= 1:
        return ()
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    member_deps = _member_stage_dependencies(manifest, sources) if manifest is not None else {}
    reducing_by_source = (
        _collect_member_reducing_edges(manifest, mappings, sources, intent, source_by_table, schema=schema)
        if manifest is not None
        else {}
    )
    member_stages = [
        FederatedStage(
            stage_id=f"member_{source_id}",
            kind="member",
            source_ids=(source_id,),
            depends_on=member_deps.get(source_id, ()),
            reducing_edges=reducing_by_source.get(source_id, ()),
        )
        for source_id in sorted(sources)
    ]
    stage_list: list[FederatedStage] = list(member_stages)
    coordinator_depends = tuple(stage.stage_id for stage in member_stages)
    spanning: tuple[str, ...] = ()
    if intent is not None and source_by_table is not None:
        spanning = spanning_cte_names(intent.cte_steps or (), source_by_table)
        if spanning:
            cte_sources = _spanning_cte_source_ids(intent.cte_steps or (), spanning, source_by_table)
            if not cte_sources:
                cte_sources = tuple(sorted(sources))
            cte_source_set = set(cte_sources)
            cte_depends = tuple(f"member_{source_id}" for source_id in cte_sources)
            cte_stage = FederatedStage(
                stage_id="coordinator_cte",
                kind="cte",
                source_ids=cte_sources,
                depends_on=cte_depends,
                spanning_cte_names=spanning,
            )
            stage_list.append(cte_stage)
            remaining = tuple(f"member_{source_id}" for source_id in sorted(sources) if source_id not in cte_source_set)
            coordinator_depends = (cte_stage.stage_id,) + remaining
    promotes_windows = False
    if intent is not None and source_by_table is not None:
        promotes_windows = _coordinator_promotes_spanning_windows(intent, source_by_table, manifest=manifest)
    if promotes_windows:
        coordinator_depends = tuple(stage.stage_id for stage in stage_list)
    grain = (intent.grain or "row_level") if intent is not None else "row_level"
    coordinator_id = "coordinator_scalar" if residual is not None and grain == "scalar" else "coordinator"
    coordinator = FederatedStage(
        stage_id=coordinator_id, kind="coordinator", source_ids=tuple(sorted(sources)), depends_on=coordinator_depends
    )
    stage_list.append(coordinator)
    return tuple(stage_list)


def _spanning_cte_source_ids(
    cte_steps: Sequence[RuntimeCteStep], spanning_names: Sequence[str], source_by_table: Mapping[str, str]
) -> tuple[str, ...]:
    """Return member source ids that feed any spanning CTE in *spanning_names*."""
    if not cte_steps or not spanning_names:
        return ()
    spanning_set = set(spanning_names)
    owners = assign_cte_sources(cte_steps, source_by_table)
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}
    cte_names_lower = {name.lower() for name in cte_names}
    collected: set[str] = set()
    for cte in cte_steps:
        name = cte.cte_name
        if not name or name not in spanning_set:
            continue
        refs = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            PredicateGroup.where_leaves(cte.where),
            PredicateGroup.having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            include_unreferenced_registries=False,
        )
        base_tables = {table for table in refs if table not in cte_names and table.lower() not in cte_names_lower}
        prior_ctes = {table for table in refs if table in cte_names or table.lower() in cte_names_lower}
        for table in base_tables:
            source_id = source_by_table.get(table, "")
            if source_id:
                collected.add(source_id)
        for prior in prior_ctes:
            canonical = next((candidate for candidate in cte_names if candidate.lower() == prior.lower()), prior)
            owner = owners.get(canonical)
            if owner:
                collected.add(owner)
            elif canonical in spanning_set:
                collected.update(
                    source_by_table.get(table, "") for table in (cte.tables or []) if source_by_table.get(table, "")
                )
        for table in cte.tables or []:
            source_id = source_by_table.get(table, "")
            if source_id:
                collected.add(source_id)
    return tuple(sorted(sid for sid in collected if sid))


def _coordinator_promotes_spanning_windows(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when any window must run at the coordinator after combine."""
    for entry in intent.window_registry or []:
        if _window_requires_coordinator(
            entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
        ):
            return True
    return False


def derive_execution_order_from_stages(plan: FederatedPlan) -> tuple[str, ...]:
    """
    Return member source ids in topological ``depends_on`` order.

    Raises:

        FederationInvariantError: When member-stage ``depends_on`` edges form a cycle.
    """
    member_stages = [stage for stage in plan.stages if stage.kind == "member"]
    if not member_stages:
        return tuple(step.source_id for step in sorted(plan.steps, key=lambda s: s.source_id))
    stage_by_id = {stage.stage_id: stage for stage in member_stages}
    ordered: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise FederationInvariantError(f"federated stage dependency cycle involving {stage_id!r}")
        stage = stage_by_id.get(stage_id)
        if stage is None:
            return
        visiting.add(stage_id)
        for dep in stage.depends_on:
            if dep in stage_by_id:
                visit(dep)
        visiting.remove(stage_id)
        visited.add(stage_id)
        if stage.source_ids:
            ordered.append(stage.source_ids[0])

    for stage in sorted(member_stages, key=lambda item: item.stage_id):
        visit(stage.stage_id)
    return tuple(ordered)


def _collect_member_reducing_edges(
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: set[str],
    intent: RuntimeIntent | None,
    source_by_table: Mapping[str, str] | None,
    *,
    schema: SchemaGraph | None = None,
) -> dict[str, tuple[FederationReducingEdge, ...]]:
    """Collect semi-join and filter-as-key reducing edges for each member stage."""
    edges: dict[str, list[FederationReducingEdge]] = {source_id: [] for source_id in sources}
    for join in manifest.cross_source_joins:
        left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
        right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if right_src in sources and source_semijoin_enabled(manifest, right_src):
            if reducing_edge_allowed_for_target(right_src, join, manifest, schema=schema):
                edges[right_src].append(
                    FederationReducingEdge(
                        driving_source_id=left_src,
                        target_source_id=right_src,
                        driving_key=left_col,
                        target_key=right_col,
                        edge_kind="semijoin",
                    )
                )
        if left_src in sources and source_semijoin_enabled(manifest, left_src):
            if reducing_edge_allowed_for_target(left_src, join, manifest, schema=schema):
                edges[left_src].append(
                    FederationReducingEdge(
                        driving_source_id=right_src,
                        target_source_id=left_src,
                        driving_key=right_col,
                        target_key=left_col,
                        edge_kind="semijoin",
                    )
                )
    if intent is not None and source_by_table is not None:
        for fp in _cross_source_where(intent, manifest, mappings, source_by_table):
            if not _cross_where_relates_to_join(fp, manifest):
                continue
            filter_cols = _param_qualified_columns(fp)
            for join in manifest.cross_source_joins:
                left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
                right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
                left_src = manifest.table_namespace.get(left_tbl, "")
                right_src = manifest.table_namespace.get(right_tbl, "")
                if not left_src or not right_src or left_src == right_src:
                    continue
                filter_tables = {split_qualified_column(col, manifest=manifest)[0] for col in filter_cols}
                filter_sources = sources_for_refs(filter_tables, manifest, mappings, source_by_table or {})
                if right_src in filter_sources and left_src in sources and source_semijoin_enabled(manifest, right_src):
                    if reducing_edge_allowed_for_target(right_src, join, manifest, schema=schema):
                        edges[right_src].append(
                            FederationReducingEdge(
                                driving_source_id=left_src,
                                target_source_id=right_src,
                                driving_key=left_col,
                                target_key=right_col,
                                edge_kind="filter_keys",
                            )
                        )
                if left_src in filter_sources and right_src in sources and source_semijoin_enabled(manifest, left_src):
                    if reducing_edge_allowed_for_target(left_src, join, manifest, schema=schema):
                        edges[left_src].append(
                            FederationReducingEdge(
                                driving_source_id=right_src,
                                target_source_id=left_src,
                                driving_key=right_col,
                                target_key=left_col,
                                edge_kind="filter_keys",
                            )
                        )
    if intent is not None and source_by_table is not None:
        owners = assign_cte_sources(intent.cte_steps or (), source_by_table)
        for cte in intent.cte_steps or []:
            if CteEmissionKind.coerce(getattr(cte, "emission", "join_table")) != "semi_join":
                continue
            owner = owners.get(cte.cte_name or "")
            if not owner:
                continue
            for key in cte_probe_join_keys(cte):
                if "." not in key:
                    continue
                left_tbl, left_col = split_qualified_column(
                    key, manifest=manifest, schema=None, source_by_table=source_by_table
                )
                left_src = source_by_table.get(left_tbl, "")
                for join in manifest.cross_source_joins:
                    j_left_tbl, j_left_col = split_qualified_column(join.left, manifest=manifest)
                    j_right_tbl, j_right_col = split_qualified_column(join.right, manifest=manifest)
                    j_left_src = manifest.table_namespace.get(j_left_tbl, "")
                    j_right_src = manifest.table_namespace.get(j_right_tbl, "")
                    if owner == j_right_src and left_src == j_left_src and j_left_col == left_col:
                        if j_right_src in sources and source_semijoin_enabled(manifest, j_right_src):
                            if reducing_edge_allowed_for_target(j_right_src, join, manifest, schema=schema):
                                edges[j_right_src].append(
                                    FederationReducingEdge(
                                        driving_source_id=j_left_src,
                                        target_source_id=j_right_src,
                                        driving_key=j_left_col,
                                        target_key=j_right_col,
                                        edge_kind="semijoin",
                                    )
                                )
                    if owner == j_left_src and left_src == j_right_src and j_right_col == left_col:
                        if j_left_src in sources and source_semijoin_enabled(manifest, j_left_src):
                            if reducing_edge_allowed_for_target(j_left_src, join, manifest, schema=schema):
                                edges[j_left_src].append(
                                    FederationReducingEdge(
                                        driving_source_id=j_right_src,
                                        target_source_id=j_left_src,
                                        driving_key=j_right_col,
                                        target_key=j_left_col,
                                        edge_kind="semijoin",
                                    )
                                )
    return {source_id: tuple(edge_list) for source_id, edge_list in edges.items() if edge_list}


def derive_federation_stages_in_order(plan: FederatedPlan) -> tuple[FederatedStage, ...]:
    """
    Return all federated stages in topological execution order.

    Raises:

        FederationInvariantError: When ``depends_on`` edges form a cycle.
    """
    if not plan.stages:
        return ()
    stage_by_id = {stage.stage_id: stage for stage in plan.stages}
    ordered: list[FederatedStage] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise FederationInvariantError(f"federated stage dependency cycle involving {stage_id!r}")
        stage = stage_by_id.get(stage_id)
        if stage is None:
            return
        visiting.add(stage_id)
        for dep in stage.depends_on:
            visit(dep)
        visiting.remove(stage_id)
        visited.add(stage_id)
        ordered.append(stage)

    for stage in sorted(
        plan.stages, key=lambda item: (0 if item.kind == "member" else 1 if item.kind == "cte" else 2, item.stage_id)
    ):
        visit(stage.stage_id)
    return tuple(ordered)


def member_stage_for_source(plan: FederatedPlan, source_id: str) -> FederatedStage | None:
    """Return the single-source member stage for *source_id*, if present. Used by the pipeline when attributing work to one federation member. Args: plan: Federated execution plan whose ``stages`` may include member, CTE, and coordinator stages. source_id: Member registration key to match against a stage whose ``source_ids`` is exactly ``(source_id)``. Returns: The matching ``kind=="member"`` :class:`~aetherdialect._contracts_core.FederatedStage`, or ``None`` when no such stage exists."""
    for stage in plan.stages:
        if stage.kind == "member" and stage.source_ids == (source_id,):
            return stage
    return None


def _combine_join_hub_source(join_specs: tuple[JoinSpec, ...], sources: set[str]) -> str:
    degree: dict[str, int] = {source_id: 0 for source_id in sources}
    for spec in join_specs:
        degree[spec.left_source] = degree.get(spec.left_source, 0) + 1
        degree[spec.right_source] = degree.get(spec.right_source, 0) + 1
    return max(sources, key=lambda source_id: (degree.get(source_id, 0), source_id))


def build_combine_join_tree(join_specs: tuple[JoinSpec, ...], sources: set[str]) -> Any:
    """Build a join tree from declared edges; refuse spanned sources without connectivity."""

    @dataclass(frozen=True, slots=True)
    class _CombineJoinTree:
        source_id: str
        children: tuple[tuple[JoinSpec, Any], ...] = ()

    if not join_specs:
        if len(sources) == 1:
            return _CombineJoinTree(source_id=next(iter(sources)))
        raise FederationRuntimeError("federation combine requires join edges when multiple sources are spanned")
    adjacency: dict[str, list[JoinSpec]] = defaultdict(list)
    for spec in join_specs:
        adjacency[spec.left_source].append(spec)
        adjacency[spec.right_source].append(spec)
    root = _combine_join_hub_source(join_specs, sources)
    visited: set[str] = {root}

    def walk(source_id: str) -> Any:
        children: list[tuple[JoinSpec, Any]] = []
        for spec in sorted(adjacency.get(source_id, []), key=lambda item: (item.right_source, item.left_source)):
            other = spec.right_source if spec.left_source == source_id else spec.left_source
            if other in visited:
                continue
            visited.add(other)
            children.append((spec, walk(other)))
        return _CombineJoinTree(source_id=source_id, children=tuple(children))

    tree = walk(root)
    orphan = sources - visited
    if orphan:
        raise FederationRuntimeError(
            f"federation combine missing declared edges for sources: {', '.join(sorted(orphan))}"
        )
    return tree


def _render_combine_tree_sql(
    tree: Any,
    step_ids: Mapping[str, str],
    plan: FederatedPlan,
    *,
    schema: SchemaGraph | None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
    explicit_cols: list[str] | None,
    alias_counter: list[int] | None = None,
) -> str:
    """Render combine SQL from a join tree with explicit projection on every hop."""
    counter = alias_counter if alias_counter is not None else [0]
    reg = step_ids.get(tree.source_id, "")
    if not reg:
        raise FederationRuntimeError(f"federation join missing frame for source {tree.source_id!r}")
    if not tree.children:
        select_kw = render_combine_select_keyword(explicit_cols)
        return f"SELECT {select_kw} FROM {reg} AS s{tree.source_id}"
    sql = ""
    for idx, (spec, child) in enumerate(tree.children):
        left_reg = reg if idx == 0 else f"({sql})"
        left_alias = "l" if idx == 0 else "prev"
        right_reg = step_ids.get(child.source_id, "")
        if not right_reg:
            raise FederationRuntimeError(f"federation join missing frame for source {child.source_id!r}")
        left_table = (
            resolve_source_column_table(
                schema,
                spec.left_source,
                spec.left_key,
                manifest=manifest,
                source_by_table=source_by_table,
                declared_table=declared_table_for_source_column(
                    plan,
                    spec.left_source,
                    spec.left_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                ),
            )
            if schema
            else None
        )
        right_table = (
            resolve_source_column_table(
                schema,
                spec.right_source,
                spec.right_key,
                manifest=manifest,
                source_by_table=source_by_table,
                declared_table=declared_table_for_source_column(
                    plan,
                    spec.right_source,
                    spec.right_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                ),
            )
            if schema
            else None
        )
        left_key_source = spec.left_source if spec.left_source == tree.source_id else child.source_id
        right_key_source = spec.right_source if spec.right_source == child.source_id else tree.source_id
        left_expr, right_expr = _coordinator_join_key_pair_exprs(
            left_alias,
            spec.left_key if left_key_source == tree.source_id else spec.right_key,
            left_table=left_table,
            right_alias="r",
            right_key=spec.right_key if right_key_source == child.source_id else spec.left_key,
            right_table=right_table,
            schema=schema,
        )
        raw_kind = (spec.kind or "inner").strip().lower()
        if raw_kind == FEDERATION_COMBINE_SEMI_KIND:
            join_kind = "INNER"
            probe_key = spec.right_key if right_key_source == child.source_id else spec.left_key
            probe_key_sql = Dialect.sqlglot_quote_identifier(unqualified_column_name(probe_key) or probe_key)
            right_reg = f"(SELECT DISTINCT {probe_key_sql} FROM {right_reg})"
        else:
            join_kind = validate_federation_cross_source_join_kind(spec.kind).upper()
        left_cols = _source_column_names_for_step(plan, tree.source_id if idx == 0 else child.source_id)
        right_cols = _source_column_names_for_step(plan, child.source_id)
        select_kw = _render_join_select_keyword(
            explicit_cols, left_alias=left_alias, right_alias="r", left_cols=left_cols, right_cols=right_cols
        )
        if idx == 0:
            sql = (
                f"SELECT {select_kw} FROM {left_reg} AS {left_alias} {join_kind} JOIN {right_reg} AS r "
                f"ON {left_expr} = {right_expr}"
            )
        else:
            sql = (
                f"SELECT {select_kw} FROM ({sql}) AS {left_alias} {join_kind} JOIN {right_reg} AS r "
                f"ON {left_expr} = {right_expr}"
            )
        if child.children:
            counter[0] += 1
            nested = _render_combine_tree_sql(
                child,
                step_ids,
                plan,
                schema=schema,
                manifest=manifest,
                source_by_table=source_by_table,
                explicit_cols=explicit_cols,
                alias_counter=counter,
            )
            sql = (
                f"SELECT {render_combine_select_keyword(explicit_cols)} FROM ({sql}) AS {left_alias} "
                f"{join_kind} JOIN ({nested}) AS r ON {left_expr} = {right_expr}"
            )
    return sql


def _render_combine_sql_for_sources(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    sources: set[str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    """Render join-tree combine SQL for a subset of member sources."""
    if not sources:
        return ""
    scoped_steps = tuple(step for step in plan.steps if step.source_id in sources)
    if not scoped_steps and len(sources) == 1:
        only_source = next(iter(sources))
        reg = step_ids.get(only_source, "")
        if not reg:
            return ""
        scoped_plan = replace(
            plan,
            steps=tuple(step for step in plan.steps if step.source_id == only_source),
            combine=None,
            union_specs=(),
            residual=None,
            stages=(),
        )
        explicit_cols = combine_select_column_names(scoped_plan)
        select_kw = render_combine_select_keyword(explicit_cols)
        return f"SELECT {select_kw} FROM {reg}"
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    scoped_joins: tuple[JoinSpec, ...] | None = None
    if join_specs:
        scoped_joins = tuple(
            spec for spec in join_specs if spec.left_source in sources and spec.right_source in sources
        )
    scoped_plan = replace(
        plan,
        steps=scoped_steps,
        combine=scoped_joins if scoped_joins else None,
        union_specs=(),
        residual=None,
        stages=(),
    )
    scoped_ids = {source_id: reg for source_id, reg in step_ids.items() if source_id in sources}
    return _render_federation_combine_sql(
        scoped_plan, scoped_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
    )


def _render_coordinator_spanning_cte_sql(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    """Compose spanning CTE bodies through the declared join graph for contributing members."""
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is None or not cte_stage.spanning_cte_names:
        return ""
    cte_sources = set(cte_stage.source_ids)
    if not cte_sources:
        cte_sources = {step.source_id for step in plan.steps}
    body = _render_combine_sql_for_sources(
        plan, step_ids, cte_sources, schema=schema, manifest=manifest, source_by_table=source_by_table
    )
    if not body:
        return ""
    cte_defs = [f"{name} AS ({body})" for name in cte_stage.spanning_cte_names]
    first = cte_stage.spanning_cte_names[0]
    all_sources = {step.source_id for step in plan.steps}
    remaining = all_sources - cte_sources
    cte_projected: list[str] = []
    for step in plan.steps:
        if step.source_id not in cte_sources:
            continue
        cte_projected.extend(step.projected_keys)
    cte_select_cols = [name for name in dict.fromkeys(unqualified_column_name(key) for key in cte_projected) if name]
    cte_select_kw = render_combine_select_keyword(cte_select_cols or None)
    if not remaining:
        return f"WITH {', '.join(cte_defs)} SELECT {cte_select_kw} FROM {first}"
    remapped_joins: list[JoinSpec] = []
    join_specs = plan.combine if isinstance(plan.combine, tuple) else ()
    for spec in join_specs or ():
        left_in = spec.left_source in cte_sources
        right_in = spec.right_source in cte_sources
        if left_in and right_in:
            continue
        if not left_in and not right_in:
            if spec.left_source in remaining and spec.right_source in remaining:
                remapped_joins.append(spec)
            continue
        remapped_joins.append(
            JoinSpec(
                left_source=first if left_in else spec.left_source,
                right_source=first if right_in else spec.right_source,
                left_key=spec.left_key,
                right_key=spec.right_key,
                logical_key=spec.logical_key,
                kind=spec.kind,
            )
        )
    remaining_steps = tuple(step for step in plan.steps if step.source_id in remaining)
    cte_step = SourceStep(
        source_id=first,
        sub_intent=RuntimeIntent(
            tables=[], grain="row_level", select_cols=[], group_by_cols=[], order_by_cols=[], where=None
        ),
        projected_keys=tuple(dict.fromkeys(cte_projected)),
    )
    outer_plan = replace(
        plan,
        steps=remaining_steps + (cte_step,),
        combine=tuple(remapped_joins) if remapped_joins else None,
        union_specs=(),
        residual=None,
        stages=(),
    )
    outer_ids = {step.source_id: step_ids[step.source_id] for step in remaining_steps if step.source_id in step_ids}
    outer_ids[first] = first
    outer_sql = _render_federation_combine_sql(
        outer_plan, outer_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
    )
    if not outer_sql:
        return f"WITH {', '.join(cte_defs)} SELECT {cte_select_kw} FROM {first}"
    return f"WITH {', '.join(cte_defs)} {outer_sql}"


def _semijoin_reduction_stage_dependencies(
    manifest: FederationManifest, sources: set[str]
) -> dict[str, tuple[str, ...]]:
    """Return member-stage depends_on edges for semi-join reduction across sources."""
    deps: dict[str, set[str]] = {source_id: set() for source_id in sources}
    for join in manifest.cross_source_joins:
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if right_src in sources and source_semijoin_enabled(manifest, right_src):
            deps.setdefault(right_src, set()).add(f"member_{left_src}")
    return {source_id: tuple(sorted(stage_ids)) for source_id, stage_ids in deps.items() if stage_ids}


def _drop_member_dependency_edges_to_dag(deps: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """Drop reverse or cycle-forming member ``depends_on`` edges until the graph is a DAG."""
    working = {source_id: set(stage_ids) for source_id, stage_ids in deps.items()}

    def _cycle_edge() -> tuple[str, str] | None:
        color: dict[str, int] = {}
        stack: list[str] = []

        def _dfs(node: str) -> tuple[str, str] | None:
            color[node] = 1
            stack.append(node)
            for dep_stage in sorted(working.get(node, ())):
                peer = dep_stage.removeprefix("member_")
                state = color.get(peer, 0)
                if state == 1:
                    cycle_nodes = stack[stack.index(peer) :]
                    cycle_edges: list[tuple[str, str]] = []
                    for idx, src in enumerate(cycle_nodes):
                        nxt = cycle_nodes[(idx + 1) % len(cycle_nodes)]
                        stage = f"member_{nxt}"
                        if stage in working.get(src, ()):
                            cycle_edges.append((src, stage))
                    if cycle_edges:
                        return max(cycle_edges)
                    return (node, dep_stage)
                if state == 0:
                    found = _dfs(peer)
                    if found is not None:
                        return found
            stack.pop()
            color[node] = 2
            return None

        for source_id in sorted(working):
            if color.get(source_id, 0) == 0:
                found = _dfs(source_id)
                if found is not None:
                    return found
        return None

    for source_id in list(working):
        for dep_stage in list(working.get(source_id, ())):
            peer = dep_stage.removeprefix("member_")
            reverse = f"member_{source_id}"
            if reverse in working.get(peer, ()):
                forward = (source_id, dep_stage)
                backward = (peer, reverse)
                drop_src, drop_stage = max(forward, backward)
                working.setdefault(drop_src, set()).discard(drop_stage)

    while True:
        edge = _cycle_edge()
        if edge is None:
            break
        src, stage = edge
        working.setdefault(src, set()).discard(stage)
    return {source_id: tuple(sorted(stage_ids)) for source_id, stage_ids in working.items() if stage_ids}


def _member_stage_dependencies(manifest: FederationManifest, sources: set[str]) -> dict[str, tuple[str, ...]]:
    """Return member-stage ``depends_on`` edges from manifest join orientation only."""
    raw = _semijoin_reduction_stage_dependencies(manifest, sources)
    as_sets = {source_id: set(stage_ids) for source_id, stage_ids in raw.items()}
    return _drop_member_dependency_edges_to_dag(as_sets)


def effective_union_specs(plan: FederatedPlan) -> tuple[UnionSpec, ...]:
    """Return union combine specs from ``union_specs`` only."""
    return plan.union_specs or ()


def _render_federation_union_cte_defs(
    union_specs: tuple[UnionSpec, ...], step_ids: Mapping[str, str], *, explicit_cols: list[str] | None
) -> tuple[str, ...]:
    """Return ``WITH`` CTE definitions materializing each union spec before join combine."""
    cte_defs: list[str] = []
    select_kw = render_combine_select_keyword(explicit_cols)
    for idx, spec in enumerate(union_specs):
        cte_name = f"fed_u{idx}"
        union_rel = _render_union_relation_sql(spec, step_ids, explicit_cols=explicit_cols)
        cte_defs.append(f"{cte_name} AS (SELECT {select_kw} FROM {union_rel} AS _u)")
    return tuple(cte_defs)


def _render_union_relation_sql(
    union_spec: UnionSpec, step_ids: Mapping[str, str], *, explicit_cols: list[str] | None = None
) -> str:
    members = [step_ids[sid] for sid in union_spec.member_source_ids if sid in step_ids]
    if not members:
        raise FederationRuntimeError(f"federation union {union_spec.logical_table!r} missing member frames")
    if union_spec.semantics == "replica" or len(members) == 1:
        return members[0]
    select_kw = render_combine_select_keyword(explicit_cols)
    return "(" + " UNION ALL ".join(f"SELECT {select_kw} FROM {m}" for m in members) + ")"


def source_by_table_from_schema(schema: SchemaGraph | None) -> dict[str, str]:
    """Build table-to-member map from a composite schema graph."""
    if schema is None:
        return {}
    return {
        table_name: str(table.source_id or "")
        for table_name, table in schema.tables.items()
        if str(table.source_id or "").strip()
    }


def _apply_coordinator_probe_joins(
    base_sql: str, probe_ctes: Sequence[RuntimeCteStep], step_ids: Mapping[str, str], source_by_table: Mapping[str, str]
) -> str:
    """Lift cross-source semi/anti probes onto materialised member frames at the coordinator."""
    if not probe_ctes or not base_sql.strip():
        return base_sql
    owners = assign_cte_sources(probe_ctes, source_by_table)
    sql = base_sql
    for cte in probe_ctes:
        owner = owners.get(cte.cte_name or "")
        if not owner or owner not in step_ids:
            continue
        probe_rel = step_ids[owner]
        keys = [unqualified_column_name(key) for key in cte_probe_join_keys(cte)]
        keys = [key for key in keys if key]
        if not keys:
            continue
        alias = (cte.cte_name or f"probe_{owner}").replace(".", "_")
        distinct_keys = ", ".join(Dialect.sqlglot_quote_identifier(key) for key in keys)
        on_parts = [
            f"drv.{Dialect.sqlglot_quote_identifier(key)} = {alias}.{Dialect.sqlglot_quote_identifier(key)}"
            for key in keys
        ]
        emission = CteEmissionKind.coerce(getattr(cte, "emission", "join_table"))
        if emission == "semi_join":
            probe_subquery = f"(SELECT DISTINCT {distinct_keys} FROM {probe_rel})"
            sql = f"SELECT drv.* FROM ({sql}) AS drv INNER JOIN {probe_subquery} AS {alias} ON {' AND '.join(on_parts)}"
        elif emission == "anti_join":
            presence = anti_join_presence_column(alias)
            anti_subquery = (
                f"(SELECT DISTINCT {distinct_keys}, 1 AS {Dialect.sqlglot_quote_identifier(presence)} FROM {probe_rel})"
            )
            sql = (
                f"SELECT drv.* FROM ({sql}) AS drv "
                f"LEFT JOIN {anti_subquery} AS {alias} ON {' AND '.join(on_parts)} "
                f"WHERE {alias}.{Dialect.sqlglot_quote_identifier(presence)} IS NULL"
            )
    return sql


def render_federation_glue(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    param_values: Mapping[str, Any] | None = None,
) -> str:
    """Render deterministic DuckDB SQL glue for a federated plan."""
    if plan.ineligible_reason:
        return ""
    source_by_table = source_by_table_from_schema(schema)
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is not None and cte_stage.spanning_cte_names:
        base_sql = _render_coordinator_spanning_cte_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    else:
        base_sql = _render_federation_combine_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    if not base_sql:
        return ""
    base_sql = _apply_coordinator_probe_joins(
        base_sql, plan.lifted_probe_ctes, step_ids, source_by_table_from_schema(schema)
    )
    return render_federation_residual_sql(base_sql, plan.residual, param_values=param_values, schema=schema)


def _render_federation_combine_only_glue(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
) -> str:
    """Render coordinator combine SQL without residual clauses."""
    source_by_table = source_by_table_from_schema(schema)
    cte_stage = next((stage for stage in plan.stages if stage.kind == "cte"), None)
    if cte_stage is not None and cte_stage.spanning_cte_names:
        base_sql = _render_coordinator_spanning_cte_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    else:
        base_sql = _render_federation_combine_sql(
            plan, step_ids, schema=schema, manifest=manifest, source_by_table=source_by_table
        )
    if not base_sql:
        return ""
    return _apply_coordinator_probe_joins(
        base_sql, plan.lifted_probe_ctes, step_ids, source_by_table_from_schema(schema)
    )


def _join_signature_from_combine_plan(
    plan: FederatedPlan,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
) -> list[str]:
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return []
    source_by_table = source_by_table_from_schema(schema)
    segments: list[str] = []
    for spec in combine:
        if (spec.kind or "inner").strip().lower() == FEDERATION_COMBINE_SEMI_KIND:
            continue
        left_table = declared_table_for_source_column(
            plan,
            spec.left_source,
            spec.left_key,
            manifest=manifest,
            schema=schema,
            source_by_table=source_by_table,
        )
        right_table = declared_table_for_source_column(
            plan,
            spec.right_source,
            spec.right_key,
            manifest=manifest,
            schema=schema,
            source_by_table=source_by_table,
        )
        if not left_table or not right_table:
            continue
        segments.append(f"{left_table}.{spec.left_key}->{right_table}.{spec.right_key}")
    return segments


def _residual_tables_for_plan(plan: FederatedPlan) -> list[str]:
    tables: list[str] = []
    for step in plan.steps:
        for key in step.projected_keys:
            if "." not in key:
                continue
            table, _ = key.split(".", 1)
            if table not in tables:
                tables.append(table)
    return tables


def _aggregate_table_sources(
    agg_table: str,
    manifest: FederationManifest | None,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph,
) -> frozenset[str]:
    """Return member sources that own *agg_table* for residual fan-out checks."""
    direct = str(source_by_table.get(agg_table, "") or "").strip()
    if direct:
        return frozenset({direct})
    if manifest is not None:
        sources = intent_table_sources({agg_table}, manifest, mappings, source_by_table, schema)
        if sources:
            return frozenset(sources)
        namespace_source = str(manifest.table_namespace.get(agg_table, "") or "").strip()
        if namespace_source:
            return frozenset({namespace_source})
    return frozenset()


def _combine_edge_relevant_to_residual_aggregate(
    spec: JoinSpec,
    agg_tables: set[str],
    *,
    manifest: FederationManifest | None,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph,
) -> bool:
    """Return True when *spec* can multiply rows for one of *agg_tables*."""
    if not agg_tables:
        return False
    edge_sources = {spec.left_source, spec.right_source}
    for agg_table in agg_tables:
        if _aggregate_table_sources(agg_table, manifest, mappings, source_by_table, schema) & edge_sources:
            return True
    return False


def rewrite_federated_residual_aggregate_fan_out(
    plan: FederatedPlan,
    schema: SchemaGraph,
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
) -> FederatedPlan:
    """
    Rewrite multiplying coordinator combines under residual

    aggregates into semi-join combines.

    When a residual aggregate would see rows duplicated by an inner/left

    combine edge, convert that edge to

    :data:`FEDERATION_COMBINE_SEMI_KIND` so the non-unique side is

    DISTINCT-key filtered without multiplying the aggregate grain.

    Raises :class:`FederationJoinFanOutError` only when rewrite cannot
        clear the residual fan-out.
    """
    residual = plan.residual
    if residual is None or not residual.select_cols:
        return plan
    if not any(_looks_aggregated(sc) for sc in residual.select_cols):
        return plan
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return plan
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    source_by_table = source_by_table_from_schema(schema)
    agg_tables = {
        table for sc in residual.select_cols if _looks_aggregated(sc) for table in _tables_referenced_by_select_col(sc)
    }
    rewritten: list[JoinSpec] = []
    changed = False
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        if kind == FEDERATION_COMBINE_SEMI_KIND:
            rewritten.append(spec)
            continue
        if not _combine_edge_relevant_to_residual_aggregate(
            spec,
            agg_tables,
            manifest=manifest,
            mappings=mappings,
            source_by_table=source_by_table,
            schema=schema,
        ):
            rewritten.append(spec)
            continue
        needs_semi = False
        for agg_table in sorted(agg_tables):
            agg_sources = _aggregate_table_sources(agg_table, manifest, mappings, source_by_table, schema)
            if not agg_sources:
                continue
            if kind == "inner":
                targets: tuple[tuple[str, str, str], ...] = (
                    (spec.left_source, spec.right_source, spec.right_key),
                    (spec.right_source, spec.left_source, spec.left_key),
                )
            elif kind == "left":
                targets = ((spec.left_source, spec.right_source, spec.right_key),)
            elif kind == "right":
                targets = ((spec.right_source, spec.left_source, spec.left_key),)
            else:
                continue
            for preserved_source, other_source, other_key in targets:
                if preserved_source not in agg_sources:
                    continue
                other_table = declared_table_for_source_column(
                    plan,
                    other_source,
                    other_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                )
                unique = source_join_key_is_unique(
                    schema,
                    other_source,
                    f"{other_table}.{other_key}" if other_table else other_key,
                    manifest=manifest,
                )
                if unique is False:
                    needs_semi = True
                    break
            if needs_semi:
                break
        if needs_semi:
            rewritten.append(replace(spec, kind=FEDERATION_COMBINE_SEMI_KIND))
            changed = True
        else:
            rewritten.append(spec)
    out = replace(plan, combine=tuple(rewritten)) if changed else plan
    validate_federated_residual_aggregate_fan_out(out, schema, manifest, mappings)
    return out


def validate_federated_residual_aggregate_fan_out(
    plan: FederatedPlan,
    schema: SchemaGraph,
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
) -> None:
    """Refuse coordinator residual aggregates that would see join- multiplied rows."""
    residual = plan.residual
    if residual is None or not residual.select_cols:
        return
    if not any(_looks_aggregated(sc) for sc in residual.select_cols):
        return
    combine = plan.combine
    if not isinstance(combine, tuple) or not combine:
        return
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    source_by_table = source_by_table_from_schema(schema)
    agg_tables = {
        table for sc in residual.select_cols if _looks_aggregated(sc) for table in _tables_referenced_by_select_col(sc)
    }
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        if kind == FEDERATION_COMBINE_SEMI_KIND:
            continue
        if not _combine_edge_relevant_to_residual_aggregate(
            spec,
            agg_tables,
            manifest=manifest,
            mappings=mappings,
            source_by_table=source_by_table,
            schema=schema,
        ):
            continue
        for agg_table in sorted(agg_tables):
            agg_sources = _aggregate_table_sources(agg_table, manifest, mappings, source_by_table, schema)
            if not agg_sources:
                continue
            if kind == "inner":
                targets: tuple[tuple[str, str, str], ...] = (
                    (spec.left_source, spec.right_source, spec.right_key),
                    (spec.right_source, spec.left_source, spec.left_key),
                )
            elif kind == "left":
                targets = ((spec.left_source, spec.right_source, spec.right_key),)
            elif kind == "right":
                targets = ((spec.right_source, spec.left_source, spec.left_key),)
            else:
                continue
            for preserved_source, other_source, other_key in targets:
                if preserved_source not in agg_sources:
                    continue
                other_table = declared_table_for_source_column(
                    plan,
                    other_source,
                    other_key,
                    manifest=manifest,
                    schema=schema,
                    source_by_table=source_by_table,
                )
                unique = source_join_key_is_unique(
                    schema,
                    other_source,
                    f"{other_table}.{other_key}" if other_table else other_key,
                    manifest=manifest,
                )
                if unique is not False:
                    continue
                raise FederationJoinFanOutError(
                    f"federation coordinator residual: aggregate over {agg_table!r} would see rows duplicated by "
                    f"join edge {preserved_source!r}->{other_source!r} on key {other_key!r}",
                    source_id=other_source,
                    phase="coordinator",
                )
    signature = _join_signature_from_combine_plan(plan, manifest=manifest, schema=schema)
    if not signature:
        return
    tables = _residual_tables_for_plan(plan)
    if not tables:
        return
    intent = RuntimeIntent(
        tables=tables,
        grain=plan.grain or "row_level",
        select_cols=list(residual.select_cols),
        group_by_cols=list(residual.group_by_cols),
        order_by_cols=list(residual.order_by_cols),
        where=residual.where,
        having=residual.having,
        distinct_on=list(residual.distinct_on),
        distinct_select_index=residual.distinct_select_index,
        limit=residual.limit,
        limit_param_key=residual.limit_param_key,
        window_registry=list(residual.window_registry),
        case_registry=list(residual.case_registry),
        chosen_join_path_signature=signature,
    )
    validation_execute = importlib.import_module("aetherdialect._validation_sql")
    issues = validation_execute.validate_aggregate_join_fan_out(
        intent,
        schema,
        "federation coordinator residual",
        join_signature=signature,
        from_anchor=tables[0],
    )
    errors = [issue for issue in issues if getattr(issue, "severity", "") == "error"]
    if not errors:
        return
    source_id = plan.steps[0].source_id if plan.steps else ""
    raise FederationJoinFanOutError(errors[0].message, source_id=source_id, phase="coordinator")


def _residual_referenced_param_keys(residual: ResidualSpec | None) -> frozenset[str]:
    if residual is None:
        return frozenset()
    keys: set[str] = set()
    for fp in PredicateGroup.where_leaves(residual.where):
        if fp.param_key:
            keys.add(fp.param_key)
        if fp.param_key_hi:
            keys.add(fp.param_key_hi)
        if fp.param_key_unit:
            keys.add(fp.param_key_unit)
    for hp in PredicateGroup.having_leaves(residual.having):
        if hp.param_key:
            keys.add(hp.param_key)
        if hp.param_key_unit:
            keys.add(hp.param_key_unit)
    lpk = (residual.limit_param_key or "").strip()
    if lpk:
        keys.add(lpk)
    return frozenset(keys)


def coordinator_residual_bind_map(plan: FederatedPlan, parent_params: Mapping[str, Any]) -> dict[str, Any]:
    """Narrow *parent_params* to handles referenced by the coordinator residual."""
    residual_keys = _residual_referenced_param_keys(plan.residual)
    if not residual_keys:
        return {}
    return {k: v for k, v in parent_params.items() if k in residual_keys}


def _explicit_residual_order_col(col: OrderByCol) -> OrderByCol:
    """Make coordinator residual null placement explicit for DuckDB rendering."""
    if col.nulls in ("first", "last"):
        return col
    return OrderByCol(
        expr=col.expr,
        direction=col.direction,
        nulls=OrderByNullPlacement.default_for_direction(col.direction),
    )


def _extract_rendered_select_alias(rendered: str) -> str:
    """Return the trailing ``AS`` alias from a rendered select expression."""
    match = RENDERED_SELECT_ALIAS_RE.search(rendered.strip())
    if not match:
        return ""
    return (match.group(1) or match.group(2) or "").strip()


def _residual_fallback_order_expr(
    idx: int,
    expr_sql: str,
    residual: ResidualSpec,
    dialect: Any,
    *,
    schema: SchemaGraph | None = None,
) -> str:
    """Build a deterministic ORDER BY expression from a residual select projection."""
    if idx >= len(residual.select_cols):
        return expr_sql
    sc = residual.select_cols[idx]
    alias = (sc.output_alias or "").strip() or _extract_rendered_select_alias(expr_sql)
    if alias:
        return Dialect.sqlglot_quote_identifier(alias)
    source_expr = sc.expr
    rendered = render_expr_sql(source_expr, dialect)
    return maybe_pin_order_expr_collation(
        rendered,
        source_expr,
        dialect,
        pin_collation=True,
        schema=schema,
    )


def render_federation_residual_sql(
    base_sql: str,
    residual: ResidualSpec | None,
    *,
    param_values: Mapping[str, Any] | None = None,
    schema: SchemaGraph | None = None,
) -> str:
    """Render coordinator residual clauses as DuckDB SQL wrapping *base_sql*. When *residual* is ``None`` or carries no clauses, returns *base_sql* unchanged. A limit-only residual without ``select_cols`` appends ``LIMIT`` directly; any other non-empty residual requires an explicit select projection and is rendered as ``SELECT ... FROM (<base_sql>) AS fed_base`` plus optional WHERE / GROUP BY / HAVING / ORDER BY / LIMIT. Args: base_sql: Inner SQL produced by the federation combine / member path. residual: Coordinator-spanning clauses, or ``None`` for a passthrough. param_values: Optional bind map used when rendering parameterised filter / having predicates. Defaults to an empty mapping. Returns: Either *base_sql* unchanged, *base_sql* with a trailing ``LIMIT``, or a full outer SELECT wrapping *base_sql* as ``fed_base``. Raises: FederationRuntimeError: *residual* has non-limit clauses (or a limit together with other clause kinds) but ``select_cols`` is empty."""
    if residual is None:
        return base_sql
    has_clauses = bool(
        residual.select_cols
        or residual.group_by_cols
        or residual.order_by_cols
        or residual.where
        or residual.having
        or residual.distinct_on
        or residual.distinct_select_index >= 0
        or residual.limit is not None
        or residual.window_registry
        or residual.case_registry
    )
    if not has_clauses:
        return base_sql
    dialect = DialectRegistry.get_dialect("duckdb")
    bind_values = dict(param_values or {})
    if not residual.select_cols:
        limit_only = (
            residual.limit is not None
            and not residual.group_by_cols
            and not residual.order_by_cols
            and not residual.where
            and not residual.having
            and not residual.distinct_on
            and residual.distinct_select_index < 0
            and not residual.window_registry
            and not residual.case_registry
        )
        if limit_only and residual.limit is not None:
            lpk = (residual.limit_param_key or "").strip()
            if lpk:
                return f"{base_sql} LIMIT :{lpk}"
            return f"{base_sql} LIMIT {int(residual.limit)}"
        raise FederationRuntimeError("federated residual requires explicit select_cols projection")
    select_exprs = [_render_residual_select_expr(sc, dialect, schema=schema) for sc in residual.select_cols]
    select_keyword = "SELECT DISTINCT" if residual.distinct_select_index >= 0 else "SELECT"
    parts = [f"{select_keyword} {', '.join(select_exprs)} FROM ({base_sql}) AS fed_base"]

    def _render_where_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect, is_having=False, param_values=bind_values)

    where_sql = render_predicate_group_sql(residual.where, _render_where_leaf)
    if where_sql:
        parts.append("WHERE " + where_sql)
    if residual.group_by_cols:
        gb_exprs = [render_expr_sql(g, dialect) for g in residual.group_by_cols]
        parts.append("GROUP BY " + ", ".join(gb_exprs))

    def _render_having_leaf(pred: WhereParam | HavingParam) -> str:
        return render_predicate_clause(pred, dialect, is_having=True, param_values=bind_values)

    having_sql = render_predicate_group_sql(residual.having, _render_having_leaf)
    if having_sql:
        parts.append("HAVING " + having_sql)
    if residual.order_by_cols:
        order_cols = [_explicit_residual_order_col(obc) for obc in residual.order_by_cols]
        parts.append(
            "ORDER BY "
            + render_order_by_sql(
                order_cols,
                dialect,
                pin_collation=True,
                schema=schema,
            )
        )
    elif select_exprs:
        pinned_exprs = [
            _residual_fallback_order_expr(idx, expr_sql, residual, dialect, schema=schema)
            for idx, expr_sql in enumerate(select_exprs)
        ]
        parts.append(
            "ORDER BY "
            + ", ".join(
                dialect.render_order_by_col(expr, "ASC", OrderByNullPlacement.default_for_direction("ASC"))
                for expr in pinned_exprs
            )
        )
    elif residual.group_by_cols:
        gb_exprs = []
        for gb_expr in residual.group_by_cols:
            rendered = render_expr_sql(gb_expr, dialect)
            gb_exprs.append(
                maybe_pin_order_expr_collation(
                    rendered,
                    gb_expr,
                    dialect,
                    pin_collation=True,
                    schema=schema,
                )
            )
        parts.append(
            "ORDER BY "
            + ", ".join(
                dialect.render_order_by_col(expr, "ASC", OrderByNullPlacement.default_for_direction("ASC"))
                for expr in gb_exprs
            )
        )
    if residual.limit is not None:
        lpk = (residual.limit_param_key or "").strip()
        if lpk:
            parts.append(f"LIMIT :{lpk}")
        else:
            parts.append(f"LIMIT {int(residual.limit)}")
    sql = " ".join(parts)
    if residual.distinct_on:
        order_cols = [_explicit_residual_order_col(col) for col in residual.order_by_cols]
        if not order_cols:
            order_cols = [
                OrderByCol(expr=expr, direction="ASC", nulls=OrderByNullPlacement.default_for_direction("ASC"))
                for expr in residual.distinct_on
            ]
        sql = wrap_core_sql_with_distinct_on(
            sql,
            select_exprs=select_exprs,
            distinct_on=list(residual.distinct_on),
            order_by_cols=order_cols,
            limit=None,
            dialect=dialect,
        )
    return sql


def _coordinator_residual_agg_inner(inner: str) -> str:
    """Strip table qualifiers from aggregate column refs in coordinator residual SQL."""
    text = str(inner or "").strip()
    if not text or text == "*":
        return text
    match = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if match:
        return match.group(2)
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        return three.group(3)
    if "." in text and "(" not in text:
        return text.rsplit(".", 1)[-1]
    return text


def _cross_source_avg_decomposes_to_sum_count(
    func: str | None,
    *,
    inner: str,
    has_distinct: bool,
) -> bool:
    return (
        func == "avg"
        and func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS
        and bool(inner)
        and inner != "*"
        and not has_distinct
    )


def _column_metadata_for_graph_ref(schema: SchemaGraph | None, col_ref: str) -> ColumnMetadata | None:
    """Resolve column metadata for a qualified or bare column reference on *schema*."""
    if schema is None:
        return None
    ref = str(col_ref or "").strip()
    if not ref:
        return None
    if "." in ref:
        table_name, col_name = ref.rsplit(".", 1)
        table = schema.tables.get(table_name)
        if table is not None and col_name in table.columns:
            return table.columns[col_name]
    for table in schema.tables.values():
        if ref in table.columns:
            return table.columns[ref]
    return None


def _federation_average_decimal_scale(source_scale: int | None) -> int:
    """Return coordinator DECIMAL scale for a federated average over an exact column."""
    scale = (source_scale or 0) + FEDERATION_AVERAGE_SCALE_HEADROOM
    return min(scale, FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION)


def _render_residual_select_expr(sc: SelectCol, dialect: Any, *, schema: SchemaGraph | None = None) -> str:
    """Render one residual projection, decomposing cross-source ``avg`` into sum and count."""
    func, has_distinct = _select_col_agg_meta(sc)
    raw_inner = _aggregate_inner_column(sc) if func else ""
    if _cross_source_avg_decomposes_to_sum_count(func, inner=raw_inner, has_distinct=has_distinct):
        col_meta = _column_metadata_for_graph_ref(schema, raw_inner)
        inner = _coordinator_residual_agg_inner(raw_inner)
        sum_sql = render_expr_sql(NormalizedExpr.from_column(f"sum({inner})"), dialect)
        count_sql = render_expr_sql(NormalizedExpr.from_column(f"count({inner})"), dialect)
        alias = (sc.output_alias or "").strip() or f"avg_{inner.replace('.', '_')}"
        if col_meta is not None and col_meta.is_exact_numeric:
            scale = _federation_average_decimal_scale(col_meta.numeric_scale)
            precision = FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION
            cast_type = f"DECIMAL({precision}, {scale})"
            return (
                f"CAST({sum_sql} AS {cast_type}) / NULLIF({count_sql}, 0) AS {Dialect.sqlglot_quote_identifier(alias)}"
            )
        return f"CAST({sum_sql} AS DOUBLE) / NULLIF({count_sql}, 0) AS {Dialect.sqlglot_quote_identifier(alias)}"
    inner = raw_inner
    if func and inner:
        inner_sql = "*" if inner == "*" else Dialect.sqlglot_quote_identifier(_coordinator_residual_agg_inner(inner))
        distinct_kw = "DISTINCT " if has_distinct else ""
        alias = (sc.output_alias or "").strip() or f"{func}_{inner.replace('.', '_')}"
        agg_core = f"{func}({distinct_kw}{inner_sql})"
        round_args = _select_col_round_args(sc)
        if round_args is not None:
            args_sql = ", ".join(str(arg) for arg in round_args)
            return f"ROUND({agg_core}, {args_sql}) AS {Dialect.sqlglot_quote_identifier(alias)}"
        return f"{agg_core} AS {Dialect.sqlglot_quote_identifier(alias)}"
    return render_select_col_sql(sc, dialect)


def _residual_group_by_column_names(plan: FederatedPlan) -> tuple[str, ...]:
    residual = plan.residual
    if residual is None or not residual.group_by_cols:
        return ()
    headers: list[str] = []
    for expr in residual.group_by_cols:
        col_ref = (expr.column_ref or expr.primary_column or expr.primary_term or "").strip()
        if col_ref:
            headers.append(col_ref.rsplit(".", 1)[-1])
    return tuple(headers)


def residual_is_aggregate_only(residual: ResidualSpec) -> bool:
    if not residual.select_cols:
        return False
    return all(_looks_aggregated(sc) for sc in residual.select_cols)


def aggregate_identity_row_for_residual(residual: ResidualSpec) -> tuple[Any, ...]:
    """Return the SQL aggregate identity row for an empty coordinator combine."""
    values: list[Any] = []
    for sc in residual.select_cols:
        if not _looks_aggregated(sc):
            values.append(None)
            continue
        func = _select_col_agg_func(sc)
        if func == "count":
            values.append(0)
        elif func == "sum":
            values.append(None)
        elif func in {"avg", "min", "max"}:
            values.append(None)
        else:
            values.append(None)
    return tuple(values)


def enforce_coordinator_result_grain(result_df: pd.DataFrame, plan: FederatedPlan) -> None:
    """Raise when the coordinator frame cardinality disagrees with the declared plan grain."""
    grain = plan.grain if plan.grain in VALID_GRAINS else "row_level"
    row_count = len(result_df)
    if grain == "scalar" and row_count != 1:
        raise FederationRuntimeError(f"federated scalar result has {row_count} rows, expected 1")
    if grain == "grouped" and row_count > 0:
        gb_cols = _residual_group_by_column_names(plan)
        if gb_cols and all(col in result_df.columns for col in gb_cols):
            if result_df.duplicated(subset=list(gb_cols), keep=False).any():
                raise FederationRuntimeError("federated grouped result has duplicate group keys")


def _render_federation_combine_sql(
    plan: FederatedPlan,
    step_ids: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str:
    union_specs = effective_union_specs(plan)
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    explicit_cols = combine_select_column_names(plan)
    if union_specs and not join_specs:
        if len(union_specs) == 1:
            select_kw = render_combine_select_keyword(explicit_cols)
            return f"SELECT {select_kw} FROM {_render_union_relation_sql(union_specs[0], step_ids, explicit_cols=explicit_cols)} AS u0"
        union_parts = [
            f"SELECT {render_combine_select_keyword(explicit_cols)} FROM {_render_union_relation_sql(spec, step_ids, explicit_cols=explicit_cols)} AS u{idx}"
            for idx, spec in enumerate(union_specs)
        ]
        return " UNION ALL ".join(union_parts)
    if not join_specs:
        if len(step_ids) == 1:
            only = next(iter(step_ids.values()))
            select_kw = render_combine_select_keyword(explicit_cols)
            return f"SELECT {select_kw} FROM {only}"
        return ""
    sources = {step.source_id for step in plan.steps}
    tree = build_combine_join_tree(join_specs, sources)
    join_sql = _render_combine_tree_sql(
        tree,
        step_ids,
        plan,
        schema=schema,
        manifest=manifest,
        source_by_table=source_by_table,
        explicit_cols=explicit_cols,
    )
    if union_specs:
        cte_defs = _render_federation_union_cte_defs(union_specs, step_ids, explicit_cols=explicit_cols)
        return f"WITH {', '.join(cte_defs)} {join_sql}"
    return join_sql


def semijoin_key_is_allowed(schema: SchemaGraph, table_name: str, column_name: str) -> bool:
    """Return True when *column_name* on *table_name* may participate in semi-join reduction."""
    table = schema.tables.get(table_name)
    if table is None:
        return False
    column = table.columns.get(column_name)
    if column is None:
        return False
    return column.sensitivity == SensitivityClassification.NONE


def semijoin_key_distinct_count(schema: SchemaGraph, table_name: str, column_name: str) -> int | None:
    """Return profiled distinct count for a semi-join key column when known."""
    table = schema.tables.get(table_name)
    if table is None:
        return None
    column = table.columns.get(column_name)
    if column is None:
        return None
    if table.row_count <= 0:
        return None
    return int(column.distinct_count)


def semijoin_key_passes_distinct_floor(schema: SchemaGraph, table_name: str, column_name: str, *, floor: int) -> bool:
    """Return False when profiled cardinality is below *floor*."""
    distinct = semijoin_key_distinct_count(schema, table_name, column_name)
    if distinct is None:
        return True
    return distinct >= int(floor)


def declared_table_for_source_column(
    plan: FederatedPlan,
    source_id: str,
    column_name: str,
    *,
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    source_by_table: Mapping[str, str] | None = None,
) -> str | None:
    """Return a plan-declared table for *column_name* on *source_id*, when known."""
    col = str(column_name or "").strip()
    if "." in col:
        table, _ = split_qualified_column(col, manifest=manifest, schema=schema, source_by_table=source_by_table)
        return table
    step = next((item for item in plan.steps if item.source_id == source_id), None)
    if step is None:
        return None
    for key in step.projected_keys:
        if "." not in key:
            continue
        table, name = split_qualified_column(key, manifest=manifest, schema=schema, source_by_table=source_by_table)
        if name == col:
            return table
    return None


def resolve_source_column_table(
    schema: SchemaGraph,
    source_id: str,
    column_name: str,
    *,
    manifest: FederationManifest | None = None,
    source_by_table: Mapping[str, str] | None = None,
    declared_table: str | None = None,
) -> str | None:
    """Return the composite table name carrying *column_name* for *source_id*."""
    col = str(column_name or "").strip()
    table_hint = str(declared_table or "").strip() or None
    if "." in col:
        table_hint, col = split_qualified_column(col, manifest=manifest, schema=schema, source_by_table=source_by_table)
    if table_hint:
        table = schema.tables.get(table_hint)
        if table is not None and table.source_id == source_id and col in table.columns:
            return table_hint
        return None
    matches = [name for name, table in schema.tables.items() if table.source_id == source_id and col in table.columns]
    if len(matches) == 1:
        return matches[0]
    return None


def source_timeout_for_source(manifest: FederationManifest, source_id: str) -> int:
    """Resolve per-source execution timeout from binding limits or coordinator defaults."""
    return resolve_member_limits_for_source(manifest, source_id).timeout_ms


def coordinator_frame_required_sources(plan: FederatedPlan) -> frozenset[str]:
    """Return source ids that must register coordinator frames (``src_<id>``) for glue SQL."""
    if plan.ineligible_reason:
        return frozenset()
    if len(plan.steps) <= 1:
        return frozenset()
    sources = frozenset(step.source_id for step in plan.steps)
    union_specs = effective_union_specs(plan)
    join_specs = plan.combine if isinstance(plan.combine, tuple) else None
    if join_specs:
        return sources
    if union_specs:
        return frozenset(sid for spec in union_specs for sid in spec.member_source_ids if sid in sources)
    if len(sources) > 1:
        return sources
    return frozenset()


def validate_federation_scalar_grain_member_frames(plan: FederatedPlan) -> None:
    """Refuse multi-member plans where a scalar-grain member must supply a coordinator frame."""
    required = coordinator_frame_required_sources(plan)
    if not required:
        return
    for step in plan.steps:
        grain = step.sub_intent.grain or "row_level"
        if grain not in VALID_GRAINS:
            grain = "row_level"
        if grain != "scalar":
            continue
        if step.source_id in required:
            raise FederationDeclarationError(
                f"federation member {step.source_id!r} has scalar grain and contributes no "
                f"coordinator frame, but combine requires src_{step.source_id}"
            )


def validate_federation_coordinator_column_types(
    plan: FederatedPlan,
    schema: SchemaGraph,
    *,
    manifest: FederationManifest | None = None,
) -> None:
    """Refuse when a projected coordinator column cannot be mapped to DuckDB."""
    if len(plan.steps) <= 1:
        return
    seen: set[tuple[str, str]] = set()
    for step in plan.steps:
        for key in step.projected_keys:
            pair = (step.source_id, key)
            if pair in seen:
                continue
            seen.add(pair)
            col_name = key.rsplit(".", 1)[-1] if "." in key else key
            table_names = [
                name
                for name, table in schema.tables.items()
                if table.source_id == step.source_id and col_name in table.columns
            ]
            if not table_names:
                continue
            table_name = table_names[0]
            table_meta = schema.tables.get(table_name)
            if table_meta is None:
                continue
            column = table_meta.columns.get(col_name)
            if column is None:
                continue
            data_type = str(column.data_type or "").strip()
            if not data_type:
                continue
            lossy_reason = _coordinator_column_type_lossy_reason(data_type, column_meta=column)
            if lossy_reason:
                raise FederationDeclarationError(
                    f"federation coordinator column {col_name!r} has lossy data_type "
                    f"{data_type!r} for member {step.source_id!r}: {lossy_reason}"
                )
            if schema_column_duckdb_type(data_type, column_meta=column) is None:
                raise FederationDeclarationError(
                    f"federation coordinator column {col_name!r} has unsupported data_type "
                    f"{data_type!r} for member {step.source_id!r}"
                )


def schema_column_duckdb_type(
    data_type: str,
    *,
    column_meta: ColumnMetadata | None = None,
    column_name: str = "",
    source_id: str = "",
) -> str | None:
    """Map composite schema ``data_type`` text to a DuckDB column type."""
    raw = str(data_type or "").strip()
    if not raw:
        return None
    base = raw.lower().split("(", 1)[0].strip()
    mapped = FEDERATION_COORDINATOR_DUCKDB_TYPE_MAP.get(base)
    if callable(mapped):
        precision = column_meta.numeric_precision if column_meta is not None else None
        scale = column_meta.numeric_scale if column_meta is not None else None
        if precision is None and scale is None:
            precision, scale = parse_numeric_type_arguments(raw)
        if precision is not None and scale is not None:
            return mapped(precision, scale)
        label = column_name or (column_meta.name if column_meta is not None else "")
        if label:
            notify(
                (
                    f"federation coordinator column {label!r} missing numeric precision/scale; "
                    f"using {FEDERATION_COORDINATOR_DECIMAL_FALLBACK}"
                ),
                stage="federation",
                code=DIAGNOSTIC_CODE_FEDERATION_COORDINATOR_DECIMAL_FALLBACK,
                level="warning",
                source_id=source_id,
                details=(
                    ("phase", "transfer"),
                    ("column", label),
                    ("data_type", raw),
                    ("fallback_type", FEDERATION_COORDINATOR_DECIMAL_FALLBACK),
                ),
            )
        return FEDERATION_COORDINATOR_DECIMAL_FALLBACK
    if mapped is not None:
        if column_data_type_is_timezone_aware(raw):
            label = column_name or (column_meta.name if column_meta is not None else "")
            if label:
                notify(
                    (
                        f"federation coordinator normalised timestamp values for column {label!r} "
                        "to UTC for coordinator transfer"
                    ),
                    stage="federation",
                    code=DIAGNOSTIC_CODE_FEDERATION_TIMESTAMP_NORMALISED,
                    level="info",
                    source_id=source_id,
                    details=(("phase", "transfer"), ("column", label)),
                )
        return mapped
    return None


def _coordinator_column_type_lossy_reason(data_type: str, *, column_meta: ColumnMetadata | None = None) -> str | None:
    """Return a refusal reason when *data_type* would be lossy in the coordinator frame."""
    raw = str(data_type or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    base = lowered.split("(", 1)[0].strip()
    if base in {"timestamptz", "timetz"} or "with time zone" in lowered:
        return None
    if base == "tinyint":
        return "tinyint width is lossy in the federation coordinator"
    if base in {"decimal", "numeric", "number", "money"}:
        if column_meta is not None and column_meta.is_exact_numeric:
            precision = column_meta.numeric_precision
            scale = column_meta.numeric_scale
            if precision is None or scale is None:
                precision, scale = parse_numeric_type_arguments(raw)
            if precision is not None and scale is not None:
                return None
        if "(" in lowered:
            inner = lowered.split("(", 1)[1].rstrip(")").strip()
            if inner and inner != "38,9" and inner != "38, 9":
                return "decimal precision is lossy in the federation coordinator"
        elif base == "money":
            return None
        else:
            return "decimal precision is lossy in the federation coordinator"
    return None


def resolve_member_limits_for_source(manifest: FederationManifest, source_id: str) -> FederationMemberResolvedLimits:
    """Resolve per-member limits using member, coordinator, then global policy fallbacks."""
    binding_limits: FederationSourceLimits | None = None
    for binding in manifest.sources:
        if binding.source_id == source_id:
            binding_limits = binding.limits
            break
    row_cap = int(manifest.coordinator.default_source_row_cap)
    if binding_limits is not None and binding_limits.row_cap is not None:
        row_cap = int(binding_limits.row_cap)
    timeout_ms = int(manifest.coordinator.default_source_timeout_ms)
    if binding_limits is not None and binding_limits.timeout_ms is not None:
        timeout_ms = int(binding_limits.timeout_ms)
    max_query_cost_rows = PolicyConfig.MAX_QUERY_COST_ROWS
    if binding_limits is not None and binding_limits.max_query_cost_rows is not None:
        max_query_cost_rows = float(binding_limits.max_query_cost_rows)
    max_query_cost_bytes = PolicyConfig.MAX_QUERY_COST_BYTES
    if binding_limits is not None and binding_limits.max_query_cost_bytes is not None:
        max_query_cost_bytes = float(binding_limits.max_query_cost_bytes)
    profile_timeout_ms = effective_profile_timeout_ms()
    if binding_limits is not None and binding_limits.profile_timeout_ms is not None:
        profile_timeout_ms = int(binding_limits.profile_timeout_ms)
    return FederationMemberResolvedLimits(
        source_id=source_id,
        row_cap=row_cap,
        timeout_ms=timeout_ms,
        max_query_cost_rows=max_query_cost_rows,
        max_query_cost_bytes=max_query_cost_bytes,
        profile_timeout_ms=profile_timeout_ms,
    )


def source_semijoin_enabled(manifest: FederationManifest, source_id: str) -> bool:
    """Return whether semi-join reduction is enabled for *source_id*."""
    for binding in manifest.sources:
        if binding.source_id == source_id:
            if binding.limits is not None:
                return bool(binding.limits.semijoin_enabled)
            return True
    return True


def _planning_sources_for_table(
    table: str,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> frozenset[str]:
    """Return member sources that should receive a planning step for *table*."""
    all_sources = sources_for_table(table, manifest, mappings, source_by_table, schema)
    if len(all_sources) <= 1:
        return all_sources
    for lt in mappings.logical_tables:
        if lt.logical != table:
            continue
        if lt.semantics == "replica":
            auth = select_replica_member_source(lt)
            return frozenset({auth})
        return all_sources
    return all_sources


def _table_owned_by_source(
    table: str,
    source_id: str,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    schema: SchemaGraph | None = None,
) -> bool:
    """Return True when *source_id* should execute a sub-plan over *table*."""
    return source_id in _planning_sources_for_table(table, manifest, mappings, source_by_table, schema)


def _widen_federation_scope_tables(tables: set[str], intent: RuntimeIntent, manifest: FederationManifest) -> None:
    """Add tables from the resolved join path and cross-source join endpoints when scope is multi-source."""
    path_tables = {str(entry).strip() for entry in (intent.chosen_join_path_signature or []) if str(entry).strip()}
    tables.update(path_tables)
    scope = set(tables) | path_tables
    if not scope:
        return
    source_ids: set[str] = set()
    for table in scope:
        owner = manifest.table_namespace.get(table, "")
        if owner:
            source_ids.add(owner)
    for join in manifest.cross_source_joins:
        left_tbl, _left_col = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _right_col = split_qualified_column(join.right, manifest=manifest)
        left_src = manifest.table_namespace.get(left_tbl, "")
        right_src = manifest.table_namespace.get(right_tbl, "")
        if not left_src or not right_src or left_src == right_src:
            continue
        if left_tbl in scope and right_src in source_ids:
            tables.add(left_tbl)
            tables.add(right_tbl)
        elif right_tbl in scope and left_src in source_ids:
            tables.add(left_tbl)
            tables.add(right_tbl)


def _expand_scope_along_foreign_keys(tables: set[str], schema: SchemaGraph) -> None:
    """Add tables reachable by one foreign-key hop from tables already in scope."""
    for tbl in list(tables):
        table_meta = schema.tables.get(tbl)
        if table_meta is None:
            continue
        for edge in table_meta.foreign_keys:
            if edge.inference_tag == InferenceTag.CROSS_SOURCE:
                continue
            if edge.dst_table and edge.dst_table not in tables:
                tables.add(edge.dst_table)
            if edge.src_table and edge.src_table not in tables:
                tables.add(edge.src_table)


def federation_table_set(
    intent: RuntimeIntent, schema: SchemaGraph, manifest: FederationManifest, mappings: FederationMappings | None = None
) -> FederationTableSet:
    """Return intent tables and their owning sources using the composite graph."""
    mappings = mappings or FederationMappings(version=FEDERATION_MAPPINGS_VERSION)
    tables = set(intent.tables or [])
    refs = collect_referenced_tables(
        list(intent.select_cols or []),
        list(intent.order_by_cols or []),
        list(intent.group_by_cols or []),
        list(PredicateGroup.where_leaves(intent.where) or []),
        list(PredicateGroup.having_leaves(intent.having) or []),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        include_unreferenced_registries=False,
    )
    tables.update(refs)
    _widen_federation_scope_tables(tables, intent, manifest)
    _expand_scope_along_foreign_keys(tables, schema)
    source_by_table = table_source_index(schema, mappings, manifest)
    sources: set[str] = set()
    for table in tables:
        sources.update(sources_for_table(table, manifest, mappings, source_by_table, schema))
    return FederationTableSet(tables=frozenset(tables), source_by_table=source_by_table, sources=frozenset(sources))


def _column_metadata_for_table_key(
    schema: SchemaGraph | None, table_name: str | None, key: str
) -> ColumnMetadata | None:
    if schema is None or not table_name:
        return None
    table = schema.tables.get(table_name)
    if table is None:
        return None
    return table.columns.get(key)


def _harmonized_join_key_decimal_scale(
    left_meta: ColumnMetadata | None, right_meta: ColumnMetadata | None
) -> int | None:
    if left_meta is None or right_meta is None:
        return None
    if not (left_meta.is_exact_numeric and right_meta.is_exact_numeric):
        return None
    return max(left_meta.numeric_scale or 0, right_meta.numeric_scale or 0)


def _join_key_duckdb_cast_type(meta: ColumnMetadata, *, harmonized_scale: int | None = None) -> str | None:
    """Map a join-key column to a DuckDB cast target."""
    value_type = (meta.value_type or "").strip().lower()
    if not value_type:
        dtype = str(meta.data_type or "").lower()
        if any(token in dtype for token in ("int", "bigint", "smallint")):
            value_type = "integer"
        elif any(token in dtype for token in ("decimal", "numeric", "double", "float", "real")):
            value_type = "number"
        elif any(token in dtype for token in ("date", "timestamp", "time")):
            value_type = "date"
        elif "bool" in dtype:
            value_type = "boolean"
        else:
            return None
    if value_type == "integer":
        return "BIGINT"
    if value_type == "number":
        if meta.is_exact_numeric:
            precision = meta.numeric_precision
            scale = meta.numeric_scale
            if precision is None or scale is None:
                precision, scale = parse_numeric_type_arguments(str(meta.data_type or ""))
            if harmonized_scale is not None:
                scale = harmonized_scale
            if precision is None:
                precision = FEDERATION_COORDINATOR_DECIMAL_MAX_PRECISION
            if scale is None:
                scale = 0
            return f"DECIMAL({precision}, {scale})"
        return "DOUBLE"
    if value_type == "date":
        return "TIMESTAMP"
    if value_type == "boolean":
        return "BOOLEAN"
    return None


def _coordinator_join_key_pair_exprs(
    left_alias: str,
    left_key: str,
    *,
    left_table: str | None,
    right_alias: str,
    right_key: str,
    right_table: str | None,
    schema: SchemaGraph | None,
) -> tuple[str, str]:
    """Render typed join-key expressions for both sides of a coordinator join."""
    left_meta = _column_metadata_for_table_key(schema, left_table, left_key)
    right_meta = _column_metadata_for_table_key(schema, right_table, right_key)
    harmonized_scale = _harmonized_join_key_decimal_scale(left_meta, right_meta)
    return (
        _coordinator_join_key_expr(
            left_alias,
            left_key,
            schema=schema,
            table_name=left_table,
            harmonized_scale=harmonized_scale,
        ),
        _coordinator_join_key_expr(
            right_alias,
            right_key,
            schema=schema,
            table_name=right_table,
            harmonized_scale=harmonized_scale,
        ),
    )


def _coordinator_join_key_expr(
    alias: str,
    key: str,
    *,
    schema: SchemaGraph | None,
    table_name: str | None,
    harmonized_scale: int | None = None,
) -> str:
    """Render a typed coordinator join key expression for *alias*.*key*."""
    ident = f"{alias}.{Dialect.sqlglot_quote_identifier(key)}"
    meta = _column_metadata_for_table_key(schema, table_name, key)
    if meta is None:
        return ident
    cast_type = _join_key_duckdb_cast_type(meta, harmonized_scale=harmonized_scale)
    if cast_type is not None:
        return f"CAST({ident} AS {cast_type})"
    return ident


def _join_specs_connect_sources(sources: frozenset[str], join_specs: Sequence[JoinSpec]) -> bool:
    if len(sources) <= 1:
        return True
    parent = {source_id: source_id for source_id in sources}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    for spec in join_specs:
        if spec.left_source in parent and spec.right_source in parent:
            union(spec.left_source, spec.right_source)
    return len({find(source_id) for source_id in sources}) == 1


def _group_by_tables(intent: RuntimeIntent) -> set[str]:
    return collect_referenced_tables(
        [],
        [],
        intent.group_by_cols or [],
        [],
        [],
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
        include_unreferenced_registries=False,
    )


def _expr_has_unattributable_raw_sql(expr: NormalizedExpr) -> bool:
    """Return True when *expr* carries raw SQL with no recoverable column references."""
    if not (expr.raw_sql or "").strip():
        return False
    return not extract_columns_from_expr(expr)


def _unattributable_raw_sql_reason(intent: RuntimeIntent) -> str | None:
    """Return an ineligibility reason when the intent contains unattributable raw SQL."""
    for sc in intent.select_cols or []:
        if _expr_has_unattributable_raw_sql(sc.expr):
            return "expression contains unattributable raw_sql fragment"
    for obc in intent.order_by_cols or []:
        if _expr_has_unattributable_raw_sql(obc.expr):
            return "expression contains unattributable raw_sql fragment"
    for group in intent.group_by_cols or []:
        if _expr_has_unattributable_raw_sql(group):
            return "expression contains unattributable raw_sql fragment"
    for fp in PredicateGroup.where_leaves(intent.where) or []:
        if _expr_has_unattributable_raw_sql(fp.left_expr):
            return "expression contains unattributable raw_sql fragment"
        if fp.right_expr and _expr_has_unattributable_raw_sql(fp.right_expr):
            return "expression contains unattributable raw_sql fragment"
    for hp in PredicateGroup.having_leaves(intent.having) or []:
        if _expr_has_unattributable_raw_sql(hp.left_expr):
            return "expression contains unattributable raw_sql fragment"
        if hp.right_expr and _expr_has_unattributable_raw_sql(hp.right_expr):
            return "expression contains unattributable raw_sql fragment"
    return None


def _clause_referenced_sources(
    *,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    schema: SchemaGraph | None = None,
    window_registry: Sequence[Any] | None = None,
    case_registry: Sequence[Any] | None = None,
) -> set[str]:
    refs = collect_referenced_tables(
        list(select_cols or []),
        list(order_by_cols or []),
        list(group_by_cols or []),
        list(where_params or []),
        list(having_param or []),
        window_registry=window_registry,
        case_registry=case_registry,
    )
    if manifest is not None and mappings is not None:
        return sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
    sources: set[str] = set()
    for table in refs:
        source_id = source_by_table.get(table, "")
        if source_id:
            sources.add(source_id)
    return sources


def _clause_spans_multiple_sources(
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    mappings: FederationMappings | None = None,
    schema: SchemaGraph | None = None,
    window_registry: Sequence[Any] | None = None,
    case_registry: Sequence[Any] | None = None,
    select_cols: Sequence[SelectCol] | None = None,
    order_by_cols: Sequence[OrderByCol] | None = None,
    group_by_cols: Sequence[NormalizedExpr] | None = None,
    where_params: Sequence[WhereParam] | None = None,
    having_param: Sequence[HavingParam] | None = None,
) -> bool:
    return (
        len(
            _clause_referenced_sources(
                select_cols=select_cols,
                order_by_cols=order_by_cols,
                group_by_cols=group_by_cols,
                where_params=where_params,
                having_param=having_param,
                source_by_table=source_by_table,
                manifest=manifest,
                mappings=mappings,
                schema=schema,
                window_registry=window_registry,
                case_registry=case_registry,
            )
        )
        > 1
    )


def _intent_cross_source_aggregate_shape(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> tuple[bool, bool]:
    """Return ``(has_cross_source_shape, has_decomposable_cross_source_aggregate)``."""
    intent_tables = set(intent.tables or [])
    if not intent_tables:
        return False, False
    intent_sources = intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return False, False
    registry_kw = intent_registry_kw(intent)
    has_shape = False
    has_decomposable = False
    group_tables = _group_by_tables(intent)
    if group_tables:
        group_sources = intent_table_sources(group_tables, manifest, mappings, source_by_table, schema=schema)
        if len(group_sources) > 1:
            has_shape = True
        elif len(group_sources) == 1 and intent_sources - group_sources:
            has_shape = True
    for sc in intent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        agg_tables = _tables_referenced_by_select_col(sc, **registry_kw)
        agg_sources = intent_table_sources(agg_tables, manifest, mappings, source_by_table, schema=schema)
        cross = len(agg_sources) > 1 or (len(agg_sources) == 1 and bool(intent_sources - agg_sources))
        if not cross:
            continue
        has_shape = True
        func, has_distinct = _select_col_agg_meta(sc)
        if func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS and not has_distinct:
            has_decomposable = True
    return has_shape, has_decomposable


def _intent_has_cross_source_aggregate(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> bool:
    has_shape, _ = _intent_cross_source_aggregate_shape(intent, manifest, mappings, source_by_table, schema=schema)
    return has_shape


def _cross_source_aggregate_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse cross-source aggregate shapes the coordinator cannot fold."""
    intent_tables = set(intent.tables or [])
    if not intent_tables:
        return None
    intent_sources = intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    registry_kw = intent_registry_kw(intent)
    group_tables = _group_by_tables(intent)
    if group_tables:
        group_sources = intent_table_sources(group_tables, manifest, mappings, source_by_table, schema=schema)
        if len(group_sources) > 1:
            for sc in intent.select_cols or []:
                if not _is_sql_aggregate_select_col(sc):
                    continue
                func, has_distinct = _select_col_agg_meta(sc)
                if has_distinct or func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
                    return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
    for sc in intent.select_cols or []:
        if not _is_sql_aggregate_select_col(sc):
            continue
        agg_tables = _tables_referenced_by_select_col(sc, **registry_kw)
        agg_sources = intent_table_sources(agg_tables, manifest, mappings, source_by_table, schema=schema)
        cross = len(agg_sources) > 1 or (len(agg_sources) == 1 and bool(intent_sources - agg_sources))
        if not cross:
            continue
        func, has_distinct = _select_col_agg_meta(sc)
        union_targets = {
            lt.logical for lt in mappings.logical_tables if lt.semantics == "union" and lt.logical in agg_tables
        }
        if not func:
            if union_targets:
                func = "count"
            elif sc.is_aggregated and (intent.grain or "") == "scalar":
                func = "sum"
        if has_distinct:
            return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
        if len(agg_sources) > 1:
            if union_targets and func in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
                continue
        if func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
            return f"cross-source aggregate not supported: {_select_col_agg_label(sc)}"
    return None


def _source_is_left_combine_nullable_side(source_id: str, combine: Sequence[JoinSpec]) -> bool:
    """Return True when *source_id* is the nullable side of a declared left combine edge."""
    for spec in combine:
        kind = (spec.kind or "inner").strip().lower()
        if kind == "left" and source_id == spec.right_source:
            return True
    return False


def reducing_edge_allowed_for_target(
    target_source_id: str,
    join: FederationCrossSourceJoin,
    manifest: FederationManifest,
    *,
    schema: SchemaGraph | None = None,
) -> bool:
    """Return True when semi-join reduction may filter *target_source_id* for *join*."""
    spec = _cross_source_join_to_spec(join, manifest, schema=schema)
    return not _member_rows_final_after_combine(
        target_source_id,
        combine=(spec,),
        schema=schema,
        manifest=manifest,
    )


def _join_preserves_member_rows(
    source_id: str, spec: JoinSpec, *, schema: SchemaGraph | None, manifest: FederationManifest | None
) -> bool | None:
    """Return whether *spec* leaves *source_id* row membership and multiplicity unchanged."""
    if source_id not in (spec.left_source, spec.right_source):
        return True
    this_is_left = source_id == spec.left_source
    kind = (spec.kind or "inner").strip().lower()
    if kind == "inner":
        return False
    if kind == "left" and not this_is_left:
        return False
    if schema is None:
        return None
    other_source = spec.right_source if this_is_left else spec.left_source
    other_key = spec.right_key if this_is_left else spec.left_key
    return source_join_key_is_unique(schema, other_source, other_key, manifest=manifest)


def _member_rows_final_after_combine(
    source_id: str,
    *,
    combine: tuple[JoinSpec, ...] | None,
    schema: SchemaGraph | None,
    manifest: FederationManifest | None,
) -> bool:
    """Return True only when every combine edge involving *source_id* preserves its rows."""
    if not combine:
        return True
    for spec in combine:
        if source_id not in (spec.left_source, spec.right_source):
            continue
        preserved = _join_preserves_member_rows(source_id, spec, schema=schema, manifest=manifest)
        if preserved is not True:
            return False
    return True


def _window_owner_source(entry: Any, source_by_table: Mapping[str, str]) -> str | None:
    """Return the single member owning *entry* columns, or None when unowned or spanning."""
    refs = collect_referenced_tables([], [], [], [], [], window_registry=[entry], case_registry=[])
    sources = {source_by_table.get(table, "") for table in refs if source_by_table.get(table, "")}
    sources.discard("")
    if len(sources) != 1:
        return None
    return next(iter(sources))


def _member_window_rows_are_final(
    source_id: str,
    entry: Any,
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when *entry* is local to *source_id* and that member's rows are final."""
    owner = _window_owner_source(entry, source_by_table)
    if owner != source_id:
        return False
    if len(set(source_by_table.values())) <= 1:
        return True
    if manifest is None or schema is None:
        return False
    return _member_rows_final_after_combine(source_id, combine=combine, schema=schema, manifest=manifest)


def _window_requires_coordinator(
    entry: Any,
    *,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest | None = None,
    schema: SchemaGraph | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> bool:
    """Return True when *entry* must be evaluated after the cross-source combine."""
    owner = _window_owner_source(entry, source_by_table)
    if owner is None:
        return True
    return not _member_window_rows_are_final(
        owner, entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
    )


def _partition_cte_steps_for_source(
    cte_steps: Sequence[RuntimeCteStep], source_id: str, source_by_table: Mapping[str, str]
) -> list[RuntimeCteStep]:
    """Keep only CTE steps owned by *source_id*, with registries filtered to clauses."""
    if not cte_steps:
        return []
    owners = assign_cte_sources(cte_steps, source_by_table)
    kept: list[RuntimeCteStep] = []
    for cte in cte_steps:
        name = cte.cte_name
        if not name or owners.get(name) != source_id:
            continue
        cte_copy = copy.deepcopy(cte)
        window_registry, case_registry = where_scope_registries_to_referenced(
            select_cols=cte_copy.select_cols,
            order_by_cols=cte_copy.order_by_cols,
            group_by_cols=cte_copy.group_by_cols,
            where_params=(PredicateGroup.where_leaves(cte_copy.where)),
            having_param=(PredicateGroup.having_leaves(cte_copy.having)),
            window_registry=cte_copy.window_registry,
            case_registry=cte_copy.case_registry,
        )
        if window_registry != list(cte_copy.window_registry or []) or case_registry != list(
            cte_copy.case_registry or []
        ):
            cte_copy = replace(cte_copy, window_registry=window_registry, case_registry=case_registry)
        kept.append(cte_copy)
    return kept


def _partition_registries_for_source(
    intent: RuntimeIntent, source_id: str, source_by_table: Mapping[str, str]
) -> RuntimeIntent:
    """Keep window/case rows local to *source_id* only when member rows are final after combine."""
    source_tables = {table for table, sid in source_by_table.items() if sid == source_id}
    ctx = _WINDOW_FINALITY_CTX

    def _registry_local(registry: Sequence[Any], *, field: str) -> list[Any]:
        kept: list[Any] = []
        for entry in registry or []:
            if field == "window":
                refs = collect_referenced_tables([], [], [], [], [], window_registry=[entry], case_registry=[])
            else:
                refs = collect_referenced_tables([], [], [], [], [], window_registry=[], case_registry=[entry])
            if not refs or not refs.issubset(source_tables):
                continue
            if field == "window":
                if not _member_window_rows_are_final(
                    source_id,
                    entry,
                    source_by_table=source_by_table,
                    manifest=ctx.manifest if ctx is not None else None,
                    schema=ctx.schema if ctx is not None else None,
                    combine=ctx.combine if ctx is not None else None,
                ):
                    continue
            kept.append(entry)
        return kept

    window_registry, case_registry = where_scope_registries_to_referenced(
        select_cols=intent.select_cols,
        order_by_cols=intent.order_by_cols,
        group_by_cols=intent.group_by_cols,
        where_params=PredicateGroup.where_leaves(intent.where),
        having_param=PredicateGroup.having_leaves(intent.having),
        window_registry=intent.window_registry,
        case_registry=intent.case_registry,
    )
    window_registry = _registry_local(window_registry, field="window")
    case_registry = _registry_local(case_registry, field="case")
    if window_registry == list(intent.window_registry or []) and case_registry == list(intent.case_registry or []):
        return intent
    return replace(intent, window_registry=window_registry, case_registry=case_registry)


def unqualified_column_name(qualified: str) -> str:
    text = str(qualified or "").strip()
    if not text or "(" in text:
        return ""
    match = FEDERATION_QUALIFIED_COLUMN_REF_RE.match(text)
    if match:
        return match.group(2)
    three = FEDERATION_QUALIFIED_THREE_PART_REF_RE.match(text)
    if three:
        return three.group(3)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _source_column_names_for_step(plan: FederatedPlan, source_id: str) -> set[str]:
    step = next((item for item in plan.steps if item.source_id == source_id), None)
    if step is None:
        return set()
    names: set[str] = set()
    for key in step.projected_keys:
        col = unqualified_column_name(key)
        if col:
            names.add(col)
    for sc in step.sub_intent.select_cols or []:
        if _looks_aggregated(sc):
            continue
        col = unqualified_column_name(_select_col_term(sc))
        if col:
            names.add(col)
    return names


def combine_select_column_names(plan: FederatedPlan) -> list[str] | None:
    """Derive explicit coordinator column names from projected/residual keys."""
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        col = unqualified_column_name(name)
        if col and col not in seen:
            seen.add(col)
            names.append(col)

    for step in plan.steps:
        for key in step.projected_keys:
            _add(key)
        for sc in step.sub_intent.select_cols or []:
            if _looks_aggregated(sc):
                continue
            _add(_select_col_term(sc))
    residual = plan.residual
    if residual is not None:
        for fp in PredicateGroup.where_leaves(residual.where):
            left_ref = fp.left_expr.column_ref or fp.left_expr.primary_term or ""
            _add(left_ref)
            right_ref = getattr(fp.right_expr, "column_ref", None) or getattr(fp.right_expr, "primary_term", None) or ""
            if right_ref:
                _add(str(right_ref))
        for sc in residual.select_cols or []:
            if _looks_aggregated(sc):
                alias = (sc.output_alias or "").strip()
                if alias:
                    if alias not in seen:
                        seen.add(alias)
                        names.append(alias)
                continue
            alias = (sc.output_alias or "").strip()
            if alias:
                if alias not in seen:
                    seen.add(alias)
                    names.append(alias)
                continue
            _add(_select_col_term(sc))
    if not names and isinstance(plan.combine, tuple):
        for spec in plan.combine:
            if isinstance(spec, JoinSpec):
                _add(spec.left_key)
                _add(spec.right_key)
                if spec.logical_key:
                    _add(spec.logical_key)
    return names or None


def render_combine_select_keyword(cols: list[str] | None) -> str:
    if not cols:
        raise FederationRuntimeError("federation combine requires explicit column projection")
    return ", ".join(Dialect.sqlglot_quote_identifier(col) for col in cols)


def _render_join_select_keyword(
    cols: list[str] | None, *, left_alias: str, right_alias: str, left_cols: set[str], right_cols: set[str]
) -> str:
    if not cols:
        raise FederationRuntimeError("federation combine requires explicit column projection")
    exprs: list[str] = []
    for col in cols:
        ident = Dialect.sqlglot_quote_identifier(col)
        in_left = col in left_cols
        in_right = col in right_cols
        if in_left and in_right:
            exprs.append(f"{left_alias}.{ident}, {right_alias}.{ident}")
        elif in_right:
            exprs.append(f"{right_alias}.{ident}")
        elif in_left:
            exprs.append(f"{left_alias}.{ident}")
        else:
            exprs.append(f"{left_alias}.{ident}")
    return ", ".join(exprs)


def _cross_source_where(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> list[WhereParam]:
    cross: list[WhereParam] = []
    registry_kw = intent_registry_kw(intent)
    for fp in PredicateGroup.where_leaves(intent.where) or []:
        refs = collect_referenced_tables([], [], [], [fp], [], **registry_kw)
        srcs = sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            cross.append(fp)
    return cross


def _cross_source_having(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> list[HavingParam]:
    cross: list[HavingParam] = []
    registry_kw = intent_registry_kw(intent)
    for hp in PredicateGroup.having_leaves(intent.having) or []:
        refs = collect_referenced_tables([], [], [], [], [hp], **registry_kw)
        srcs = sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            cross.append(hp)
    return cross


def _param_qualified_columns(param: WhereParam | HavingParam) -> set[str]:
    cols: set[str] = set()
    for expr in (param.left_expr, param.right_expr):
        if expr is None:
            continue
        for col in extract_columns_from_expr(expr):
            if "." in col:
                cols.add(col)
    return cols


def _cross_where_relates_to_join(fp: WhereParam, manifest: FederationManifest) -> bool:
    filter_cols = _param_qualified_columns(fp)
    if not filter_cols:
        return False
    for join in manifest.cross_source_joins:
        join_cols = {join.left, join.right}
        if filter_cols & join_cols:
            return True
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        filter_tables = {split_qualified_column(c, manifest=manifest)[0] for c in filter_cols if "." in c}
        if left_tbl in filter_tables and right_tbl in filter_tables:
            return True
    return False


def _predicate_is_literal_comparison(param: WhereParam | HavingParam) -> bool:
    """Return True when *param* compares a column to a literal rather than another column."""
    if (param.value_type or "").strip().lower() == "column":
        if param.right_expr is None:
            return False
        return not extract_columns_from_expr(param.right_expr)
    return True


def _join_covered_literal_push_allowed(
    param: WhereParam,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
    registry_kw: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when a join-covered literal filter should execute on its owning member."""
    if not _predicate_is_literal_comparison(param):
        return False
    srcs = predicate_param_sources(param, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw)
    if len(srcs) != 1:
        return False
    return _cross_where_relates_to_join(param, manifest)


def _cross_source_window_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse windows that need post-combine evaluation when no combine path exists."""
    intent_tables = set(intent.tables or [])
    intent_sources = intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    combine = _join_specs_for_sources(
        manifest, mappings, frozenset(intent_sources), schema=schema, scope_tables=frozenset(intent_tables)
    )
    union_specs = _union_specs_for_intent(intent_tables, mappings, source_by_table)
    has_combine = bool(combine) or bool(union_specs)
    for entry in intent.window_registry or []:
        if not _window_requires_coordinator(
            entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
        ):
            continue
        if has_combine:
            continue
        rid = str(getattr(entry, "registry_id", "") or "").strip() or "window"
        spec = getattr(entry, "window_spec", None)
        func = str(getattr(spec, "function", "") or "").strip() or "window"
        return f"cross-source window is not supported: {rid} ({func})"
    return None


def _cross_source_scalar_subquery_ineligible_reason(
    intent: RuntimeIntent, source_by_table: Mapping[str, str]
) -> str | None:
    """Refuse scalar-subquery CTE steps whose body is not owned by a single member."""
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return None
    owners = assign_cte_sources(cte_steps, source_by_table)
    cte_names = {step.cte_name for step in cte_steps if step.cte_name}
    cte_names_lower = {name.lower() for name in cte_names}
    for cte in cte_steps:
        emission = getattr(cte, "emission", "join_table")
        if emission != "scalar_subquery":
            continue
        name = (cte.cte_name or "").strip()
        if not name:
            continue
        if name not in owners:
            return f"cross-source correlated subquery is not supported: {name}"
        declared = {table for table in (cte.tables or []) if table}
        refs = collect_referenced_tables(
            cte.select_cols,
            cte.order_by_cols,
            cte.group_by_cols,
            PredicateGroup.where_leaves(cte.where),
            PredicateGroup.having_leaves(cte.having),
            window_registry=cte.window_registry,
            case_registry=cte.case_registry,
            include_unreferenced_registries=False,
        )
        base_tables = {
            table for table in (declared | refs) if table not in cte_names and table.lower() not in cte_names_lower
        }
        sources = {source_by_table[table] for table in base_tables if table in source_by_table}
        if len(sources) > 1:
            return f"cross-source correlated subquery is not supported: {name}"
    return None


def _federation_clause_ineligible_reason(
    intent: RuntimeIntent,
    manifest: FederationManifest,
    mappings: FederationMappings,
    source_by_table: Mapping[str, str],
    *,
    schema: SchemaGraph | None = None,
) -> str | None:
    """Refuse clause shapes federation cannot compose instead of silently dropping them."""
    registry_kw = intent_registry_kw(intent)
    predicate_reason = predicate_group_spans_sources(
        intent.where, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
    )
    if predicate_reason:
        return predicate_reason
    having_predicate_reason = predicate_group_spans_sources(
        intent.having, manifest, mappings, source_by_table, schema=schema, registry_kw=registry_kw
    )
    if having_predicate_reason:
        return having_predicate_reason
    window_reason = _cross_source_window_ineligible_reason(intent, manifest, mappings, source_by_table, schema=schema)
    if window_reason:
        return window_reason
    subquery_reason = _cross_source_scalar_subquery_ineligible_reason(intent, source_by_table)
    if subquery_reason:
        return subquery_reason
    probe_reason = cross_source_probe_cte_ineligible_reason(intent, manifest, source_by_table)
    if probe_reason:
        return probe_reason
    if intent.distinct_on:
        if distinct_on_spans_sources(intent, manifest, mappings, source_by_table, schema=schema):
            combine = _join_specs_for_sources(
                manifest,
                mappings,
                frozenset(
                    intent_table_sources(set(intent.tables or []), manifest, mappings, source_by_table, schema=schema)
                ),
                schema=schema,
                scope_tables=frozenset(intent.tables or []),
            )
            union_specs = _union_specs_for_intent(set(intent.tables or []), mappings, source_by_table)
            if not combine and not union_specs:
                return "cross-source distinct_on requires declared join"
        elif not intent.order_by_cols:
            return "distinct_on requires order_by_cols"
    cross_having = _cross_source_having(intent, manifest, mappings, source_by_table, schema=schema)
    if cross_having:
        uncovered = [hp for hp in cross_having if not _cross_having_covered(hp, manifest)]
        if uncovered:
            labels = ", ".join(predicate_clause_label(hp) for hp in uncovered)
            return f"cross-source HAVING requires declared join: {labels}"
    return None


def _cross_having_covered(hp: HavingParam, manifest: FederationManifest) -> bool:
    if not manifest.cross_source_joins:
        return False
    cols = _param_qualified_columns(hp)
    if not cols:
        return False
    for join in manifest.cross_source_joins:
        join_cols = {join.left, join.right}
        if cols & join_cols:
            return True
        left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
        right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
        filter_tables = {split_qualified_column(c, manifest=manifest)[0] for c in cols if "." in c}
        if left_tbl in filter_tables and right_tbl in filter_tables:
            return True
    return False


def _union_specs_for_intent(
    tables: set[str], mappings: FederationMappings, source_by_table: Mapping[str, str]
) -> list[UnionSpec]:
    del source_by_table
    specs: list[UnionSpec] = []
    for lt in mappings.logical_tables:
        if lt.logical not in tables:
            continue
        if lt.semantics == "replica":
            auth = select_replica_member_source(lt)
            member_sources: tuple[str, ...] = (auth,)
        else:
            member_sources = tuple(sorted({m.source for m in lt.members}))
        specs.append(UnionSpec(logical_table=lt.logical, member_source_ids=member_sources, semantics=lt.semantics))
    return specs


def _copy_runtime_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Return a deep copy of *intent* so sub-intents do not share mutable IR nodes."""
    return copy.deepcopy(intent)


def _isolate_sub_intent_decisions(intent: RuntimeIntent) -> RuntimeIntent:
    """Clear parent join and validation decisions a member sub-intent cannot honour."""
    return replace(
        intent,
        chosen_join_candidate_id="",
        chosen_join_path_signature=[],
        sql_shape=None,
        schema_invalid=False,
        interpret_cte_names=[],
        grain="row_level",
    )


def _member_schema_for_sub_intent_repair(
    source_id: str,
    composite_schema: SchemaGraph,
    *,
    manifest: FederationManifest,
    member_graphs: Mapping[str, SchemaGraph] | None,
) -> SchemaGraph:
    """Return the schema a sub-intent repair must be judged against. Prefers the loaded member graph. When that graph is absent, uses the per-source composite slice — never the full composite graph."""
    if member_graphs is not None and source_id in member_graphs:
        return member_graphs[source_id]
    return member_schema_slice(composite_schema, source_id, manifest=manifest, member_graph=None)


def _remap_member_schema_logical_tables(
    member_schema: SchemaGraph,
    source_id: str,
    mappings: FederationMappings,
) -> SchemaGraph:
    """Align a composite member slice with physical table and column names after logical rewrite."""
    table_map = _member_logical_table_map(source_id, mappings)
    column_rewrites = _member_logical_column_rewrites(source_id, mappings)
    if not table_map and not column_rewrites:
        return member_schema
    col_by_table: dict[str, dict[str, str]] = {}
    for logical_table, logical_col, _physical_table, physical_col in column_rewrites:
        col_by_table.setdefault(logical_table, {})[logical_col] = physical_col
    tables: dict[str, TableMetadata] = {}
    for name, table in member_schema.tables.items():
        physical_name = table_map.get(name, name)
        columns = dict(table.columns)
        for logical_col, physical_col in col_by_table.get(name, {}).items():
            if logical_col not in columns:
                continue
            meta = copy.deepcopy(columns.pop(logical_col))
            columns[physical_col] = replace(meta, name=physical_col)
        tbl = replace(copy.deepcopy(table), name=physical_name, columns=columns)
        tables[physical_name] = tbl
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=member_schema.schema_graph_id,
        effective_structural_hash=member_schema.effective_structural_hash,
        structural_hash=member_schema.structural_hash,
    )


def _finalize_member_sub_intent(sub: RuntimeIntent, member_schema: SchemaGraph) -> RuntimeIntent:
    """Run shared-key expansion and post-compose processing against *member_schema*."""
    intent_process = importlib.import_module("aetherdialect._intent_loop")
    expanded = expand_shared_pk_tables_for_refs(sub, member_schema)
    question_fallback = (sub.natural_language or "").strip()
    processed, post_issues = cast(
        tuple[RuntimeIntent | None, list[Any]],
        intent_process.apply_runtime_post_processing(expanded, member_schema, question_fallback=question_fallback),
    )
    if processed is None:
        raise FederationRuntimeError("federated member sub-intent post-processing incomplete")
    blocking = [issue for issue in post_issues if getattr(issue, "severity", "") == "error"]
    if blocking:
        messages = "; ".join(str(getattr(issue, "message", issue)) for issue in blocking)
        raise FederationRuntimeError(f"federated member sub-intent post-processing failed: {messages}")
    return processed


def _tables_in_federation_space(tables: set[str], space: SpaceContext | None) -> set[str]:
    """Return *tables* restricted to the active federation space allow/deny lists."""
    if space is None:
        return set(tables)
    scoped = set(tables)
    if space.tables:
        scoped &= set(space.tables)
    if space.deny_objects:
        scoped -= set(space.deny_objects)
    return scoped


def _build_source_sub_intent(
    intent: RuntimeIntent,
    source_id: str,
    tables: set[str],
    source_by_table: Mapping[str, str],
    mappings: FederationMappings,
    schema: SchemaGraph,
    manifest: FederationManifest,
    *,
    multi_source: bool = False,
    member_schema: SchemaGraph,
    chosen_specs: Sequence[JoinSpec] | None = None,
    space: SpaceContext | None = None,
) -> SourceStep | None:
    intent = _isolate_sub_intent_decisions(_copy_runtime_intent(intent))
    source_tables = {
        t for t in tables if _table_owned_by_source(t, source_id, manifest, mappings, source_by_table, schema)
    }
    source_tables = _tables_in_federation_space(source_tables, space)
    if not source_tables:
        return None
    cte_steps = _partition_cte_steps_for_source(intent.cte_steps or [], source_id, source_by_table)
    if space is not None:
        allowed_cte_names = _tables_in_federation_space(
            {cte.cte_name for cte in cte_steps if cte.cte_name},
            space,
        )
        cte_steps = [cte for cte in cte_steps if cte.cte_name in allowed_cte_names]
    partitioned_cte_steps = copy.deepcopy(cte_steps)
    source_tables |= {cte.cte_name for cte in cte_steps if cte.cte_name}
    source_tables = _tables_in_federation_space(source_tables, space)
    if not source_tables:
        return None

    def _predicate_local(param: WhereParam | HavingParam) -> bool:
        if isinstance(param, HavingParam):
            refs = collect_referenced_tables([], [], [], [], [param])
        else:
            refs = collect_referenced_tables([], [], [], [param], [])
        if not refs.issubset(source_tables):
            return False
        if chosen_specs and _source_is_left_combine_nullable_side(
            source_id,
            chosen_specs,
        ):
            srcs = sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
            if len(srcs) == 1 and source_id in srcs:
                if isinstance(param, WhereParam) and _join_covered_literal_push_allowed(
                    param, manifest, mappings, source_by_table, schema=schema
                ):
                    return True
                return False
        return True

    local_where, _ = PredicateGroup.partition(intent.where, _predicate_local)
    local_having, _ = PredicateGroup.partition(intent.having, _predicate_local)
    local_preserve_tables = sorted(
        {str(table).strip() for table in (intent.preserve_tables or []) if str(table).strip() in source_tables}
    )
    sub = replace(
        intent,
        tables=sorted(source_tables),
        where=local_where,
        having=local_having,
        preserve_tables=local_preserve_tables,
        cte_steps=cte_steps,
    )
    if multi_source:
        parent_cross_agg = _intent_has_cross_source_aggregate(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        _cross_shape, has_decomposable_partial = _intent_cross_source_aggregate_shape(
            intent, manifest, mappings, source_by_table, schema=schema
        )
        sub = _intent_exprs_local_to_tables(
            sub,
            source_tables,
            multi_source=True,
            residual_fold=parent_cross_agg,
            combine_key_cols=_join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs),
        )
        if has_decomposable_partial:
            sub = _apply_member_partial_aggregation(
                sub,
                intent,
                source_id,
                source_tables,
                manifest,
                chosen_specs=chosen_specs,
            )
        sub = _strip_coordinator_clauses_from_sub_intent(sub)
        if multi_source:
            sub = _strip_member_round_from_sub_intent(sub)
        if chosen_specs:
            sub = replace(sub, order_by_cols=[])
    sub = _partition_registries_for_source(sub, source_id, source_by_table)
    sub = reconcile_tables(sub)
    sub_grain = sub.grain or "row_level"
    if sub_grain not in VALID_GRAINS:
        sub = replace(sub, grain="row_level")
    sub = _rewrite_logical_references(sub, source_id, mappings, schema, manifest)
    if not (sub.select_cols or []):
        local_gb = list(sub.group_by_cols or [])
        if local_gb:
            sub = replace(sub, select_cols=[SelectCol(expr=col) for col in local_gb])
        elif multi_source:
            key_cols = [
                SelectCol(expr=NormalizedExpr.from_column(key))
                for key in _join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs)
                if "." in key and key.split(".", 1)[0] in source_tables
            ]
            if key_cols:
                sub = replace(sub, select_cols=key_cols, grain="row_level")
            elif PredicateGroup.where_leaves(sub.where) or []:
                raise FederationRuntimeError("federated member sub-intent requires at least one select column")
    sub = _finalize_member_sub_intent(sub, _remap_member_schema_logical_tables(member_schema, source_id, mappings))
    if partitioned_cte_steps:
        kept_names = {cte.cte_name for cte in (sub.cte_steps or []) if cte.cte_name}
        merged_ctes = list(sub.cte_steps or [])
        for cte in partitioned_cte_steps:
            if cte.cte_name and cte.cte_name not in kept_names:
                merged_ctes.append(cte)
        if merged_ctes != list(sub.cte_steps or []):
            sub = replace(sub, cte_steps=merged_ctes)
    source_tables = set(sub.tables or []) or source_tables
    keys = _projected_keys_for_step(
        source_id,
        manifest,
        sub,
        parent=intent,
        source_tables=source_tables,
        chosen_specs=chosen_specs,
    )
    return SourceStep(source_id=source_id, sub_intent=sub, projected_keys=keys)


def _looks_aggregated(select_col: SelectCol) -> bool:
    """Return whether *select_col* carries structured aggregate metadata."""
    return select_col.is_aggregated


def _select_col_agg_meta(select_col: SelectCol) -> tuple[str | None, bool]:
    """Return ``(agg_func, has_distinct)`` from structured select-column IR."""
    expr = select_col.expr
    if expr.agg_func:
        return str(expr.agg_func).lower(), False
    for group in expr.add_groups:
        if group.agg_func:
            return str(group.agg_func).lower(), bool(group.distinct)
    for group in expr.sub_groups:
        if group.agg_func:
            return str(group.agg_func).lower(), bool(group.distinct)
    return None, False


def _select_col_round_args(select_col: SelectCol) -> list[Any] | None:
    """Return ``round`` precision args when the select column wraps an aggregate."""
    for group in select_col.expr.add_groups or []:
        if (group.scalar_func or "").strip().lower() == "round":
            return list(group.scalar_func_args or [])
    if (select_col.expr.scalar_func or "").strip().lower() == "round":
        return list(select_col.expr.scalar_func_args or [])
    return None


def _select_col_agg_func(select_col: SelectCol) -> str | None:
    """Return the structured aggregate function name from select-column IR metadata."""
    func, _ = _select_col_agg_meta(select_col)
    return func


def _select_col_agg_label(select_col: SelectCol) -> str:
    """Human-readable aggregate label derived from structured IR (not raw SQL text)."""
    func, has_distinct = _select_col_agg_meta(select_col)
    term = _select_col_term(select_col).strip()
    if func and term:
        distinct_kw = "distinct " if has_distinct else ""
        return f"{func}({distinct_kw}{term})"
    return term or func or "aggregate"


def _is_sql_aggregate_select_col(select_col: SelectCol) -> bool:
    """Return True for SQL aggregate select columns, excluding bare registry refs."""
    if select_col.expr.registry_ref() is not None:
        return False
    return _looks_aggregated(select_col)


def _tables_referenced_by_select_col(
    sc: SelectCol, *, window_registry: Sequence[Any] | None = None, case_registry: Sequence[Any] | None = None
) -> set[str]:
    tables: set[str] = set()
    if _looks_aggregated(sc):
        inner = _aggregate_inner_column(sc)
        if inner and inner != "*":
            if "." in inner:
                tables.add(inner.split(".", 1)[0])
            return tables
    return collect_referenced_tables([sc], [], [], [], [], window_registry=window_registry, case_registry=case_registry)


def _strip_coordinator_clauses_from_sub_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Remove limit, distinct, and distinct_on from per-source sub- intents; coordinator applies them."""
    unchanged = intent.limit is None and intent.distinct_select_index < 0 and not intent.distinct_on
    if unchanged:
        return intent
    return replace(intent, limit=None, distinct_select_index=-1, distinct_on=[])


def _strip_round_from_mulgroup(group: MulGroup) -> MulGroup:
    """Drop a ``round`` scalar wrapper from one aggregate group for member decomposition."""
    if (group.scalar_func or "").strip().lower() != "round":
        return group
    return replace(group, scalar_func=None, scalar_func_args=[], sarg_param_keys=[])


def _strip_round_from_expr(expr: NormalizedExpr) -> NormalizedExpr:
    """Remove ``round`` scalar wrappers from an expression tree."""
    new_groups = [_strip_round_from_mulgroup(g) for g in (expr.add_groups or [])]
    scalar_func = expr.scalar_func
    scalar_func_args = expr.scalar_func_args
    sarg_param_keys = expr.sarg_param_keys
    changed = new_groups != list(expr.add_groups or [])
    if (scalar_func or "").strip().lower() == "round":
        scalar_func = None
        scalar_func_args = []
        sarg_param_keys = []
        changed = True
    if not changed:
        return expr
    return replace(
        expr,
        add_groups=new_groups,
        scalar_func=scalar_func,
        scalar_func_args=scalar_func_args,
        sarg_param_keys=sarg_param_keys,
    )


def _strip_member_round_from_sub_intent(intent: RuntimeIntent) -> RuntimeIntent:
    """Keep raw member aggregates; coordinator residual applies ``round`` once."""
    new_select = [replace(sc, expr=_strip_round_from_expr(sc.expr)) for sc in (intent.select_cols or [])]
    having_leaves = list(PredicateGroup.having_leaves(intent.having) or [])
    new_having_leaves: list[HavingParam] = []
    for hp in having_leaves:
        stripped = _strip_round_from_expr(hp.left_expr)
        new_having_leaves.append(replace(hp, left_expr=stripped) if stripped is not hp.left_expr else hp)
    having_group = PredicateGroup.from_list(new_having_leaves)
    if new_select == list(intent.select_cols or []) and having_group == intent.having:
        return intent
    return replace(intent, select_cols=new_select, having=having_group)


def _expr_uses_round(expr: NormalizedExpr) -> bool:
    if (expr.scalar_func or "").strip().lower() == "round":
        return True
    return any((g.scalar_func or "").strip().lower() == "round" for g in (expr.add_groups or []))


def _intent_uses_round(intent: RuntimeIntent) -> bool:
    for sc in intent.select_cols or []:
        if _expr_uses_round(sc.expr):
            return True
    for hp in PredicateGroup.having_leaves(intent.having) or []:
        if _expr_uses_round(hp.left_expr):
            return True
    return False


def _rounded_logical_columns(intent: RuntimeIntent) -> list[str]:
    columns: list[str] = []
    for sc in intent.select_cols or []:
        if not _expr_uses_round(sc.expr):
            continue
        for group in sc.expr.add_groups or []:
            for term in group.multiply or []:
                ref = (getattr(term, "column_ref", None) or "").strip()
                if ref and "." in ref:
                    columns.append(ref.rsplit(".", 1)[-1])
    return list(dict.fromkeys(columns))


def emit_federation_rounding_mode_mixed_diagnostics(
    manifest: FederationManifest,
    plan: FederatedPlan,
    intent: RuntimeIntent,
    *,
    schema: SchemaGraph | None = None,
) -> None:
    """Emit ``ROUNDING_MODE_MIXED`` when federated members disagree on rounding tie-breaking."""
    _ = plan, schema
    if not _intent_uses_round(intent):
        return
    modes = {
        str(src.engine).strip().lower(): DialectRegistry.engine_rounding_mode(str(src.engine))
        for src in manifest.sources
        if str(src.engine).strip()
    }
    if len(set(modes.values())) <= 1:
        return
    for logical_column in _rounded_logical_columns(intent):
        notify(
            (
                f"federation rounding mode differs across members for logical column "
                f"{logical_column!r}; coordinator applies half-up rounding"
            ),
            stage="federation",
            code=DIAGNOSTIC_CODE_ROUNDING_MODE_MIXED,
            level="warning",
            details=(("logical_column", logical_column),),
        )


def _join_key_columns_for_source(
    source_id: str, manifest: FederationManifest, *, chosen_specs: Sequence[JoinSpec] | None = None
) -> list[str]:
    """Return qualified join-key columns declared on *source_id* for cross-source joins."""
    cols: list[str] = []
    for join in manifest.cross_source_joins:
        spec = _cross_source_join_to_spec(join, manifest)
        if chosen_specs is not None and spec not in chosen_specs:
            continue
        for qualified in (join.left, join.right):
            tbl, col = split_qualified_column(qualified, manifest=manifest)
            if manifest.table_namespace.get(tbl, "") != source_id:
                continue
            ref = f"{tbl}.{col}"
            if ref not in cols:
                cols.append(ref)
    return cols


def _aggregate_inner_column(select_col: SelectCol) -> str:
    """Return the aggregated column/star target from structured select- column IR."""
    if not select_col.is_aggregated:
        return ""
    col = select_col.expr.primary_column
    return col if col else ""


def _aggregate_columns_for_source(parent: RuntimeIntent, source_tables: set[str]) -> list[str]:
    cols: list[str] = []
    for sc in parent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        col_ref = _aggregate_inner_column(sc)
        if not col_ref or "." not in col_ref:
            continue
        tbl = col_ref.split(".", 1)[0]
        if tbl in source_tables:
            cols.append(col_ref)
    return cols


def _select_col_term(sc: SelectCol) -> str:
    expr = sc.expr
    term = (expr.primary_term or "").strip()
    if term:
        return term
    col = (expr.primary_column or "").strip()
    if col:
        return col
    return str(expr)


def remap_distinct_select_index(old_cols: Sequence[SelectCol], new_cols: Sequence[SelectCol], old_index: int) -> int:
    """Translate ``distinct_select_index`` after select-col append/reorder."""
    if old_index < 0:
        return old_index
    if old_index >= len(old_cols):
        return -1
    target = old_cols[old_index]
    for idx, sc in enumerate(new_cols):
        if sc is target:
            return idx
    return -1


def apply_projected_keys_to_intent(intent: RuntimeIntent, projected_keys: tuple[str, ...]) -> RuntimeIntent:
    """Ensure *intent* projects every coordinator-required column."""
    if not projected_keys:
        return intent
    old_cols = list(intent.select_cols or [])
    existing = {_select_col_term(sc) for sc in old_cols}
    new_cols = [SelectCol(expr=NormalizedExpr.from_column(ref)) for ref in projected_keys if ref not in existing]
    if not new_cols:
        return intent
    merged = old_cols + new_cols
    distinct_idx = remap_distinct_select_index(old_cols, merged, intent.distinct_select_index)
    return replace(intent, select_cols=merged, distinct_select_index=distinct_idx)


def _normalized_expr_identity(expr: NormalizedExpr) -> str:
    return (expr.column_ref or expr.primary_column or expr.primary_term or "").strip()


def _apply_member_partial_aggregation(
    sub: RuntimeIntent,
    parent: RuntimeIntent,
    source_id: str,
    source_tables: set[str],
    manifest: FederationManifest,
    *,
    chosen_specs: Sequence[JoinSpec] | None = None,
) -> RuntimeIntent:
    """Pre-aggregate decomposable member aggregates before the coordinator folds them."""
    group_exprs: list[NormalizedExpr] = []
    seen_group: set[str] = set()
    for key in _join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs):
        expr = NormalizedExpr.from_column(key)
        ident = _normalized_expr_identity(expr)
        if ident and ident not in seen_group:
            seen_group.add(ident)
            group_exprs.append(expr)
    for col in parent.group_by_cols or []:
        refs = collect_referenced_tables([], [], [col], [], [])
        if refs and refs.issubset(source_tables):
            ident = _normalized_expr_identity(col)
            if ident and ident not in seen_group:
                seen_group.add(ident)
                group_exprs.append(col)

    select_cols: list[SelectCol] = []
    seen_select: set[str] = set()
    for expr in group_exprs:
        ident = _normalized_expr_identity(expr)
        if ident and ident not in seen_select:
            seen_select.add(ident)
            select_cols.append(SelectCol(expr=expr))

    for sc in parent.select_cols or []:
        if not _looks_aggregated(sc):
            continue
        inner = _aggregate_inner_column(sc)
        if not inner or inner == "*":
            continue
        if "." in inner and inner.split(".", 1)[0] not in source_tables:
            continue
        func, has_distinct = _select_col_agg_meta(sc)
        if has_distinct or func not in FEDERATION_DECOMPOSABLE_CROSS_SOURCE_AGGS:
            continue
        if func == "avg":
            for partial_func in ("sum", "count"):
                partial_expr = NormalizedExpr.from_agg(partial_func, inner)
                ident = f"{partial_func}:{inner}"
                if ident not in seen_select:
                    seen_select.add(ident)
                    select_cols.append(SelectCol(expr=partial_expr))
            continue
        partial_expr = NormalizedExpr.from_agg(func, inner)
        ident = f"{func}:{inner}"
        if ident not in seen_select:
            seen_select.add(ident)
            select_cols.append(SelectCol(expr=partial_expr))

    if not any(_looks_aggregated(sc) for sc in select_cols):
        return sub
    grain = "grouped" if group_exprs else (parent.grain or "row_level")
    return replace(
        sub,
        tables=sorted(source_tables),
        grain=grain,
        select_cols=select_cols,
        group_by_cols=group_exprs,
        order_by_cols=[],
    )


def _intent_exprs_local_to_tables(
    intent: RuntimeIntent,
    source_tables: set[str],
    *,
    multi_source: bool = False,
    residual_fold: bool = False,
    combine_key_cols: Sequence[str] | None = None,
) -> RuntimeIntent:
    """Drop parent aggregates/projections that reference tables outside *source_tables*."""
    allowed = set(source_tables)

    def _expr_refs_local(
        select_cols: Sequence[SelectCol],
        order_by_cols: Sequence[OrderByCol],
        group_by_cols: Sequence[NormalizedExpr],
        where_params: Sequence[WhereParam],
        having_param: Sequence[HavingParam],
    ) -> bool:
        refs = collect_referenced_tables(
            list(select_cols),
            list(order_by_cols),
            list(group_by_cols),
            list(where_params),
            list(having_param),
            window_registry=intent.window_registry,
            case_registry=intent.case_registry,
            include_unreferenced_registries=False,
        )
        return bool(refs) and refs.issubset(allowed)

    select_cols = [
        sc
        for sc in (intent.select_cols or [])
        if not _is_sql_aggregate_select_col(sc) and _expr_refs_local([sc], [], [], [], [])
    ]
    group_by_cols = [col for col in (intent.group_by_cols or []) if _expr_refs_local([], [], [col], [], [])]
    order_by_cols = [col for col in (intent.order_by_cols or []) if _expr_refs_local([], [col], [], [], [])]
    having_leaves = [
        hp for hp in (PredicateGroup.having_leaves(intent.having) or []) if _expr_refs_local([], [], [], [], [hp])
    ]
    having_group = PredicateGroup.from_list(having_leaves)
    parent_refs = collect_referenced_tables(
        intent.select_cols,
        intent.order_by_cols,
        intent.group_by_cols,
        PredicateGroup.where_leaves(intent.where),
        PredicateGroup.having_leaves(intent.having),
    )
    parent_had_agg = any(_is_sql_aggregate_select_col(sc) for sc in (intent.select_cols or []))
    fold_to_residual = multi_source and (residual_fold or (intent.grain or "") == "scalar" or parent_had_agg)

    def _local_combine_key_cols() -> list[SelectCol]:
        local_keys = [key for key in (combine_key_cols or ()) if "." in key and key.split(".", 1)[0] in allowed]
        return [SelectCol(expr=NormalizedExpr.from_column(key)) for key in local_keys]

    if fold_to_residual:
        projected = list(select_cols)
        if not projected:
            projected = _local_combine_key_cols()
        return replace(
            intent,
            grain="row_level",
            select_cols=projected,
            group_by_cols=[] if residual_fold else group_by_cols,
            order_by_cols=order_by_cols,
            having=having_group,
            limit=None,
            distinct_select_index=-1,
        )
    if parent_refs - allowed and not select_cols and not group_by_cols:
        key_cols = _local_combine_key_cols()
        if key_cols:
            return replace(
                intent,
                grain="row_level",
                select_cols=key_cols,
                group_by_cols=[],
                order_by_cols=[],
                having=having_group,
                limit=None,
            )
        raise FederationRuntimeError("federated member sub-intent requires at least one select column")
    if multi_source and not select_cols:
        key_cols = _local_combine_key_cols()
        if key_cols:
            return replace(
                intent,
                grain="row_level",
                select_cols=key_cols,
                group_by_cols=group_by_cols,
                order_by_cols=order_by_cols,
                having=having_group,
            )
        if group_by_cols:
            return replace(
                intent,
                select_cols=[SelectCol(expr=col) for col in group_by_cols],
                group_by_cols=group_by_cols,
                order_by_cols=order_by_cols,
                having=having_group,
            )
    return replace(
        intent, select_cols=select_cols, group_by_cols=group_by_cols, order_by_cols=order_by_cols, having=having_group
    )


def _member_logical_column_map(
    source_id: str,
    mappings: FederationMappings,
    schema: SchemaGraph,
    base_map: Mapping[str, str] | None = None,
    manifest: FederationManifest | None = None,
) -> dict[str, str]:
    """Build logical-to-physical column aliases for one federation member."""
    column_map = dict(base_map or {})
    for col_map in mappings.logical_columns:
        for member in col_map.members:
            tbl, col = split_qualified_column(member, manifest=manifest)
            if schema.tables.get(tbl) and schema.tables[tbl].source_id == source_id:
                column_map[col_map.logical] = col
    for lt in mappings.logical_tables:
        for table_member in lt.members:
            if table_member.source != source_id:
                continue
            for logical, physical in table_member.columns.items():
                column_map[logical] = physical
    return column_map


def _member_logical_table_map(source_id: str, mappings: FederationMappings) -> dict[str, str]:
    """Map logical table names to the physical table name used on *source_id*."""
    out: dict[str, str] = {}
    for lt in mappings.logical_tables:
        logical = str(lt.logical or "").strip()
        if not logical:
            continue
        for table_member in lt.members:
            if table_member.source != source_id:
                continue
            physical = str(table_member.table or "").strip()
            if physical and physical != logical:
                out[logical] = physical
            break
    return out


def _member_logical_column_rewrites(source_id: str, mappings: FederationMappings) -> list[tuple[str, str, str, str]]:
    """Return ``(logical_table, logical_col, physical_table, physical_col)`` alias tuples."""
    out: list[tuple[str, str, str, str]] = []
    for lt in mappings.logical_tables:
        logical_table = str(lt.logical or "").strip()
        if not logical_table:
            continue
        for table_member in lt.members:
            if table_member.source != source_id:
                continue
            physical_table = str(table_member.table or logical_table).strip()
            for logical_col, physical_col in (table_member.columns or {}).items():
                logical_name = str(logical_col or "").strip()
                physical_name = str(physical_col or "").strip()
                if logical_name and physical_name and logical_name != physical_name:
                    out.append((logical_table, logical_name, physical_table, physical_name))
            break
    return out


def _rewrite_logical_references(
    intent: RuntimeIntent,
    source_id: str,
    mappings: FederationMappings,
    schema: SchemaGraph,
    manifest: FederationManifest | None = None,
) -> RuntimeIntent:
    """Rewrite logical column aliases and logical→physical table names for one member."""
    column_map = _member_logical_column_map(source_id, mappings, schema, intent.column_map, manifest)
    table_map = _member_logical_table_map(source_id, mappings)
    cte_steps_out: list[RuntimeCteStep] = []
    cte_changed = False
    for cte in intent.cte_steps or []:
        merged = dict(cte.column_map or {})
        for logical, physical in column_map.items():
            merged[logical] = physical
        rewritten_cte = cte
        if merged != dict(cte.column_map or {}):
            rewritten_cte = replace(rewritten_cte, column_map=merged)
            cte_changed = True
        if table_map:
            cte_tables = [table_map.get(t, t) for t in (rewritten_cte.tables or [])]
            if cte_tables != list(rewritten_cte.tables or []):
                rewritten_cte = replace(rewritten_cte, tables=cte_tables)
                cte_changed = True
        cte_steps_out.append(rewritten_cte)
    out = intent
    if column_map != dict(intent.column_map or {}) or cte_changed:
        out = replace(
            out,
            column_map=column_map,
            cte_steps=cte_steps_out if cte_changed else intent.cte_steps,
        )
    if not table_map and not _member_logical_column_rewrites(source_id, mappings):
        return out
    for logical_table, logical_col, physical_table, physical_col in _member_logical_column_rewrites(
        source_id, mappings
    ):
        out = apply_column_replacer_to_intent(
            out,
            build_column_term_replacer(logical_table, logical_col, physical_table, physical_col),
        )
    tables = [table_map.get(t, t) for t in (out.tables or [])]
    preserve = [table_map.get(t, t) for t in (out.preserve_tables or [])]
    out = replace(out, tables=tables, preserve_tables=preserve)
    for logical, physical in table_map.items():
        out = apply_column_replacer_to_intent(out, build_table_term_replacer(logical, physical))
    for _logical_table, logical_col, physical_table, physical_col in _member_logical_column_rewrites(
        source_id, mappings
    ):
        out = apply_column_replacer_to_intent(
            out,
            build_column_term_replacer(physical_table, logical_col, physical_table, physical_col),
        )
    return out


def _projected_keys_for_step(
    source_id: str,
    manifest: FederationManifest,
    sub_intent: RuntimeIntent,
    *,
    parent: RuntimeIntent | None = None,
    source_tables: set[str] | None = None,
    chosen_specs: Sequence[JoinSpec] | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    keys.extend(_join_key_columns_for_source(source_id, manifest, chosen_specs=chosen_specs))
    if parent is not None and source_tables:
        keys.extend(_aggregate_columns_for_source(parent, source_tables))
    for sc in sub_intent.select_cols or []:
        term = _select_col_term(sc)
        if term:
            keys.append(term)
    return tuple(dict.fromkeys(keys))


def _residual_spec_for_intent(
    intent: RuntimeIntent,
    source_by_table: Mapping[str, str],
    manifest: FederationManifest,
    mappings: FederationMappings,
    *,
    schema: SchemaGraph | None = None,
    scope_tables: Iterable[str] | None = None,
    combine: tuple[JoinSpec, ...] | None = None,
) -> ResidualSpec | None:
    intent_tables = set(scope_tables or intent.tables or [])
    if not intent_tables:
        return None
    intent_sources = intent_table_sources(intent_tables, manifest, mappings, source_by_table, schema=schema)
    if len(intent_sources) <= 1:
        return None
    registry_kw = intent_registry_kw(intent)
    source_kw: dict[str, Any] = {
        "source_by_table": source_by_table,
        "manifest": manifest,
        "mappings": mappings,
        "schema": schema,
        **registry_kw,
    }
    has_cross_agg = _intent_has_cross_source_aggregate(intent, manifest, mappings, source_by_table, schema=schema)

    def _cross_source(**clause: Any) -> bool:
        return _clause_spans_multiple_sources(**source_kw, **clause)

    if has_cross_agg:
        select_cols = tuple(
            sc for sc in (intent.select_cols or []) if _looks_aggregated(sc) or _cross_source(select_cols=[sc])
        )
        parent_cross_select_agg = any(_looks_aggregated(sc) for sc in (intent.select_cols or []))

        def _group_by_needed_post_join(col: NormalizedExpr) -> bool:
            if _cross_source(group_by_cols=[col]):
                return True
            col_sources = _clause_referenced_sources(**source_kw, group_by_cols=[col])
            if not col_sources:
                return False
            return bool(intent_sources - col_sources)

        if parent_cross_select_agg and (intent.grain or "") != "grouped":
            group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _cross_source(group_by_cols=[col]))
        else:
            group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _group_by_needed_post_join(col))
    else:
        select_cols = tuple(sc for sc in (intent.select_cols or []) if _cross_source(select_cols=[sc]))
        group_by_cols = tuple(col for col in (intent.group_by_cols or []) if _cross_source(group_by_cols=[col]))
    order_by_cols = (
        tuple(_explicit_residual_order_col(col) for col in (intent.order_by_cols or []))
        if combine
        else tuple(
            _explicit_residual_order_col(col)
            for col in (intent.order_by_cols or [])
            if _cross_source(order_by_cols=[col])
        )
    )

    def _predicate_spans_sources(param: WhereParam | HavingParam) -> bool:
        if isinstance(param, HavingParam):
            refs = collect_referenced_tables([], [], [], [], [param], **registry_kw)
        else:
            refs = collect_referenced_tables([], [], [], [param], [], **registry_kw)
        srcs = sources_for_refs(refs, manifest, mappings, source_by_table, schema=schema)
        if len(srcs) > 1:
            return True
        if combine and len(srcs) == 1:
            if isinstance(param, WhereParam) and _join_covered_literal_push_allowed(
                param,
                manifest,
                mappings,
                source_by_table,
                schema=schema,
                registry_kw=registry_kw,
            ):
                return False
            return _source_is_left_combine_nullable_side(next(iter(srcs)), combine)
        return False

    _, residual_where = PredicateGroup.partition(intent.where, lambda param: not _predicate_spans_sources(param))
    _, residual_having = PredicateGroup.partition(intent.having, lambda param: not _predicate_spans_sources(param))
    distinct_on: tuple[NormalizedExpr, ...] = ()
    if intent.distinct_on and distinct_on_spans_sources(intent, manifest, mappings, source_by_table, schema=schema):
        distinct_on = tuple(intent.distinct_on)
    distinct_select_index = intent.distinct_select_index if intent.distinct_select_index >= 0 else -1
    limit = intent.limit

    def _registry_entry_cross_source(entry: Any, *, field: str) -> bool:
        if field == "window":
            return _window_requires_coordinator(
                entry, source_by_table=source_by_table, manifest=manifest, schema=schema, combine=combine
            )
        return _clause_spans_multiple_sources(
            source_by_table=source_by_table, manifest=manifest, mappings=mappings, schema=schema, case_registry=[entry]
        )

    window_registry = tuple(
        entry for entry in (intent.window_registry or []) if _registry_entry_cross_source(entry, field="window")
    )
    case_registry = tuple(
        entry for entry in (intent.case_registry or []) if _registry_entry_cross_source(entry, field="case")
    )
    if window_registry:
        window_ids = {
            str(getattr(entry, "registry_id", "") or "")
            for entry in window_registry
            if getattr(entry, "registry_id", "")
        }
        promoted: list[SelectCol] = list(select_cols)
        seen_refs: set[str] = set()
        for sc in select_cols:
            ref = sc.expr.registry_ref()
            if ref:
                seen_refs.add(ref)
        for sc in intent.select_cols or []:
            ref = sc.expr.registry_ref()
            if ref and ref in window_ids and ref not in seen_refs:
                promoted.append(sc)
                seen_refs.add(ref)
        select_cols = tuple(promoted)
    if not select_cols and (residual_where or window_registry):
        select_cols = tuple(intent.select_cols or [])
    if not any(
        (
            select_cols,
            group_by_cols,
            order_by_cols,
            residual_where,
            residual_having,
            distinct_on,
            distinct_select_index >= 0,
            limit is not None,
            window_registry,
            case_registry,
        )
    ):
        return None
    return ResidualSpec(
        select_cols=select_cols,
        group_by_cols=group_by_cols,
        order_by_cols=order_by_cols,
        where=residual_where,
        having=residual_having,
        distinct_on=distinct_on,
        distinct_select_index=distinct_select_index,
        limit=limit,
        limit_param_key=(intent.limit_param_key or "").strip(),
        window_registry=window_registry,
        case_registry=case_registry,
    )


def _cross_source_join_to_spec(
    join: FederationCrossSourceJoin, manifest: FederationManifest, *, schema: SchemaGraph | None = None
) -> JoinSpec:
    if schema is not None:
        left_ref = resolve_federation_qualified_ref(join.left, manifest=manifest, schema=schema)
        right_ref = resolve_federation_qualified_ref(join.right, manifest=manifest, schema=schema)
        return JoinSpec(
            left_source=left_ref.source_id,
            right_source=right_ref.source_id,
            left_key=left_ref.column,
            right_key=right_ref.column,
            logical_key=join.logical_key,
            kind=join.kind,
        )
    left_tbl, left_col = split_qualified_column(join.left, manifest=manifest)
    right_tbl, right_col = split_qualified_column(join.right, manifest=manifest)
    left_source = manifest.table_namespace.get(left_tbl, "")
    right_source = manifest.table_namespace.get(right_tbl, "")
    return JoinSpec(
        left_source=left_source,
        right_source=right_source,
        left_key=left_col,
        right_key=right_col,
        logical_key=join.logical_key,
        kind=join.kind,
    )


def _opaque_join_choice_scope_key(scope_index: int) -> str:
    return f"jc{scope_index}"


def _cross_source_join_path_signature(join: FederationCrossSourceJoin) -> list[str]:
    return [f"{join.left}->{join.right}"]


def _cross_source_join_table_pair(
    join: FederationCrossSourceJoin, manifest: FederationManifest | None = None
) -> tuple[str, str]:
    left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
    right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
    return (min(left_tbl, right_tbl), max(left_tbl, right_tbl))


def _cross_source_join_in_scope(
    join: FederationCrossSourceJoin, scope_tables: frozenset[str] | None, manifest: FederationManifest | None = None
) -> bool:
    if not scope_tables:
        return True
    left_tbl, _ = split_qualified_column(join.left, manifest=manifest)
    right_tbl, _ = split_qualified_column(join.right, manifest=manifest)
    return left_tbl in scope_tables and right_tbl in scope_tables


def _is_cross_source_join_key_nonsensitive(
    join: FederationCrossSourceJoin, schema: SchemaGraph, manifest: FederationManifest | None = None
) -> bool:
    for qualified in (join.left, join.right):
        tbl, col = split_qualified_column(qualified, manifest=manifest)
        table = schema.tables.get(tbl)
        if table is None:
            continue
        cm = table.columns.get(col)
        if cm is not None and cm.sensitivity != SensitivityClassification.NONE:
            return False
    return True


def _eligible_cross_source_joins(
    manifest: FederationManifest, sources: frozenset[str], schema: SchemaGraph | None = None
) -> list[tuple[FederationCrossSourceJoin, JoinSpec]]:
    eligible: list[tuple[FederationCrossSourceJoin, JoinSpec]] = []
    for join in manifest.cross_source_joins:
        spec = _cross_source_join_to_spec(join, manifest, schema=schema)
        if spec.left_source not in sources or spec.right_source not in sources:
            continue
        if schema is not None and not _is_cross_source_join_key_nonsensitive(join, schema, manifest):
            continue
        eligible.append((join, spec))
    return eligible


def _deterministic_cross_source_choice(entries: Sequence[tuple[FederationCrossSourceJoin, JoinSpec, str]]) -> str:
    return sorted(entries, key=lambda row: row[2])[0][2]


def _join_specs_for_sources(
    manifest: FederationManifest,
    mappings: FederationMappings,
    sources: frozenset[str],
    *,
    schema: SchemaGraph | None = None,
    join_choices: Mapping[str, str] | None = None,
    scope_tables: frozenset[str] | None = None,
) -> tuple[JoinSpec, ...] | None:
    _ = mappings
    if len(sources) < 2:
        return None
    eligible = [
        row
        for row in _eligible_cross_source_joins(manifest, sources, schema)
        if _cross_source_join_in_scope(row[0], scope_tables, manifest)
    ]
    by_pair: dict[tuple[str, str], list[tuple[FederationCrossSourceJoin, JoinSpec, str]]] = {}
    for join, spec in sorted(eligible, key=lambda row: (row[0].logical_key, row[0].left, row[0].right)):
        pair = _cross_source_join_table_pair(join, manifest)
        candidate_id = f"J{len(by_pair.get(pair, [])):02d}"
        by_pair.setdefault(pair, []).append((join, spec, candidate_id))
    if not by_pair:
        return None
    choices = dict(join_choices or {})
    chosen: list[JoinSpec] = []
    for scope_index, pair in enumerate(sorted(by_pair.keys())):
        entries = by_pair[pair]
        scope = _opaque_join_choice_scope_key(scope_index)
        if len(entries) == 1:
            chosen.append(entries[0][1])
            continue
        chosen_id: str | None = choices.get(scope)
        if chosen_id is None:
            chosen_id = _deterministic_cross_source_choice(entries)
        picked: JoinSpec | None = None
        for _join, spec, cid in entries:
            if cid == chosen_id:
                picked = spec
                break
        if picked is None:
            fallback_cid = _deterministic_cross_source_choice(entries)
            for _join, spec, cid in entries:
                if cid == fallback_cid:
                    picked = spec
                    break
        if picked is None:
            picked = entries[0][1]
        chosen.append(picked)
    if chosen and not _join_specs_connect_sources(sources, chosen):
        return None
    return tuple(chosen)


def resolve_federated_combine(
    q_norm: str,
    plan: FederatedPlan,
    manifest: FederationManifest,
    composite_schema: SchemaGraph,
    *,
    preset_choices: Mapping[str, str] | None = None,
    temporal_bind: AnchoredTemporalBind | None = None,
) -> FederatedPlan:
    """Disambiguate declared cross-source joins and refresh coordinator projections."""
    if plan.ineligible_reason or not plan.steps:
        return plan
    if temporal_bind is None and plan.steps:
        parent_intent = plan.steps[0].sub_intent
        temporal_bind = resolve_anchored_temporal_bind(parent_intent)
    sources = frozenset(step.source_id for step in plan.steps)
    eligible = [
        row
        for row in _eligible_cross_source_joins(manifest, sources, composite_schema)
        if _cross_source_join_in_scope(
            row[0], frozenset(table for step in plan.steps for table in (step.sub_intent.tables or [])), manifest
        )
    ]
    by_pair: dict[tuple[str, str], list[tuple[FederationCrossSourceJoin, JoinSpec, str]]] = {}
    for join, spec in sorted(eligible, key=lambda row: (row[0].logical_key, row[0].left, row[0].right)):
        pair = _cross_source_join_table_pair(join, manifest)
        candidate_id = f"J{len(by_pair.get(pair, [])):02d}"
        by_pair.setdefault(pair, []).append((join, spec, candidate_id))
    if not by_pair:
        return replace(plan, combine=None)
    join_choices = dict(preset_choices or {})
    llm_scopes: list[dict[str, Any]] = []
    for scope_index, pair in enumerate(sorted(by_pair.keys())):
        entries = by_pair[pair]
        scope = _opaque_join_choice_scope_key(scope_index)
        if len(entries) == 1:
            join_choices[scope] = entries[0][2]
            continue
        if scope in join_choices:
            continue
        tables = sorted(
            {split_qualified_column(join.left, manifest=manifest)[0] for join, _spec, _cid in entries}
            | {split_qualified_column(join.right, manifest=manifest)[0] for join, _spec, _cid in entries}
        )
        llm_scopes.append(
            {
                "scope": scope,
                "tables": tables,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "join_path_signature": _cross_source_join_path_signature(join),
                    }
                    for join, _spec, candidate_id in entries
                ],
            }
        )
    if llm_scopes:
        resolved = get_join_choice_from_llm(
            q_norm, "SELECT 1", llm_scopes=llm_scopes, preset_choices=join_choices, schema=composite_schema
        )
        join_choices.update(resolved)
    combine = _join_specs_for_sources(
        manifest,
        FederationMappings(version=FEDERATION_MAPPINGS_VERSION),
        sources,
        schema=composite_schema,
        join_choices=join_choices,
        scope_tables=frozenset(table for step in plan.steps for table in (step.sub_intent.tables or [])),
    )
    refreshed_steps: list[SourceStep] = []
    for step in plan.steps:
        join_keys = _join_key_columns_for_source(step.source_id, manifest, chosen_specs=combine)
        output_keys = [key for key in step.projected_keys if key not in join_keys]
        keys = tuple(dict.fromkeys([*join_keys, *output_keys]))
        refreshed_steps.append(replace(step, projected_keys=keys))
    return replace(plan, combine=combine, steps=tuple(refreshed_steps))
