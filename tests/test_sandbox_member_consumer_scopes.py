"""Hermetic four-consumer RBAC matrix — EngineContext allow_objects egress scopes."""

from __future__ import annotations

import pytest

from aetherdialect._constants import SESSION_KIND_RESULT
from aetherdialect._constants_runtime import SANDBOX_MEMBER_SPACE_TABLES
from aetherdialect._contracts_base import Diagnostic, EngineContext, KnowledgeScope
from aetherdialect._contracts_core import AuditEvent, SessionStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from tests.support.egress_assert import (
    assert_no_forbidden_identifiers,
    forbidden_identifiers_for_scope,
)

_ALL_SANDBOX_TABLES = frozenset().union(*SANDBOX_MEMBER_SPACE_TABLES.values())


def _master_schema_graph() -> SchemaGraph:
    tables = {
        name: TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
        for name in sorted(_ALL_SANDBOX_TABLES)
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sandbox_member_consumer_scopes",
        effective_structural_hash="sandbox_member_consumer_scopes",
    )


def _scope_for_member(member: str) -> tuple[EngineContext, KnowledgeScope]:
    allow = SANDBOX_MEMBER_SPACE_TABLES[member]
    graph = _master_schema_graph()
    ctx = EngineContext(allow_objects=allow)
    scope = KnowledgeScope.from_engine_context(graph, ctx.allow_objects)
    return ctx, scope


def _clean_observables(allowed_table: str) -> tuple[SessionStep, AuditEvent]:
    step = SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        sql=f"SELECT id FROM {allowed_table}",
        answer=f"Returned rows from {allowed_table}.",
        diagnostics=(
            Diagnostic(
                stage="execute",
                level="info",
                code="ROWS",
                message=f"Queried {allowed_table}.",
                subject=f"{allowed_table}.id",
                details=(("table", allowed_table), ("column", f"{allowed_table}.id")),
            ),
        ),
    )
    audit = AuditEvent(
        event_type="turn_complete",
        timestamp_iso="2026-01-01T00:00:00Z",
        question=f"How many rows in {allowed_table}?",
        schema_hash="sandbox_member_consumer_scopes",
        provider="sandbox",
        details=(("table", allowed_table), ("outcome", "ok")),
    )
    return step, audit


def _leaky_session_step(forbidden_table: str) -> SessionStep:
    return SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        answer=f"Unable to join {forbidden_table} to the requested data.",
        diagnostics=(
            Diagnostic(
                stage="validate",
                level="warning",
                code="JOIN_PATH_MISSING",
                message="No join path found.",
                subject=f"{forbidden_table}.id",
                details=(("table", forbidden_table),),
            ),
        ),
    )


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_member_consumer_engine_context_matches_bundled_allow_set(member: str) -> None:
    """Each consumer EngineContext must mirror the bundled member table frozenset."""
    ctx, _scope = _scope_for_member(member)
    assert ctx.allow_objects == SANDBOX_MEMBER_SPACE_TABLES[member]


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_member_consumer_egress_allows_in_scope_identifiers(member: str) -> None:
    """In-scope table identifiers pass the hermetic egress scan for each consumer."""
    _ctx, scope = _scope_for_member(member)
    forbidden = forbidden_identifiers_for_scope(_master_schema_graph(), scope)
    allowed_table = sorted(SANDBOX_MEMBER_SPACE_TABLES[member])[0]
    step, audit = _clean_observables(allowed_table)

    assert_no_forbidden_identifiers(step, forbidden)
    assert_no_forbidden_identifiers(audit, forbidden)


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_member_consumer_egress_forbids_out_of_scope_tables(member: str) -> None:
    """Out-of-scope table names are forbidden in consumer egress for each member scope."""
    _ctx, scope = _scope_for_member(member)
    forbidden = forbidden_identifiers_for_scope(_master_schema_graph(), scope)
    leaked_table = sorted(_ALL_SANDBOX_TABLES - SANDBOX_MEMBER_SPACE_TABLES[member])[0]
    leaked = _leaky_session_step(leaked_table)

    with pytest.raises(AssertionError, match="forbidden identifier"):
        assert_no_forbidden_identifiers(leaked, forbidden)


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize("member", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_member_consumer_reflective_scope_excludes_foreign_space_tables(member: str) -> None:
    """KnowledgeScope derived from allow_objects excludes every other member's tables."""
    graph = _master_schema_graph()
    _ctx, scope = _scope_for_member(member)
    foreign = next(name for name in sorted(SANDBOX_MEMBER_SPACE_TABLES) if name != member)
    leaked = sorted(SANDBOX_MEMBER_SPACE_TABLES[foreign] - SANDBOX_MEMBER_SPACE_TABLES[member])[0]
    assert leaked not in scope.tables
    assert leaked in forbidden_identifiers_for_scope(graph, scope)
