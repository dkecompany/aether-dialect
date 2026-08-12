"""Hermetic sandbox egress matrix — scope-bound observables without corpus pack."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED, SESSION_KIND_RESULT
from aetherdialect._constants_runtime import SANDBOX_MEMBER_SPACE_TABLES
from aetherdialect._contracts_base import Diagnostic, KnowledgeScope, SpaceContext
from aetherdialect._contracts_core import AuditEvent, SessionStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._utils import (
    sanitize_audit_details_for_egress,
    sanitize_session_step_for_egress,
)
from tests.support.egress_assert import (
    assert_no_forbidden_identifiers,
    forbidden_identifiers_for_scope,
)

_ALL_SANDBOX_TABLES = frozenset().union(*SANDBOX_MEMBER_SPACE_TABLES.values())

_CONSUMER_PROFILES = (
    "owner_writer",
    "consumer_reader",
    *(f"member:{space}" for space in sorted(SANDBOX_MEMBER_SPACE_TABLES)),
)

_SPACE_PROFILES = (
    ("master", None),
    *((name, name) for name in sorted(SANDBOX_MEMBER_SPACE_TABLES)),
)


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
        schema_graph_id="sandbox_egress_matrix",
        effective_structural_hash="sandbox_egress_matrix",
    )


def _visible_tables_for_profile(profile: str) -> frozenset[str] | None:
    if profile in {"owner_writer", "consumer_reader"}:
        return None
    if profile.startswith("member:"):
        space = profile.split(":", 1)[1]
        return SANDBOX_MEMBER_SPACE_TABLES[space]
    raise ValueError(f"unknown consumer profile: {profile!r}")


def _forbidden_for_visible_tables(visible_tables: frozenset[str] | None) -> frozenset[str]:
    graph = _master_schema_graph()
    scope = KnowledgeScope.from_visible_tables(graph, visible_tables)
    return forbidden_identifiers_for_scope(graph, scope)


def _sample_allowed_table(visible_tables: frozenset[str] | None) -> str:
    pool = sorted(visible_tables or _ALL_SANDBOX_TABLES)
    assert pool, "expected at least one visible table"
    return pool[0]


def _sample_forbidden_table(visible_tables: frozenset[str] | None) -> str:
    allowed = visible_tables or _ALL_SANDBOX_TABLES
    forbidden = sorted(_ALL_SANDBOX_TABLES - allowed)
    assert forbidden, "expected at least one out-of-scope table"
    return forbidden[0]


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
        schema_hash="sandbox_egress_matrix",
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
@pytest.mark.parametrize("profile", _CONSUMER_PROFILES)
def test_hermetic_consumer_egress_matrix_clean_observables(profile: str) -> None:
    """Scoped consumer egress must not contain identifiers outside the caller allow-list."""
    visible = _visible_tables_for_profile(profile)
    forbidden = _forbidden_for_visible_tables(visible)
    allowed = _sample_allowed_table(visible)
    step, audit = _clean_observables(allowed)

    assert_no_forbidden_identifiers(step, forbidden)
    assert_no_forbidden_identifiers(audit, forbidden)


@pytest.mark.fast
@pytest.mark.parametrize("profile", [p for p in _CONSUMER_PROFILES if p.startswith("member:")])
def test_hermetic_consumer_egress_matrix_detects_table_leak(profile: str) -> None:
    """Member-scoped consumers must fail the egress scan when an out-of- scope table appears."""
    visible = _visible_tables_for_profile(profile)
    forbidden = _forbidden_for_visible_tables(visible)
    leaked = _leaky_session_step(_sample_forbidden_table(visible))

    with pytest.raises(AssertionError, match="forbidden identifier"):
        assert_no_forbidden_identifiers(leaked, forbidden)


@pytest.mark.fast
@pytest.mark.sandbox
@pytest.mark.parametrize(("space_name", "space_key"), _SPACE_PROFILES)
def test_hermetic_space_egress_matrix_clean_observables(space_name: str, space_key: str | None) -> None:
    """Space-scoped callers only see identifiers from their locked table set."""
    graph = _master_schema_graph()
    if space_key is None:
        scope = KnowledgeScope.from_schema_graph(graph)
        visible = None
    else:
        scope = KnowledgeScope.from_space_context(
            graph,
            SpaceContext(tables=SANDBOX_MEMBER_SPACE_TABLES[space_key]),
        )
        visible = SANDBOX_MEMBER_SPACE_TABLES[space_key]
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    allowed = _sample_allowed_table(visible)
    step, audit = _clean_observables(allowed)

    assert_no_forbidden_identifiers(step, forbidden)
    assert_no_forbidden_identifiers(audit, forbidden)


@pytest.mark.fast
@pytest.mark.parametrize("space_key", sorted(SANDBOX_MEMBER_SPACE_TABLES))
def test_hermetic_space_egress_matrix_detects_cross_space_leak(space_key: str) -> None:
    """Member spaces must not leak tables owned by another bundled space."""
    graph = _master_schema_graph()
    scope = KnowledgeScope.from_space_context(
        graph,
        SpaceContext(tables=SANDBOX_MEMBER_SPACE_TABLES[space_key]),
    )
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    foreign_space = next(name for name in sorted(SANDBOX_MEMBER_SPACE_TABLES) if name != space_key)
    leaked_table = sorted(SANDBOX_MEMBER_SPACE_TABLES[foreign_space] - SANDBOX_MEMBER_SPACE_TABLES[space_key])[0]
    leaked = _leaky_session_step(leaked_table)

    with pytest.raises(AssertionError, match="forbidden identifier"):
        assert_no_forbidden_identifiers(leaked, forbidden)


@pytest.mark.fast
def test_sanitize_session_step_for_egress_scrubs_federation_identity() -> None:
    """Federation egress sanitizer replaces member source ids before consumer-visible scan."""
    raw = SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        sql={"storefront_db": "SELECT 1", "catalog_db": "SELECT 2"},
        diagnostics=(
            Diagnostic(
                stage="execution",
                level="info",
                code=DIAGNOSTIC_CODE_FEDERATION_SOURCES_QUERIED,
                message="Federated turn queried sources: storefront_db,catalog_db",
                details=(("phase", "execution"), ("sources_queried", "storefront_db,catalog_db")),
                source_id="storefront_db",
                subject="storefront_db.payment",
            ),
        ),
    )
    redacted = sanitize_session_step_for_egress(raw)
    member_tokens = frozenset({"storefront_db", "catalog_db"})
    assert_no_forbidden_identifiers(redacted, member_tokens)
    assert set(redacted.sql or {}) == {"member_0", "member_1"}


@pytest.mark.fast
def test_sanitize_audit_details_for_egress_strips_member_source_ids() -> None:
    details = (
        ("source_id", "catalog_db"),
        ("phase", "execution"),
        ("outcome", "ok"),
    )
    redacted = sanitize_audit_details_for_egress(details)
    assert_no_forbidden_identifiers(redacted, frozenset({"catalog_db"}))
    assert ("phase", "execution") in redacted


@pytest.mark.sandbox
@pytest.mark.needs_corpus
@pytest.mark.not_fast
def test_sandbox_egress_matrix_corpus_sweep() -> None:
    """Full question-inventory egress sweep; requires packed sandbox corpus."""
    pytest.skip("corpus not packed")
