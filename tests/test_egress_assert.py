"""Fast unit tests for the hermetic egress assertion helper."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import Diagnostic, KnowledgeScope
from aetherdialect._contracts_core import SessionStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from tests.support.egress_assert import (
    assert_no_forbidden_identifiers,
    forbidden_identifiers_for_scope,
)


def _rental_shop_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "customer": TableMetadata(
                name="customer",
                columns={"email": ColumnMetadata(name="email", data_type="text")},
                primary_key=[],
                foreign_keys=[],
            ),
            "staff": TableMetadata(
                name="staff",
                columns={"email": ColumnMetadata(name="email", data_type="text")},
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )


@pytest.mark.fast
def test_detects_leak_in_session_step_message() -> None:
    step = SessionStep(
        done=True,
        prompt="Continue?",
        kind="result",
        answer="Unable to join staff to customer without a path.",
    )
    with pytest.raises(AssertionError, match=r"root\.answer"):
        assert_no_forbidden_identifiers(step, frozenset({"staff"}))


@pytest.mark.fast
def test_detects_leak_in_diagnostic_subject() -> None:
    diagnostic = Diagnostic(
        stage="validate",
        level="warning",
        code="JOIN_PATH_MISSING",
        message="No join path found.",
        subject="staff.email",
    )
    with pytest.raises(AssertionError, match=r"root\.subject"):
        assert_no_forbidden_identifiers(diagnostic, frozenset({"staff"}))


@pytest.mark.fast
def test_detects_leak_in_diagnostic_details() -> None:
    diagnostic = Diagnostic(
        stage="validate",
        level="warning",
        code="JOIN_PATH_MISSING",
        message="No join path found.",
        details=(("table", "staff"), ("column", "staff.email")),
    )
    with pytest.raises(AssertionError, match=r"root\.details"):
        assert_no_forbidden_identifiers(diagnostic, frozenset({"staff"}))


@pytest.mark.fast
def test_passes_when_only_allowed_identifiers_present() -> None:
    graph = _rental_shop_graph()
    scope = KnowledgeScope.from_visible_tables(graph, frozenset({"customer"}))
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    assert "staff" in forbidden

    step = SessionStep(
        done=True,
        prompt=None,
        kind="result",
        answer="Showing customer email rows.",
        diagnostics=(
            Diagnostic(
                stage="execute",
                level="info",
                code="ROWS",
                message="Returned customer rows.",
                details=(("table", "customer"),),
            ),
        ),
    )
    assert_no_forbidden_identifiers(step, forbidden)


@pytest.mark.fast
def test_walks_nested_federated_bundle_dicts() -> None:
    bundle = {
        "members": [
            {"source_id": "storefront", "status": "ok"},
            {"source_id": "catalog", "status": "ok"},
        ],
        "sql_by_member": {"catalog": "SELECT payment_id FROM payment"},
        "nested": {"rows": [{"member": "logistics", "count": 3}]},
    }
    forbidden = frozenset({"catalog", "logistics", "storefront"})
    with pytest.raises(AssertionError, match=r"root\[.members.\]\[0\]\[.source_id.\]"):
        assert_no_forbidden_identifiers(bundle, forbidden)


@pytest.mark.fast
def test_forbidden_identifiers_for_scope_includes_out_of_scope_tables() -> None:
    graph = _rental_shop_graph()
    scope = KnowledgeScope.from_visible_tables(graph, frozenset({"customer"}))
    forbidden = forbidden_identifiers_for_scope(
        graph,
        scope,
        extra_forbidden=("storefront",),
    )
    assert forbidden >= frozenset({"staff", "storefront"})


@pytest.mark.fast
def test_dataframe_checks_column_names_only() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({"customer_id": [1], "staff_id": [2]})
    with pytest.raises(AssertionError, match=r"root\.columns\['staff_id'\]"):
        assert_no_forbidden_identifiers(frame, frozenset({"staff_id"}))


@pytest.mark.fast
def test_session_step_smoke_catches_forbidden_table_name() -> None:
    """Minimal SessionStep smoke: one forbidden table name in answer must fail."""
    step = SessionStep(
        done=True,
        prompt=None,
        kind="error",
        answer="Query referenced staff, which is not available.",
    )
    graph = _rental_shop_graph()
    scope = KnowledgeScope.from_visible_tables(graph, frozenset({"customer"}))
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    with pytest.raises(AssertionError, match="staff"):
        assert_no_forbidden_identifiers(step, forbidden)
