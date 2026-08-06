"""Named-context execution scope gate activation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import MASTER_AETHERSPACE_NAME
from aetherdialect._contracts_base import EngineContext, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import parse_federation_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline import _execution_scope_gate_active, generate_and_validate_sql


def _two_table_graph() -> SchemaGraph:
    return SchemaGraph(
        join_paths_multi={},
        tables={
            "allowed": TableMetadata(
                name="allowed",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
            "secret": TableMetadata(
                name="secret",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        effective_structural_hash="eff_t33",
    )


@pytest.mark.fast
def test_execution_scope_gate_active_for_named_context_without_scope_lists() -> None:
    ctx = EngineContext()
    assert _execution_scope_gate_active(ctx, None, "owner", context_name="team_a") is True


@pytest.mark.fast
def test_execution_scope_gate_inactive_for_master_without_scope_lists() -> None:
    ctx = EngineContext()
    assert _execution_scope_gate_active(ctx, None, "owner", context_name=MASTER_AETHERSPACE_NAME) is False


@pytest.mark.fast
def test_consumer_sql_gate_kwargs_includes_context_name() -> None:
    owner = MagicMock()
    owner._schema_role = "owner"
    owner._context_name = "team_a"
    owner._runtime_config = MagicMock(engine_context=EngineContext(), execution_context=EngineContext())
    port = MagicMock(_owner=owner, execution_visible_objects=None, space_tables=None, space_columns=None)
    kwargs = MainExecutionOps._consumer_sql_gate_kwargs(port)
    assert kwargs["context_name"] == "team_a"


@pytest.mark.fast
def test_federation_gate_kwargs_named_context_sets_context_name(tmp_path) -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_t33",
            "sources": [
                {"source_id": "alpha", "engine": "duckdb", "context": "restricted", "role": "owner"},
            ],
            "table_namespace": {"entity_a": "alpha"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    member_dir = tmp_path / "aetherdialect" / "conn_alpha"
    member_dir.mkdir(parents=True)
    (member_dir / "schema_context.restricted.json").write_text(json.dumps({"version": "0.2.1"}), encoding="utf-8")
    owner = MagicMock()
    owner._artifacts_root = tmp_path
    owner._runtime_config = MagicMock(engine_context=EngineContext())
    owner._federation_source_runtimes = {"alpha": MagicMock(artifacts_dir=str(member_dir))}
    gates = MainExecutionOps._federation_gate_kwargs_by_source(owner, None, manifest)
    assert gates["alpha"]["context_name"] == "restricted"
    assert gates["alpha"]["schema_context"] == EngineContext()


@pytest.mark.fast
@patch("aetherdialect._pipeline._run_sql_validation_cascade", return_value=(True, None, None, []))
@patch("aetherdialect._pipeline.assert_consumer_intent_in_scope", return_value=True)
def test_generate_and_validate_sql_runs_scope_gate_for_named_empty_context(
    mock_assert_scope: MagicMock,
    _mock_validate: MagicMock,
) -> None:
    schema = _two_table_graph()
    intent = RuntimeIntent(
        tables=["allowed"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("allowed.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    generate_and_validate_sql(
        "how many allowed",
        intent,
        schema,
        {},
        {},
        MagicMock(),
        {},
        schema_context=EngineContext(),
        context_name="team_a",
        persist_template_learning=False,
    )
    mock_assert_scope.assert_called()
    assert mock_assert_scope.call_count >= 1
