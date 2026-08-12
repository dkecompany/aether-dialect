"""Federated session steps expose member SQL as a source_id mapping (or a single string)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_core import FederatedSqlBundle, FederatedStatementRecord, GenerationPath
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_session import PipelineSession


@pytest.mark.fast
def test_multi_member_federated_step_has_member_sql_mapping_and_bundle() -> None:
    bundle = FederatedSqlBundle(
        statements=(
            FederatedStatementRecord(source_id="a", engine="duckdb", statement="SELECT id FROM left_t"),
            FederatedStatementRecord(source_id="b", engine="duckdb", statement="SELECT id FROM right_t"),
        ),
        display_sql="SELECT a.id FROM left_t a JOIN right_t b ON a.id = b.id",
        column_names=("id",),
    )
    owner = MagicMock()
    owner._schema_graph = None
    owner._audit_emit = MagicMock()
    sess = PipelineSession(owner)
    sess._turn_question = "join left and right"
    sess._last_turn_outcome = {
        "outcome": "success",
        "error": None,
        "sql": bundle.display_sql,
        "rows": [(1,)],
        "columns": ("id",),
        "federated_bundle": bundle,
        "generation_path": GenerationPath.FEDERATION_PLAN,
    }
    step = sess._completed_step()
    assert step.sql == {"a": "SELECT id FROM left_t", "b": "SELECT id FROM right_t"}
    assert step.data is not None
    assert list(step.data.columns) == ["id"]


@pytest.mark.fast
def test_single_member_federated_step_carries_member_statement_sql() -> None:
    bundle = FederatedSqlBundle(
        statements=(FederatedStatementRecord(source_id="a", engine="duckdb", statement="SELECT id FROM left_t"),),
        display_sql="SELECT id FROM left_t",
        column_names=("id",),
    )
    assert MainExecutionOps._federation_session_step_sql(federated_bundle=bundle) == "SELECT id FROM left_t"
    assert (
        MainExecutionOps.resolved_session_step_sql("SELECT display", federated_bundle=bundle) == "SELECT id FROM left_t"
    )


@pytest.mark.fast
def test_single_engine_step_keeps_display_sql() -> None:
    assert MainExecutionOps.resolved_session_step_sql("SELECT 1") == "SELECT 1"
    assert MainExecutionOps._federation_session_step_sql() is None
