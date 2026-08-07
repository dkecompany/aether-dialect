"""Federation member liveness is re-checked at turn start."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import SESSION_KIND_ERROR, PipelineSession
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import TemplateOps


def _member_table(name: str, source_id: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )


def _federation_owner(*, members: dict[str, MagicMock] | None = None) -> MagicMock:
    tables = {"t": _member_table("t", "west")}
    schema = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    owner = MagicMock()
    owner._is_aether_federation = True
    owner._schema_graph = schema
    owner._store = TemplateOps.empty_template_store(schema.schema_graph_id)
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._dialect = "duckdb"
    owner._artifacts_dir = None
    owner._audit_emit = None
    owner._pipeline_writer_lock = None
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._engine_identity = None
    owner._members = members if members is not None else {}
    return owner


def _mock_db_member() -> MagicMock:
    member = MagicMock()
    member.dialect = "postgresql"
    sa_engine = MagicMock()
    mock_conn = MagicMock()
    sa_engine.connect.return_value.__enter__.return_value = mock_conn
    member._execution_engine = sa_engine
    return member


@pytest.mark.fast
def test_federation_turn_start_probes_member_connections() -> None:
    members = {"west": _mock_db_member()}
    owner = _federation_owner(members=members)
    session = PipelineSession(owner)

    with patch("aetherdialect._main_execution.probe_federation_member_liveness") as probe_mock:
        with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once"):
            session.ask("how many t")

    probe_mock.assert_called_once_with(members)


@pytest.mark.fast
def test_stale_connection_surfaces_as_probe_error_not_partial_failure() -> None:
    member = _mock_db_member()
    member._execution_engine.connect.side_effect = RuntimeError("connection reset by peer")
    owner = _federation_owner(members={"west": member})
    session = PipelineSession(owner)

    with patch("aetherdialect._main_execution.MainExecutionOps.interactive_run_once"):
        step = session.ask("how many t")

    assert step.done is True
    assert step.kind == SESSION_KIND_ERROR
    assert step.federation_source_id == "west"
    assert step.federation_phase == "prepare"
    assert step.federation_succeeded == ()
    assert step.error is not None
    assert "connection reset" not in step.error
    assert step.status != "federation_partial_failure"
