"""Terminal meta answer failure returns status meta_error."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import SESSION_KIND_ERROR
from aetherdialect._contracts_base import Diagnostic
from aetherdialect._contracts_core import QuestionRoute
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import drain_diagnostic_collector, reset_diagnostic_collector, set_diagnostic_collector


def _schema() -> SchemaGraph:
    t = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(
                name="order_id",
                data_type="integer",
                value_type="integer",
                role="identifier",
                is_primary_key=True,
                distinct_count=5,
                distinct_ratio=1.0,
                row_count=5,
                null_ratio=0.0,
            )
        },
        primary_key=["order_id"],
        foreign_keys=[],
        description="Orders",
    )
    return SchemaGraph(
        tables={"orders": t},
        join_paths_multi={},
        schema_graph_id="sg-fail",
        effective_structural_hash="hash-fail",
    )


@pytest.mark.fast
def test_failed_meta_step_status_meta_error() -> None:
    bad = {
        "response_kind": "schema_catalog",
        "headline": "Invented.",
        "counts": {
            "tables": None,
            "columns": None,
            "members": None,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [
            {
                "name": "not_a_real_table",
                "source_id": "default",
                "description": "",
                "columns": [],
            }
        ],
        "relationships": [],
        "notes": [],
    }
    buf: list[Diagnostic] = []
    tok = set_diagnostic_collector(buf)
    try:
        with patch("aetherdialect._llm_provider.LLMProvider.json", return_value=bad) as llm_mock:
            step = MainExecutionOps.answer_metadata_question(
                MagicMock(), "how many tables", QuestionRoute.SCHEMA_CATALOG, _schema(), None, None
            )
        codes = (
            {d.code for d in step.diagnostics} | {d.code for d in buf} | {d.code for d in drain_diagnostic_collector()}
        )
    finally:
        reset_diagnostic_collector(tok)

    assert llm_mock.call_count == 2
    assert step.done is True
    assert step.kind == SESSION_KIND_ERROR
    assert step.error is not None
    assert step.sql is None
    assert step.error.detail_code == "meta.answer.failed" or "meta.answer.failed" in {d.code for d in step.diagnostics}
    assert "meta.answer.failed" in codes
    assert "meta.answer.repair" in codes
