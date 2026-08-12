"""Enum prompt truncation must reach host diagnostics via notify."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED, FEDERATION_ENUM_PROMPT_CAP
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import _emit_schema_enum_truncation_diagnostic
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


def _graph_with_capped_enum() -> SchemaGraph:
    values = [f"v{i}" for i in range(FEDERATION_ENUM_PROMPT_CAP + 5)]
    table = TableMetadata(
        name="tbl",
        columns={
            "col_0": ColumnMetadata(name="col_0", data_type="varchar", description="short"),
        },
        primary_key=[],
        foreign_keys=[],
        description="table purpose",
    )
    return SchemaGraph(
        tables={"tbl": table},
        join_paths_multi={},
        enum_values={"status_enum": values},
        effective_structural_hash="eff_hash",
    )


@pytest.mark.fast
def test_enum_truncation_reaches_diagnostics() -> None:
    """Enum prompt cap must emit ENUM_PROMPT_TRUNCATED via notify."""
    graph = _graph_with_capped_enum()
    json.loads(graph.schema_payload_interpret(owner_master_scope=True))

    token = set_diagnostic_collector([])
    try:
        _emit_schema_enum_truncation_diagnostic(graph)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert any(d.code == DIAGNOSTIC_CODE_ENUM_PROMPT_TRUNCATED for d in diags)
