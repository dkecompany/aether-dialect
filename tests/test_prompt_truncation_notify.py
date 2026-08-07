"""Schema prompt truncation must reach host diagnostics via notify."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_DESCRIPTION_PROMPT_TRUNCATED,
    SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._intent_process import _emit_schema_description_truncation_diagnostic


def _graph_with_long_description() -> SchemaGraph:
    long_desc = "w" * (SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS + 50)
    table = TableMetadata(
        name="tbl",
        columns={
            "col_0": ColumnMetadata(name="col_0", data_type="varchar", description=long_desc),
        },
        primary_key=[],
        foreign_keys=[],
        description="table purpose",
    )
    return SchemaGraph(
        tables={"tbl": table},
        join_paths_multi={},
        effective_structural_hash="eff_hash",
    )


@pytest.mark.fast
def test_description_truncation_reaches_diagnostics() -> None:
    """Description prompt cap must emit DESCRIPTION_PROMPT_TRUNCATED via notify."""
    graph = _graph_with_long_description()
    json.loads(graph.schema_payload_interpret(owner_master_scope=True))

    token = set_diagnostic_collector([])
    try:
        _emit_schema_description_truncation_diagnostic(graph)
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert any(d.code == DIAGNOSTIC_CODE_DESCRIPTION_PROMPT_TRUNCATED for d in diags)
