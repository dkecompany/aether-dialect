"""Member schema overrides name cross-source FK limits explicitly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_overrides import apply_schema_overrides_to_graph, load_schema_overrides_file
from tests.test_schema import _ov_doc


def _member_graph() -> SchemaGraph:
    table = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id="storefront",
    )
    tables = {"orders": table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_orders",
        effective_structural_hash="eff_orders",
    )


@pytest.mark.fast
def test_member_foreign_key_override_to_other_member_table_names_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    graph = _member_graph()
    editor = tmp_path / "schema_overrides.json"
    editor.write_text(
        json.dumps(
            _ov_doc(
                foreign_keys_add=[
                    {
                        "from": "orders.id",
                        "to": "inventory.id",
                        "kind": "structural",
                    },
                ],
            ),
        ),
        encoding="utf-8",
    )
    document = load_schema_overrides_file(editor)
    report = apply_schema_overrides_to_graph(graph, document, strict=False)
    reasons = [skip.reason for skip in report.skipped]
    assert any("federation manifest" in reason for reason in reasons)
