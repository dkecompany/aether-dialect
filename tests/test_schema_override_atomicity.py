"""Atomic structure-document application and writer-lock serialization."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata, TableRole
from aetherdialect._schema_finalize import apply_structure_to_graph
from aetherdialect._schema_graph import assign_schema_graph_hashes, recompute_join_paths_multi
from tests.test_aetherdialect import _make_aether_stub
from tests.test_schema import _ov_doc


def _store_table() -> TableMetadata:
    return TableMetadata(
        name="store",
        columns={
            "store_id": ColumnMetadata(
                name="store_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=5,
                distinct_ratio=0.05,
                row_count=100,
            ),
        },
        primary_key=["store_id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )


def _customers_with_store_fk_target(schema_graph: SchemaGraph) -> None:
    schema_graph.tables["store"] = _store_table()
    store_col = ColumnMetadata(
        name="store_id",
        data_type="integer",
        value_type="integer",
        role=ColumnRole.IDENTIFIER.value,
        distinct_count=5,
        distinct_ratio=0.05,
        row_count=100,
    )
    store_col._owner_table = schema_graph.tables["customers"]
    schema_graph.tables["customers"].columns["store_id"] = store_col
    assign_schema_graph_hashes(schema_graph, EngineContext(), "")
    schema_graph.join_paths_multi = recompute_join_paths_multi(schema_graph.tables)


def _paths_match_tables(sg: SchemaGraph) -> bool:
    return sg.join_paths_multi == recompute_join_paths_multi(sg.tables)


def test_apply_structure_to_graph_live_graph_consistent_mid_apply(
    schema_graph: SchemaGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live graph must stay path-consistent while overrides mutate a working copy."""
    _customers_with_store_fk_target(schema_graph)
    live = schema_graph
    pre_recompute_checks: list[bool] = []
    real_recompute = recompute_join_paths_multi

    def probe_recompute(tables: dict[str, Any], **kwargs: Any) -> Any:
        pre_recompute_checks.append(_paths_match_tables(live))
        return real_recompute(tables, **kwargs)

    monkeypatch.setattr("aetherdialect._schema_finalize.recompute_join_paths_multi", probe_recompute)
    monkeypatch.setattr("aetherdialect._config.EngineConfig.llm_credentials_configured", lambda: False)
    report = apply_structure_to_graph(
        schema_graph,
        _ov_doc(
            foreign_keys_add=[
                {
                    "from": "customers.store_id",
                    "to": "store.store_id",
                    "kind": "structural",
                }
            ],
        ),
    )
    assert report.fks_added == 1
    assert pre_recompute_checks, "expected join-path recompute hook"
    assert all(pre_recompute_checks)


def test_apply_structure_acquires_pipeline_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner apply_structure must serialize on the pipeline writer lock."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "schema_structure.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")

    lock = threading.Lock()
    acquired: list[bool] = []

    class TrackingLock:
        def __enter__(self):
            acquired.append(lock.acquire())
            return self

        def __exit__(self, *args: object) -> None:
            lock.release()

    engine = _make_aether_stub(_pipeline_writer_lock=TrackingLock(), _artifacts_dir=art_dir)
    engine._schema_graph = MagicMock()
    engine._schema_graph.schema_graph_id = "sg_test000000000001__abcd1234"
    engine._schema_graph.effective_structural_hash = "eff"

    with patch(
        "aetherdialect.aetherdialect.apply_structure_document",
        return_value=MagicMock(
            table_edits=0,
            column_edits=0,
            fks_added=0,
            fks_removed=0,
            pks_added=0,
            pks_endorsed=0,
            pks_blocked=0,
            coerced_columns=0,
            collapsed_inferences=0,
            domain_knowledge_entries=None,
        ),
    ):
        engine.apply_structure(
            {
                "version": "0.2.3",
                "tables": {},
                "foreign_keys_add": [],
                "foreign_keys_remove": [],
                "primary_keys_add": [],
                "primary_keys_remove": [],
            }
        )
    assert acquired == [True]
