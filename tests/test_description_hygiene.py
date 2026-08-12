"""Description enrichment diagnostics and enriched schema line caps."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED,
    DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP,
    SCHEMA_ENRICHED_LINES_MAX_CHARS,
)
from aetherdialect._contracts_base import SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaDiff, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_finalize import _refresh_existing_descriptions_after_addition
from aetherdialect._schema_profile import NotesExtractionLedger, NotesExtractionResult
from aetherdialect._utils import (
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)
from aetherdialect._utils_intent import schema_context_enriched_lines_for_tables


def _table(name: str, *, description: str = "", column_description: str = "") -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                description=column_description,
            )
        },
        primary_key=["id"],
        foreign_keys=[],
        description=description,
    )


def _graph(*names: str) -> SchemaGraph:
    return SchemaGraph(
        tables={name: _table(name) for name in names},
        join_paths_multi={},
        effective_structural_hash="eff_hash",
    )


def _space_snapshot(*tables: str) -> dict:
    return {
        "tables": list(tables),
        "table_descriptions": {},
        "column_meta": {},
    }


@pytest.mark.fast
def test_enrich_space_notes_failure_propagates(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("domain notes\n", encoding="utf-8")
    graph = _graph("orders")
    snapshot = _space_snapshot("orders")
    space = SpaceContext(tables=frozenset({"orders"}), notes_file=str(notes))

    sk_fact = __import__("aetherdialect._contracts_base", fromlist=["StructuralKnowledgeFact"]).StructuralKnowledgeFact(
        kind="relation",
        text="order rows",
        referenced_entities=frozenset({"orders"}),
    )
    with (
        patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True),
        patch(
            "aetherdialect._main_spaces.extract_knowledge_from_notes",
            return_value=NotesExtractionResult((), (sk_fact,), NotesExtractionLedger(())),
        ),
        patch(
            "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
            side_effect=RuntimeError("model unavailable"),
        ),
        pytest.raises(RuntimeError, match="model unavailable"),
    ):
        MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, graph, space, str(notes))


@pytest.mark.fast
def test_enrich_space_notes_noop_emits_diagnostic(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("domain notes\n", encoding="utf-8")
    graph = _graph("orders")
    snapshot = _space_snapshot("orders")
    space = SpaceContext(tables=frozenset({"orders"}), notes_file=str(notes))

    token = set_diagnostic_collector([])
    try:
        with (
            patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True),
            patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=True),
            patch(
                "aetherdialect._main_spaces.extract_knowledge_from_notes",
                return_value=NotesExtractionResult((), (), NotesExtractionLedger(())),
            ),
            patch(
                "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
                return_value={},
            ),
        ):
            out = MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, graph, space, str(notes))
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert out.get("table_descriptions") == {}
    assert out.get("column_meta") == {}
    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP
    assert dict(diags[0].details).get("scope") == "aetherspace_notes"


@pytest.mark.fast
def test_refresh_existing_descriptions_failure_emits_diagnostic() -> None:
    cached = _graph("legacy", "new_table")
    diff = SchemaDiff(added_tables=("new_table",))

    token = set_diagnostic_collector([])
    try:
        with patch(
            "aetherdialect._schema_finalize.llm_classify_schema",
            side_effect=RuntimeError("classify failed"),
        ):
            _refresh_existing_descriptions_after_addition(cached, diff, "notes")
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_FAILED
    assert dict(diags[0].details).get("scope") == "schema_migration_refresh"


@pytest.mark.fast
def test_refresh_existing_descriptions_noop_emits_diagnostic() -> None:
    cached = _graph("legacy", "new_table")
    diff = SchemaDiff(added_tables=("new_table",))
    noop_classify = {
        "legacy": ("dimension", "", {"id": ("identifier", "", None)}),
        "new_table": ("fact", "", {"id": ("identifier", "", None)}),
    }

    token = set_diagnostic_collector([])
    try:
        with patch(
            "aetherdialect._schema_finalize.llm_classify_schema",
            return_value=noop_classify,
        ):
            _refresh_existing_descriptions_after_addition(cached, diff, "notes")
        diags = drain_diagnostic_collector()
    finally:
        reset_diagnostic_collector(token)

    assert len(diags) == 1
    assert diags[0].code == DIAGNOSTIC_CODE_DESCRIPTION_ENRICHMENT_NOOP
    assert dict(diags[0].details).get("scope") == "schema_migration_refresh"


@pytest.mark.fast
def test_enriched_schema_lines_total_char_cap() -> None:
    long_desc = "x" * 400
    columns = {
        f"col_{i}": ColumnMetadata(name=f"col_{i}", data_type="varchar", description=long_desc) for i in range(200)
    }
    table = TableMetadata(
        name="wide",
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        description=long_desc,
    )
    graph = SchemaGraph(tables={"wide": table}, join_paths_multi={})
    rendered = schema_context_enriched_lines_for_tables(graph, ["wide"])
    assert len(rendered) <= SCHEMA_ENRICHED_LINES_MAX_CHARS


@pytest.mark.fast
def test_enriched_schema_lines_preserve_full_description_under_block_budget() -> None:
    long_desc = "w" * 300
    graph = SchemaGraph(
        tables={"tbl": _table("tbl", description=long_desc, column_description=long_desc)},
        join_paths_multi={},
    )
    rendered = schema_context_enriched_lines_for_tables(graph, ["tbl"])
    assert long_desc in rendered
    assert len(rendered) <= SCHEMA_ENRICHED_LINES_MAX_CHARS
