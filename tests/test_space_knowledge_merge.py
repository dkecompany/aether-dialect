"""Space knowledge merge: SPACE_NOTES wins descriptions; space BK overrides by key."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION
from aetherdialect._contracts_base import (
    BusinessKnowledgeEntry,
    DescriptionOwner,
    SpaceContext,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import business_knowledge_digest
from aetherdialect._main_execution import MainExecutionOps


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
                description="engine notes description",
                description_owner=DescriptionOwner.NOTES,
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g1",
    )


@pytest.mark.fast
def test_space_description_wins() -> None:
    assert DescriptionOwner._rank(DescriptionOwner.SPACE_NOTES) > DescriptionOwner._rank(DescriptionOwner.NOTES)
    assert DescriptionOwner._rank(DescriptionOwner.SPACE_NOTES) < DescriptionOwner._rank(DescriptionOwner.USER_OVERRIDE)
    engine_desc = "from engine notes"
    space_desc = "from space notes"
    text, owner = DescriptionOwner.resolve(
        (engine_desc, DescriptionOwner.NOTES),
        (space_desc, DescriptionOwner.SPACE_NOTES),
    )
    assert text == space_desc
    assert owner == DescriptionOwner.SPACE_NOTES
    assert ARTIFACT_FORMAT_VERSION == "0.2.1"


@pytest.mark.fast
def test_space_bk_key_overrides() -> None:
    engine = (
        BusinessKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),
        BusinessKnowledgeEntry(key="fy", text="engine fy", kind="policy"),
    )
    space = (
        BusinessKnowledgeEntry(key="arr", text="space arr wins", kind="glossary"),
        BusinessKnowledgeEntry(key="nrr", text="space only", kind="metric"),
    )
    merged = MainExecutionOps.merge_business_knowledge(engine, space)
    by_key = {e.key: e.text for e in merged}
    assert by_key["arr"] == "space arr wins"
    assert by_key["fy"] == "engine fy"
    assert by_key["nrr"] == "space only"
    from aetherdialect._contracts_base import BusinessKnowledgeState

    assert business_knowledge_digest(merged) == BusinessKnowledgeState.digest_for(merged)


@pytest.mark.fast
def test_engine_honours_space_context_notes_file(tmp_path: Path) -> None:
    notes = tmp_path / "space_notes.txt"
    notes.write_text("ARR means annual recurring revenue.\n", encoding="utf-8")
    schema = _schema()
    space_ctx = SpaceContext(tables=frozenset({"orders"}), notes_file=str(notes))
    snapshot = {
        "tables": ["orders"],
        "columns": ["orders.id"],
        "table_descriptions": {"orders": "engine notes description"},
        "column_meta": {},
    }
    llm_payload = [{"key": "arr", "kind": "glossary", "text": "ARR means annual recurring revenue."}]
    classify = {
        "orders": (
            "entity",
            "space refined orders",
            {"id": ("identifier", "order id", None)},
        )
    }
    with patch("aetherdialect._main_execution.EngineConfig.llm_credentials_configured", return_value=True):
        with patch("aetherdialect._main_execution.llm_classify_schema", return_value=classify):
            with patch(
                "aetherdialect._schema_catalog.LLMProvider.chat",
                return_value=__import__("json").dumps(llm_payload),
            ):
                with patch(
                    "aetherdialect._schema_catalog.EngineConfig.llm_credentials_configured",
                    return_value=True,
                ):
                    out = MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, schema, space_ctx, str(notes))
    assert out["table_descriptions"]["orders"] == "space refined orders"
    assert "business_knowledge" in out
    assert isinstance(out["business_knowledge"], list)
    assert any(row.get("key") == "arr" for row in out["business_knowledge"])
    assert out.get("business_knowledge_digest")
    # aetherspace notes resolution: kwarg None → SpaceContext.notes_file
    from aetherdialect import AetherEngine

    engine = object.__new__(AetherEngine)
    engine._schema_graph = schema
    engine._artifacts_dir = str(tmp_path)
    engine._schema_role = "owner"
    engine._sandbox_mode = False
    engine._pipeline_writer_lock = __import__("threading").Lock()
    engine._runtime_config = MagicMock(engine_context=MagicMock())
    captured: dict[str, object] = {}

    def _fake_enrich(snap, master, ctx, path):
        captured["notes_path"] = path
        return {**snap, "business_knowledge": [], "business_knowledge_digest": ""}

    with patch.object(AetherEngine, "_require_owner", lambda self, *_a, **_k: None):
        with patch.object(AetherEngine, "_require_master_context", lambda self, *_a, **_k: None):
            with patch("aetherdialect.aetherdialect.validate_space_context_against_graph", return_value=space_ctx):
                with patch("aetherdialect.aetherdialect.subset_graph_for_space", return_value=snapshot):
                    with patch(
                        "aetherdialect.aetherdialect.enrich_space_snapshot_with_notes", side_effect=_fake_enrich
                    ):
                        with patch("aetherdialect.aetherdialect.save_aetherspace_snapshot"):
                            with patch(
                                "aetherdialect.aetherdialect.aetherspace_descriptor_from_snapshot",
                                return_value=MagicMock(),
                            ):
                                engine.aetherspace("analytics", space_ctx)
    assert captured["notes_path"] == str(notes)
