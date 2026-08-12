"""Space knowledge merge: SPACE_NOTES wins descriptions; engine DK authoritative on key clash."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import ARTIFACT_FORMAT_VERSION
from aetherdialect._contracts_base import DomainKnowledgeEntry, SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, DescriptionOwner, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._utils import domain_knowledge_digest


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
    assert ARTIFACT_FORMAT_VERSION == "0.2.3"


@pytest.mark.fast
def test_space_dk_engine_authoritative_same_key() -> None:
    engine = (
        DomainKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),
        DomainKnowledgeEntry(key="fy", text="engine fy", kind="policy"),
    )
    space = (
        DomainKnowledgeEntry(key="arr", text="space arr overlay", kind="glossary"),
        DomainKnowledgeEntry(key="nrr", text="space only", kind="metric"),
    )
    merged = MainExecutionOps.merge_domain_knowledge(engine, space)
    by_key = {e.key: e.text for e in merged}
    assert by_key["arr"] == "engine arr"
    assert by_key["fy"] == "engine fy"
    assert by_key["nrr"] == "space only"
    from aetherdialect._contracts_base import DomainKnowledgeState

    assert domain_knowledge_digest(merged) == DomainKnowledgeState.digest_for(merged)


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
    fed_dk = DomainKnowledgeEntry(
        key="arr",
        kind="glossary",
        text="ARR means annual recurring revenue.",
        referenced_entities=frozenset(),
    )
    sk_fact = __import__("aetherdialect._contracts_base", fromlist=["StructuralKnowledgeFact"]).StructuralKnowledgeFact(
        kind="relation",
        text="order header",
        referenced_entities=frozenset({"orders"}),
    )
    classify = {
        "orders": (
            "entity",
            "space refined orders",
            {"id": ("identifier", "order id", None)},
        )
    }
    with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True):
        with patch(
            "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
            return_value=classify,
        ):
            with patch(
                "aetherdialect._main_spaces.resolve_knowledge_extraction_for_schema",
                return_value=((fed_dk,), (sk_fact,)),
            ):
                with patch(
                    "aetherdialect._main_spaces.EngineConfig.llm_credentials_configured",
                    return_value=True,
                ):
                    out = MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, schema, space_ctx, str(notes))
    assert out["table_descriptions"]["orders"] == "space refined orders"
    assert "domain_knowledge" in out
    assert isinstance(out["domain_knowledge"], list)
    assert any(row.get("key") == "arr" for row in out["domain_knowledge"])
    assert out.get("domain_knowledge_digest")
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

    def _fake_enrich(snap, master, ctx, path, **kwargs):
        captured["notes_path"] = path
        return {**snap, "domain_knowledge": [], "domain_knowledge_digest": ""}

    with patch.object(AetherEngine, "_require_owner", lambda self, *_a, **_k: None):
        with patch("aetherdialect.aetherdialect.validate_space_context_against_graph", return_value=space_ctx):
            with patch(
                "aetherdialect.aetherdialect.validate_aetherspace_define_within_visibility",
                return_value=None,
            ):
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
