"""Master domain_knowledge.json artifact load/save round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_base import DomainKnowledgeEntry
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._knowledge_staleness import knowledge_artifact_save_stamps
from aetherdialect._utils import (
    domain_knowledge_artifact_path,
    load_domain_knowledge_artifact,
)
from aetherdialect._utils_artifacts import save_domain_knowledge_artifact


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "payment": TableMetadata(
                name="payment",
                columns={"amount": ColumnMetadata(name="amount", data_type="numeric")},
                primary_key=["amount"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g",
    )


@pytest.mark.fast
def test_domain_knowledge_artifact_round_trip(tmp_path: Path) -> None:
    schema = _schema()
    entries = (
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts in April.", kind="policy"),
        DomainKnowledgeEntry(key="arr", text="ARR means annualized rental revenue.", kind="metric"),
    )
    save_domain_knowledge_artifact(tmp_path, entries, **knowledge_artifact_save_stamps(schema))
    path = domain_knowledge_artifact_path(tmp_path)
    assert path.is_file()
    loaded = load_domain_knowledge_artifact(tmp_path, schema)
    assert loaded is not None
    assert [(e.key, e.kind, e.text) for e in loaded] == [(e.key, e.kind, e.text) for e in entries]


@pytest.mark.fast
def test_domain_knowledge_artifact_missing_returns_none(tmp_path: Path) -> None:
    assert load_domain_knowledge_artifact(tmp_path, _schema()) is None
