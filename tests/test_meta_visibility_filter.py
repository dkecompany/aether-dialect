"""Effective visibility filtering for meta schema dumps and structure export."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import EngineContext, SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps


def _graph_two_tables() -> SchemaGraph:
    open_col = ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")
    hidden = ColumnMetadata(
        name="secret",
        data_type="TEXT",
        value_type="text",
        sensitivity=SensitivityClassification.HIDDEN,
    )
    restricted = ColumnMetadata(
        name="ssn",
        data_type="TEXT",
        value_type="text",
        sensitivity=SensitivityClassification.RESTRICTED,
    )
    public = TableMetadata(
        name="orders",
        columns={"id": open_col, "secret": hidden, "ssn": restricted},
        primary_key=["id"],
        foreign_keys=[],
    )
    private = TableMetadata(
        name="payroll",
        columns={"id": ColumnMetadata(name="id", data_type="INTEGER", value_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(tables={"orders": public, "payroll": private}, join_paths_multi={})


@pytest.mark.fast
def test_meta_schema_dump_strips_hidden_restricted_and_invisible_tables() -> None:
    sg = _graph_two_tables()
    dump = MainExecutionOps.build_meta_schema_dump(
        sg,
        scope_ctx=EngineContext(),
        visible_objects=frozenset({"orders"}),
        exclude_restricted=True,
    )
    names = {t["name"] for t in dump["tables"]}
    assert names == {"orders"}
    cols = {c["name"] for c in dump["tables"][0]["columns"]}
    assert cols == {"id"}
    assert dump["inventory"]["table_count"] == 1


@pytest.mark.fast
def test_meta_schema_dump_applies_space_description_overlays() -> None:
    sg = _graph_two_tables()
    sg.tables["orders"].description = "master orders prose mentioning payroll"
    dump = MainExecutionOps.build_meta_schema_dump(
        sg,
        scope_ctx=EngineContext(),
        visible_objects=frozenset({"orders"}),
        space_tables={"orders"},
        table_descriptions={"orders": "space-scoped orders only"},
        column_descriptions={"orders.id": "space-scoped id"},
    )
    assert dump["tables"][0]["description"] == "space-scoped orders only"
    assert dump["tables"][0]["columns"][0]["description"] == "space-scoped id"


@pytest.mark.fast
def test_structure_export_respects_visible_objects() -> None:
    sg = _graph_two_tables()
    out = MainExecutionOps.build_structure_export(
        schema_graph=sg,
        visible_objects=frozenset({"orders"}),
        scope_ctx=EngineContext(deny_objects=frozenset()),
    )
    assert out["table_count"] == 1
    assert out["tables"][0]["name"] == "orders"
    assert {c["name"] for c in out["tables"][0]["columns"]} == {"id"}
    assert "description" not in out["tables"][0]
    assert all("description" not in c and "role" not in c for c in out["tables"][0]["columns"])


@pytest.mark.fast
def test_validate_question_accepts_schema_and_knowledge_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from aetherdialect._contracts_core import QuestionRoute
    from aetherdialect._utils_intent import validate_question

    monkeypatch.setattr(
        "aetherdialect._utils_intent.LLMProvider.json",
        lambda *_a, **_k: {
            "valid_database_question": "yes",
            "query_type": "schema_and_knowledge",
            "corrected": "which tables discuss revenue",
        },
    )
    result = validate_question("which tables discuss revenue")
    assert result.accepted is True
    assert result.route is QuestionRoute.SCHEMA_AND_KNOWLEDGE
