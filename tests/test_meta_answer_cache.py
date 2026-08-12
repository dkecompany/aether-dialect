"""meta_answers.json cache for schema_catalog / domain_knowledge answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import META_ANSWER_FORMAT_VERSION, META_ANSWERS_FILENAME, SESSION_KIND_META
from aetherdialect._contracts_base import (
    Diagnostic,
    DomainKnowledgeEntry,
    DomainKnowledgeHolder,
)
from aetherdialect._contracts_core import QuestionRoute
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_init import MainInitOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    domain_knowledge_scope,
    drain_diagnostic_collector,
    reset_diagnostic_collector,
    set_diagnostic_collector,
)


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        role="identifier",
        is_primary_key=True,
        distinct_count=5,
        distinct_ratio=1.0,
        row_count=5,
        null_ratio=0.0,
    )


def _schema() -> SchemaGraph:
    t = TableMetadata(
        name="orders",
        columns={"order_id": _col("order_id")},
        primary_key=["order_id"],
        foreign_keys=[],
        description="Orders",
    )
    return SchemaGraph(
        tables={"orders": t},
        join_paths_multi={},
        schema_graph_id="sg-cache",
        effective_structural_hash="hash-cache",
    )


def _count_answer() -> dict[str, Any]:
    return {
        "response_kind": "schema_catalog",
        "headline": "One table.",
        "counts": {
            "tables": 1,
            "columns": None,
            "members": None,
            "columns_in_table": None,
            "tables_in_member": None,
        },
        "tables": [],
        "relationships": [],
        "notes": [],
    }


def _owner(artifacts_dir: str) -> MagicMock:
    owner = MagicMock()
    owner._domain_knowledge = DomainKnowledgeHolder()
    owner._artifacts_dir = artifacts_dir
    owner._store = TemplateOps.empty_template_store("sg-cache")
    owner._templates = {}
    owner._federation_manifest = None
    return owner


def _schema_two_tables() -> SchemaGraph:
    orders = TableMetadata(
        name="orders",
        columns={"order_id": _col("order_id")},
        primary_key=["order_id"],
        foreign_keys=[],
        description="Orders",
    )
    customers = TableMetadata(
        name="customers",
        columns={"customer_id": _col("customer_id")},
        primary_key=["customer_id"],
        foreign_keys=[],
        description="Customers",
    )
    return SchemaGraph(
        tables={"orders": orders, "customers": customers},
        join_paths_multi={},
        schema_graph_id="sg-cache",
        effective_structural_hash="hash-cache",
    )


def _owner_with_visibility(artifacts_dir: str, visible: frozenset[str] | None) -> MagicMock:
    owner = _owner(artifacts_dir)
    owner._consumer_visible_objects = visible
    return owner


@pytest.mark.fast
def test_visibility_fingerprint_splits_cache_keys() -> None:
    owner_fp = MainInitOps._meta_answer_visibility_fingerprint(
        scope_ctx=None,
        visible_objects=None,
        space_tables=None,
    )
    consumer_fp = MainInitOps._meta_answer_visibility_fingerprint(
        scope_ctx=None,
        visible_objects=frozenset({"orders", "orders.order_id"}),
        space_tables=None,
    )
    assert owner_fp != consumer_fp
    base = dict(
        schema_graph_id="sg",
        federation_id="",
        space_name="master",
        domain_knowledge_digest="",
        corrected_question="how many tables",
        route="schema_catalog",
    )
    assert MainInitOps.meta_answer_cache_key(
        **base, visibility_fingerprint=owner_fp
    ) != MainInitOps.meta_answer_cache_key(**base, visibility_fingerprint=consumer_fp)


@pytest.mark.fast
def test_consumer_does_not_hit_owner_schema_catalog_cache(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    schema = _schema_two_tables()
    owner_answer = {
        "response_kind": "schema_catalog",
        "headline": "Two tables.",
        "counts": {"tables": 2, "columns": None, "members": None, "columns_in_table": None, "tables_in_member": None},
        "tables": [],
        "relationships": [],
        "notes": [],
    }
    consumer_answer = {
        "response_kind": "schema_catalog",
        "headline": "One table.",
        "counts": {"tables": 1, "columns": None, "members": None, "columns_in_table": None, "tables_in_member": None},
        "tables": [],
        "relationships": [],
        "notes": [],
    }
    owner = _owner_with_visibility(artifacts, None)
    consumer = _owner_with_visibility(artifacts, frozenset({"orders", "orders.order_id"}))
    with patch("aetherdialect._llm_provider.LLMProvider.json", side_effect=[owner_answer, consumer_answer]) as llm_mock:
        owner_step = MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
        consumer_step = MainExecutionOps.answer_metadata_question(
            consumer, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
    assert llm_mock.call_count == 2
    assert owner_step.answer != consumer_step.answer
    assert "Two tables." in owner_step.answer
    assert "One table." in consumer_step.answer


@pytest.mark.fast
def test_stale_owner_cache_rejected_for_consumer(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    schema = _schema_two_tables()
    owner = _owner_with_visibility(artifacts, None)
    consumer = _owner_with_visibility(artifacts, frozenset({"orders", "orders.order_id"}))
    owner_answer = {
        "response_kind": "schema_catalog",
        "headline": "Two tables.",
        "counts": {"tables": 2, "columns": None, "members": None, "columns_in_table": None, "tables_in_member": None},
        "tables": [{"name": "customers", "source_id": "default", "description": "", "columns": []}],
        "relationships": [],
        "notes": [],
    }
    consumer_answer = {
        "response_kind": "schema_catalog",
        "headline": "One table.",
        "counts": {"tables": 1, "columns": None, "members": None, "columns_in_table": None, "tables_in_member": None},
        "tables": [],
        "relationships": [],
        "notes": [],
    }
    with patch("aetherdialect._llm_provider.LLMProvider.json", side_effect=[owner_answer, consumer_answer]) as llm_mock:
        MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
        consumer_step = MainExecutionOps.answer_metadata_question(
            consumer, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
    assert llm_mock.call_count == 2
    assert "customers" not in (consumer_step.answer or "").lower()


@pytest.mark.fast
def test_second_question_hits_cache(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    owner = _owner(artifacts)
    schema = _schema()
    answer = _count_answer()
    with patch("aetherdialect._llm_provider.LLMProvider.json", return_value=answer) as llm_mock:
        first = MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
        second = MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
    assert llm_mock.call_count == 1
    assert first.answer == second.answer
    assert second.kind == SESSION_KIND_META
    cache_path = Path(artifacts) / META_ANSWERS_FILENAME
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["meta_answer_format_version"] == META_ANSWER_FORMAT_VERSION
    assert payload["entries"]


@pytest.mark.fast
def test_dk_digest_change_misses(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    owner = _owner(artifacts)
    e1 = (DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="metric"),)
    e2 = (DomainKnowledgeEntry(key="arr", text="Updated ARR definition.", kind="metric"),)
    llm_answers = [
        {"response_kind": "domain_knowledge", "message": "ARR is annual recurring revenue."},
        {"response_kind": "domain_knowledge", "message": "ARR was updated."},
    ]
    with patch("aetherdialect._llm_provider.LLMProvider.json", side_effect=llm_answers) as llm_mock:
        with domain_knowledge_scope(entries=e1, digest="digest-one"):
            first = MainExecutionOps.answer_metadata_question(
                owner, "what is ARR", QuestionRoute.DOMAIN_KNOWLEDGE, _schema(), "master", artifacts
            )
        with domain_knowledge_scope(entries=e2, digest="digest-two"):
            second = MainExecutionOps.answer_metadata_question(
                owner, "what is ARR", QuestionRoute.DOMAIN_KNOWLEDGE, _schema(), "master", artifacts
            )
    assert llm_mock.call_count == 2
    assert first.answer != second.answer


@pytest.mark.fast
def test_sql_template_count_unchanged(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    owner = _owner(artifacts)
    schema = _schema()
    before_store = len(TemplateOps.store_to_templates(owner._store))
    before_templates = len(owner._templates)
    answer = _count_answer()
    with patch("aetherdialect._llm_provider.LLMProvider.json", return_value=answer):
        MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
        MainExecutionOps.answer_metadata_question(
            owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
        )
    assert len(TemplateOps.store_to_templates(owner._store)) == before_store
    assert len(owner._templates) == before_templates
    assert not any(Path(artifacts).glob("**/templates*.json*"))


@pytest.mark.fast
def test_cache_hit_diagnostic(tmp_path: Path) -> None:
    artifacts = str(tmp_path)
    owner = _owner(artifacts)
    schema = _schema()
    answer = _count_answer()
    buf: list[Diagnostic] = []
    tok = set_diagnostic_collector(buf)
    try:
        with patch("aetherdialect._llm_provider.LLMProvider.json", return_value=answer):
            MainExecutionOps.answer_metadata_question(
                owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
            )
            MainExecutionOps.answer_metadata_question(
                owner, "how many tables", QuestionRoute.SCHEMA_CATALOG, schema, "master", artifacts
            )
        codes = {d.code for d in buf} | {d.code for d in drain_diagnostic_collector()}
    finally:
        reset_diagnostic_collector(tok)
    assert "meta.cache.miss" in codes
    assert "meta.cache.hit" in codes
