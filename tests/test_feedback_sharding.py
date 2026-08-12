"""Question feedback sharded out of the template store header."""

from __future__ import annotations

import gzip
import json
import os

import pytest

from aetherdialect._config import EngineConfig, PolicyConfig
from aetherdialect._constants import (
    FEEDBACK_SHARD_INDEX_KEY,
    TEMPLATE_STORE_HEADER_FILENAME,
)
from aetherdialect._contracts_base import ConfigError
from aetherdialect._contracts_core import FeedbackKind, QuestionFeedbackEntry, RejectionBucket
from aetherdialect._templates import TemplateStoreView
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import normalize_question
from aetherdialect._utils_artifacts import write_gzip_json_atomic


def _store_dir(tmp_path) -> str:
    return str(tmp_path / "intent_templates" / "spaces" / "master")


def _artifacts_dir(tmp_path) -> str:
    return str(tmp_path)


def _feedback_entry(*, ish: str = "ih_1", summary: str = "failure") -> QuestionFeedbackEntry:
    graph_id = "sg_test000000000001__abcd1234"
    return QuestionFeedbackEntry(
        summary=summary,
        buckets=(RejectionBucket.OTHER,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash=graph_id,
        intent_structural_hash=ish,
        intent_payload="{}",
        created_at="2020-01-01T00:00:00Z",
        updated_at="2020-01-01T00:00:00Z",
    )


def _distinct_partition_q_norms(n: int) -> list[str]:
    norms: list[str] = []
    seen: set[int] = set()
    for i in range(500_000):
        q = f"distinct question number {i}"
        q_norm = normalize_question(q)
        part = TemplateStoreView.question_feedback_partition_number(q_norm)
        if part in seen:
            continue
        seen.add(part)
        norms.append(q_norm)
        if len(norms) >= n:
            break
    assert len(norms) == n
    return norms


def _header_bytes(store_dir: str) -> int:
    return os.path.getsize(os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME))


def _loaded_feedback_partition_count(view: TemplateStoreView) -> int:
    return len(view._feedback_partition_cache)


@pytest.mark.not_fast
def test_header_size_independent_of_question_count(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    os.makedirs(_store_dir(tmp_path), exist_ok=True)

    store_one = TemplateOps.empty_template_store(graph_id)
    q_one = normalize_question("single question with many feedback rows")
    for i in range(PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION):
        TemplateOps.record_question_feedback(store_one, q_one, _feedback_entry(ish=f"ih_{i}"))
    TemplateOps.save_template_store(store_one)
    header_one_q = _header_bytes(store_one._store_dir)

    store_many = TemplateOps.empty_template_store(graph_id)
    for i in range(1000):
        q_norm = normalize_question(f"question number {i}")
        TemplateOps.record_question_feedback(store_many, q_norm, _feedback_entry(ish=f"ih_{i}"))
    TemplateOps.save_template_store(store_many)
    header_many_q = _header_bytes(store_many._store_dir)

    assert header_one_q < 4096
    assert header_many_q < header_one_q + 1000 * 48
    assert "question_feedback" not in json.loads(
        gzip.open(os.path.join(store_many._store_dir, TEMPLATE_STORE_HEADER_FILENAME), "rt").read()
    )


@pytest.mark.fast
def test_feedback_lookup_loads_one_shard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    os.makedirs(_store_dir(tmp_path), exist_ok=True)

    q_norms = _distinct_partition_q_norms(2)
    store = TemplateOps.empty_template_store(graph_id)
    for i, q_norm in enumerate(q_norms):
        TemplateOps.record_question_feedback(store, q_norm, _feedback_entry(ish=f"ih_{i}"))
    TemplateOps.save_template_store(store)

    loaded = TemplateOps.load_template_store(graph_id, schema=None)
    with loaded._feedback_partition_cache_lock:
        loaded._feedback_partition_cache.clear()

    rows = TemplateOps.collect_question_feedback_for_prompt(loaded, q_norms[0], graph_id)
    assert len(rows) == 1
    assert _loaded_feedback_partition_count(loaded) == 1
    part_a = loaded.feedback_shard_index[q_norms[0]]
    part_b = loaded.feedback_shard_index[q_norms[1]]
    assert part_a != part_b
    assert part_a in loaded._feedback_partition_cache
    assert part_b not in loaded._feedback_partition_cache


@pytest.mark.fast
def test_existing_header_feedback_migrated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(tmp_path / "intent_templates"))
    graph_id = "sg_test000000000001__abcd1234"
    store_dir = _store_dir(tmp_path)
    os.makedirs(store_dir, exist_ok=True)
    artifacts_dir = _artifacts_dir(tmp_path)

    q_norm = normalize_question("legacy header feedback question")
    legacy_header = {
        "format_version": 3,
        "schema_graph_id": graph_id,
        "next_id": 1,
        "partition_map": {},
        "question_feedback": {
            q_norm: [_feedback_entry(ish="ih_legacy").to_dict()],
        },
        FEEDBACK_SHARD_INDEX_KEY: {},
    }
    write_gzip_json_atomic(os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME), legacy_header, sort_keys=True)

    with pytest.raises(ConfigError, match=r"format_version"):
        TemplateOps.load_template_store(graph_id, schema=None, artifacts_dir=artifacts_dir)
