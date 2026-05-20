"""Tests for :mod:`aetherdialect._templates` (question feedback, store, demotion, fingerprints)."""

from __future__ import annotations

import hashlib
import os
from unittest.mock import patch

import pytest

from aetherdialect._config import (
    PolicyConfig,
    EngineConfig,
    SHAPE_QUESTION_INDEX_KEY,
    TEMPLATE_INTENT_KEY_INDEX_KEY,
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_UNION_FAMILY_INDEX_KEY,
)
from aetherdialect._contracts_base import SchemaGraph, SQLShape, TableMetadata
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FeedbackKind,
    NormalizedExpr,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
    Template,
    TemplateStats,
    ValueHistory,
)
from aetherdialect._templates import (
    collect_question_feedback_for_prompt,
    compute_question_feedback_penalty,
    demote_template_to_rejected,
    empty_template_store,
    join_fingerprint_from_concrete_intent,
    join_fingerprint_from_runtime_intent,
    load_template_store,
    record_question_feedback,
    save_template_store,
    summarize_failure_for_memory,
    template_is_live,
    template_partition_number,
    template_schema_refs,
    templates_to_store,
    TemplateStoreView,
)
from aetherdialect._core_utils import read_gzip_json, write_gzip_json_atomic
from aetherdialect._utils import intent_key, question_token_fingerprint_from_raw


def _minimal_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )


def _minimal_template(tid: str = "T0001") -> Template:
    ci = ConcreteIntent(
        intent_id="x",
        tables=["t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )
    return Template(
        id=tid,
        effective_structural_hash="h",
        intent_signature=ci,
        intent_key=intent_key(_minimal_intent()),
        tables_used=["t"],
        sql_param="SELECT 1",
        sql_fp="fp",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _tiny_schema() -> SchemaGraph:
    tbl = TableMetadata(name="t", columns={}, primary_key=[], foreign_keys=[], row_count=1)
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="sh1", tables={"t": tbl})


class TestRecordQuestionFeedback:
    def test_appends_and_trims_per_question(self):
        store: dict = {"question_feedback": {}, "next_id": 1}
        q = "how many"
        for i in range(PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION + 3):
            ent = QuestionFeedbackEntry(
                summary=str(i),
                buckets=(RejectionBucket.OTHER,),
                kind=FeedbackKind.VALIDATION_FAILURE,
                effective_structural_hash="h",
                intent_structural_hash=f"h{i}",
                intent_payload="{}",
                created_at=f"t{i}",
                updated_at=f"t{i}",
            )
            record_question_feedback(store, q, ent)
        assert len(store["question_feedback"][q]) == PolicyConfig.MAX_QUESTION_FEEDBACK_ENTRIES_PER_QUESTION

    def test_skips_duplicate_validation_failure_hash(self):
        store: dict = {"question_feedback": {}}
        ent = QuestionFeedbackEntry(
            summary="dup",
            buckets=(RejectionBucket.OTHER,),
            kind=FeedbackKind.VALIDATION_FAILURE,
            effective_structural_hash="h1",
            intent_structural_hash="same",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        record_question_feedback(store, "q", ent)
        record_question_feedback(store, "q", ent)
        assert len(store["question_feedback"]["q"]) == 1

    def test_intent_rejected_merges_second_bucket_same_hash(self):
        store: dict = {"question_feedback": {}}
        ent_a = QuestionFeedbackEntry(
            summary="first",
            buckets=(RejectionBucket.WRONG_AGGREGATION,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="h1",
            intent_structural_hash="same",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        ent_b = QuestionFeedbackEntry(
            summary="second",
            buckets=(RejectionBucket.MISSING_FILTER,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="h1",
            intent_structural_hash="same",
            intent_payload="{}",
            created_at="t2",
            updated_at="t2",
        )
        record_question_feedback(store, "q", ent_a)
        record_question_feedback(store, "q", ent_b)
        rows = store["question_feedback"]["q"]
        assert len(rows) == 1
        rebuilt = QuestionFeedbackEntry.from_dict(rows[0])
        assert RejectionBucket.WRONG_AGGREGATION in rebuilt.buckets
        assert RejectionBucket.MISSING_FILTER in rebuilt.buckets


class TestCollectQuestionFeedbackForPrompt:
    def test_filters_hash_and_preserves_insertion_order(self):
        store = {
            "question_feedback": {
                "q1": [
                    {
                        "summary": "first",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "keep",
                        "intent_structural_hash": "a",
                        "intent_payload": "{}",
                        "created_at": "2000-01-01T00:00:00Z",
                        "updated_at": "2000-01-01T00:00:00Z",
                    },
                    {
                        "summary": "second",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "keep",
                        "intent_structural_hash": "b",
                        "intent_payload": "{}",
                        "created_at": "2001-01-01T00:00:00Z",
                        "updated_at": "2001-01-01T00:00:00Z",
                    },
                    {
                        "summary": "wrong_hash",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "drop",
                        "intent_structural_hash": "c",
                        "intent_payload": "{}",
                        "created_at": "2002-01-01T00:00:00Z",
                        "updated_at": "2002-01-01T00:00:00Z",
                    },
                ],
            },
        }
        rows = collect_question_feedback_for_prompt(store, "q1", "keep")
        assert [r["summary"] for r in rows] == ["first", "second"]


class TestSummarizeFailureForMemory:
    def test_llm_chat_runtime_error_propagates(self):
        with (
            patch("aetherdialect._templates.llm_credentials_configured", return_value=True),
            patch("aetherdialect._templates.llm_chat", side_effect=RuntimeError("down")),
        ):
            with pytest.raises(RuntimeError, match="down"):
                summarize_failure_for_memory(
                    question="q",
                    intent=_minimal_intent(),
                    kind=FeedbackKind.VALIDATION_FAILURE,
                    schema_hash="h",
                    validator_errors=["e1"],
                )

    def test_bad_json_coerces_to_other(self):
        with (
            patch("aetherdialect._templates.llm_credentials_configured", return_value=True),
            patch("aetherdialect._templates.llm_chat", return_value="not json"),
        ):
            ent = summarize_failure_for_memory(
                question="q",
                intent=_minimal_intent(),
                kind=FeedbackKind.VALIDATION_FAILURE,
                schema_hash="h",
                validator_errors=["err"],
            )
            assert ent.buckets[0] == RejectionBucket.OTHER
            assert "err" in ent.summary

    def test_no_credentials_uses_user_reason(self):
        with patch("aetherdialect._templates.llm_credentials_configured", return_value=False):
            ent = summarize_failure_for_memory(
                question="q",
                intent=None,
                kind=FeedbackKind.INTENT_REJECTED,
                schema_hash="h",
                user_reason="  my note  ",
                validator_errors=["ignored"],
            )
            assert ent.summary == "my note"
            assert ent.buckets[0] == RejectionBucket.OTHER


class TestComputeQuestionFeedbackPenalty:
    def test_scales_and_caps(self):
        store = {
            "question_feedback": {
                "mine": [
                    {
                        "summary": "a",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "h",
                        "intent_structural_hash": "ia",
                        "intent_payload": "{}",
                        "created_at": "t",
                        "updated_at": "t",
                    },
                    {
                        "summary": "b",
                        "buckets": [RejectionBucket.OTHER.value],
                        "kind": FeedbackKind.VALIDATION_FAILURE.value,
                        "effective_structural_hash": "h",
                        "intent_structural_hash": "ib",
                        "intent_payload": "{}",
                        "created_at": "t2",
                        "updated_at": "t2",
                    },
                ],
            },
        }
        pen = compute_question_feedback_penalty(store, "mine", "h")
        assert pen == pytest.approx(2 * PolicyConfig.PEN_BY_THREE_SOURCE_UNIT)


class TestDemoteTemplateToRejected:
    @patch("aetherdialect._templates.summarize_failure_for_memory")
    def test_removes_template_and_records_feedback(self, mock_sum):
        mock_sum.return_value = QuestionFeedbackEntry(
            summary="x",
            buckets=(RejectionBucket.WRONG_AGGREGATION,),
            kind=FeedbackKind.INTENT_REJECTED,
            effective_structural_hash="sh1",
            intent_structural_hash="ih",
            intent_payload="{}",
            created_at="t",
            updated_at="t",
        )
        tmpl = _minimal_template()
        templates = {tmpl.id: tmpl}
        store = empty_template_store("sh1")
        demote_template_to_rejected(
            store,
            templates,
            tmpl,
            _tiny_schema(),
            _minimal_intent(),
            "SELECT 1",
            "q1",
            "bad template",
        )
        assert tmpl.id not in templates
        assert "q1" in store.get("question_feedback", {})


class TestJoinFingerprints:
    def test_runtime_deterministic(self):
        ri = _minimal_intent()
        a = join_fingerprint_from_runtime_intent(ri)
        b = join_fingerprint_from_runtime_intent(ri)
        assert a == b
        assert len(a) == 64

    def test_concrete_matches_runtime_skeleton(self):
        ci = ConcreteIntent(
            intent_id="i",
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert join_fingerprint_from_concrete_intent(ci) == join_fingerprint_from_runtime_intent(_minimal_intent())


class TestSchemaMigrationMapParse:
    """``schema_migration_map.json`` optional fields."""

    def test_refresh_descriptions_defaults_false(self):
        from aetherdialect._templates import _parse_schema_migration_map_payload

        m = _parse_schema_migration_map_payload({"version": 1, "action": "remap"})
        assert m.refresh_existing_descriptions_on_addition is False

    def test_refresh_descriptions_true(self):
        from aetherdialect._templates import _parse_schema_migration_map_payload

        m = _parse_schema_migration_map_payload(
            {"version": 1, "action": "remap", "refresh_existing_descriptions_on_addition": True}
        )
        assert m.refresh_existing_descriptions_on_addition is True

    def test_refresh_descriptions_rejects_non_bool(self):
        from aetherdialect._contracts_base import MigrationPendingError
        from aetherdialect._templates import _parse_schema_migration_map_payload

        with pytest.raises(MigrationPendingError, match="refresh_existing_descriptions_on_addition"):
            _parse_schema_migration_map_payload(
                {"version": 1, "action": "remap", "refresh_existing_descriptions_on_addition": "yes"}
            )


class TestTemplateIsLiveHelpers:
    def test_template_is_live_empty_refs(self):
        sg = _tiny_schema()
        refs = template_schema_refs(_minimal_template())
        ok, dead = template_is_live(refs, sg)
        assert ok is True
        assert dead == ()


class TestLoadTemplateStoreIndexes:
    def test_load_recomputes_missing_inverted_indexes(self, tmp_path, monkeypatch):
        """Partitioned stores with missing inverted indexes get rebuilt on load."""
        monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
        store_dir = str(tmp_path / "intent_templates")
        os.makedirs(store_dir, exist_ok=True)
        monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", store_dir)
        tmpl = _minimal_template()
        eff = tmpl.effective_structural_hash
        store = empty_template_store(eff)
        templates_to_store(store, {tmpl.id: tmpl})
        save_template_store(store)
        from aetherdialect._config import TEMPLATE_STORE_HEADER_FILENAME

        hdr_path = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        header = read_gzip_json(hdr_path)
        assert isinstance(header, dict)
        for k in (
            SHAPE_QUESTION_INDEX_KEY,
            TEMPLATE_INTENT_KEY_INDEX_KEY,
            TEMPLATE_UNION_FAMILY_INDEX_KEY,
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
        ):
            header.pop(k, None)
        write_gzip_json_atomic(hdr_path, header, sort_keys=True)
        out = load_template_store(eff, schema=None)
        for k in (
            SHAPE_QUESTION_INDEX_KEY,
            TEMPLATE_INTENT_KEY_INDEX_KEY,
            TEMPLATE_UNION_FAMILY_INDEX_KEY,
            TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
        ):
            assert k in out and isinstance(out[k], dict)
        fp = question_token_fingerprint_from_raw("q")
        assert fp in out[TEMPLATE_QUESTION_TOKEN_INDEX_KEY]
        assert [tmpl.id, "0"] in out[TEMPLATE_QUESTION_TOKEN_INDEX_KEY][fp]


class TestPartitionedTemplateStore:
    def test_partition_number_matches_sha256_prefix(self) -> None:
        tid = "T0042"
        assert template_partition_number(tid) == int(
            hashlib.sha256(tid.encode("utf-8")).hexdigest()[:2],
            16,
        )

    def test_load_template_store_returns_view_with_indexes(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
        sd = str(tmp_path / "intent_templates")
        os.makedirs(sd, exist_ok=True)
        monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", sd)
        tmpl = _minimal_template()
        eff = tmpl.effective_structural_hash
        v = empty_template_store(eff)
        templates_to_store(v, {tmpl.id: tmpl})
        save_template_store(v)
        loaded = load_template_store(eff, schema=None)
        assert isinstance(loaded, TemplateStoreView)
        assert loaded.partition_map
        assert isinstance(loaded[TEMPLATE_INTENT_KEY_INDEX_KEY], dict)

    def test_second_save_without_mutation_leaves_no_residual_dirty_partitions(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(PolicyConfig, "REGENERATE_TEMPLATE_STORE", False)
        sd = str(tmp_path / "intent_templates")
        os.makedirs(sd, exist_ok=True)
        monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", sd)
        tmpl = _minimal_template()
        eff = tmpl.effective_structural_hash
        v = empty_template_store(eff)
        templates_to_store(v, {tmpl.id: tmpl})
        save_template_store(v)
        assert not v.dirty_partitions()
        save_template_store(v)
        assert not v.dirty_partitions()
