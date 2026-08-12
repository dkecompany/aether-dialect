"""Hygiene: docs/TROUBLESHOOTING.md stays aligned with live enums and catalogues."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._constants import (
    SESSION_KIND_AWAITING_INTENT_CONFIRM,
    SESSION_KIND_AWAITING_INTENT_FEEDBACK,
    SESSION_KIND_AWAITING_REUSE_CONFIRM,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_AWAITING_SQL_FEEDBACK,
    SESSION_KIND_ERROR,
    SESSION_KIND_EXECUTE,
    SESSION_KIND_IDLE,
    SESSION_KIND_META,
    SESSION_KIND_RESULT,
)
from aetherdialect._contracts_base import DiagnosticCode, SqlDiagnosticCode
from aetherdialect._contracts_core import SessionOutcome

_REPO = Path(__file__).resolve().parents[2]
_TROUBLESHOOTING = _REPO / "docs" / "TROUBLESHOOTING.md"

_AUDIT_EVENT_TYPES: tuple[str, ...] = (
    "init",
    "data_quality",
    "domain_knowledge_ingest",
    "refresh",
    "ask_begin",
    "ask_suspend",
    "ask_cancelled",
    "ask_done",
    "ask_error",
    "ask_blocked",
    "llm_call",
    "llm_turn",
    "sql_execution",
    "federation_semijoin_key_transfer",
    "write_queue_feedback_record",
    "write_queue_template_reject",
    "write_queue_template_accept",
    "write_queue_structure_proposal",
    "apply_structure",
    "clear_template_store",
    "clear_simulation_caches",
    "clear_all_learning",
    "close",
    "export_federation",
    "export_knowledge",
    "export_structure",
)

_AUDIT_DETAIL_KEYS: tuple[str, ...] = (
    "engine",
    "federation",
    "members",
    "status",
    "kept",
    "space",
    "removed",
    "existed",
    "outcome",
    "kind",
    "result_columns",
    "sources_queried",
    "message",
    "source_id",
    "phase",
    "limit_key",
    "scope",
    "task",
    "logical_model",
    "api_model",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "attempt",
    "elapsed_ms",
    "statement_hash",
    "row_count",
    "schema_hash",
    "cache_write_tokens",
    "cost_usd",
    "requests",
    "removed_files",
    "keep_structure",
    "issues_json",
    "ok",
    "issue_count",
    "path",
    "migration_tier",
    "schema_changed",
    "orphans_removed",
    "bytes_reclaimed",
    "table_edits",
    "column_edits",
)

_DIAGNOSTIC_DETAIL_KEYS: tuple[str, ...] = (
    "phase",
    "attempt",
    "reason",
    "sources_queried",
    "cap",
    "types",
    "logical_column",
    "path",
    "issue_code",
    "requests",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "cost_usd",
    "price_table_as_of",
    "unpriced_models",
)

_SESSION_KINDS: tuple[str, ...] = (
    SESSION_KIND_IDLE,
    SESSION_KIND_AWAITING_INTENT_CONFIRM,
    SESSION_KIND_AWAITING_INTENT_FEEDBACK,
    SESSION_KIND_AWAITING_REUSE_CONFIRM,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SESSION_KIND_AWAITING_SQL_FEEDBACK,
    SESSION_KIND_EXECUTE,
    SESSION_KIND_RESULT,
    SESSION_KIND_META,
    SESSION_KIND_ERROR,
)


@pytest.mark.fast
def test_troubleshooting_doc_documents_apply_structure_api() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    assert "apply_structure" in text
    assert "apply_schema_overrides" not in text
    assert "clear_persisted_overrides" not in text
    assert "write_queue_structure_proposal" in text
    assert "STRUCTURE_EDIT_SKIP" in text
    assert "STRUCTURE_NEEDS_RECONFIRMATION" in text


@pytest.mark.fast
def test_troubleshooting_doc_exists() -> None:
    assert _TROUBLESHOOTING.is_file(), "docs/TROUBLESHOOTING.md is missing"


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_session_outcome() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(member.value for member in SessionOutcome if member.value not in text)
    assert not missing, f"TROUBLESHOOTING.md missing SessionOutcome values: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_refusal_catalogue_key() -> None:
    from aetherdialect._constants_runtime import REFUSAL_CATALOGUE

    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(key for key in REFUSAL_CATALOGUE if key not in text)
    assert not missing, f"TROUBLESHOOTING.md missing REFUSAL_CATALOGUE keys: {missing}"


_DIAGNOSTIC_CODES_EXCLUDED_FROM_DOCS: frozenset[str] = frozenset(
    {
        "COMPOSE_REPAIR",
        "FALLBACK_FRESH_RESTART",
        "INTERPRET_GROUND_RETRY",
    }
)


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_diagnostic_code() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(
        code.value
        for code in DiagnosticCode
        if code.value not in text and code.value not in _DIAGNOSTIC_CODES_EXCLUDED_FROM_DOCS
    )
    assert not missing, f"TROUBLESHOOTING.md missing DiagnosticCode values: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_sql_diagnostic_code() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(code.value for code in SqlDiagnosticCode if code.value not in text)
    assert not missing, f"TROUBLESHOOTING.md missing SqlDiagnosticCode values: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_audit_event_type() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(event for event in _AUDIT_EVENT_TYPES if event not in text)
    assert not missing, f"TROUBLESHOOTING.md missing AuditEvent.event_type values: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_audit_event_detail_keys() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(key for key in _AUDIT_DETAIL_KEYS if key not in text)
    assert not missing, f"TROUBLESHOOTING.md missing AuditEvent details keys: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_diagnostic_detail_keys() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(key for key in _DIAGNOSTIC_DETAIL_KEYS if key not in text)
    assert not missing, f"TROUBLESHOOTING.md missing Diagnostic details keys: {missing}"


@pytest.mark.fast
def test_troubleshooting_doc_lists_every_session_step_kind() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(kind for kind in _SESSION_KINDS if kind not in text)
    assert not missing, f"TROUBLESHOOTING.md missing SessionStep kind values: {missing}"
