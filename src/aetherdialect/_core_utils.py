"""Hashing, stable JSON, SQL cleanup, artifact manifest I/O, LLM clients, and display helpers."""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

from ._config import (
    LlmExecutionConfig,
    PolicyConfig,
)
from ._constants import (
    AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV,
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_LOCK_FILENAME,
    ARTIFACT_LOCK_POLL_INTERVAL_SECONDS,
    ARTIFACT_LOCK_TIMEOUT_SECONDS,
    ARTIFACT_MANIFEST_FILENAME,
    AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY,
    AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT,
    AZURE_OPENAI_ENV_DEPLOYMENT_MEDIUM,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    JSON_COMPACT_SEPARATORS,
    LEGACY_ARTIFACT_FILENAMES,
    LEGACY_ARTIFACT_GLOBS,
    MIGRATION_DATA_OVERLAP_MIN,
    MIN_COMPATIBLE_PACKAGE_VERSION,
    QUERY_RESULTS_HEADER,
    REPHRASE_HINT_MESSAGES,
    SQL_BIND_TOKEN_RE,
    STRUCTURAL_IDENTITY_VALUES,
    STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE,
    STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE,
    TEMPLATE_STORE_SEGMENT,
    UNBOUND_PYFORMAT_PLACEHOLDER_RE,
    USER_ERROR_PREFIX,
    USER_INVALID_INPUT_LINE,
    USER_REJECTED_RESULT_BUCKET_TIPS,
    USER_TERMINATED_LINE,
    VALID_VALUE_TYPES,
    VALUE_TYPE_NORMALIZATION,
    WRITE_QUEUE_FILENAME,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations, product
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from unittest.mock import patch

from packaging.version import InvalidVersion, Version

from ._contracts_base import (
    Diagnostic,
    EngineContext,
    MigrationTier,
    WriteQueueEvent,
)
from ._contracts_core import RuntimeIntent
from ._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
)

_DIAGNOSTIC_COLLECTOR: ContextVar[list[Diagnostic] | None] = ContextVar(
    "aetherdialect_diagnostic_collector",
    default=None,
)

_ORPHAN_DIAGNOSTICS: list[Diagnostic] = []

_DIAGNOSTIC_PRINT_LISTENER: ContextVar[Callable[[str], None] | None] = ContextVar(
    "aetherdialect_diagnostic_print_listener",
    default=None,
)

LLM_EXECUTION_CONTEXT: ContextVar[LlmExecutionConfig | None] = ContextVar(
    "aetherdialect_llm_execution",
    default=None,
)


def is_structural_param_key(key: str) -> bool:
    """Return True when *key* is a structural bind name (``s`` followed by digits)."""
    return len(key) >= 2 and key[0] == "s" and key[1:].isdigit()


def effective_explain_timeout_ms() -> int | None:
    """Statement timeout for ``EXPLAIN`` paths only. Prefers:data:`PolicyConfig.EXPLAIN_TIMEOUT_MS` when set and positive; otherwise uses :data:`PolicyConfig.STATEMENT_TIMEOUT_MS`. Returns ``None`` when neither bound is active."""
    explain_tm = PolicyConfig.EXPLAIN_TIMEOUT_MS
    if cost_cap_active(explain_tm) and explain_tm is not None:
        return int(explain_tm)
    statement_tm = PolicyConfig.STATEMENT_TIMEOUT_MS
    if cost_cap_active(statement_tm) and statement_tm is not None:
        return int(statement_tm)
    return None


def effective_llm_timeout_ms() -> int:
    """Resolved HTTP timeout for OpenAI-compatible clients and :func:`aetherdialect._llm_provider.llm_chat`. Uses:data:`PolicyConfig.LLM_TIMEOUT_MS` when positive; otherwise ``60_000`` ms."""
    tm = PolicyConfig.LLM_TIMEOUT_MS
    if cost_cap_active(tm) and tm is not None:
        return int(tm)
    return 60_000


def seed_warmup_failure_code_from_validate_sql_error(
    message: str | None,
    *,
    failure_category: str | None = None,
) -> str:
    """Map ``validate_sql`` outcome to a seed-warmup validation failure code."""
    if failure_category:
        exec_bucket = {
            "execution_explain_failed": "explain_failed",
            "execution_timeout": "explain_transient",
            "execution_cost_exceeded": "explain_failed",
            "execution_schema_error": "explain_schema",
            "execution_semantic_error": "explain_semantic",
            "execution_other_error": "explain_failed",
        }
        hit = exec_bucket.get(failure_category)
        if hit is not None:
            return hit
        if failure_category == "schema" and (message or "").strip() == "not_select":
            return "ast_validate_unsupported_construct"
        if failure_category == "other" and (message or "").strip() == "forbidden_sql":
            return "ast_validate_other"
        if failure_category == "unbound_placeholder":
            return "ast_validate_unbound_placeholder"

    if not message:
        return "ast_validate_other"
    m = message.strip()
    for tag in (
        "explain_schema",
        "explain_semantic",
        "explain_transient",
        "explain_failed",
    ):
        if m.startswith(f"[{tag}]"):
            return tag
    if m == "not_select":
        return "ast_validate_unsupported_construct"
    if m == "forbidden_sql":
        return "ast_validate_other"
    if m == "unbound_placeholder":
        return "ast_validate_unbound_placeholder"
    low = m.lower()
    if "sql structure error:" in low:
        tail = m.split("SQL structure error:", 1)[-1].strip().lower()
        if "cte" in tail or "with " in tail:
            return "ast_validate_cte_error"
        if "from" in tail and ("missing" in tail or "no from" in tail):
            return "ast_validate_missing_from_clause"
        if "column" in tail and ("not exist" in tail or "undefined" in tail or "bad" in tail):
            return "ast_validate_bad_identifier"
        if "syntax" in tail or "parse" in tail:
            return "ast_validate_pglast_syntax"
        return "ast_validate_other"
    if "syntax" in low or "parse" in low:
        return "ast_validate_pglast_syntax"
    return "ast_validate_other"


@contextmanager
def llm_execution_scope(cfg: LlmExecutionConfig) -> Iterator[None]:
    """Bind *cfg* as the active :class:`LlmExecutionConfig` for nested LLM calls."""
    tok = LLM_EXECUTION_CONTEXT.set(cfg)
    try:
        yield
    finally:
        LLM_EXECUTION_CONTEXT.reset(tok)


def diagnostic_segment() -> list[Diagnostic]:
    """Return a new collector list and bind it as the active diagnostic buffer (call :func:`reset_diagnostic_collector` when done)."""
    buf: list[Diagnostic] = []
    return buf


def set_diagnostic_collector(buf: list[Diagnostic] | None) -> Any:
    """Bind *buf* as the active diagnostic collector; returns a token for :func:`reset_diagnostic_collector`."""
    return _DIAGNOSTIC_COLLECTOR.set(buf)


def reset_diagnostic_collector(token: Any) -> None:
    """Restore the previous diagnostic collector."""
    _DIAGNOSTIC_COLLECTOR.reset(token)


def take_and_clear_orphan_diagnostics() -> tuple[Diagnostic, ...]:
    """Drain diagnostics emitted before any collector was bound (for example during construction)."""
    out = tuple(_ORPHAN_DIAGNOSTICS)
    _ORPHAN_DIAGNOSTICS.clear()
    return out


@contextmanager
def diagnostic_print_listener(fn: Callable[[str], None] | None) -> Iterator[None]:
    """Bind *fn* to receive human-readable copies of notify lines (used by ``run_interactive``)."""
    tok = _DIAGNOSTIC_PRINT_LISTENER.set(fn)
    try:
        yield
    finally:
        _DIAGNOSTIC_PRINT_LISTENER.reset(tok)


def drain_diagnostic_collector() -> tuple[Diagnostic, ...]:
    """Extract and clear diagnostics from the active collector (used when building :class:`SessionStep`)."""
    buf = _DIAGNOSTIC_COLLECTOR.get()
    if not buf:
        return ()
    out = tuple(buf)
    buf.clear()
    return out


def notify(
    message: str,
    *,
    stage: str | None = None,
    code: str = DIAGNOSTIC_CODE_ENGINE_INFO,
    level: str = "info",
    duration_ms: int | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> None:
    """Append a diagnostic to the active collector and optionally mirror the line to a print listener."""
    eff_stage = stage or "notify"
    diag = Diagnostic(
        stage=eff_stage,
        level=level,
        code=code,
        message=message,
        details=details,
        duration_ms=duration_ms,
    )
    buf = _DIAGNOSTIC_COLLECTOR.get()
    if buf is not None:
        buf.append(diag)
    else:
        _ORPHAN_DIAGNOSTICS.append(diag)
    if prev_suppress:
        return
    fn = _DIAGNOSTIC_PRINT_LISTENER.get()
    if fn is not None:
        fn(message)
    if diagnostic_debug_enabled():
        rec: dict[str, Any] = {
            "kind": "notify",
            "stage": eff_stage,
            "code": code,
            "level": level,
            "message": message,
        }
        if duration_ms is not None:
            rec["duration_ms"] = duration_ms
        print(json.dumps(rec, ensure_ascii=False), file=sys.stderr, flush=True)


_progress_depth = 0


def progress(message: str) -> None:
    """Emit a progress diagnostic when :func:`progress_enabled` is active and mirror it to a print listener."""
    if _progress_depth <= 0:
        return
    notify(message, stage="progress", code=DIAGNOSTIC_CODE_ENGINE_INFO)


@contextmanager
def progress_enabled() -> Iterator[None]:
    """Enable :func:`progress` writes for the duration of the block (supports nesting)."""
    global _progress_depth
    _progress_depth += 1
    try:
        yield
    finally:
        _progress_depth -= 1


def _running_in_jupyter() -> bool:
    """Return True when the current process runs inside a Jupyter or IPython kernel front-end."""
    return "ipykernel" in sys.modules


def echo_yes_no_answer(raw: str) -> None:
    """Echo ``Yes`` or ``No`` for *raw* input answer; emit nothing for invalid tokens or in a TTY terminal."""
    if not _running_in_jupyter():
        return
    token = raw.strip().lower()
    if token in {"y", "yes"}:
        print("Yes", flush=True)
    elif token in {"n", "no"}:
        print("No", flush=True)


def echo_user_text(raw: str) -> None:
    """Echo *raw* on its own line for Jupyter front-ends (terminal already shows what the user typed)."""
    if not _running_in_jupyter() or not raw:
        return
    print(raw, flush=True)


def _result(message: str) -> None:
    """Emit a query-_result line through :func:`notify` (mirrored to the diagnostic print listener when bound)."""
    notify(message, stage="user_result", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def error(message: str) -> None:
    """Emit an error line through :func:`notify`, prefixed with ``Error: ``."""
    notify(
        f"{USER_ERROR_PREFIX}{message}",
        stage="user_error",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="_error",
    )


def prompt(message: str) -> str:
    """Display ``message``, read one line from stdin, and return it stripped."""
    return input(message).strip()


def terminated() -> None:
    """Emit the canonical user-termination line through :func:`notify`."""
    notify(
        USER_TERMINATED_LINE,
        stage="user_terminated",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def invalid_input(detail: str | None = None) -> None:
    """Emit the canonical invalid-input line through :func:`notify`, or *detail* when provided."""
    if detail:
        notify(
            detail.strip(),
            stage="user_invalid_input",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="warn",
        )
    else:
        notify(
            USER_INVALID_INPUT_LINE,
            stage="user_invalid_input",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="warn",
        )


class InteractiveChoicePort(Protocol):
    """Bridges yes/no prompts to a session queue or stdin."""

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        ...

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Return a normalised choice or raise ``PipelineSuspended`` when the queue is empty."""
        ...


def note_interactive_turn(
    choice_port: InteractiveChoicePort | None,
    *,
    outcome: str,
    error: str | None = None,
    sql: str | None = None,
    rows: list[tuple[Any, ...]] | None = None,
    columns: tuple[str, ...] | None = None,
    rejection_bucket: str | None = None,
    intent: RuntimeIntent | None = None,
    matched_template: Any | None = None,
    template_history_index: int | None = None,
) -> None:
    """Record turn outcome on *choice_port* when it implements ``note_turn_outcome``."""
    fn = getattr(choice_port, "note_turn_outcome", None)
    if callable(fn):
        fn(
            outcome=outcome,
            error=error,
            sql=sql,
            rows=rows,
            columns=columns,
            rejection_bucket=rejection_bucket,
            intent=intent,
            matched_template=matched_template,
            template_history_index=template_history_index,
        )


prev_sink: list[str] | None = None
prev_suppress: bool = False


@contextmanager
def telemetry_capture(
    *,
    suppress_console: bool = False,
    force_diagnostic_flags: bool = False,
) -> Iterator[list[str]]:
    """Collect ``debug`` / ``pipeline_trace`` lines into a buffer."""
    global prev_sink, prev_suppress
    buf: list[str] = []
    saved_sink = prev_sink
    saved_suppress = prev_suppress
    if force_diagnostic_flags:
        diagnostic_force_enter()
    prev_sink = buf
    prev_suppress = suppress_console
    try:
        yield buf
    finally:
        prev_sink = saved_sink
        prev_suppress = saved_suppress
        if force_diagnostic_flags:
            diagnostic_force_exit()


def debug(msg: str) -> None:
    """Print ``[DEBUG]`` + *msg* when ``PolicyConfig.DEBUG`` or debug. diagnostics are on (see ``diagnostic_debug_enabled``)."""
    line = f"[DEBUG] {msg}"
    if prev_sink is not None:
        prev_sink.append(line)
    if prev_suppress:
        return
    if diagnostic_debug_enabled():
        print(line)
        print(
            json.dumps({"kind": "debug", "message": msg}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )


def _format_pipeline_trace_block(heading: str, body: str) -> str:
    """Format a pipeline-trace block without performing I/O."""
    return f"[PIPELINE_TRACE] {heading}\n{body}"


def pipeline_trace(heading: str, body: str | Callable[[], str]) -> None:
    """Emit a ``[PIPELINE_TRACE]`` block when ``PolicyConfig.DEBUG`` or diagnostic capture is active."""
    sink_on = prev_sink is not None
    console_on = not prev_suppress and diagnostic_debug_enabled()
    if not sink_on and not console_on:
        return
    resolved = body() if callable(body) else body
    block = _format_pipeline_trace_block(heading, resolved)
    if prev_sink is not None:
        prev_sink.append(block)
    if prev_suppress:
        return
    if not diagnostic_debug_enabled():
        return
    print(block)


def sha256(s: str) -> str:
    """SHA-256 hex digest of UTF-8 *s*."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _strip_fences(s: str) -> str:
    """Strip leading/trailing ``` fences and surrounding whitespace."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def canonicalize_sql(sql: str) -> str:
    """Normalize SQL whitespace, formatting, and join equality operand. order."""
    s = _strip_fences(sql).strip()
    s = s.rstrip(";").strip()
    s = re.sub(r"^EXPLAIN\s+(?:ANALYZE\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(?<![><!=])=(?![>=])", " = ", s)
    s = re.sub(r"\s+", " ", s).strip()

    def normalize_equality(m: re.Match[str]) -> str:
        left, right = m.group(1).strip(), m.group(2).strip()
        if left > right:
            left, right = right, left
        return f"{left} = {right}"

    s = re.sub(r"([^\s()><!=]+)\s*=\s*([^\s()><!=]+)", normalize_equality, s)
    return s


def stable_json(o: Any) -> str:
    """``json.dumps`` with sorted keys and minimal separators."""
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def colmap_signature(column_map: dict[str, str]) -> str:
    """SHA-256 of stable JSON for sorted ``column_map`` items."""
    return sha256(stable_json(sorted(column_map.items())))


def intent_id(d: dict[str, Any]) -> str:
    """Short intent id: first 16 hex chars of ``stable_json(d)`` hash."""
    return sha256(stable_json(d))[:16]


def structural_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint DDL-stable table payloads (kinds, columns, keys, FK. edges)."""
    return sha256(stable_json({"tables": tables_payload}))


def profiling_hash_fp(tables_payload: dict[str, Any]) -> str:
    """Fingerprint profiling-only payloads (counts, roles, top values, semantics)."""
    return sha256(stable_json({"tables": tables_payload}))


def _schema_scope_file_content_sha256(path: str | None) -> str:
    """Return a SHA-256 hex digest of the UTF-8 file at *path*, or ``""`` when the path is missing or not a readable file."""
    if path is None or not str(path).strip():
        return ""
    expanded = os.path.expanduser(str(path).strip())
    if not os.path.isfile(expanded):
        return ""
    with open(expanded, encoding="utf-8") as fh:
        return sha256(fh.read())


def scope_hash_fp(schema_context: EngineContext) -> str:
    """Fingerprint scope inputs: include mode, allow list, deny lists, and inlined DDL or notes file contents. ``EngineContext.name`` is intentionally excluded so master and named subset contexts share one graph id."""
    deny_cols = sorted(schema_context.deny_columns)
    allow_cols = sorted(schema_context.allow_columns)
    payload = {
        "allow_objects": sorted(schema_context.allow_objects),
        "deny_objects": sorted(schema_context.deny_objects),
        "deny_columns": deny_cols,
        "allow_columns": allow_cols,
        "include": schema_context.include,
        "sql_file_content_sha256": _schema_scope_file_content_sha256(schema_context.sql_file),
        "notes_file_content_sha256": _schema_scope_file_content_sha256(schema_context.notes_file),
    }
    return sha256(stable_json(payload))


def effective_structural_hash_fp(structural_hash: str, scope_hash: str) -> str:
    """Combine structural and scope fingerprints into the template-store key."""
    return sha256(structural_hash + "|" + scope_hash)


def schema_hash_fp(tables_dict: dict[str, Any]) -> str:
    """Legacy SHA-256 of ``{"tables": tables_dict}`` JSON. Used by cache diagnostics and tests that pass arbitrary table-shaped dicts."""
    return sha256(stable_json({"tables": tables_dict}))


def normalize_question(q: str) -> str:
    """Lowercase and clean *q*; restore single-quoted spans to original. case."""
    q = q.strip()
    q = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", q)

    quoted_values = []

    def preserve_quoted(m: re.Match[str]) -> str:
        quoted_values.append(m.group(1))
        return f"__QUOTED_{len(quoted_values) - 1}__"

    q = re.sub(r"'([^']*)'", preserve_quoted, q)

    q = q.lower()
    q = re.sub(r"[^a-z0-9\s_:/\-\.,\?]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    for i, val in enumerate(quoted_values):
        q = q.replace(f"__quoted_{i}__", f"'{val}'")

    return q


def _extract_first_json_object(s: str) -> str | None:
    """First top-level ``{...}`` substring via brace depth (after fence. strip)."""
    s = _strip_fences(s)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def safe_json_loads(s: str) -> Any | None:
    """Try ``json.loads``; on failure, parse first ``{...}`` fragment."""
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    frag = _extract_first_json_object(s)
    if frag:
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None


def _reconfigure_console_streams_to_utf8() -> None:
    """Force ``sys.stdout`` / ``sys.stderr`` to UTF-8 with replacement on undefined glyphs."""
    for _stream in (sys.stdout, sys.stderr):
        if _stream is None:
            continue
        if getattr(_stream, "encoding", "").lower() == "utf-8":
            continue
        reconfigure = getattr(_stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_reconfigure_console_streams_to_utf8()


def engine_connect_likely_transient(exc: BaseException) -> bool:
    """Heuristic for cold-start or transport failures suitable for :class:`DatabasePingFailed`."""
    s = str(exc).lower()
    needles = (
        "timeout",
        "timed out",
        "temporarily unavailable",
        "503",
        "502",
        "429",
        "connection refused",
        "connection reset",
        "eof",
        "broken pipe",
        "network",
        "unreachable",
        "name or service not known",
        "could not translate host name",
        "warehouse",
        "cold-start",
        "cold start",
    )
    if any(n in s for n in needles):
        return True
    if isinstance(exc, OSError):
        errn = getattr(exc, "errno", None)
        if errn in {10060, 10061, 11001, 11002, 111, 113, 115, 116}:
            return True
    return False


def _normalize_sql_operator_spaces(sql: str) -> str:
    """Merge split operators: ``> =``, ``< =``, ``! =`` → ``>=``, ``<=``, ``!=``."""
    if not sql or not sql.strip():
        return sql
    s = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return s


def normalize_sql(sql: str) -> str:
    """After ``canonicalize_sql``, append default ``ASC`` where ORDER. BY. lacks direction."""
    s = _normalize_sql_operator_spaces(canonicalize_sql(sql))
    if not s:
        return s

    s_upper = s.upper()
    order_by_pos = s_upper.find("ORDER BY")
    if order_by_pos != -1:
        before_order = s[: order_by_pos + 8]
        after_order = s[order_by_pos + 8 :]

        limit_pos = after_order.upper().find("LIMIT")
        if limit_pos != -1:
            order_clause = after_order[:limit_pos].strip()
            rest = after_order[limit_pos:]
        else:
            order_clause = after_order.strip()
            rest = ""

        normalized_items = []
        for item in order_clause.split(","):
            item = item.strip()
            if not item:
                continue
            item_upper = item.upper()
            if item_upper.endswith(" ASC") or item_upper.endswith(" DESC"):
                normalized_items.append(item)
            else:
                normalized_items.append(f"{item} ASC")

        if normalized_items:
            s = f"{before_order} {', '.join(normalized_items)}"
            if rest:
                s = f"{s} {rest}"

    return s


def normalize_array_contains_param_value(value: Any) -> Any:
    """Strip whitespace and redundant surrounding quotes from array. ``contains`` operands. Keeps bind values free of decorative quotes; SQL generation also normalizes stored array elements per dialect so membership stays stable across data encodings."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.strip("%")
    return s


def _escape_sql_single_quoted_literal(value: str) -> str:
    """Escape a UTF-8 string for safe use inside single-quoted SQL literals."""
    return value.replace("'", "''")


def substitute_params(sql_param: str, params: dict[str, Any]) -> str:
    """Replace ``:key``, ``@key``, and ``$key`` placeholders with formatted parameter values."""
    _result = sql_param
    for key in sorted(params.keys(), key=lambda k: -len(k)):
        val = params[key]
        if not key:
            continue
        if isinstance(val, list):
            formatted_items = []
            for item in val:
                if isinstance(item, str):
                    formatted_items.append(f"'{_escape_sql_single_quoted_literal(item)}'")
                else:
                    formatted_items.append(str(item))
            formatted = ", ".join(formatted_items)
        elif isinstance(val, bool):
            formatted = "TRUE" if val else "FALSE"
        elif isinstance(val, str):
            if not val.strip():
                raise ValueError("unbound_placeholder")
            if val.startswith("'") and val.endswith("'") and "','" in val:
                formatted = val
            elif re.match(r"^-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)*$", val):
                formatted = val
            else:
                formatted = f"'{_escape_sql_single_quoted_literal(val)}'"
        else:
            formatted = str(val)
        for prefix in (":", "$", "@"):
            _result = _result.replace(f"{prefix}{key}", formatted)
    if SQL_BIND_TOKEN_RE.search(_result) or UNBOUND_PYFORMAT_PLACEHOLDER_RE.search(_result):
        raise ValueError("unbound_placeholder")
    return _result


def _format_scalar_for_structural_sql_inline(val: Any) -> str:
    """Format a single bind value for structural placeholder inlining."""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, str):
        if val.startswith("'") and val.endswith("'") and "','" in val:
            return val
        if STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE.match(val):
            return val
        return f"'{_escape_sql_single_quoted_literal(val)}'"
    return str(val)


def reduce_structural_sql_placeholders(
    sql_param: str,
    params: dict[str, Any],
    structural_defaults: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Inline ``:sN`` placeholders when values match defaults or identity structural values. Returns the reduced SQL string and the parameter map with structural keys removed."""
    sd = structural_defaults or {}
    keys_in_sql = sorted(
        set(STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE.findall(sql_param)),
        key=lambda k: (-len(k), k),
    )
    reduced = sql_param
    removed: set[str] = set()
    for sk in keys_in_sql:
        if sk not in params:
            continue
        val = params[sk]
        if sk in sd:
            if sd[sk] != val:
                continue
        elif val not in STRUCTURAL_IDENTITY_VALUES:
            continue
        lit = _format_scalar_for_structural_sql_inline(val)
        reduced = re.sub(rf":{re.escape(sk)}\b", lit, reduced)
        removed.add(sk)
    remaining = {k: v for k, v in params.items() if k not in removed}
    return reduced, remaining


def _format_cell(v: object) -> str:
    """String for one _result cell: ``NULL``, decimals, or ``str(v)``."""
    if v is None:
        return "NULL"
    if isinstance(v, Decimal):
        return f"{v:f}" if v == v.to_integral_value() else f"{v}"
    if isinstance(v, str):
        return v
    return str(v)


def print_query_result(
    rows: list[tuple[Any, ...]],
    sql: str,
    *,
    headers: list[str] | None = None,
) -> None:
    """Emit SQL and scalar answer or up to five aligned rows through. :func:`notify`."""
    out_lines: list[str] = [f"\n{QUERY_RESULTS_HEADER}\n", f"SQL:\n  {sql}\n"]
    if len(rows) == 1 and len(rows[0]) == 1:
        val = rows[0][0]
        out_lines.append(f"Answer: {_format_cell(val)}\n")
    else:
        sample = rows[:5]
        formatted = [[_format_cell(v) for v in row] for row in sample]
        num_cols = max(len(r) for r in formatted) if formatted else 0
        col_headers = (headers or [])[:num_cols]
        while len(col_headers) < num_cols:
            col_headers.append(f"col{len(col_headers) + 1}")
        widths = [len(h) for h in col_headers]
        for row in formatted:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(col_headers))
        sep_line = "  ".join("-" * widths[i] for i in range(num_cols))
        out_lines.append(f"  {header_line}")
        out_lines.append(f"  {sep_line}")
        for row in formatted:
            line = "  ".join((row[i] if i < len(row) else "").ljust(widths[i]) for i in range(num_cols))
            out_lines.append(f"  {line}")
        if len(rows) > 5:
            out_lines.append(f"  ... ({len(rows)} total rows)")
    notify(
        "\n".join(out_lines),
        stage="query_result",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def dataframe_to_row_tuples(df: Any) -> list[tuple[Any, ...]]:
    """Convert a pandas ``DataFrame`` to plain row tuples for :func:`print_query_result`."""
    if df is None:
        return []
    return [tuple(row) for row in df.values]


def interactive_yes_no(
    stage: str,
    prompt: str,
    options: list[str],
    silent_no: bool = False,
    *,
    choice_port: InteractiveChoicePort | None = None,
) -> str | None:
    """Resolve a yes/no prompt via an optional session port or stdin."""
    if choice_port is not None:
        return choice_port.take_yes_no(stage, prompt, options, silent_no)
    return ask_user_choice(prompt, options, silent_no)


def ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
    """Interactive ``input()`` for yes/no style choices (y/n/yes/no)."""
    options_display = "/".join(options)
    notify(
        f"{prompt} ({options_display}): ",
        stage="interactive_choice",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )
    try:
        user_input = input().strip()
    except (EOFError, KeyboardInterrupt):
        terminated()
        return None

    if not user_input or not user_input.strip():
        invalid_input()
        return None

    normalized = user_input.lower()
    if normalized in ("y", "yes"):
        notify(
            "Yes",
            stage="interactive_choice",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return "y"
    elif normalized in ("n", "no"):
        notify(
            "No",
            stage="interactive_choice",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        if not silent_no:
            terminated()
        return "n"

    invalid_input()
    return None


class RephraseHint(Enum):
    """User-facing rephrase hint categories printed when the pipeline cannot continue. Each value maps to a short, non-technical, suggestive message intended to help the user produce a better question without exposing internal validation output."""

    INTENT_PARSE_FAILED = "intent_parse_failed"
    SCHEMA_INVALID_DECLINED = "schema_invalid_declined"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    USER_REJECTED_INTENT = "user_rejected_intent"
    USER_REJECTED_RESULT = "user_rejected_result"
    RESTRICTED_QUESTION = "restricted_question"
    VAGUE_QUESTION = "vague_question"


def print_rephrase_hint(
    reason: RephraseHint,
    *,
    rejection_bucket: str | None = None,
) -> None:
    """Print a tiered, suggestive rephrase hint for *reason*. Uses a fixed catalogue of short non-technical messages; never exposes validation logs or repair-loop internals to the user."""
    if rejection_bucket and reason in (
        RephraseHint.USER_REJECTED_RESULT,
        RephraseHint.USER_REJECTED_INTENT,
    ):
        tip = USER_REJECTED_RESULT_BUCKET_TIPS.get(
            rejection_bucket,
            USER_REJECTED_RESULT_BUCKET_TIPS["OTHER"],
        )
        notify(
            f"\nPlease retry or rephrase your question.\n{tip}",
            stage="rephrase_hint",
            code=DIAGNOSTIC_CODE_ENGINE_INFO,
            level="info",
        )
        return
    notify(
        f"\n{REPHRASE_HINT_MESSAGES[reason.value]}",
        stage="rephrase_hint",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def print_info(title: str, items: dict[str, Any] | None = None, footer: str | None = None) -> None:
    """Emit *title*, optional indented *items*, and optional *footer* through :func:`notify`."""
    lines: list[str] = [f"\n{title}"]
    if items:
        for key, val in items.items():
            if isinstance(val, list | tuple | set):
                val = ", ".join(str(v) for v in val)
            lines.append(f"  {key}: {val}")
    if footer:
        lines.append(f"\n{footer}")
    notify(
        "\n".join(lines),
        stage="print_info",
        code=DIAGNOSTIC_CODE_ENGINE_INFO,
        level="info",
    )


def normalize_op(op: str) -> str:
    """Lowercase/whitespace-trim *op*; map ``==``, ``gte``, etc. to SQL. ops."""
    op_lower = re.sub(r"\s+", " ", op.lower().strip())
    mapping = {
        "==": "=",
        "<>": "!=",
        "ne": "!=",
        "eq": "=",
        "gt": ">",
        "lt": "<",
        "ge": ">=",
        "le": "<=",
        "gte": ">=",
        "lte": "<=",
    }
    return mapping.get(op_lower, op_lower)


def read_gzip_json(path: str) -> Any:
    """Load a JSON value from a UTF-8 document stored as gzip."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def write_gzip_json_atomic(path: str, obj: Any, *, sort_keys: bool) -> None:
    """Serialize ``obj`` to compact UTF-8 JSON, gzip it, and replace. ``path`` atomically."""
    raw = json.dumps(obj, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=sort_keys).encode("utf-8")
    compressed = gzip.compress(raw)
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    lock_path = abs_path + ".__write.lock"
    with _file_lock(lock_path, timeout=ARTIFACT_LOCK_TIMEOUT_SECONDS):
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json.gz", dir=directory)
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(compressed)
            os.replace(tmp_path, abs_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


_LOCK_REENTRY = threading.local()


def _get_reentry_map() -> dict[str, int]:
    m = getattr(_LOCK_REENTRY, "map", None)
    if m is None:
        m = {}
        _LOCK_REENTRY.map = m
    return m


@contextmanager
def _file_lock(lock_path: str, *, timeout: float) -> Iterator[None]:
    """Acquire an exclusive OS-level lock on ``lock_path`` for the duration of the context. Blocks up to ``timeout`` seconds, raising ``TimeoutError`` if the lock cannot be acquired. Works on both POSIX (``fcntl.flock``) and Windows (``msvcrt.locking``) without external dependencies."""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        deadline = time.monotonic() + max(timeout, 0.0)
        if sys.platform == "win32":
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire artifact lock {lock_path!r} within {timeout:.1f}s",
                        ) from None
                    time.sleep(ARTIFACT_LOCK_POLL_INTERVAL_SECONDS)
            try:
                yield
            finally:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire artifact lock {lock_path!r} within {timeout:.1f}s",
                        ) from None
                    time.sleep(ARTIFACT_LOCK_POLL_INTERVAL_SECONDS)
            try:
                yield
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        fh.close()


@contextmanager
def artifact_lock(
    artifacts_dir: str,
    *,
    timeout: float = ARTIFACT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Reentrant per-``artifacts_dir`` lock covering load, mutate, and save sequences for template learning. The lock file path joins *artifacts_dir* with :data:`ARTIFACT_LOCK_FILENAME` from ``aetherdialect._config``. Nested ``with artifact_lock`` blocks on the same directory bump a per-thread refcount without deadlocking. Cross-thread and cross- process callers serialize at the OS level."""
    abs_dir = os.path.abspath(artifacts_dir)
    os.makedirs(abs_dir, exist_ok=True)
    lock_path = os.path.join(abs_dir, ARTIFACT_LOCK_FILENAME)
    key = os.path.normcase(lock_path)
    reentry = _get_reentry_map()
    depth = reentry.get(key, 0)
    if depth > 0:
        reentry[key] = depth + 1
        try:
            yield
        finally:
            reentry[key] -= 1
            if reentry[key] <= 0:
                reentry.pop(key, None)
        return
    reentry[key] = 1
    try:
        with _file_lock(lock_path, timeout=timeout):
            yield
    finally:
        reentry[key] -= 1
        if reentry[key] <= 0:
            reentry.pop(key, None)


def _artifact_package_version_string() -> str:
    try:
        return version("aetherdialect")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _manifest_path(artifacts_dir: str) -> str:
    """Return the absolute path to ``artifact_manifest.json`` under ``artifacts_dir``."""
    return os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)


@dataclass(frozen=True, slots=True)
class _ArtifactManifest:
    """Typed view of ``artifact_manifest.json`` fields used by migration checks."""

    artifact_format_version: int = 0
    created_with_package_version: str = ""
    min_compatible_package_version: str = ""
    last_action: str = ""
    last_action_at: str = ""
    structural_hash: str = ""
    profiling_hash: str = ""
    scope_hash: str = ""
    effective_structural_hash: str = ""
    schema_graph_id: str = ""
    notes_hash: str = ""
    semantic_edges_hash: str = ""
    last_migration_tier: str = ""
    last_migration_at: str = ""


def read_artifact_manifest(artifacts_dir: str) -> _ArtifactManifest | None:
    """Load artifact manifest JSON if present."""
    path = _manifest_path(artifacts_dir)
    with artifact_lock(artifacts_dir):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            ver = int(data.get("artifact_format_version", 0) or 0)
        except (TypeError, ValueError):
            ver = 0
        return _ArtifactManifest(
            artifact_format_version=ver,
            created_with_package_version=str(data.get("created_with_package_version", "") or ""),
            min_compatible_package_version=str(data.get("min_compatible_package_version", "") or ""),
            last_action=str(data.get("last_action", "") or ""),
            last_action_at=str(data.get("last_action_at", "") or ""),
            structural_hash=str(data.get("structural_hash", "") or ""),
            profiling_hash=str(data.get("profiling_hash", "") or ""),
            scope_hash=str(data.get("scope_hash", "") or ""),
            effective_structural_hash=str(data.get("effective_structural_hash", "") or ""),
            schema_graph_id=str(data.get("schema_graph_id", "") or ""),
            notes_hash=str(data.get("notes_hash", "") or ""),
            semantic_edges_hash=str(data.get("semantic_edges_hash", "") or ""),
            last_migration_tier=str(data.get("last_migration_tier", "") or ""),
            last_migration_at=str(data.get("last_migration_at", "") or ""),
        )


def write_artifact_manifest(
    artifacts_dir: str,
    *,
    structural_hash: str = "",
    profiling_hash: str = "",
    scope_hash: str = "",
    effective_structural_hash: str = "",
    schema_graph_id: str = "",
    notes_hash: str = "",
    semantic_edges_hash: str = "",
    last_migration_tier: str = "",
    last_migration_at: str | None = None,
    last_action: str = "compat_wipe",
    last_corruption_at: str = "",
) -> None:
    """Write manifest with format version, package version, optional. hashes, and last action. Persists atomically via a temporary file in *artifacts_dir* followed by ``os.replace``."""
    os.makedirs(artifacts_dir, exist_ok=True)
    path = _manifest_path(artifacts_dir)
    mig_at = last_migration_at if last_migration_at is not None else ""
    if last_migration_tier and not mig_at:
        mig_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_with_package_version": _artifact_package_version_string(),
        "min_compatible_package_version": MIN_COMPATIBLE_PACKAGE_VERSION,
        "last_action": last_action,
        "last_action_at": datetime.now(timezone.utc).isoformat(),
        "structural_hash": structural_hash,
        "profiling_hash": profiling_hash,
        "scope_hash": scope_hash,
        "effective_structural_hash": effective_structural_hash,
        "schema_graph_id": schema_graph_id,
        "notes_hash": notes_hash,
        "semantic_edges_hash": semantic_edges_hash,
        "last_migration_tier": last_migration_tier,
        "last_migration_at": mig_at,
        "last_corruption_at": last_corruption_at or "",
    }
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json.tmp",
            prefix=".artifact_manifest_",
            dir=artifacts_dir,
            delete=False,
        ) as tf:
            tmp_path = tf.name
            json.dump(payload, tf, ensure_ascii=False, indent=2)
        assert tmp_path is not None
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    debug(f"[core_utils.write_artifact_manifest] path={path}")


def emit_write_queue_event(artifacts_dir: str, event: WriteQueueEvent) -> None:
    """Append one JSON line representing a deferred writer event to the artifact write queue."""
    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    obj = {
        "kind": event.kind,
        "schema_graph_id": event.schema_graph_id,
        "schema_hash": event.schema_hash,
        "produced_at": event.produced_at,
        "payload": [list(pair) for pair in event.payload],
    }
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    with artifact_lock(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def decode_write_queue_event(obj: dict[str, Any]) -> WriteQueueEvent | None:
    """Parse one write-queue JSON object into a :class:`WriteQueueEvent`, or return ``None`` when invalid."""
    kinds = {
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    }
    kind = str(obj.get("kind") or "")
    if kind not in kinds:
        return None
    schema_graph_id = str(obj.get("schema_graph_id") or "")
    schema_hash = str(obj.get("schema_hash") or "")
    produced_at = str(obj.get("produced_at") or "")
    if not schema_graph_id:
        return None
    raw_pl = obj.get("payload")
    if not isinstance(raw_pl, list):
        return None
    pairs: list[tuple[str, str]] = []
    for row in raw_pl:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        pairs.append((str(row[0]), str(row[1])))
    write_kind = cast(
        Literal[
            "template_accept",
            "template_reject",
            "paraphrase_emit",
            "override_proposal",
            "feedback_record",
        ],
        kind,
    )
    return WriteQueueEvent(
        kind=write_kind,
        schema_graph_id=schema_graph_id,
        schema_hash=schema_hash,
        produced_at=produced_at,
        payload=tuple(pairs),
    )


def wipe_filenames(artifacts_dir: str, names: tuple[str, ...]) -> int:
    """Remove named files directly under *artifacts_dir*; return count removed."""
    removed = 0
    for name in names:
        fp = os.path.join(artifacts_dir, name)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    return removed


def wipe_globs(artifacts_dir: str, patterns: tuple[str, ...]) -> int:
    """Remove files matching glob patterns relative to *artifacts_dir*; return count removed."""
    removed = 0
    for pattern in patterns:
        for fp in glob.glob(os.path.join(artifacts_dir, pattern)):
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1
    return removed


def wipe_versioned_artifacts(artifacts_dir: str) -> None:
    """Remove on-disk template and simulation cache files under *artifacts_dir*."""
    wipe_filenames(artifacts_dir, LEGACY_ARTIFACT_FILENAMES)
    wipe_globs(artifacts_dir, LEGACY_ARTIFACT_GLOBS)
    partitioned = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    if os.path.isdir(partitioned):
        shutil.rmtree(partitioned, ignore_errors=True)


def detect_legacy_artifacts(artifacts_dir: str) -> list[str]:
    """Return artifact filenames suggesting a pre-manifest install. populated this directory. A directory is considered "legacy" when at least one versioned artifact (schema graph snapshot, template store, qsim run, seed warmup cache) is present *but* no ``artifact_manifest.json`` exists alongside it. Such artifacts were produced by an earlier release of this package whose on-disk format predates the migration manifest, and they cannot be safely loaded by the current code path."""
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return []
    if os.path.isfile(os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)):
        return []
    found: set[str] = set()
    for name in LEGACY_ARTIFACT_FILENAMES:
        if os.path.isfile(os.path.join(artifacts_dir, name)):
            found.add(name)
    for pattern in LEGACY_ARTIFACT_GLOBS:
        for fp in glob.glob(os.path.join(artifacts_dir, pattern)):
            if os.path.isfile(fp):
                found.add(os.path.basename(fp))
    return sorted(found)


def _fk_edge_key(fk: FKEdge) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    return (fk.src_table, tuple(fk.src_cols), fk.dst_table, tuple(fk.dst_cols))


def _all_fk_multiset(
    sg: SchemaGraph,
    *,
    catalog_only: bool = False,
) -> Counter[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    ctr: Counter[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = Counter()
    for tm in sg.tables.values():
        for fk in tm.foreign_keys:
            if catalog_only and fk.inference_tag is not None:
                continue
            ctr[_fk_edge_key(fk)] += 1
    return ctr


def _map_fk_key_full(
    key: tuple[str, tuple[str, ...], str, tuple[str, ...]],
    tmap: dict[str, str],
    colmap: dict[str, dict[str, str]],
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    st, sc, dt, dc = key
    nst = tmap.get(st, st)
    ndt = tmap.get(dt, dt)
    sm = colmap.get(st, {})
    dm = colmap.get(dt, {})
    nsc = tuple(sm.get(c, c) for c in sc)
    ndc = tuple(dm.get(c, c) for c in dc)
    return (nst, nsc, ndt, ndc)


def _anon_col_sig(col: ColumnMetadata) -> tuple[Any, ...]:
    fk = col.fk_target
    return (
        (col.data_type or "").lower().strip(),
        bool(col.is_primary_key),
        bool(col.is_foreign_key),
        (fk[0], fk[1]) if fk else None,
    )


def _table_anon_sig(tm: TableMetadata) -> tuple[Any, ...]:
    cmul = tuple(sorted(Counter(_anon_col_sig(c) for c in tm.columns.values()).items()))
    return (tm.kind, len(tm.columns), tuple(sorted(tm.primary_key)), cmul)


def _prof_jaccard_sets(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u


def _col_value_overlap_frozen(col: ColumnMetadata) -> frozenset[str]:
    vals = col.value_overlap_sample or []
    cleaned = {str(v).strip() for v in vals if v is not None and str(v).strip() != ""}
    cap = PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT
    return frozenset(sorted(cleaned)[:cap])


def _profiling_value_overlap(older: SchemaGraph, newer: SchemaGraph) -> float:
    """Aggregate Jaccard overlap of value-overlap samples on shared ``(table, column)`` keys."""
    inter = 0
    union = 0
    for t in older.tables:
        if t not in newer.tables:
            continue
        ot = older.tables[t]
        nt = newer.tables[t]
        for c in ot.columns:
            if c not in nt.columns:
                continue
            a = _col_value_overlap_frozen(ot.columns[c])
            b = _col_value_overlap_frozen(nt.columns[c])
            u = len(a | b)
            if u == 0:
                continue
            union += u
            inter += len(a & b)
    if union == 0:
        return 1.0
    return inter / union


def _match_columns_between_tables(old_t: TableMetadata, new_t: TableMetadata) -> dict[str, str] | None:
    ocols = list(old_t.columns.values())
    ncols = list(new_t.columns.values())
    if len(ocols) != len(ncols):
        return None
    if Counter(_anon_col_sig(c) for c in ocols) != Counter(_anon_col_sig(c) for c in ncols):
        return None
    by_sig_o: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    by_sig_n: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for c in ocols:
        by_sig_o[_anon_col_sig(c)].append(c.name)
    for c in ncols:
        by_sig_n[_anon_col_sig(c)].append(c.name)
    mapping: dict[str, str] = {}
    for sig, onames in by_sig_o.items():
        nnames = list(by_sig_n.get(sig, []))
        if len(onames) != len(nnames):
            return None
        if len(onames) == 1:
            mapping[onames[0]] = nnames[0]
            continue
        used: set[str] = set()
        for ocn in onames:
            ocol = old_t.columns[ocn]
            best: str | None = None
            best_score = -1.0
            for ncn in nnames:
                if ncn in used:
                    continue
                ncol = new_t.columns[ncn]
                sc = _prof_jaccard_sets(_col_value_overlap_frozen(ocol), _col_value_overlap_frozen(ncol))
                if sc > best_score:
                    best_score = sc
                    best = ncn
            if best is None or best_score < MIGRATION_DATA_OVERLAP_MIN:
                return None
            mapping[ocn] = best
            used.add(best)
    return mapping


def _enumerate_tmap_candidates(old: SchemaGraph, new: SchemaGraph) -> Iterator[dict[str, str]]:
    olds = list(old.tables.values())
    news = list(new.tables.values())
    if len(olds) != len(news):
        return
    ob: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    nb: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for t in olds:
        ob[_table_anon_sig(t)].append(t.name)
    for t in news:
        nb[_table_anon_sig(t)].append(t.name)
    if sorted(ob.keys()) != sorted(nb.keys()):
        return
    sigs = sorted(ob.keys())
    bucket_iters: list[list[dict[str, str]]] = []
    for sig in sigs:
        o_names = sorted(ob[sig])
        n_names = sorted(nb[sig])
        if len(o_names) != len(n_names):
            return
        if len(o_names) > 6:
            return
        bucket_iters.append([dict(zip(o_names, perm, strict=True)) for perm in permutations(n_names)])
    for parts in product(*bucket_iters):
        merged: dict[str, str] = {}
        for part in parts:
            merged.update(part)
        yield merged


def _fk_maps_consistent(
    old: SchemaGraph,
    new: SchemaGraph,
    tmap: dict[str, str],
    colmap: dict[str, dict[str, str]],
) -> bool:
    old_ctr = _all_fk_multiset(old, catalog_only=True)
    new_ctr = _all_fk_multiset(new, catalog_only=True)
    mapped: Counter[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = Counter()
    for k, v in old_ctr.items():
        mapped[_map_fk_key_full(k, tmap, colmap)] += v
    return mapped == new_ctr


def try_rename_migration_plan(
    old: SchemaGraph,
    new: SchemaGraph,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]] | None:
    """Return ``(renamed_tables, renamed_columns)`` when *old* maps to *new* by renames only."""
    for tmap in _enumerate_tmap_candidates(old, new):
        colmap: dict[str, dict[str, str]] = {}
        col_renames: list[tuple[str, str, str]] = []
        ok = True
        for otn in sorted(tmap.keys()):
            ntn = tmap[otn]
            m = _match_columns_between_tables(old.tables[otn], new.tables[ntn])
            if m is None:
                ok = False
                break
            colmap[otn] = m
            for oc, nc in m.items():
                if oc != nc:
                    col_renames.append((otn, oc, nc))
        if not ok:
            continue
        if not _fk_maps_consistent(old, new, tmap, colmap):
            continue
        rtuples = tuple(sorted((o, n) for o, n in tmap.items() if o != n))
        ctuples = tuple(sorted(col_renames))
        if not rtuples and not ctuples:
            continue
        return (rtuples, ctuples)
    return None


def _manifest_matches_schema(manifest: _ArtifactManifest, schema: SchemaGraph) -> bool:
    if manifest.schema_graph_id and schema.schema_graph_id:
        if manifest.schema_graph_id != schema.schema_graph_id:
            return False
    return (
        manifest.structural_hash == schema.structural_hash
        and manifest.profiling_hash == schema.profiling_hash
        and manifest.scope_hash == schema.scope_hash
        and manifest.effective_structural_hash == schema.effective_structural_hash
        and (manifest.notes_hash or "") == (schema.notes_hash or "")
        and (manifest.semantic_edges_hash or "") == (schema.semantic_edges_hash or "")
    )


def _schema_diff_implies_remap(schema_diff: Any) -> bool:
    """True when a non-empty structural diff carries rename signals (tables or columns)."""
    if schema_diff is None:
        return False
    if getattr(schema_diff, "is_empty", True):
        return False
    impl = getattr(schema_diff, "implies_rename_remapping", None)
    return bool(impl()) if callable(impl) else False


def artifact_manifest_incompatible_with_package(manifest: _ArtifactManifest | None) -> bool:
    """Return True when the manifest requires a newer package or unknown artifact format."""
    if manifest is None:
        return False
    fmt = manifest.artifact_format_version
    if fmt not in (0, ARTIFACT_FORMAT_VERSION):
        return True
    min_cv = (manifest.min_compatible_package_version or "").strip()
    if not min_cv:
        return False
    try:
        return Version(_artifact_package_version_string()) < Version(min_cv)
    except (InvalidVersion, TypeError, ValueError):
        return True


def classify_migration_tier(
    manifest: _ArtifactManifest | None,
    schema: SchemaGraph,
    *,
    previous_schema: SchemaGraph | None = None,
    schema_diff: Any | None = None,
) -> MigrationTier:
    """Compare stored manifest fingerprints to the live schema graph."""
    if manifest is None:
        return MigrationTier.NO_CHANGE
    man_id = str(manifest.schema_graph_id or "")
    live_id = str(schema.schema_graph_id or "")
    if man_id and live_id and man_id == live_id and _manifest_matches_schema(manifest, schema):
        return MigrationTier.NO_CHANGE
    if not manifest.effective_structural_hash and not man_id:
        return MigrationTier.NO_CHANGE
    if _manifest_matches_schema(manifest, schema):
        return MigrationTier.NO_CHANGE
    if artifact_manifest_incompatible_with_package(manifest):
        return MigrationTier.DESTRUCTIVE
    same_effective = manifest.effective_structural_hash == schema.effective_structural_hash
    if same_effective:
        if (manifest.notes_hash or "") != (schema.notes_hash or ""):
            return MigrationTier.SOFT_REFRESH
        if (manifest.semantic_edges_hash or "") != (schema.semantic_edges_hash or ""):
            return MigrationTier.SOFT_REFRESH
        if manifest.profiling_hash != schema.profiling_hash:
            if previous_schema is None:
                return MigrationTier.DESTRUCTIVE
            if _profiling_value_overlap(previous_schema, schema) >= MIGRATION_DATA_OVERLAP_MIN:
                return MigrationTier.SOFT_REFRESH
            return MigrationTier.DESTRUCTIVE
        return MigrationTier.SOFT_REFRESH
    if (
        previous_schema is not None
        and manifest.scope_hash == schema.scope_hash
        and (try_rename_migration_plan(previous_schema, schema) is not None or _schema_diff_implies_remap(schema_diff))
    ):
        return MigrationTier.REMAP
    return MigrationTier.DESTRUCTIVE


@dataclass
class StepResult:
    """Mutable capture of pipeline outputs, metrics, and logs for one scenario run."""

    scenario_id: str
    question: str
    status: str = "unknown"
    intent: RuntimeIntent | None = None
    sql: str | None = None
    rows: list[tuple[Any, ...]] | None = None
    confidence: float | None = None
    reuse_type: str | None = None
    template_id: str | None = None
    validation_failed: bool = False
    feedback: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    captured_logs: list[str] = field(default_factory=list)
    semantic_warnings: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    reject_reason_actual: str | None = None
    classified_category: str | None = None
    classified_reason: str | None = None
    generation_path: str | None = None
    pending_feedback: Any | None = None
    diagnostics: tuple[Any, ...] = ()
    kind: str | None = None


def format_failure_trace(step: StepResult | list[StepResult] | object) -> str:
    """Format a step result or list of results into a diagnostic string for results.txt."""
    if isinstance(step, list):
        parts = [format_failure_trace(s) for s in step]
        return "\n\n".join(p for p in parts if p)
    lines: list[str] = []
    question = getattr(step, "question", None)
    if question:
        lines.append(f"question: {question}")
    error = getattr(step, "error", None)
    if error:
        lines.extend(str(error).strip().splitlines())
    intent = getattr(step, "intent", None)
    if intent is None:
        summary = getattr(step, "intent_summary", None)
        if summary is not None:
            tables = getattr(summary, "tables", None)
            if tables:
                lines.append(f"tables: {list(tables)}")
            grain = getattr(summary, "grain", None)
            if grain:
                lines.append(f"grain: {grain}")
    elif intent:
        it = intent
        lines.append(f"tables: {it.tables}")
        lines.append(f"grain: {it.grain}")
        gb_terms = [getattr(g, "primary_term", str(g)) for g in (it.group_by_cols or [])]
        if gb_terms:
            lines.append(f"group_by: {gb_terms}")
        agg_cols = [
            getattr(sc.expr, "primary_term", str(sc.expr))
            for sc in (it.select_cols or [])
            if getattr(sc, "is_aggregated", False)
        ]
        if agg_cols:
            lines.append(f"agg_select: {agg_cols}")
    sql = getattr(step, "sql", None)
    if sql:
        lines.append("sql:")
        lines.extend(str(sql).splitlines())
    llm_calls = getattr(step, "llm_calls", None)
    if llm_calls is not None:
        lines.append(f"llm_calls: {llm_calls}")
    status = getattr(step, "status", None)
    kind = getattr(step, "kind", None)
    if status is not None or kind is not None:
        lines.append(f"status: {status!r} kind: {kind!r}")
    captured_logs = getattr(step, "captured_logs", None)
    if captured_logs:
        for ln in captured_logs:
            lines.append(str(ln))
    diagnostics = getattr(step, "diagnostics", None)
    if diagnostics:
        for diag in diagnostics:
            code = getattr(diag, "code", None)
            message = getattr(diag, "message", None)
            if code or message:
                lines.append(f"diagnostic: {code}: {message}")
    return "\n".join(lines)


def append_failure_trace(step: StepResult | list[StepResult] | object | None, path: str | os.PathLike[str]) -> None:
    """Append a formatted failure trace to the specified results file."""
    if step is None:
        return
    text = format_failure_trace(step)
    if not text:
        return
    p = Path(path)
    needs_sep = p.is_file() and p.stat().st_size > 0
    with open(path, "a", encoding="utf-8") as fh:
        if needs_sep:
            fh.write("\n\n" + "=" * 80 + "\n\n")
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def build_session_step_trace(
    *,
    scenario_id: str,
    question: str,
    step: object | None,
    error: str | None = None,
    captured_logs: list[str] | None = None,
) -> StepResult:
    status = "failed"
    if step is not None and getattr(step, "done", False) and not error:
        status = str(getattr(step, "status", None) or "ok")
    intent = getattr(step, "intent", None) if step is not None else None
    sql = getattr(step, "sql", None) if step is not None else None
    step_error = error
    if not step_error and step is not None:
        step_error = getattr(step, "error", None) or getattr(step, "message", None)
    diagnostics = getattr(step, "diagnostics", ()) if step is not None else ()
    return StepResult(
        scenario_id=scenario_id,
        question=question,
        status=status,
        intent=intent if isinstance(intent, RuntimeIntent) else None,
        sql=str(sql) if sql else None,
        error=str(step_error).strip() if step_error else None,
        captured_logs=list(captured_logs or []),
        diagnostics=tuple(diagnostics) if diagnostics else (),
        kind=str(getattr(step, "kind", "") or "") or None,
    )


def run_with_pipeline_capture(
    fn: Callable[[], tuple[object | None, str]],
    *,
    auto_responses: list[str] | None = None,
) -> tuple[object | None, str, list[str]]:
    responses = ["y"] if auto_responses is None else auto_responses
    with pipeline_capture(auto_responses=responses) as capture:
        step, err = fn()
    return step, err, list(capture.get("logs", []))


def _make_prompt_responders(
    responses: list[str],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build FIFO auto-responders for ``ask_user_choice`` and ``interactive_yes_no`` sharing one queue."""
    queue = list(responses)

    def _ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        if queue:
            return queue.pop(0)
        return "y"

    def _interactive_yes_no(
        stage: str,
        prompt: str,
        options: list[str],
        silent_no: bool = False,
        *,
        choice_port: Any = None,
    ) -> str | None:
        if choice_port is not None:
            if queue:
                return queue.pop(0)
            return "y"
        if queue:
            return queue.pop(0)
        return "y"

    return _ask_user_choice, _interactive_yes_no


def _make_input_responder(reject_reason: str = "incorrect results") -> Callable[[str], str]:
    """Build a replacement for ``builtins.input`` that supplies canned text."""
    call_count = {"n": 0}

    def _fake_input(prompt: str = "") -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return reject_reason
        return "n"

    return _fake_input


@contextmanager
def pipeline_capture(
    auto_responses: list[str],
    reject_reason: str = "incorrect results",
    csv_dir: str = "",
) -> Iterator[dict[str, Any]]:
    """Patch interactive I/O for programmatic pipeline runs."""
    import importlib

    capture: dict[str, Any] = {"logs": []}
    ask_uc, iyn = _make_prompt_responders(auto_responses)
    input_responder = _make_input_responder(reject_reason)

    def _import_mod(short: str) -> Any:
        return importlib.import_module(f"aetherdialect.{short}")

    def _capturing_debug(msg: str) -> None:
        capture["logs"].append(f"[DEBUG] {msg}")

    _debug_module_names = (
        "_core_utils",
        "_pipeline",
        "_sql_gen",
        "_validation_execute",
        "_validation_schema",
        "_validation_semantic",
        "_intent_expr",
        "_intent_process",
        "_intent_repair",
        "_intent_resolve",
        "_dialect",
        "_expansion_ops",
        "_utils",
        "_templates",
        "_schema_graph",
        "_schema_catalog",
        "_qsim",
        "_qsim_ops",
        "_main_execution",
        "_seed_warmup",
        "_llm_provider",
    )
    extra_patches: list[Any] = []
    for short in _debug_module_names:
        mod = _import_mod(short)
        if hasattr(mod, "debug"):
            extra_patches.append(patch.object(mod, "debug", _capturing_debug))

    def _capturing_pipeline_trace(heading: str, body: str | Callable[[], str]) -> None:
        resolved = body() if callable(body) else body
        capture["logs"].append(f"[PIPELINE_TRACE] {heading}\n{resolved}")

    _trace_module_names = (
        "_core_utils",
        "_pipeline",
        "_intent_process",
        "_sql_gen",
        "_validation_execute",
        "_intent_resolve",
        "_dialect",
        "_intent_repair",
        "_llm_provider",
    )
    for short in _trace_module_names:
        mod = _import_mod(short)
        if hasattr(mod, "pipeline_trace"):
            extra_patches.append(patch.object(mod, "pipeline_trace", _capturing_pipeline_trace))

    core_utils_mod = _import_mod("_core_utils")
    pipeline_mod = _import_mod("_pipeline")
    main_exec_mod = _import_mod("_main_execution")

    if csv_dir:
        _original_save = pipeline_mod.save_result_csv

        def _redirected_save(df: Any) -> None:
            orig_cwd = os.getcwd()
            try:
                os.chdir(csv_dir)
                _original_save(df)
            finally:
                os.chdir(orig_cwd)

        extra_patches.append(patch.object(pipeline_mod, "save_result_csv", _redirected_save))
        live_testing_mod = _import_mod("_live_testing")
        if hasattr(live_testing_mod, "save_result_csv"):
            extra_patches.append(patch.object(live_testing_mod, "save_result_csv", _redirected_save))

    with (
        patch.object(core_utils_mod, "ask_user_choice", ask_uc),
        patch.object(core_utils_mod, "interactive_yes_no", iyn),
        patch.object(pipeline_mod, "interactive_yes_no", iyn),
        patch.object(main_exec_mod, "interactive_yes_no", iyn),
        patch("builtins.input", input_responder),
    ):
        for p in extra_patches:
            p.start()
        try:
            yield capture
        finally:
            for p in extra_patches:
                p.stop()


_structural_migration_handler: Callable[..., None] | None = None


def register_structural_migration_handler(handler: Callable[..., None]) -> None:
    """Register the owner-side structural migration callback."""
    global _structural_migration_handler
    _structural_migration_handler = handler


def apply_structural_migration_to_persisted_scopes(
    engine_dir: str,
    *,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Apply table/column migration to persisted aetherspace and named context specs."""
    if _structural_migration_handler is None:
        raise RuntimeError("structural migration handler is not registered")
    _structural_migration_handler(
        engine_dir,
        dropped_tables=dropped_tables,
        dropped_columns=dropped_columns,
        table_renames=table_renames,
        column_renames=column_renames,
    )


def bind_params_for_sql(sql: str, param_values: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return bind map for EXPLAIN only when *sql* still contains ``:pN`` / ``@pN`` / ``$pN`` placeholders."""
    if not param_values:
        return None
    return param_values if SQL_BIND_TOKEN_RE.search(sql) else None


def reconcile_execute_bind_params(
    sql: str,
    param_values: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Filter *param_values* to tokens present in *sql* and fail when any bind token lacks a value."""
    if not SQL_BIND_TOKEN_RE.search(sql) and not UNBOUND_PYFORMAT_PLACEHOLDER_RE.search(sql):
        return None
    bind_map = bind_params_for_sql(sql, param_values) or {}
    required: set[str] = set()
    for match in SQL_BIND_TOKEN_RE.finditer(sql):
        required.add(match.group(1))
    for match in re.finditer(r"%\((\w+)\)s", sql):
        required.add(match.group(1))
    missing = sorted(required - set(bind_map.keys()))
    if missing:
        raise ValueError(f"unbound_placeholder: {', '.join(missing)}")
    return bind_map if bind_map else None


def cost_cap_active(v: float | int | None) -> bool:
    """True when *v* is a positive finite bound; ``None``, ``0``, and negatives disable the cap."""
    if v is None:
        return False
    try:
        return float(v) > 0.0
    except (TypeError, ValueError):
        return False


_DIAGNOSTIC_FORCE_DEPTH: int = 0


def diagnostic_debug_enabled() -> bool:
    """True when ``PolicyConfig.DEBUG`` or diagnostic capture (``telemetry_capture`` depth) is active."""
    return _DIAGNOSTIC_FORCE_DEPTH > 0 or PolicyConfig.DEBUG


def diagnostic_pipeline_trace_full_enabled() -> bool:
    """True when full pipeline trace logging is enabled."""
    return diagnostic_debug_enabled()


def diagnostic_force_enter() -> None:
    """Increment nested diagnostic capture depth (used by ``telemetry_capture``)."""
    global _DIAGNOSTIC_FORCE_DEPTH
    _DIAGNOSTIC_FORCE_DEPTH += 1


def diagnostic_force_exit() -> None:
    """Decrement nested diagnostic capture depth."""
    global _DIAGNOSTIC_FORCE_DEPTH
    if _DIAGNOSTIC_FORCE_DEPTH > 0:
        _DIAGNOSTIC_FORCE_DEPTH -= 1


def permission_denied_detail_logging_enabled() -> bool:
    """Return whether permission-denied failures may log SQL and driver detail at DEBUG."""
    raw = os.environ.get(AETHERDIALECT_LOG_PERMISSION_DENIED_DETAIL_ENV, "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def normalize_value_type(value_type: str) -> str:
    """Map a raw value-type string onto a canonical pipeline type."""
    if not value_type:
        return "string"
    vt_lower = value_type.lower().strip()
    if vt_lower in VALUE_TYPE_NORMALIZATION:
        return VALUE_TYPE_NORMALIZATION[vt_lower]
    if vt_lower in VALID_VALUE_TYPES:
        return vt_lower
    return "string"


def load_runtime_config(
    *,
    merged_env: Mapping[str, str],
) -> LlmExecutionConfig:
    """
    Merge built-in defaults with a caller-supplied environment.

    snapshot into one frozen LLM execution config. Resolution order is defaults first, then the environment layer keyed by the canonical Azure OpenAI and execution-limit variable names. Args: merged_env: Mapping of effective environment strings used for the environment merge layer. Returns: The frozen :class:`LlmExecutionConfig`.

    Raises: ValueError: When numeric fields are negative after merge.
    """

    def _env_text(name: str) -> str:
        return str(merged_env.get(name, "") or "").strip()

    defaults: dict[str, Any] = {
        "azure_endpoint": "",
        "azure_api_key": "",
        "azure_api_version": "",
        "deployment_light": "",
        "deployment_medium": "",
        "deployment_heavy": "",
        "max_query_cost_rows": 50_000_000,
        "max_query_cost_bytes": 50_000_000_000,
        "statement_timeout_ms": 30_000,
        "llm_timeout_ms": 60_000,
        "profile_timeout_ms": 120_000,
        "explain_timeout_ms": None,
    }
    env_map: dict[str, str] = {
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_api_key": "AZURE_OPENAI_API_KEY",
        "azure_api_version": "AZURE_OPENAI_API_VERSION",
        "deployment_light": AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT,
        "deployment_medium": AZURE_OPENAI_ENV_DEPLOYMENT_MEDIUM,
        "deployment_heavy": AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY,
        "max_query_cost_rows": "AETHERDIALECT_MAX_QUERY_COST_ROWS",
        "max_query_cost_bytes": "AETHERDIALECT_MAX_QUERY_COST_BYTES",
        "statement_timeout_ms": "AETHERDIALECT_STATEMENT_TIMEOUT_MS",
        "llm_timeout_ms": "AETHERDIALECT_LLM_TIMEOUT_MS",
        "profile_timeout_ms": "AETHERDIALECT_PROFILE_TIMEOUT_MS",
        "explain_timeout_ms": "AETHERDIALECT_EXPLAIN_TIMEOUT_MS",
    }
    merged: dict[str, Any] = dict(defaults)
    for canon, env_name in env_map.items():
        raw = _env_text(env_name)
        if not raw:
            continue
        if canon in {
            "max_query_cost_rows",
            "max_query_cost_bytes",
            "statement_timeout_ms",
            "llm_timeout_ms",
            "profile_timeout_ms",
        }:
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            if iv < 0:
                raise ValueError(f"Invalid non-negative integer for {env_name}")
            merged[canon] = iv
        elif canon == "explain_timeout_ms":
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            merged[canon] = None if iv <= 0 else iv
        else:
            merged[canon] = raw
    for name in (
        "max_query_cost_rows",
        "max_query_cost_bytes",
        "statement_timeout_ms",
        "llm_timeout_ms",
        "profile_timeout_ms",
    ):
        v = merged.get(name)
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"Invalid runtime config for {name}")
    exm = merged.get("explain_timeout_ms")
    if exm is not None and (not isinstance(exm, int) or exm < 0):
        raise ValueError("Invalid runtime config for explain_timeout_ms")
    cfg = LlmExecutionConfig(
        azure_endpoint=str(merged.get("azure_endpoint") or ""),
        azure_api_key=str(merged.get("azure_api_key") or ""),
        azure_api_version=str(merged.get("azure_api_version") or ""),
        deployment_light=str(merged.get("deployment_light") or ""),
        deployment_medium=str(merged.get("deployment_medium") or ""),
        deployment_heavy=str(merged.get("deployment_heavy") or ""),
        max_query_cost_rows=int(merged["max_query_cost_rows"]),
        max_query_cost_bytes=int(merged["max_query_cost_bytes"]),
        statement_timeout_ms=int(merged["statement_timeout_ms"]),
        llm_timeout_ms=int(merged["llm_timeout_ms"]),
        profile_timeout_ms=int(merged["profile_timeout_ms"]),
        explain_timeout_ms=merged.get("explain_timeout_ms"),
    )
    return cfg
