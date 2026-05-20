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

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations, product
from typing import Any, Protocol

from openai import AzureOpenAI, OpenAI
from packaging.version import InvalidVersion, Version

from ._config import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_LOCK_FILENAME,
    ARTIFACT_LOCK_POLL_INTERVAL_SECONDS,
    ARTIFACT_LOCK_TIMEOUT_SECONDS,
    ARTIFACT_MANIFEST_FILENAME,
    DIAGNOSTIC_CODE_ENGINE_INFO,
    JSON_COMPACT_SEPARATORS,
    LEGACY_ARTIFACT_FILENAMES,
    LEGACY_ARTIFACT_GLOBS,
    LlmExecutionConfig,
    MIGRATION_DATA_OVERLAP_MIN,
    MIN_COMPATIBLE_PACKAGE_VERSION,
    PROFILING_TOP_K,
    STRUCTURAL_IDENTITY_VALUES,
    STRUCTURAL_INLINE_SQL_LITERAL_LIST_RE,
    STRUCTURAL_SQL_PLACEHOLDER_PARAM_RE,
    TEMPLATE_STORE_SEGMENT,
    EngineConfig,
    PolicyConfig,
    diagnostic_debug_enabled,
    effective_llm_timeout_ms,
    diagnostic_force_enter,
    diagnostic_force_exit,
    diagnostic_pipeline_trace_full_enabled,
    diagnostic_verbose_enabled,
)
from ._contracts_base import (
    ColumnMetadata,
    ConfigError,
    Diagnostic,
    FKEdge,
    LlmJsonExhausted,
    LlmTransientFailure,
    MigrationTier,
    SchemaContext,
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

_LLM_EXECUTION_CONTEXT: ContextVar[LlmExecutionConfig | None] = ContextVar(
    "aetherdialect_llm_execution",
    default=None,
)

_TASK_MODEL_TO_DEPLOYMENT_FIELD: dict[str, str] = {
    "gpt-4o-mini": "deployment_light",
    "gpt-4.1-mini": "deployment_medium",
    "gpt-5.4-mini": "deployment_heavy",
}


REPHRASE_HINT_MESSAGES: dict[str, str] = {
    "intent_parse_failed": (
        "Please rephrase your question.\n"
        "Tips: mention specific tables or columns, keep filters simple, "
        "and avoid ambiguous references."
    ),
    "schema_invalid_declined": (
        "Please rephrase your question.\n"
        "Tips: use tables and columns that exist in this database, "
        "or ask about a related concept."
    ),
    "sql_validation_failed": (
        "Please rephrase or retry.\n"
        "Tips: simplify filters, be explicit about columns, "
        "or split a complex question into smaller ones."
    ),
    "user_rejected_intent": (
        "Please rephrase your question.\n"
        "Tips: be more specific about which columns, filters, grouping, "
        "or time range you want."
    ),
    "user_rejected_result": (
        "Please retry or rephrase your question.\n"
        "Tips: be more specific about columns, filters, grouping, or time range."
    ),
    "restricted_question": (
        "This question references columns or tables outside the visible schema. Try rephrasing using only "
        "the table and column names you see in the schema notes, or update `deny_columns` / `allow_columns` "
        "if you want them in scope."
    ),
    "vague_question": (
        "I could not pin this question to specific tables or columns. Try naming the entity (a table or "
        "business object), the metric you want, and any filter (date range, status, region) so I have "
        "something concrete to map."
    ),
}

USER_REJECTED_RESULT_BUCKET_TIPS: dict[str, str] = {
    "MISSING_FILTER": (
        "Tips: name the filter or dimension you care about (time range, status, category)."
    ),
    "WRONG_GROUPING": (
        "Tips: say whether you want totals per entity, per period, or overall."
    ),
    "WRONG_AGGREGATION": (
        "Tips: specify sum, average, count, or another metric clearly."
    ),
    "WRONG_TIME_RANGE": (
        "Tips: give an explicit date range or relative window."
    ),
    "WRONG_TABLES_OR_JOINS": (
        "Tips: name the tables or relationships that should connect your answer."
    ),
    "WRONG_SORT_OR_LIMIT": (
        "Tips: say how results should be ordered or how many rows you need."
    ),
    "OTHER": (
        "Tips: be more specific about columns, filters, grouping, or time range."
    ),
}

QUERY_RESULTS_HEADER: str = "Query Results"

USER_ERROR_PREFIX: str = "Error: "
USER_WARN_PREFIX: str = "! "
USER_TERMINATED_LINE: str = "\nUser terminated."
USER_INVALID_INPUT_LINE: str = "\nInvalid input."


@contextmanager
def llm_execution_scope(cfg: LlmExecutionConfig) -> Iterator[None]:
    """Bind *cfg* as the active :class:`LlmExecutionConfig` for nested LLM calls."""

    tok = _LLM_EXECUTION_CONTEXT.set(cfg)
    try:
        yield
    finally:
        _LLM_EXECUTION_CONTEXT.reset(tok)


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


def running_in_jupyter() -> bool:
    """Return True when the current process runs inside a Jupyter or IPython kernel front-end."""

    return "ipykernel" in sys.modules


def echo_yes_no_answer(raw: str) -> None:
    """Echo ``Yes`` or ``No`` for *raw* input answer; emit nothing for invalid tokens or in a TTY terminal."""

    if not running_in_jupyter():
        return
    token = raw.strip().lower()
    if token in {"y", "yes"}:
        print("Yes", flush=True)
    elif token in {"n", "no"}:
        print("No", flush=True)


def echo_user_text(raw: str) -> None:
    """Echo *raw* on its own line for Jupyter front-ends (terminal already shows what the user typed)."""

    if not running_in_jupyter() or not raw:
        return
    print(raw, flush=True)


def result(message: str) -> None:
    """Emit a query-result line through :func:`notify` (mirrored to the diagnostic print listener when bound)."""

    notify(message, stage="user_result", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def warn(message: str) -> None:
    """Emit a non-fatal warning through :func:`notify`, prefixed with ``! ``."""

    notify(f"{USER_WARN_PREFIX}{message}", stage="user_warn", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="warn")


def error(message: str) -> None:
    """Emit an error line through :func:`notify`, prefixed with ``Error: ``."""

    notify(f"{USER_ERROR_PREFIX}{message}", stage="user_error", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="error")


def prompt(message: str) -> str:
    """Display ``message``, read one line from stdin, and return it stripped."""

    return input(message).strip()


def terminated() -> None:
    """Emit the canonical user-termination line through :func:`notify`."""

    notify(USER_TERMINATED_LINE, stage="user_terminated", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def invalid_input(detail: str | None = None) -> None:
    """Emit the canonical invalid-input line through :func:`notify`, or *detail* when provided."""

    if detail:
        notify(detail.strip(), stage="user_invalid_input", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="warn")
    else:
        notify(USER_INVALID_INPUT_LINE, stage="user_invalid_input", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="warn")


class InteractiveChoicePort(Protocol):
    """Bridges yes/no prompts to a session queue or stdin."""

    def has_pending_choice(self) -> bool:
        """Return True when at least one queued answer is available for the next prompt."""
        ...

    def take_yes_no(self, stage: str, prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        """Return a normalised choice or raise ``PipelineSuspended`` when the queue is empty."""
        ...


_clients: dict[tuple[str, int], OpenAI | AzureOpenAI] = {}


def clear_llm_clients() -> None:
    """Remove cached OpenAI clients so a new environment configuration takes effect."""

    _clients.clear()


def _azure_deployment_for_model(model_id: str) -> str:
    """Return the Azure deployment name for *model_id* using the active execution config or environment."""

    mid = str(model_id).strip()
    runtime_llm = _LLM_EXECUTION_CONTEXT.get()
    if runtime_llm is not None:
        field = _TASK_MODEL_TO_DEPLOYMENT_FIELD.get(mid)
        if field:
            dep = getattr(runtime_llm, field, "")
            if isinstance(dep, str) and dep.strip():
                return dep.strip()
        return mid
    env_triples: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("gpt-5.4-mini", ("AZURE_OPENAI_DEPLOYMENT_HEAVY",)),
        ("gpt-4.1-mini", ("AZURE_OPENAI_DEPLOYMENT_MEDIUM",)),
        ("gpt-4o-mini", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
    )
    for known_id, env_keys in env_triples:
        if known_id != mid:
            continue
        for key in env_keys:
            value = (os.environ.get(key) or "").strip()
            if value:
                return value
        return mid
    return mid


_telemetry_sink: list[str] | None = None
_telemetry_suppress_console: bool = False


@contextmanager
def telemetry_capture(
    *,
    suppress_console: bool = False,
    force_diagnostic_flags: bool = False,
) -> Iterator[list[str]]:
    """
    Collect ``log`` / ``debug`` / ``pipeline_trace`` / ``pipeline_trace_lazy`` lines into a buffer.

    Args:

        suppress_console: When true, ``log`` / ``debug`` / ``pipeline_trace`` / ``pipeline_trace_lazy`` skip ``print`` even when flags are on (lines still append to the buffer).

        force_diagnostic_flags: When true, temporarily enables diagnostic output (nested ``diagnostic_force_enter`` / ``diagnostic_force_exit``) so the pipeline emits into the capture buffer while the block runs.

    Yields:

        The list that receives captured lines (same object for the whole block).
    """
    global _telemetry_sink, _telemetry_suppress_console
    buf: list[str] = []
    prev_sink = _telemetry_sink
    prev_suppress = _telemetry_suppress_console
    if force_diagnostic_flags:
        diagnostic_force_enter()
    _telemetry_sink = buf
    _telemetry_suppress_console = suppress_console
    try:
        yield buf
    finally:
        _telemetry_sink = prev_sink
        _telemetry_suppress_console = prev_suppress
        if force_diagnostic_flags:
            diagnostic_force_exit()


def _provider_order() -> list[str]:
    """Return the single resolved provider stored on :class:`EngineConfig`."""
    if EngineConfig.LLM_PROVIDER in {"openai", "azure"}:
        return [EngineConfig.LLM_PROVIDER]
    return ["openai"]


def _provider_is_configured(provider: str) -> bool:
    """Return whether a provider has required credentials configured."""
    if provider == "openai":
        return bool(EngineConfig.API_TOKEN and EngineConfig.OPENAI_BASE_URL)
    if provider == "azure":
        has_token = bool(EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN)
        has_endpoint = bool(EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL)
        has_version = bool((EngineConfig.AZURE_OPENAI_API_VERSION or "").strip())
        return has_token and has_endpoint and has_version
    return False


def _resolve_llm_timeout_ms() -> int:
    """Return HTTP timeout milliseconds from the active execution config or policy defaults."""

    llm = _LLM_EXECUTION_CONTEXT.get()
    if llm is not None and isinstance(llm.llm_timeout_ms, int) and llm.llm_timeout_ms > 0:
        return int(llm.llm_timeout_ms)
    return effective_llm_timeout_ms()


def _build_client(provider: str) -> OpenAI | AzureOpenAI:
    """Build and cache an OpenAI-compatible client for *provider*."""
    llm = _LLM_EXECUTION_CONTEXT.get()
    timeout_ms = _resolve_llm_timeout_ms()
    timeout_s = timeout_ms / 1000.0
    endpoint_sig = ""
    if llm is not None and isinstance(llm.azure_endpoint, str) and llm.azure_endpoint.strip():
        endpoint_sig = llm.azure_endpoint.strip()
    cache_key = (provider, timeout_ms, endpoint_sig)
    if cache_key in _clients:
        return _clients[cache_key]
    if provider == "openai":
        client: OpenAI | AzureOpenAI = OpenAI(
            api_key=EngineConfig.API_TOKEN,
            base_url=EngineConfig.OPENAI_BASE_URL,
            timeout=timeout_s,
        )
        _clients[cache_key] = client
        return client
    if provider == "azure":
        if llm is not None and llm.azure_endpoint.strip():
            endpoint = llm.azure_endpoint.strip()
            api_version = (llm.azure_api_version or "").strip()
            api_key = (llm.azure_api_key or EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN or "").strip()
        else:
            endpoint = EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL
            api_version = (EngineConfig.AZURE_OPENAI_API_VERSION or "").strip()
            api_key = EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN
        if not endpoint:
            raise RuntimeError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE_URL")
        if not api_version:
            raise RuntimeError("Azure OpenAI requires AZURE_OPENAI_API_VERSION")
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout_s,
        )
        _clients[cache_key] = client
        return client
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def log(msg: str) -> None:
    """
    Print ``[LOG]`` + *msg* when ``PolicyConfig.VERBOSE`` or verbose diagnostics are on (see ``diagnostic_verbose_enabled``).

    Args:

        msg: Text to print.

    Returns:

        None.
    """
    line = f"[LOG] {msg}"
    if _telemetry_sink is not None:
        _telemetry_sink.append(line)
    if _telemetry_suppress_console:
        return
    if diagnostic_verbose_enabled():
        print(line)


def debug(msg: str) -> None:
    """
    Print ``[DEBUG]`` + *msg* when ``PolicyConfig.DEBUG`` or debug diagnostics are on (see ``diagnostic_debug_enabled``).

    Args:

        msg: Text to print.

    Returns:

        None.
    """
    line = f"[DEBUG] {msg}"
    if _telemetry_sink is not None:
        _telemetry_sink.append(line)
    if _telemetry_suppress_console:
        return
    if diagnostic_debug_enabled():
        print(line)
        print(
            json.dumps({"kind": "debug", "message": msg}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )


def pipeline_trace(heading: str, body: str) -> None:
    """
    Print a full ``[PIPELINE_TRACE]`` block when debug + full trace are on.

    Args:

        heading: Event label.

        body: Untruncated payload (SQL, JSON, prompts, etc.).

    Returns:

        None.
    """
    block = f"[PIPELINE_TRACE] {heading}\n{body}"
    if _telemetry_sink is not None:
        _telemetry_sink.append(block)
    if _telemetry_suppress_console:
        return
    if not (diagnostic_debug_enabled() and diagnostic_pipeline_trace_full_enabled()):
        return
    print(block)


def pipeline_trace_lazy(heading: str, body_factory: Callable[[], str]) -> None:
    """
    Like :func:`pipeline_trace`, but *body_factory* runs only when output is needed.

    Args:

        heading: Event label.

        body_factory: Callable returning the trace body (often ``lambda: stable_json(...)``).

    Returns:

        None.
    """

    sink_on = _telemetry_sink is not None
    console_on = (
        not _telemetry_suppress_console
        and diagnostic_debug_enabled()
        and diagnostic_pipeline_trace_full_enabled()
    )
    if not sink_on and not console_on:
        return
    pipeline_trace(heading, body_factory())


def sha256(s: str) -> str:
    """
    SHA-256 hex digest of UTF-8 *s*.

    Args:

        s: String to hash.

    Returns:

        64-character hex string.
    """

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _strip_fences(s: str) -> str:
    """
    Strip leading/trailing ``` fences and surrounding whitespace.

    Args:

        s: Possibly fenced text.

    Returns:

        Inner content, stripped.
    """
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def canonicalize_sql(sql: str) -> str:
    """
    Normalize SQL whitespace, formatting, and join equality operand order.

    Args:

        sql: Raw SQL string, possibly with extra whitespace or markdown fences.

    Returns:

        Canonicalized SQL string with consistent spacing and canonical operand order in equality conditions.
    """
    s = _strip_fences(sql).strip()
    s = s.rstrip(";").strip()
    s = re.sub(r"^EXPLAIN\s+(?:ANALYZE\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(?<![><!=])=(?![>=])", " = ", s)
    s = re.sub(r"\s+", " ", s).strip()

    def normalize_equality(m: re.Match) -> str:
        left, right = m.group(1).strip(), m.group(2).strip()
        if left > right:
            left, right = right, left
        return f"{left} = {right}"

    s = re.sub(r"([^\s()><!=]+)\s*=\s*([^\s()><!=]+)", normalize_equality, s)
    return s


def stable_json(o: Any) -> str:
    """
    ``json.dumps`` with sorted keys and minimal separators.

    Args:

        o: JSON-serialisable value.

    Returns:

        Deterministic compact JSON string.
    """
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def colmap_signature(column_map: dict[str, str]) -> str:
    """
    SHA-256 of stable JSON for sorted ``column_map`` items.

    Args:

        column_map: Bare column -> table name.

    Returns:

        Hex digest string.
    """
    return sha256(stable_json(sorted(column_map.items())))


def intent_id(d: dict[str, Any]) -> str:
    """
    Short intent id: first 16 hex chars of ``stable_json(d)`` hash.

    Args:

        d: Canonical intent dict.

    Returns:

        16-character prefix of SHA-256 hex.
    """
    return sha256(stable_json(d))[:16]


def structural_hash_fp(tables_payload: dict[str, Any]) -> str:
    """
    Fingerprint DDL-stable table payloads (kinds, columns, keys, FK edges).

    Args:

        tables_payload: Mapping of table name to structural column/table dicts.

    Returns:

        Hex digest string.
    """

    return sha256(stable_json({"tables": tables_payload}))


def profiling_hash_fp(tables_payload: dict[str, Any]) -> str:
    """
    Fingerprint profiling-only payloads (counts, roles, top values, semantics).

    Args:

        tables_payload: Mapping of table name to profiling dicts.

    Returns:

        Hex digest string.
    """

    return sha256(stable_json({"tables": tables_payload}))


def _schema_scope_file_content_sha256(path: str | None) -> str:
    """Return a SHA-256 hex digest of the UTF-8 file at *path*, or ``\"\"`` when the path is missing or not a readable file."""

    if path is None or not str(path).strip():
        return ""
    expanded = os.path.expanduser(str(path).strip())
    if not os.path.isfile(expanded):
        return ""
    with open(expanded, encoding="utf-8") as fh:
        return sha256(fh.read())


def scope_hash_fp(schema_context: SchemaContext) -> str:
    """
    Fingerprint scope inputs: include mode, allow list, deny lists, and inlined DDL or notes file contents.

    Args:

        schema_context: Frozen schema scope descriptor.

    Returns:

        Hex digest string.
    """

    deny_cols = sorted(schema_context.deny_columns)
    allow_cols = sorted(schema_context.allow_columns)
    payload = {
        "allow_objects": sorted(schema_context.allow_objects),
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
    """
    Legacy SHA-256 of ``{"tables": tables_dict}`` JSON.

    Used by cache diagnostics and tests that pass arbitrary table-shaped dicts.
    """

    return sha256(stable_json({"tables": tables_dict}))


def normalize_question(q: str) -> str:
    """
    Lowercase and clean *q*; restore single-quoted spans to original case.

    Args:

        q: Raw user question.

    Returns:

        Normalised string for fuzzy matching.
    """
    q = q.strip()
    q = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", q)

    quoted_values = []

    def preserve_quoted(m):
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
    """
    First top-level ``{...}`` substring via brace depth (after fence strip).

    Args:

        s: Text that may embed JSON.

    Returns:

        JSON object substring, or ``None``.
    """
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
    """
    Try ``json.loads``; on failure, parse first ``{...}`` fragment.

    Args:

        s: Raw model or file text.

    Returns:

        Parsed object, or ``None``.
    """
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


def _task_model_for_profile(task: str) -> str:
    """Return the configured logical model name for *task* from ``EngineConfig``."""

    if task == "intent":
        return str(EngineConfig.OPENAI_MODEL_INTENT)
    if task == "feedback":
        return str(EngineConfig.OPENAI_MODEL_INTENT)
    if task == "schema":
        return str(EngineConfig.OPENAI_MODEL_SCHEMA)
    if task == "schema_base":
        return str(EngineConfig.OPENAI_MODEL_SCHEMA_BASE)
    if task == "ddl":
        return str(EngineConfig.OPENAI_MODEL_DDL)
    if task == "join":
        return str(EngineConfig.OPENAI_MODEL_JOIN)
    if task == "judge":
        return str(EngineConfig.OPENAI_MODEL_JOIN)
    if task == "conversation":
        return str(EngineConfig.OPENAI_MODEL_INTENT)
    return str(EngineConfig.OPENAI_MODEL)


_TASK_PROFILES: dict[str, dict[str, Any]] = {
    "intent": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "feedback": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "schema": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "schema_base": {
        "temperature": 0,
    },
    "ddl": {
        "temperature": 0,
    },
    "join": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "judge": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "conversation": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "default": {
        "temperature": 0,
    },
}


def _reconfigure_console_streams_to_utf8() -> None:
    """Force ``sys.stdout`` / ``sys.stderr`` to UTF-8 with replacement on undefined glyphs."""
    for _stream in (sys.stdout, sys.stderr):
        if _stream is None:
            continue
        if getattr(_stream, "encoding", "").lower() == "utf-8":
            continue
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_console_streams_to_utf8()

_DEFAULT_LLM_CHAT_TIMEOUT = object()


def _llm_error_likely_transient(exc: BaseException) -> bool:
    """Heuristic for HTTP/network overload signals suitable for :class:`LlmTransientFailure`."""

    s = str(exc).lower()
    needles = (
        "429",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "503",
        "502",
    )
    return any(n in s for n in needles)


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


_LLM_SENSITIVITY_STRIP_KEYS: frozenset[str] = frozenset({"sensitivity", "pii"})


def _omit_sensitivity_classification_for_llm_json(value: Any) -> Any:
    """Return a copy with sensitivity tier keys removed for outbound LLM user payloads."""

    if isinstance(value, dict):
        return {
            k: _omit_sensitivity_classification_for_llm_json(v)
            for k, v in value.items()
            if str(k) not in _LLM_SENSITIVITY_STRIP_KEYS
        }
    if isinstance(value, list):
        return [_omit_sensitivity_classification_for_llm_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_omit_sensitivity_classification_for_llm_json(v) for v in value)
    return value


def _llm_user_text_without_sensitivity_classification(user: str) -> str:
    """When *user* parses as JSON, strip sensitivity classification keys recursively."""

    s = user.strip()
    if not s or s[0] not in "{[":
        return user
    try:
        parsed: Any = json.loads(s)
    except json.JSONDecodeError:
        return user
    if isinstance(parsed, (dict, list)):
        return stable_json(_omit_sensitivity_classification_for_llm_json(parsed))
    return user


def llm_chat(
    system: str,
    user: str,
    max_retries: int = 3,
    timeout: Any = _DEFAULT_LLM_CHAT_TIMEOUT,
    task: str = "default",
) -> str:
    """
    JSON-mode chat completion with task-based model profile and retries.

    Args:

        system: System prompt.

        user: User message.

        max_retries: Attempts before raising.

        timeout: Seconds per request; when left as the default, resolved from :data:`PolicyConfig.LLM_TIMEOUT_MS`.

        task: ``intent`` / ``feedback`` / ``join`` / ``judge`` / ``schema`` / ``schema_base`` / ``ddl`` / ``conversation`` / ``default``.

    Returns:

        Stripped model text.
    """
    if timeout is _DEFAULT_LLM_CHAT_TIMEOUT:
        timeout = _resolve_llm_timeout_ms() / 1000.0
    profile = _TASK_PROFILES.get(task, _TASK_PROFILES["default"])
    model = _task_model_for_profile(task)
    api_model = _azure_deployment_for_model(model) if EngineConfig.LLM_PROVIDER == "azure" else model
    user_for_llm = _llm_user_text_without_sensitivity_classification(user)

    kwargs: dict[str, Any] = {
        "model": api_model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": user_for_llm}]},
        ],
        "timeout": timeout,
        "text": {"format": {"type": "json_object"}},
    }

    if "reasoning" in profile:
        kwargs["reasoning"] = profile["reasoning"]
    else:
        kwargs["temperature"] = profile.get("temperature", 0)

    debug(f"[core_utils.llm_chat] task={task} system_len={len(system)} user_len={len(user_for_llm)}")
    pipeline_trace_lazy(f"llm_chat.request task={task} system_message", lambda: system)
    pipeline_trace_lazy(f"llm_chat.request task={task} user_message", lambda: user_for_llm)

    providers = [p for p in _provider_order() if _provider_is_configured(p)]
    if not providers:
        raise RuntimeError("No configured OpenAI/Azure OpenAI provider found")

    for attempt in range(max_retries):
        last_error: Exception | None = None
        for provider in providers:
            client = _build_client(provider)
            try:
                start = time.time()
                r = client.responses.create(**kwargs)
                elapsed = time.time() - start
                output = r.output_text.strip()
                debug(f"[core_utils.llm_chat] provider={provider} RAW OUTPUT:\n{output}")
                pipeline_trace_lazy(
                    f"llm_chat.response task={task} attempt={attempt + 1}",
                    lambda: output,
                )
                usage = getattr(r, "usage", None)
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
                tot_tok = getattr(usage, "total_tokens", None)
                tok_str = f" tokens(in={in_tok},out={out_tok},total={tot_tok})" if usage is not None else ""
                debug(
                    f"[core_utils.llm_chat] provider={provider} model={api_model} task={task} "
                    f"completed in {elapsed:.1f}s (attempt {attempt + 1}/{max_retries}){tok_str}"
                )
                return output
            except Exception as e:
                elapsed = time.time() - start
                last_error = e
                err_full = str(e)
                log(
                    f"[core_utils.llm_chat] provider={provider} timeout or error after {elapsed:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries}): "
                    f"{err_full if diagnostic_pipeline_trace_full_enabled() else err_full[:100]}"
                )
                pipeline_trace_lazy(
                    f"llm_chat.error task={task} provider={provider} attempt={attempt + 1}",
                    lambda: err_full,
                )
        if attempt < max_retries - 1:
            wait = 2**attempt
            log(f"[core_utils.llm_chat] retrying in {wait}s...")
            time.sleep(wait)
        else:
            msg = f"LLM call failed after {max_retries} attempts: {str(last_error)}"
            if last_error is not None and _llm_error_likely_transient(last_error):
                raise LlmTransientFailure(msg) from last_error
            raise RuntimeError(msg) from last_error


def normalize_sql_operator_spaces(sql: str) -> str:
    """
    Merge split operators: ``> =``, ``< =``, ``! =`` → ``>=``, ``<=``, ``!=``.

    Args:

        sql: Raw SQL.

    Returns:

        Same string with operator tokens fixed, or unchanged if empty.
    """
    if not sql or not sql.strip():
        return sql
    s = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return s


def normalize_sql(sql: str) -> str:
    """
    After ``canonicalize_sql``, append default ``ASC`` where ORDER BY lacks direction.

    Args:

        sql: Raw SQL.

    Returns:

        SQL with explicit ``ASC``/``DESC`` on each ORDER BY item.
    """
    s = normalize_sql_operator_spaces(canonicalize_sql(sql))
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


def llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, Any]:
    """
    Call ``llm_chat`` and parse JSON; retry with format hint; wrap bare SELECT.

    Raises ``LlmJsonExhausted`` when no attempt produces valid JSON (or a bare
    SQL SELECT that we wrap); callers are responsible for deciding whether
    exhaustion is recoverable or terminal.

    Args:

        system: System prompt.

        user: User payload.

        retries: Extra attempts after the initial call (minimum zero).

        task: Profile key for ``llm_chat``.

    Returns:

        Parsed dict payload on success.
    """
    total_attempts = 1 + max(0, retries)
    raw = llm_chat(system, user, task=task)
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        debug(f"[core_utils.llm_json] parsed keys={list(parsed.keys())}")
        return parsed

    if raw.strip().upper().startswith("SELECT"):
        debug("[core_utils.llm_json] raw_sql_detected wrapping")
        sql_statement = raw.strip()
        return {"sql": sql_statement, "chosen_join_candidate_id": "J00"}

    debug("[core_utils.llm_json] parse_failed: retrying")
    for attempt in range(max(0, retries)):
        debug(f"[core_utils.llm_json] retry: {attempt + 1}")
        raw = llm_chat(
            system,
            user + "\n\nFORMAT_ERROR: Output ONLY valid JSON that matches the required schema. Do NOT output raw SQL.",
            task=task,
        )
        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict):
            debug(f"[core_utils.llm_json] retry_success: keys={list(parsed.keys())}")
            return parsed

        if raw.strip().upper().startswith("SELECT"):
            debug("[core_utils.llm_json] retry_sql_detected: wrapping")
            sql_statement = raw.strip()
            return {"sql": sql_statement, "chosen_join_candidate_id": "J00"}

    debug("[core_utils.llm_json] all_retries_failed")
    raise LlmJsonExhausted(task=task, attempts=total_attempts)


def normalize_array_contains_param_value(value: Any) -> Any:
    """
    Strip whitespace and redundant surrounding quotes from array ``contains`` operands.

    Keeps bind values free of decorative quotes; SQL generation also normalizes stored array elements per dialect so membership stays stable across data encodings.

    Args:

        value: Bound parameter value (typically a string).

    Returns:

        Normalized value; non-strings returned unchanged.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _escape_sql_single_quoted_literal(value: str) -> str:
    """Escape a UTF-8 string for safe use inside single-quoted SQL literals."""

    return value.replace("'", "''")


def substitute_params(sql_param: str, params: dict[str, Any]) -> str:
    """
    Replace ``:key`` placeholders with formatted parameter values.

    Args:

        sql_param: SQL with ``:pN`` style keys.

        params: Key -> value map.

    Returns:

        SQL with all ``:key`` placeholders substituted.
    """
    result = sql_param
    for key in sorted(params.keys(), key=lambda k: -len(k)):
        val = params[key]
        if not key:
            continue
        placeholder = f":{key}"
        if isinstance(val, list):
            formatted_items = []
            for item in val:
                if isinstance(item, str):
                    formatted_items.append(f"'{_escape_sql_single_quoted_literal(item)}'")
                else:
                    formatted_items.append(str(item))
            result = result.replace(placeholder, ", ".join(formatted_items))
        elif isinstance(val, bool):
            result = result.replace(placeholder, "TRUE" if val else "FALSE")
        elif isinstance(val, str):
            if val.startswith("'") and val.endswith("'") and "','" in val:
                result = result.replace(placeholder, val)
            elif re.match(r"^-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)*$", val):
                result = result.replace(placeholder, val)
            else:
                result = result.replace(placeholder, f"'{_escape_sql_single_quoted_literal(val)}'")
        else:
            result = result.replace(placeholder, str(val))
    return result


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
    """
    Inline ``:sN`` placeholders when values match defaults or identity structural values.

    Returns the reduced SQL string and the parameter map with structural keys removed.
    """

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


def _format_cell(v) -> str:
    """
    String for one result cell: ``NULL``, decimals, or ``str(v)``.

    Args:

        v: Driver cell value.

    Returns:

        Display string.
    """
    if v is None:
        return "NULL"
    if isinstance(v, Decimal):
        return f"{v:f}" if v == v.to_integral_value() else f"{v}"
    if isinstance(v, str):
        return v
    return str(v)


def print_query_result(
    rows: list[tuple],
    sql: str,
    *,
    headers: list[str] | None = None,
) -> None:
    """
    Emit SQL and scalar answer or up to five aligned rows through :func:`notify`.

    Args:

        rows: Result tuples from the driver.

        sql: Query text.

        headers: Optional column names (padded with ``colN``).

    Returns:

        None.
    """
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
    notify("\n".join(out_lines), stage="query_result", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


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
    """
    Resolve a yes/no prompt via an optional session port or stdin.

    Args:

        stage: Stable stage label for session mapping (ignored for stdin).

        prompt: User-facing prompt line.

        options: Allowed token labels such as ``"y"`` and ``"n"``.

        silent_no: Forwarded to ``ask_user_choice`` when using stdin.

        choice_port: When set, delegates to the port (may raise ``PipelineSuspended``).

    Returns:

        ``"y"``, ``"n"``, or ``None`` when cancelled.
    """

    if choice_port is not None:
        return choice_port.take_yes_no(stage, prompt, options, silent_no)
    return ask_user_choice(prompt, options, silent_no)


def ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
    """
    Interactive ``input()`` for yes/no style choices (y/n/yes/no).

    Args:

        prompt: Line printed before the bracketed options.

        options: Shown as ``opt1/opt2/...`` in the prompt.

        silent_no: If True, skip "User terminated." on ``n``.

    Returns:

        ``"y"``, ``"n"``, or ``None`` on EOF/invalid.
    """
    options_display = "/".join(options)
    notify(f"{prompt} ({options_display}): ", stage="interactive_choice", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
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
        notify("Yes", stage="interactive_choice", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        return "y"
    elif normalized in ("n", "no"):
        notify("No", stage="interactive_choice", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")
        if not silent_no:
            terminated()
        return "n"

    invalid_input()
    return None


class RephraseHint(Enum):
    """
    User-facing rephrase hint categories printed when the pipeline cannot continue.

    Each value maps to a short, non-technical, suggestive message intended to help the user produce a better question without exposing internal validation output.
    """

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
    """
    Print a tiered, suggestive rephrase hint for *reason*.

    Uses a fixed catalogue of short non-technical messages; never exposes validation logs or repair-loop internals to the user.
    """
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
    """
    Emit *title*, optional indented *items*, and optional *footer* through :func:`notify`.

    Args:

        title: Heading line.

        items: Key/value lines (lists joined with commas).

        footer: Trailing paragraph.

    Returns:

        None.
    """
    lines: list[str] = [f"\n{title}"]
    if items:
        for key, val in items.items():
            if isinstance(val, list | tuple | set):
                val = ", ".join(str(v) for v in val)
            lines.append(f"  {key}: {val}")
    if footer:
        lines.append(f"\n{footer}")
    notify("\n".join(lines), stage="print_info", code=DIAGNOSTIC_CODE_ENGINE_INFO, level="info")


def join_sig_string(sig: list[str]) -> str:
    """
    Join path segments with ``|`` for stable keys.

    Args:

        sig: Ordered join signature parts.

    Returns:

        Single string, e.g. ``a.b|c.d``.
    """
    return "|".join(sig)


def normalize_op(op: str) -> str:
    """
    Lowercase/whitespace-trim *op*; map ``==``, ``gte``, etc. to SQL ops.

    Args:

        op: LLM filter/having operator token.

    Returns:

        Canonical operator string (e.g. ``=``, ``!=``, ``>=``).
    """
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
    """
    Load a JSON value from a UTF-8 document stored as gzip.

    Args:

        path: Filesystem path to the `.json.gz` file.

    Returns:

        The parsed JSON value.
    """
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def write_gzip_json_atomic(path: str, obj: Any, *, sort_keys: bool) -> None:
    """
    Serialize ``obj`` to compact UTF-8 JSON, gzip it, and replace ``path`` atomically.

    Args:

        path: Destination path for the gzip JSON artifact.

        obj: JSON-serializable value.

        sort_keys: Passed to ``json.dumps`` for deterministic key order.

    Returns:

        None.
    """
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
    """
    Acquire an exclusive OS-level lock on ``lock_path`` for the duration of the context.

    Blocks up to ``timeout`` seconds, raising ``TimeoutError`` if the lock cannot be acquired. Works on both POSIX (``fcntl.flock``) and Windows (``msvcrt.locking``) without external dependencies.
    """

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
    """
    Reentrant per-``artifacts_dir`` lock covering load, mutate, and save sequences for template learning.

    The lock file path joins *artifacts_dir* with :data:`ARTIFACT_LOCK_FILENAME` from ``aetherdialect._config``.

    Nested ``with artifact_lock`` blocks on the same directory bump a per-thread refcount without deadlocking.

    Cross-thread and cross-process callers serialize at the OS level.
    """

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
    for dist_name in ("aetherdialect", "text2sql"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "0.0.0+dev"


def manifest_path(artifacts_dir: str) -> str:
    """Return the absolute path to ``artifact_manifest.json`` under ``artifacts_dir``."""

    return os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
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
    notes_hash: str = ""
    semantic_edges_hash: str = ""
    last_migration_tier: str = ""
    last_migration_at: str = ""


def read_artifact_manifest(artifacts_dir: str) -> ArtifactManifest | None:
    """
    Load artifact manifest JSON if present.

    Returns:

        Parsed manifest, or ``None`` when missing or invalid.
    """

    path = manifest_path(artifacts_dir)
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
    return ArtifactManifest(
        artifact_format_version=ver,
        created_with_package_version=str(data.get("created_with_package_version", "") or ""),
        min_compatible_package_version=str(data.get("min_compatible_package_version", "") or ""),
        last_action=str(data.get("last_action", "") or ""),
        last_action_at=str(data.get("last_action_at", "") or ""),
        structural_hash=str(data.get("structural_hash", "") or ""),
        profiling_hash=str(data.get("profiling_hash", "") or ""),
        scope_hash=str(data.get("scope_hash", "") or ""),
        effective_structural_hash=str(data.get("effective_structural_hash", "") or ""),
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
    notes_hash: str = "",
    semantic_edges_hash: str = "",
    last_migration_tier: str = "",
    last_migration_at: str | None = None,
    last_action: str = "compat_wipe",
    last_corruption_at: str = "",
) -> None:
    """
    Write manifest with format version, package version, optional hashes, and last action.

    Persists atomically via a temporary file in *artifacts_dir* followed by ``os.replace``.

    Returns:

        None.
    """

    os.makedirs(artifacts_dir, exist_ok=True)
    path = manifest_path(artifacts_dir)
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


def _wipe_filenames(artifacts_dir: str, names: tuple[str, ...]) -> int:
    """Remove named files directly under *artifacts_dir*; return count removed."""

    removed = 0
    for name in names:
        fp = os.path.join(artifacts_dir, name)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    return removed


def _wipe_globs(artifacts_dir: str, patterns: tuple[str, ...]) -> int:
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

    _wipe_filenames(artifacts_dir, LEGACY_ARTIFACT_FILENAMES)
    _wipe_globs(artifacts_dir, LEGACY_ARTIFACT_GLOBS)
    partitioned = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    if os.path.isdir(partitioned):
        shutil.rmtree(partitioned, ignore_errors=True)


def detect_legacy_artifacts(artifacts_dir: str) -> list[str]:
    """
    Return artifact filenames suggesting a pre-manifest install populated this directory.

    A directory is considered "legacy" when at least one versioned artifact (schema graph
    snapshot, template store, qsim run, seed warmup cache) is present *but* no
    ``artifact_manifest.json`` exists alongside it. Such artifacts were produced by an
    earlier release of this package whose on-disk format predates the migration manifest,
    and they cannot be safely loaded by the current code path.

    Args:

        artifacts_dir: Directory to inspect. Missing or non-directory paths return ``[]``.

    Returns:

        Sorted list of basenames of legacy artifacts found. Empty when the directory does
        not exist, is empty, contains only user-supplied non-versioned files (notes,
        ``.env``, raw SQL), or already contains an ``artifact_manifest.json``.
    """

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
) -> Counter[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    ctr: Counter[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = Counter()
    for tm in sg.tables.values():
        for fk in tm.foreign_keys:
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


def _col_topk_frozen(col: ColumnMetadata) -> frozenset[str]:
    vals = col.top_k_values or []
    cleaned = {str(v).strip() for v in vals if v is not None and str(v).strip() != ""}
    return frozenset(sorted(cleaned)[:PROFILING_TOP_K])


def profiling_value_overlap(older: SchemaGraph, newer: SchemaGraph) -> float:
    """Aggregate Jaccard overlap of profiling Top-K sets on shared ``(table, column)`` keys."""

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
            a = _col_topk_frozen(ot.columns[c])
            b = _col_topk_frozen(nt.columns[c])
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
                sc = _prof_jaccard_sets(_col_topk_frozen(ocol), _col_topk_frozen(ncol))
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
    old_ctr = _all_fk_multiset(old)
    new_ctr = _all_fk_multiset(new)
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


def _manifest_matches_schema(manifest: ArtifactManifest, schema: SchemaGraph) -> bool:
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


def classify_migration_tier(
    manifest: ArtifactManifest | None,
    schema: SchemaGraph,
    *,
    previous_schema: SchemaGraph | None = None,
    schema_diff: Any | None = None,
) -> MigrationTier:
    """Compare stored manifest fingerprints to the live schema graph."""

    if manifest is None or not manifest.effective_structural_hash:
        return MigrationTier.NO_CHANGE
    if _manifest_matches_schema(manifest, schema):
        return MigrationTier.NO_CHANGE
    fmt = manifest.artifact_format_version
    if fmt not in (0, ARTIFACT_FORMAT_VERSION):
        return MigrationTier.DESTRUCTIVE
    min_cv = (manifest.min_compatible_package_version or "").strip()
    if min_cv:
        try:
            if Version(_artifact_package_version_string()) < Version(min_cv):
                return MigrationTier.DESTRUCTIVE
        except (InvalidVersion, TypeError, ValueError):
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
            if profiling_value_overlap(previous_schema, schema) >= MIGRATION_DATA_OVERLAP_MIN:
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
