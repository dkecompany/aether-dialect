"""LLM provider dispatch: OpenAI, Azure OpenAI, mock fixture replay, and chat/JSON helpers."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from openai import AzureOpenAI, OpenAI

from ._config import EngineConfig
from ._constants import (
    INTENT_COMPOSE_SYSTEM,
    INTENT_GROUND_SYSTEM,
    INTENT_INTERPRET_SYSTEM,
    LLM_SENSITIVITY_STRIP_KEYS,
    MOCK_FIXTURE_STUB_SCHEMA_LITERALS,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
    TASK_MODEL_TO_DEPLOYMENT_FIELD,
    TASK_PROFILES,
)
from ._contracts_base import LlmJsonExhausted, LlmTransientFailure
from ._core_utils import (
    LLM_EXECUTION_CONTEXT,
    debug,
    diagnostic_debug_enabled,
    effective_llm_timeout_ms,
    pipeline_trace,
    safe_json_loads,
    stable_json,
)

_clients: dict[tuple[str, int, str], OpenAI | AzureOpenAI] = {}

_DEFAULT_LLM_CHAT_TIMEOUT = object()


class MockFixtureMissingError(RuntimeError):
    """Raised when the mock provider has no fixture for the requested LLM call."""

    def __init__(self, *, task: str, system: str, user: str) -> None:
        self.task = task
        self.system = system
        self.user = user
        skeleton = json.dumps(
            {
                "task": task,
                "system": system[:200] + ("..." if len(system) > 200 else ""),
                "user": user[:500] + ("..." if len(user) > 500 else ""),
                "output_text": "<paste model JSON/text here>",
            },
            ensure_ascii=False,
            indent=2,
        )
        super().__init__(
            f"No mock fixture for task={task!r}. Add an entry to the fixture corpus:\n{skeleton}",
        )


class _LLMProvider(Protocol):
    def chat_text(
        self,
        system: str,
        user: str,
        *,
        task: str,
        max_retries: int,
        timeout: float,
    ) -> str: ...


def clear_llm_clients() -> None:
    """Remove cached OpenAI clients so a new environment configuration takes effect."""
    _clients.clear()


def _azure_deployment_for_model(model_id: str) -> str:
    """Return the Azure deployment name for *model_id* using the active execution config or environment."""
    mid = str(model_id).strip()
    runtime_llm = LLM_EXECUTION_CONTEXT.get()
    if runtime_llm is not None:
        field = TASK_MODEL_TO_DEPLOYMENT_FIELD.get(mid)
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


def _provider_order() -> list[str]:
    """Return the single resolved provider stored on :class:`EngineConfig`."""
    if EngineConfig.LLM_PROVIDER in {"openai", "azure", "mock"}:
        return [EngineConfig.LLM_PROVIDER]
    return ["openai"]


def _provider_is_configured(provider: str) -> bool:
    """Return whether a provider has required credentials configured."""
    if provider == "mock":
        return bool((EngineConfig.MOCK_FIXTURES_FILE or "").strip())
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
    llm = LLM_EXECUTION_CONTEXT.get()
    if llm is not None and isinstance(llm.llm_timeout_ms, int) and llm.llm_timeout_ms > 0:
        return int(llm.llm_timeout_ms)
    return effective_llm_timeout_ms()


def _build_client(provider: str) -> OpenAI | AzureOpenAI:
    """Build and cache an OpenAI-compatible client for *provider*."""
    llm = LLM_EXECUTION_CONTEXT.get()
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
            endpoint = (EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL or "").strip()
            api_version = (EngineConfig.AZURE_OPENAI_API_VERSION or "").strip()
            api_key = (EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN or "").strip()
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


def _omit_sensitivity_classification_for_llm_json(value: Any) -> Any:
    """Return a copy with sensitivity tier keys removed for outbound LLM user payloads."""
    if isinstance(value, dict):
        return {
            k: _omit_sensitivity_classification_for_llm_json(v)
            for k, v in value.items()
            if str(k) not in LLM_SENSITIVITY_STRIP_KEYS
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


def _normalize_mock_lookup_text(text: str) -> str:
    """Canonicalize punctuation so stored fixtures match runtime mock lookup keys. Recorded fixtures occasionally contain UTF-8 punctuation decoded as cp1252."""
    normalized = unicodedata.normalize("NFKC", text)
    for bad, good in (
        ("\u00e2\u20ac\u201d", "\u2014"),
        ("\u00e2\u20ac\u201c", "\u2013"),
        ("\u00e2\u20ac\u2122", "\u2019"),
        ("\u00e2\u20ac\u0153", "\u201c"),
        ("\u00e2\u20ac\u009d", "\u201d"),
        ("\u00e2\u20ac\u00a6", "\u2026"),
    ):
        normalized = normalized.replace(bad, good)

    def _sort_valid_list(match: re.Match[str]) -> str:
        inner = match.group(1)
        try:
            items = json.loads(inner.replace("'", '"'))
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            return match.group(0)
        return f"Valid: {repr(sorted(items))}"

    return re.sub(r"Valid: (\[[^\]]+\])", _sort_valid_list, normalized)


_literals_cache: dict[str, str] | None = None
_interpret_domain_cache: dict[str, Any] | None = None


def clear_canonical_schema_literals_cache() -> None:
    """Clear cached bundled mock-fixture key inputs (schema literals + interpret domain)."""
    global _literals_cache, _interpret_domain_cache
    _literals_cache = None
    _interpret_domain_cache = None


def _pin_mock_interpret_domain(domain: dict[str, Any]) -> None:
    global _interpret_domain_cache
    _interpret_domain_cache = domain


def load_canonical_interpret_domain() -> dict[str, Any] | None:
    """Return pinned interpret schema_domain for mock fixture user-key normalization."""
    return _interpret_domain_cache


def _pin_mock_schema_literals(literals: dict[str, str]) -> None:
    global _literals_cache
    _literals_cache = {
        "owner": stable_schema_literal(str(literals["owner"])),
        "consumer": stable_schema_literal(str(literals["consumer"])),
    }


def pin_mock_fixture_keys_from_bundle(bundle_dir: Path) -> None:
    """Load bundled schema literals and interpret domain for offline mock replay lookup."""
    literals_path = bundle_dir / SANDBOX_SCHEMA_LITERALS_FILENAME
    if literals_path.is_file():
        payload = json.loads(literals_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            owner = str(payload.get("owner", "")).strip()
            consumer = str(payload.get("consumer", "")).strip()
            if owner and consumer:
                _pin_mock_schema_literals({"owner": owner, "consumer": consumer})
    domain_path = bundle_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME
    if domain_path.is_file():
        payload = json.loads(domain_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            _pin_mock_interpret_domain(payload)


def pin_schema_literal_slot(slot: str, schema_literal_json: str) -> None:
    """Merge one owner/consumer schema literal into the mock replay lookup cache."""
    global _literals_cache
    if slot not in ("owner", "consumer"):
        raise ValueError(f"unsupported schema literal slot: {slot!r}")
    stable = stable_schema_literal(schema_literal_json)
    if _literals_cache is None:
        _literals_cache = dict(MOCK_FIXTURE_STUB_SCHEMA_LITERALS)
    _literals_cache[slot] = stable


def load_canonical_schema_literals() -> dict[str, str]:
    global _literals_cache
    if _literals_cache is not None:
        return _literals_cache
    return dict(MOCK_FIXTURE_STUB_SCHEMA_LITERALS)


def stable_schema_literal(literal: str) -> str:
    try:
        parsed = json.loads(literal)
    except json.JSONDecodeError:
        return literal
    if not isinstance(parsed, dict):
        return literal
    return stable_json(parsed)


def _schema_literal_table_count(schema_literal_json: str) -> int:
    try:
        obj = json.loads(schema_literal_json)
    except json.JSONDecodeError:
        return -1
    if not isinstance(obj, dict):
        return -1
    return sum(1 for key in obj if key != "enum_types")


def _schema_literal_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    return stable_schema_literal(left) == stable_schema_literal(right)


def _canonical_schema_literal_for_embedded(embedded: str, literals: dict[str, str]) -> str:
    count = _schema_literal_table_count(embedded)
    owner_count = _schema_literal_table_count(literals["owner"])
    consumer_count = _schema_literal_table_count(literals["consumer"])
    if count == consumer_count:
        return literals["consumer"]
    if count == owner_count:
        return literals["owner"]
    if abs(count - consumer_count) < abs(count - owner_count):
        return literals["consumer"]
    return literals["owner"]


def rewrite_user_schema_literals(user: str, literals: dict[str, str]) -> str:
    s = user.strip()
    if not s or s[0] != "{":
        return user
    try:
        body = json.loads(s)
    except json.JSONDecodeError:
        return user
    if not isinstance(body, dict):
        return user
    changed = False
    domain = load_canonical_interpret_domain()
    if domain is not None and "schema_domain" in body:
        if body.get("schema_domain") != domain:
            body["schema_domain"] = domain
            changed = True
    for key in ("schema_literal_json", "schema_summary", "schema_info"):
        if key not in body:
            continue
        embedded = str(body[key])
        canonical = _canonical_schema_literal_for_embedded(embedded, literals)
        if not _schema_literal_equivalent(embedded, canonical):
            body[key] = canonical
            changed = True
    if not changed:
        return user
    return stable_json(body)


def _mock_fixture_user_key_body(text: str, *, stable: bool) -> str:
    stripped = text.strip()
    if not stable or not stripped.startswith("{"):
        return stripped
    try:
        body = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(body, dict):
        return stable_json(body)
    return stripped


def mock_fixture_user_key(user: str, *, literals: dict[str, str] | None = None) -> str:
    if literals is None:
        literals = load_canonical_schema_literals()
    rewritten = rewrite_user_schema_literals(user, literals)
    stripped = _llm_user_text_without_sensitivity_classification(rewritten).strip()
    canonical = _mock_fixture_user_key_body(stripped, stable=True)
    return _normalize_mock_lookup_text(canonical)


def legacy_mock_fixture_user_key(user: str, *, literals: dict[str, str] | None = None) -> str:
    """Pre-stable-json mock lookup key kept for fixtures recorded before canonical compaction."""
    if literals is None:
        literals = load_canonical_schema_literals()
    rewritten = rewrite_user_schema_literals(user, literals)
    stripped = _llm_user_text_without_sensitivity_classification(rewritten).strip()
    return _normalize_mock_lookup_text(stripped)


def mock_fixture_lookup_key(
    task: str,
    system: str,
    user: str,
    *,
    literals: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    return (
        str(task),
        str(system),
        mock_fixture_user_key(user, literals=literals),
    )


def _extract_gatekeeper_question_text(user: str) -> str | None:
    """Return embedded ``question`` text from legacy gatekeeper JSON user payloads."""
    text = str(user).strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    question = parsed.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    return None


def _mock_fixture_lookup_aliases(
    task: str,
    system: str,
    user: str,
    *,
    literals: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return lookup keys for a stored or runtime mock user payload."""
    keys = [
        mock_fixture_lookup_key(task, system, user, literals=literals),
        (
            str(task),
            str(system),
            legacy_mock_fixture_user_key(user, literals=literals),
        ),
    ]
    if task == "default":
        question = _extract_gatekeeper_question_text(user)
        if question:
            keys.append(mock_fixture_lookup_key(task, system, question, literals=literals))
            keys.append(
                (
                    str(task),
                    str(system),
                    legacy_mock_fixture_user_key(question, literals=literals),
                ),
            )
    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _llm_request_kwargs(system: str, user: str, *, task: str, timeout: float) -> dict[str, Any]:
    profile = TASK_PROFILES.get(task, TASK_PROFILES["default"])
    model = _task_model_for_profile(task)
    api_model = _azure_deployment_for_model(model) if EngineConfig.LLM_PROVIDER == "azure" else model
    kwargs: dict[str, Any] = {
        "model": api_model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "timeout": timeout,
        "text": {"format": {"type": "json_object"}},
    }
    if "reasoning" in profile:
        kwargs["reasoning"] = profile["reasoning"]
    else:
        kwargs["temperature"] = profile.get("temperature", 0)
    return kwargs


def _handcrafted_stage_for_system(system: str) -> str:
    if system == INTENT_INTERPRET_SYSTEM:
        return "interpret"
    if system == INTENT_GROUND_SYSTEM:
        return "ground"
    if system == INTENT_COMPOSE_SYSTEM:
        return "compose"
    return "default"


_handcrafted_entries_cache: list[dict[str, object]] | None = None


def _mock_bundle_path() -> Path:
    override = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "sandbox" / "data.zip"


def _read_mock_bundle_json_member(filename: str) -> dict[str, object] | None:
    bundle = _mock_bundle_path()
    if bundle.is_dir():
        path = bundle / filename
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    if not bundle.is_file():
        return None
    with zipfile.ZipFile(bundle) as zf:
        member = next((name for name in zf.namelist() if name.endswith(filename)), None)
        if member is None:
            return None
        payload = json.loads(zf.read(member).decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_handcrafted_entries() -> list[dict[str, object]]:
    global _handcrafted_entries_cache
    if _handcrafted_entries_cache is not None:
        return _handcrafted_entries_cache
    payload = _read_mock_bundle_json_member("sandbox_handcrafted_fixtures.json")
    if payload is None:
        _handcrafted_entries_cache = []
        return _handcrafted_entries_cache
    rows = payload.get("entries")
    if not isinstance(rows, list):
        _handcrafted_entries_cache = []
        return _handcrafted_entries_cache
    _handcrafted_entries_cache = [dict(row) for row in rows if isinstance(row, dict)]
    return _handcrafted_entries_cache


def _question_from_mock_user(user: str) -> str | None:
    question = _extract_gatekeeper_question_text(user)
    if question:
        return question
    text = str(user).strip()
    if not text.startswith("{"):
        return text or None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    embedded = body.get("question")
    if isinstance(embedded, str) and embedded.strip():
        return embedded.strip()
    return None


def _handcrafted_chat_text(system: str, user: str, *, task: str) -> str | None:
    question = _question_from_mock_user(user)
    if not question:
        return None
    stage = _handcrafted_stage_for_system(system)
    for row in _load_handcrafted_entries():
        if str(row.get("question", "")).strip() != question:
            continue
        if str(row.get("task", "default")) != str(task):
            continue
        if str(row.get("stage", "")).strip() != stage:
            continue
        response = row.get("response")
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
    return None


class MockProvider:
    """Replay canned LLM responses from a JSON fixtures file."""

    def __init__(self, fixtures_path: str) -> None:
        """Load fixtures; stored user prompts are already canonical from recording."""
        path = Path(fixtures_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("fixtures", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError(f"Mock fixtures file {path} must contain a fixtures list")
        self._map: dict[tuple[str, str, str], str] = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", "default"))
            system = str(item.get("system", ""))
            user = str(item.get("user", "")).strip()
            output = str(item.get("output_text", ""))
            self._map[(task, system, user)] = output
            for key in _mock_fixture_lookup_aliases(task, system, user):
                self._map[key] = output

    def chat_text(
        self,
        system: str,
        user: str,
        *,
        task: str,
        max_retries: int,
        timeout: float,
    ) -> str:
        del max_retries, timeout
        for key in _mock_fixture_lookup_aliases(task, system, user):
            if key in self._map:
                return self._map[key]
        handcrafted = _handcrafted_chat_text(system, user, task=task)
        if handcrafted is not None:
            return handcrafted
        raise MockFixtureMissingError(task=task, system=system, user=user)


_mock_provider: MockProvider | None = None


def _get_mock_provider() -> MockProvider:
    global _mock_provider
    path = (EngineConfig.MOCK_FIXTURES_FILE or "").strip()
    if not path:
        raise RuntimeError("Mock provider requires AETHERDIALECT_MOCK_FIXTURES_FILE")
    if _mock_provider is None:
        _mock_provider = MockProvider(path)
    return _mock_provider


def reset_mock_provider(*, clear_literals: bool = False) -> None:
    """Clear cached mock fixture index."""
    global _mock_provider, _handcrafted_entries_cache
    _mock_provider = None
    _handcrafted_entries_cache = None
    if clear_literals:
        clear_canonical_schema_literals_cache()


def _clear_provider_cache() -> None:
    """Clear HTTP client and mock fixture caches."""
    clear_llm_clients()
    reset_mock_provider()


def llm_chat(
    system: str,
    user: str,
    max_retries: int = 3,
    timeout: Any = _DEFAULT_LLM_CHAT_TIMEOUT,
    task: str = "default",
) -> str:
    """JSON-mode chat completion with task-based model profile and retries."""
    if timeout is _DEFAULT_LLM_CHAT_TIMEOUT or timeout is None:
        timeout_val = _resolve_llm_timeout_ms() / 1000.0
    else:
        timeout_val = float(timeout)

    profile = TASK_PROFILES.get(task, TASK_PROFILES["default"])
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
        "timeout": timeout_val,
        "text": {"format": {"type": "json_object"}},
    }

    if "reasoning" in profile:
        kwargs["reasoning"] = profile["reasoning"]
    else:
        kwargs["temperature"] = profile.get("temperature", 0)

    debug(f"[llm_provider.llm_chat] task={task} system_len={len(system)} user_len={len(user_for_llm)}")
    pipeline_trace(f"llm_chat.request task={task} system_message", lambda: system)
    pipeline_trace(f"llm_chat.request task={task} user_message", lambda: user_for_llm)

    if EngineConfig.LLM_PROVIDER == "mock":
        return _get_mock_provider().chat_text(
            system,
            user_for_llm,
            task=task,
            max_retries=max_retries,
            timeout=timeout_val,
        )

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
                output = str(r.output_text).strip()
                debug(f"[llm_provider.llm_chat] provider={provider} RAW OUTPUT:\n{output}")
                output_trace = cast(Callable[[], str], (lambda captured=output: lambda: captured)())
                pipeline_trace(
                    f"llm_chat.response task={task} attempt={attempt + 1}",
                    output_trace,
                )
                usage = getattr(r, "usage", None)
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
                tot_tok = getattr(usage, "total_tokens", None)
                tok_str = f" tokens(in={in_tok},out={out_tok},total={tot_tok})" if usage is not None else ""
                debug(
                    f"[llm_provider.llm_chat] provider={provider} model={api_model} task={task} "
                    f"completed in {elapsed:.1f}s (attempt {attempt + 1}/{max_retries}){tok_str}"
                )
                return output
            except Exception as e:
                elapsed = time.time() - start
                last_error = e
                err_full = str(e)
                debug(
                    f"[llm_provider.llm_chat] provider={provider} timeout or error after {elapsed:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries}): "
                    f"{err_full if diagnostic_debug_enabled() else err_full[:100]}"
                )
                error_trace = cast(Callable[[], str], (lambda captured=err_full: lambda: captured)())
                pipeline_trace(
                    f"llm_chat.error task={task} provider={provider} attempt={attempt + 1}",
                    error_trace,
                )
        if attempt < max_retries - 1:
            wait = 2**attempt
            debug(f"[llm_provider.llm_chat] retrying in {wait}s...")
            time.sleep(wait)
        else:
            msg = f"LLM call failed after {max_retries} attempts: {str(last_error)}"
            if last_error is not None and _llm_error_likely_transient(last_error):
                raise LlmTransientFailure(msg) from last_error
            raise RuntimeError(msg) from last_error
    raise RuntimeError("llm_chat failed without returning")


def llm_json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, Any]:
    """Call ``llm_chat`` and parse JSON; retry with format hint; wrap bare SELECT. Raises ``LlmJsonExhausted`` when no attempt produces valid JSON."""
    total_attempts = 1 + max(0, retries)
    raw = llm_chat(system, user, task=task)
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        debug(f"[llm_provider.llm_json] parsed keys={list(parsed.keys())}")
        return parsed

    if raw.strip().upper().startswith("SELECT"):
        debug("[llm_provider.llm_json] raw_sql_detected wrapping")
        sql_statement = raw.strip()
        return {"sql": sql_statement, "chosen_join_candidate_id": "J00"}

    debug("[llm_provider.llm_json] parse_failed: retrying")
    for attempt in range(max(0, retries)):
        debug(f"[llm_provider.llm_json] retry: {attempt + 1}")
        raw = llm_chat(
            system,
            user + "\n\nFORMAT_ERROR: Output ONLY valid JSON that matches the required schema. Do NOT output raw SQL.",
            task=task,
        )
        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict):
            debug(f"[llm_provider.llm_json] retry_success: keys={list(parsed.keys())}")
            return parsed

        if raw.strip().upper().startswith("SELECT"):
            debug("[llm_provider.llm_json] retry_sql_detected: wrapping")
            sql_statement = raw.strip()
            return {"sql": sql_statement, "chosen_join_candidate_id": "J00"}

    debug("[llm_provider.llm_json] all_retries_failed")
    raise LlmJsonExhausted(task=task, attempts=total_attempts)
