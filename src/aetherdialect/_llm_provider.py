"""LLM provider dispatch: OpenAI, Azure OpenAI, mock fixture replay, and chat/JSON helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
import unicodedata
import zipfile
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from openai import AzureOpenAI, OpenAI

from ._config import EngineConfig, PolicyConfig
from ._constants import (
    INTENT_COMPOSE_SYSTEM,
    INTENT_GROUND_SYSTEM,
    INTENT_INTERPRET_SYSTEM,
    LLM_SENSITIVITY_STRIP_KEYS,
    MOCK_FIXTURE_STUB_SCHEMA_LITERALS,
    SANDBOX_INTERPRET_DOMAIN_FILENAME,
    SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS,
    SANDBOX_SCHEMA_LITERALS_FILENAME,
    TASK_MODEL_TO_DEPLOYMENT_FIELD,
    TASK_PROFILES,
)
from ._contracts_base import (
    LlmBatchRequest,
    LlmExecutionConfig,
    LlmJsonExhausted,
    LlmTransientFailure,
    MockFixtureMissingError,
)
from ._core_utils import (
    LLM_EXECUTION_CONTEXT,
    active_business_knowledge_digest,
    active_prompt_cache_schema_hash,
    debug,
    diagnostic_debug_enabled,
    effective_llm_timeout_ms,
    pipeline_trace,
    record_llm_usage,
    register_after_fork_callback,
    safe_json_loads,
    stable_json,
)

_clients: dict[tuple[str, int, str], OpenAI | AzureOpenAI] = {}
_clients_lock = threading.Lock()

_DEFAULT_LLM_CHAT_TIMEOUT = object()

_sandbox_recorded_corpus_questions: int | None = None

_literals_cache: dict[str, str] | None = None
_interpret_domain_cache: dict[str, Any] | None = None

_handcrafted_entries_cache: dict[str, list[dict[str, object]]] = {}
_mock_provider_instances: dict[tuple[str, str], Any] = {}


@dataclass
class SandboxRuntimeState:
    """Per-sandbox mutable state that must not leak across Sandbox instances."""

    paraphrase_source: dict[str, list[str]] | None = None
    recorded_corpus_question_count: int | None = None
    faithfulness_by_question: dict[str, Any] = field(default_factory=dict)
    faithfulness_loaded: bool = False
    mock_provider: Any | None = None
    mock_fixtures_path: str | None = None

    @staticmethod
    def bind_sandbox_runtime(state: SandboxRuntimeState | None) -> Token[SandboxRuntimeState | None]:
        """Bind *state* for nested sandbox-scoped helpers."""
        return _ACTIVE_SANDBOX_RUNTIME.set(state)

    @staticmethod
    def reset_sandbox_runtime(token: Token[SandboxRuntimeState | None]) -> None:
        """Restore the prior sandbox runtime binding."""
        _ACTIVE_SANDBOX_RUNTIME.reset(token)

    @staticmethod
    def current_sandbox_runtime() -> SandboxRuntimeState | None:
        """Return the sandbox runtime bound for this execution context, if any."""
        return _ACTIVE_SANDBOX_RUNTIME.get()

    @staticmethod
    def set_sandbox_recorded_corpus_question_count(count: int | None) -> None:
        """Pin the recorded-corpus question count for sandbox mock-mode error text."""
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None:
            runtime.recorded_corpus_question_count = count
            return
        if count is not None:
            raise RuntimeError(
                "set_sandbox_recorded_corpus_question_count requires an active Sandbox runtime; "
                "create Sandbox() or bind_sandbox_runtime() first.",
            )
        global _sandbox_recorded_corpus_questions
        _sandbox_recorded_corpus_questions = count

    @staticmethod
    def _sandbox_recorded_corpus_question_count() -> int | None:
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None:
            return runtime.recorded_corpus_question_count
        return _sandbox_recorded_corpus_questions


_ACTIVE_SANDBOX_RUNTIME: ContextVar[SandboxRuntimeState | None] = ContextVar(
    "aetherdialect_active_sandbox_runtime",
    default=None,
)

MockFixtureMissingError.register_recorded_corpus_count(SandboxRuntimeState._sandbox_recorded_corpus_question_count)


class LLMProvider:
    """LLM provider dispatch: OpenAI, Azure OpenAI, and chat/JSON/batch helpers."""

    @staticmethod
    def _hash_credential_part(value: str) -> str:
        """Return a short stable digest for a credential string used in cache keys."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _llm_credential_identity(provider: str, llm: LlmExecutionConfig | None) -> str:
        """Return a stable cache identity for the credentials used to build *provider* clients."""
        if provider == "openai":
            token = (EngineConfig.API_TOKEN or "").strip()
            base = (EngineConfig.OPENAI_BASE_URL or "").strip()
            return f"openai:{base}:{LLMProvider._hash_credential_part(token)}"
        if provider == "azure":
            if llm is not None and llm.azure_endpoint.strip():
                endpoint = llm.azure_endpoint.strip()
                api_version = (llm.azure_api_version or "").strip()
                api_key = (llm.azure_api_key or EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN or "").strip()
            else:
                endpoint = (EngineConfig.AZURE_OPENAI_ENDPOINT or EngineConfig.AZURE_OPENAI_BASE_URL or "").strip()
                api_version = (EngineConfig.AZURE_OPENAI_API_VERSION or "").strip()
                api_key = (EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN or "").strip()
            return f"azure:{endpoint}:{api_version}:{LLMProvider._hash_credential_part(api_key)}"
        return provider

    @staticmethod
    def clear_llm_clients() -> None:
        """Remove cached clients for the active credential identity only."""
        provider = EngineConfig.LLM_PROVIDER
        if provider not in {"openai", "azure"}:
            return
        llm = LLM_EXECUTION_CONTEXT.get()
        credential_identity = LLMProvider._llm_credential_identity(provider, llm)
        with _clients_lock:
            for key in list(_clients):
                if key[0] == provider and key[2] == credential_identity:
                    del _clients[key]

    @staticmethod
    def clear_all_clients_after_fork() -> None:
        """Drop every cached LLM client after process fork."""
        with _clients_lock:
            _clients.clear()

    @staticmethod
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
        env_pairs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("gpt-5.4-mini", ("AZURE_OPENAI_DEPLOYMENT_HEAVY",)),
            ("gpt-5.4-nano", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
            ("gpt-5-mini", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
            ("gpt-5-nano", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
            ("gpt-4.1-mini", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
            ("gpt-4.1-nano", ("AZURE_OPENAI_DEPLOYMENT_LIGHT",)),
        )
        for known_id, env_keys in env_pairs:
            if known_id != mid:
                continue
            for key in env_keys:
                value = (os.environ.get(key) or "").strip()
                if value:
                    return value
            return mid
        return mid

    @staticmethod
    def _task_model_for_profile(task: str) -> str:
        """Return the configured logical model name for *task* from ``EngineConfig``."""
        if task in ("intent", "conversation"):
            return str(EngineConfig.OPENAI_MODEL_INTENT)
        if task == "intent_format":
            return str(EngineConfig.OPENAI_MODEL_INTENT_FORMAT)
        if task == "intent_schema_repair":
            return str(EngineConfig.OPENAI_MODEL_INTENT_SCHEMA_REPAIR)
        if task == "feedback":
            return str(EngineConfig.OPENAI_MODEL)
        if task == "schema":
            return str(EngineConfig.OPENAI_MODEL_SCHEMA)
        if task == "schema_base":
            return str(EngineConfig.OPENAI_MODEL_SCHEMA_BASE)
        if task == "ddl":
            return str(EngineConfig.OPENAI_MODEL_DDL)
        if task == "join":
            return str(EngineConfig.OPENAI_MODEL_JOIN)
        if task == "synth":
            return str(EngineConfig.OPENAI_MODEL_SYNTH)
        if task == "synth_variety":
            return str(EngineConfig.OPENAI_MODEL_SYNTH_VARIETY)
        return str(EngineConfig.OPENAI_MODEL)

    @staticmethod
    def _logical_model_uses_reasoning(model_id: str) -> bool:
        """Return whether *model_id* is a reasoning model that accepts ``reasoning`` instead of ``temperature``."""
        mid = str(model_id).strip().lower()
        if mid.startswith("gpt-4"):
            return False
        return mid.startswith("gpt-5") or mid.startswith("o")

    @staticmethod
    def _reasoning_effort_for_model(model_id: str, effort: str) -> str:
        """Map a profile effort token to the effort keyword supported by *model_id*."""
        mid = str(model_id).strip().lower()
        if mid.startswith("gpt-5.4") and effort == "minimal":
            return "none"
        return effort

    @staticmethod
    def _apply_generation_profile(kwargs: dict[str, Any], profile: dict[str, Any], logical_model: str) -> None:
        """Attach ``reasoning`` or ``temperature`` to *kwargs* based on the active provider and model."""
        if EngineConfig.LLM_PROVIDER == "azure":
            reasoning = profile.get("reasoning") if isinstance(profile.get("reasoning"), dict) else {}
            effort_raw = str(reasoning.get("effort", "low")) if reasoning else "low"
            effort = LLMProvider._reasoning_effort_for_model(logical_model, effort_raw)
            kwargs["reasoning"] = {
                "effort": effort,
                "summary": str(reasoning.get("summary", "concise")) if reasoning else "concise",
            }
            kwargs.pop("temperature", None)
            return
        if "reasoning" in profile and LLMProvider._logical_model_uses_reasoning(logical_model):
            reasoning = profile["reasoning"]
            effort = LLMProvider._reasoning_effort_for_model(logical_model, str(reasoning.get("effort", "low")))
            kwargs["reasoning"] = {
                "effort": effort,
                "summary": str(reasoning.get("summary", "concise")),
            }
            return
        kwargs["temperature"] = profile.get("temperature", 0)

    @staticmethod
    def _provider_order() -> list[Literal["openai", "azure", "mock"]]:
        """Return the single resolved provider stored on :class:`EngineConfig`."""
        if EngineConfig.LLM_PROVIDER in {"openai", "azure", "mock"}:
            return [cast(Literal["openai", "azure", "mock"], EngineConfig.LLM_PROVIDER)]
        return ["openai"]

    @staticmethod
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

    @staticmethod
    def _resolve_llm_timeout_ms() -> int:
        """Return HTTP timeout milliseconds from the active execution config or policy defaults."""
        llm = LLM_EXECUTION_CONTEXT.get()
        if llm is not None and isinstance(llm.llm_timeout_ms, int) and llm.llm_timeout_ms > 0:
            return int(llm.llm_timeout_ms)
        return effective_llm_timeout_ms()

    @staticmethod
    def _build_client(provider: str) -> OpenAI | AzureOpenAI:
        """Build and cache an OpenAI-compatible client for *provider*."""
        llm = LLM_EXECUTION_CONTEXT.get()
        timeout_ms = LLMProvider._resolve_llm_timeout_ms()
        timeout_s = timeout_ms / 1000.0
        cache_key = (provider, timeout_ms, LLMProvider._llm_credential_identity(provider, llm))
        with _clients_lock:
            existing = _clients.get(cache_key)
            if existing is not None:
                return existing
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
                    api_key = (
                        llm.azure_api_key or EngineConfig.AZURE_API_TOKEN or EngineConfig.API_TOKEN or ""
                    ).strip()
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

    @staticmethod
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

    @staticmethod
    def _omit_sensitivity_classification_for_llm_json(value: Any) -> Any:
        """Return a copy with sensitivity tier keys removed for outbound LLM user payloads."""
        if isinstance(value, dict):
            return {
                k: LLMProvider._omit_sensitivity_classification_for_llm_json(v)
                for k, v in value.items()
                if str(k) not in LLM_SENSITIVITY_STRIP_KEYS
            }
        if isinstance(value, list):
            return [LLMProvider._omit_sensitivity_classification_for_llm_json(v) for v in value]
        if isinstance(value, tuple):
            return tuple(LLMProvider._omit_sensitivity_classification_for_llm_json(v) for v in value)
        return value

    @staticmethod
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
            return stable_json(LLMProvider._omit_sensitivity_classification_for_llm_json(parsed))
        return user

    @staticmethod
    def _llm_request_kwargs(system: str, user: str, *, task: str, timeout: float) -> dict[str, Any]:
        profile = TASK_PROFILES.get(task, TASK_PROFILES["default"])
        model = LLMProvider._task_model_for_profile(task)
        api_model = LLMProvider._azure_deployment_for_model(model) if EngineConfig.LLM_PROVIDER == "azure" else model
        user_for_llm = LLMProvider._llm_user_text_without_sensitivity_classification(user)
        kwargs: dict[str, Any] = {
            "model": api_model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_for_llm}]},
            ],
            "timeout": timeout,
            "text": {"format": {"type": "json_object"}},
        }
        LLMProvider._apply_generation_profile(kwargs, profile, model)
        return kwargs

    @staticmethod
    def _clear_provider_cache() -> None:
        """Clear HTTP client and mock fixture caches."""
        LLMProvider.clear_llm_clients()
        MockProvider.reset_mock_provider()

    @staticmethod
    def resolve_prompt_cache_key(task: str) -> str | None:
        """Return a provider ``prompt_cache_key`` when a schema scope is active."""
        schema_hash = active_prompt_cache_schema_hash()
        if not schema_hash:
            return None
        business_digest = active_business_knowledge_digest()
        if business_digest:
            raw = f"{task}:{schema_hash}:{business_digest[:16]}"
        else:
            raw = f"{task}:{schema_hash}"
        if len(raw) <= 64:
            return raw
        return f"{task[:8]}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _usage_tokens_from_response(usage: Any) -> tuple[int, int, int]:
        """Return ``(input_tokens, output_tokens, cached_input_tokens)`` from a provider usage object."""
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cached = 0
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return in_tok, out_tok, cached

    @staticmethod
    def chat(
        system: str,
        user: str,
        max_retries: int = 3,
        timeout: Any = _DEFAULT_LLM_CHAT_TIMEOUT,
        task: str = "default",
    ) -> str:
        """JSON-mode chat completion with task-based model profile and retries."""
        if timeout is _DEFAULT_LLM_CHAT_TIMEOUT or timeout is None:
            timeout_val = LLMProvider._resolve_llm_timeout_ms() / 1000.0
        else:
            timeout_val = float(timeout)

        profile = TASK_PROFILES.get(task, TASK_PROFILES["default"])
        model = LLMProvider._task_model_for_profile(task)
        api_model = LLMProvider._azure_deployment_for_model(model) if EngineConfig.LLM_PROVIDER == "azure" else model
        user_for_llm = LLMProvider._llm_user_text_without_sensitivity_classification(user)

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

        LLMProvider._apply_generation_profile(kwargs, profile, model)

        cache_key = LLMProvider.resolve_prompt_cache_key(task)
        if cache_key is not None:
            kwargs["prompt_cache_key"] = cache_key

        debug(f"[llm_provider.llm_chat] task={task} system_len={len(system)} user_len={len(user_for_llm)}")
        pipeline_trace(f"llm_chat.request task={task} system_message", lambda: system)
        pipeline_trace(f"llm_chat.request task={task} user_message", lambda: user_for_llm)

        if EngineConfig.LLM_PROVIDER == "mock":
            return MockProvider.get().chat_text(
                system,
                user_for_llm,
                task=task,
                max_retries=max_retries,
                timeout=timeout_val,
            )

        providers = [p for p in LLMProvider._provider_order() if LLMProvider._provider_is_configured(p)]
        if not providers:
            raise RuntimeError("No configured OpenAI/Azure OpenAI provider found")

        for attempt in range(max_retries):
            last_error: Exception | None = None
            for provider in providers:
                client = LLMProvider._build_client(provider)
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
                    if usage is not None:
                        usage_in, usage_out, usage_cached = LLMProvider._usage_tokens_from_response(usage)
                        record_llm_usage(
                            task=task,
                            logical_model=model,
                            api_model=api_model,
                            provider=provider,
                            input_tokens=usage_in,
                            cached_input_tokens=usage_cached,
                            output_tokens=usage_out,
                            cache_write_tokens=None,
                            attempt=attempt + 1,
                            elapsed_ms=int(elapsed * 1000),
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
                if last_error is not None and LLMProvider._llm_error_likely_transient(last_error):
                    raise LlmTransientFailure(msg) from last_error
                raise RuntimeError(msg) from last_error
        raise RuntimeError("llm_chat failed without returning")

    @staticmethod
    def json(system: str, user: str, retries: int = 1, task: str = "default") -> dict[str, Any]:
        """Call ``chat`` and parse JSON; retry with format hint; wrap bare SELECT. Raises ``LlmJsonExhausted`` when no attempt produces valid JSON."""
        total_attempts = 1 + max(0, retries)
        raw = LLMProvider.chat(system, user, task=task)
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
            raw = LLMProvider.chat(
                system,
                user
                + "\n\nFORMAT_ERROR: Output ONLY valid JSON that matches the required schema. Do NOT output raw SQL.",
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

    @staticmethod
    def batch_enabled() -> bool:
        """Return whether offline corpus generation should use the OpenAI Batch API."""
        return bool(PolicyConfig.LLM_BATCH_ENABLED and EngineConfig.LLM_PROVIDER == "openai")

    @staticmethod
    def _batch_output_text_from_body(body: dict[str, Any]) -> str:
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
        response = body.get("response")
        if isinstance(response, dict):
            nested = response.get("body")
            if isinstance(nested, dict):
                return LLMProvider._batch_output_text_from_body(nested)
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        return ""

    @staticmethod
    def _poll_openai_batch(client: OpenAI, batch_id: str, *, timeout_s: float) -> Any:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            batch = client.batches.retrieve(batch_id)
            status = str(getattr(batch, "status", "") or "")
            if status in {"completed", "failed", "cancelled", "expired"}:
                return batch
            time.sleep(2.0)
        raise RuntimeError(f"OpenAI batch {batch_id} did not complete within {timeout_s:.0f}s")

    @staticmethod
    def batch_chat(requests: list[LlmBatchRequest]) -> dict[str, str]:
        """Run independent JSON-mode completions via the OpenAI Batch API and return outputs keyed by ``custom_id``."""
        if not requests:
            return {}
        if not LLMProvider.batch_enabled():
            return {req.custom_id: LLMProvider.chat(req.system, req.user, task=req.task) for req in requests}
        providers = [p for p in LLMProvider._provider_order() if LLMProvider._provider_is_configured(p)]
        if providers != ["openai"]:
            return {req.custom_id: LLMProvider.chat(req.system, req.user, task=req.task) for req in requests}
        timeout_val = LLMProvider._resolve_llm_timeout_ms() / 1000.0
        client = LLMProvider._build_client("openai")
        lines: list[str] = []
        for req in requests:
            body = LLMProvider._llm_request_kwargs(req.system, req.user, task=req.task, timeout=timeout_val)
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
            )
        batch_file = client.files.create(file=io.BytesIO("\n".join(lines).encode("utf-8")), purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        finished = LLMProvider._poll_openai_batch(client, batch.id, timeout_s=max(timeout_val, 300.0))
        status = str(getattr(finished, "status", "") or "")
        if status != "completed":
            raise RuntimeError(f"OpenAI batch finished with status {status!r}")
        output_file_id = getattr(finished, "output_file_id", None)
        if not isinstance(output_file_id, str) or not output_file_id.strip():
            raise RuntimeError("OpenAI batch completed without an output file")
        raw_file = client.files.content(output_file_id)
        payload_raw = raw_file.read()
        payload_text = payload_raw.decode("utf-8") if isinstance(payload_raw, bytes) else str(payload_raw)
        outputs: dict[str, str] = {}
        for line in payload_text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            custom_id = str(row.get("custom_id", "") or "")
            if not custom_id:
                continue
            response = row.get("response")
            if not isinstance(response, dict):
                continue
            response_body = response.get("body")
            if not isinstance(response_body, dict):
                continue
            text = LLMProvider._batch_output_text_from_body(response_body)
            if text:
                outputs[custom_id] = text
        return outputs

    @staticmethod
    def batch_json(requests: list[LlmBatchRequest]) -> dict[str, dict[str, Any]]:
        """Parse JSON outputs from :meth:`batch_chat` for each ``custom_id``."""
        raw_by_id = LLMProvider.batch_chat(requests)
        parsed: dict[str, dict[str, Any]] = {}
        for custom_id, raw in raw_by_id.items():
            obj = safe_json_loads(raw)
            if isinstance(obj, dict):
                parsed[custom_id] = obj
        return parsed


class MockProvider:
    """Fixture replay only: return canned LLM responses from a JSON fixtures file. ``max_retries`` and ``timeout`` are ignored — no real HTTP transport, retries, or request timeouts are applied."""

    _instance: MockProvider | None = None

    def __init__(self, fixtures_path: str) -> None:
        """Load fixtures; stored user prompts are already canonical from recording."""
        path = Path(fixtures_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("fixtures", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError(f"Mock fixtures file {path} must contain a fixtures list")
        self._map: dict[tuple[str, str, str], str] = {}
        self._malformed_mock_used_questions: set[str] = set()
        for item in entries:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", "default"))
            system = str(item.get("system", ""))
            user = str(item.get("user", "")).strip()
            output = str(item.get("output_text", ""))
            self._map[(task, system, user)] = output
            for key in MockProvider._mock_fixture_lookup_aliases(task, system, user):
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
        malformed = self._malformed_mock_compose_override(system, user, task=task)
        if malformed is not None:
            return malformed
        for key in MockProvider._mock_fixture_lookup_aliases(task, system, user):
            if key in self._map:
                return self._map[key]
        handcrafted = MockProvider._handcrafted_chat_text(system, user, task=task)
        if handcrafted is not None:
            return handcrafted
        raise MockFixtureMissingError(task=task, system=system, user=user)

    @staticmethod
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

    @staticmethod
    def clear_canonical_schema_literals_cache() -> None:
        """Clear cached bundled mock-fixture key inputs (schema literals + interpret domain)."""
        global _literals_cache, _interpret_domain_cache
        _literals_cache = None
        _interpret_domain_cache = None

    @staticmethod
    def _pin_mock_interpret_domain(domain: dict[str, Any]) -> None:
        global _interpret_domain_cache
        _interpret_domain_cache = domain

    @staticmethod
    def load_canonical_interpret_domain() -> dict[str, Any] | None:
        """Return pinned interpret schema_domain for mock fixture user- key normalization."""
        return _interpret_domain_cache

    @staticmethod
    def _pin_mock_schema_literals(literals: dict[str, str]) -> None:
        global _literals_cache
        _literals_cache = {
            "owner": MockProvider.stable_schema_literal(str(literals["owner"])),
            "consumer": MockProvider.stable_schema_literal(str(literals["consumer"])),
        }

    @staticmethod
    def pin_mock_fixture_keys_from_bundle(bundle_dir: Path) -> None:
        """Load bundled schema literals and interpret domain for offline mock replay lookup."""
        literals_path = bundle_dir / SANDBOX_SCHEMA_LITERALS_FILENAME
        if literals_path.is_file():
            payload = json.loads(literals_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                owner = str(payload.get("owner", "")).strip()
                consumer = str(payload.get("consumer", "")).strip()
                if owner and consumer:
                    MockProvider._pin_mock_schema_literals({"owner": owner, "consumer": consumer})
        domain_path = bundle_dir / SANDBOX_INTERPRET_DOMAIN_FILENAME
        if domain_path.is_file():
            payload = json.loads(domain_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                MockProvider._pin_mock_interpret_domain(payload)

    @staticmethod
    def pin_schema_literal_slot(slot: str, schema_literal_json: str) -> None:
        """Merge one owner/consumer schema literal into the mock replay lookup cache."""
        global _literals_cache
        if slot not in ("owner", "consumer"):
            raise ValueError(f"unsupported schema literal slot: {slot!r}")
        stable = MockProvider.stable_schema_literal(schema_literal_json)
        if _literals_cache is None:
            _literals_cache = dict(MOCK_FIXTURE_STUB_SCHEMA_LITERALS)
        _literals_cache[slot] = stable

    @staticmethod
    def load_canonical_schema_literals() -> dict[str, str]:
        global _literals_cache
        if _literals_cache is not None:
            return _literals_cache
        return dict(MOCK_FIXTURE_STUB_SCHEMA_LITERALS)

    @staticmethod
    def stable_schema_literal(literal: str) -> str:
        try:
            parsed = json.loads(literal)
        except json.JSONDecodeError:
            return literal
        if not isinstance(parsed, dict):
            return literal
        return stable_json(parsed)

    @staticmethod
    def _schema_literal_table_count(schema_literal_json: str) -> int:
        try:
            obj = json.loads(schema_literal_json)
        except json.JSONDecodeError:
            return -1
        if not isinstance(obj, dict):
            return -1
        return sum(1 for key in obj if key != "enum_types")

    @staticmethod
    def _schema_literal_equivalent(left: str, right: str) -> bool:
        if left == right:
            return True
        return MockProvider.stable_schema_literal(left) == MockProvider.stable_schema_literal(right)

    @staticmethod
    def _canonical_schema_literal_for_embedded(embedded: str, literals: dict[str, str]) -> str:
        count = MockProvider._schema_literal_table_count(embedded)
        owner_count = MockProvider._schema_literal_table_count(literals["owner"])
        consumer_count = MockProvider._schema_literal_table_count(literals["consumer"])
        if count == consumer_count:
            return literals["consumer"]
        if count == owner_count:
            return literals["owner"]
        if abs(count - consumer_count) < abs(count - owner_count):
            return literals["consumer"]
        return literals["owner"]

    @staticmethod
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
        domain = MockProvider.load_canonical_interpret_domain()
        if domain is not None and "schema_domain" in body:
            if body.get("schema_domain") != domain:
                body["schema_domain"] = domain
                changed = True
        for key in ("schema_literal_json", "schema_summary", "schema_info"):
            if key not in body:
                continue
            raw = body[key]
            if isinstance(raw, (dict, list)):
                embedded = stable_json(raw)
            else:
                embedded = str(raw)
            canonical = MockProvider._canonical_schema_literal_for_embedded(embedded, literals)
            if not MockProvider._schema_literal_equivalent(embedded, canonical):
                try:
                    body[key] = json.loads(canonical)
                except json.JSONDecodeError:
                    body[key] = canonical
                changed = True
        if not changed:
            return user
        return stable_json(body)

    @staticmethod
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

    @staticmethod
    def mock_fixture_user_key(user: str, *, literals: dict[str, str] | None = None) -> str:
        if literals is None:
            literals = MockProvider.load_canonical_schema_literals()
        rewritten = MockProvider.rewrite_user_schema_literals(user, literals)
        stripped = LLMProvider._llm_user_text_without_sensitivity_classification(rewritten).strip()
        canonical = MockProvider._mock_fixture_user_key_body(stripped, stable=True)
        return MockProvider._normalize_mock_lookup_text(canonical)

    @staticmethod
    def legacy_mock_fixture_user_key(user: str, *, literals: dict[str, str] | None = None) -> str:
        """Pre-stable-json mock lookup key kept for fixtures recorded before canonical compaction."""
        if literals is None:
            literals = MockProvider.load_canonical_schema_literals()
        rewritten = MockProvider.rewrite_user_schema_literals(user, literals)
        stripped = LLMProvider._llm_user_text_without_sensitivity_classification(rewritten).strip()
        return MockProvider._normalize_mock_lookup_text(stripped)

    @staticmethod
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
            MockProvider.mock_fixture_user_key(user, literals=literals),
        )

    @staticmethod
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

    @staticmethod
    def _mock_fixture_lookup_aliases(
        task: str,
        system: str,
        user: str,
        *,
        literals: dict[str, str] | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return lookup keys for a stored or runtime mock user payload."""
        keys = [
            MockProvider.mock_fixture_lookup_key(task, system, user, literals=literals),
            (
                str(task),
                str(system),
                MockProvider.legacy_mock_fixture_user_key(user, literals=literals),
            ),
        ]
        if task == "default":
            question = MockProvider._extract_gatekeeper_question_text(user)
            if question:
                keys.append(MockProvider.mock_fixture_lookup_key(task, system, question, literals=literals))
                keys.append(
                    (
                        str(task),
                        str(system),
                        MockProvider.legacy_mock_fixture_user_key(question, literals=literals),
                    ),
                )
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    @staticmethod
    def _handcrafted_stage_for_system(system: str) -> str:
        if system == INTENT_INTERPRET_SYSTEM:
            return "interpret"
        if system == INTENT_GROUND_SYSTEM:
            return "ground"
        if system == INTENT_COMPOSE_SYSTEM:
            return "compose"
        return "default"

    @staticmethod
    def _mock_bundle_path() -> Path:
        override = os.environ.get("AETHERDIALECT_SANDBOX_DATA_ZIP", "").strip()
        if override:
            return Path(override)
        return Path(__file__).resolve().parent / "sandbox" / "data.zip"

    @staticmethod
    def _read_mock_bundle_json_member(filename: str) -> dict[str, object] | None:
        bundle = MockProvider._mock_bundle_path()
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

    @staticmethod
    def _load_handcrafted_entries() -> list[dict[str, object]]:
        cache_key = str(MockProvider._mock_bundle_path())
        cached = _handcrafted_entries_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = MockProvider._read_mock_bundle_json_member("sandbox_handcrafted_fixtures.json")
        if payload is None:
            _handcrafted_entries_cache[cache_key] = []
            return _handcrafted_entries_cache[cache_key]
        rows = payload.get("entries")
        if not isinstance(rows, list):
            _handcrafted_entries_cache[cache_key] = []
            return _handcrafted_entries_cache[cache_key]
        _handcrafted_entries_cache[cache_key] = [dict(row) for row in rows if isinstance(row, dict)]
        return _handcrafted_entries_cache[cache_key]

    @staticmethod
    def _question_from_mock_user(user: str) -> str | None:
        question = MockProvider._extract_gatekeeper_question_text(user)
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

    @staticmethod
    def _handcrafted_chat_text(system: str, user: str, *, task: str) -> str | None:
        question = MockProvider._question_from_mock_user(user)
        if not question:
            return None
        stage = MockProvider._handcrafted_stage_for_system(system)
        for row in MockProvider._load_handcrafted_entries():
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

    @staticmethod
    def _malformed_mock_spec_for_question(question: str) -> dict[str, object] | None:
        normalized = " ".join(question.strip().lower().split())
        for spec in SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS:
            spec_question = str(spec.get("question", "")).strip()
            if spec_question and " ".join(spec_question.strip().lower().split()) == normalized:
                return spec
        return None

    @staticmethod
    def _normalize_malformed_mock_question(question: str) -> str:
        return " ".join(question.strip().lower().split())

    def _malformed_mock_repair_output(self, spec: dict[str, object]) -> str | None:
        repair_output = spec.get("repair_output")
        if repair_output is None:
            repair_response = spec.get("repair_response")
            if isinstance(repair_response, dict):
                repair_output = json.dumps(repair_response, ensure_ascii=False)
        if repair_output is None:
            return None
        return str(repair_output)

    def _malformed_mock_compose_override(self, system: str, user: str, *, task: str) -> str | None:
        if system != INTENT_COMPOSE_SYSTEM:
            return None
        question = MockProvider._question_from_mock_user(user)
        spec: dict[str, object] | None = None
        matched_via_malformed_user = False
        if question:
            spec = MockProvider._malformed_mock_spec_for_question(question)
        if spec is None:
            for candidate in SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS:
                malformed_output = str(candidate.get("malformed_output", "")).strip()
                if malformed_output and user.strip() == malformed_output:
                    spec = candidate
                    matched_via_malformed_user = True
                    break
        if spec is None:
            return None
        repair_output = self._malformed_mock_repair_output(spec)
        if task == "intent_format":
            if repair_output is not None and (question or str(spec.get("question", "")).strip()):
                return repair_output
            return None
        if task != "intent":
            return None
        if matched_via_malformed_user:
            if repair_output is not None:
                return repair_output
            return None
        if question:
            normalized = MockProvider._normalize_malformed_mock_question(question)
            if normalized in self._malformed_mock_used_questions:
                return None
            self._malformed_mock_used_questions.add(normalized)
            return str(spec.get("malformed_output", ""))
        return None

    @staticmethod
    def _mock_cache_token(runtime: SandboxRuntimeState | None) -> str:
        if runtime is None:
            return "process"
        return f"sandbox:{id(runtime)}"

    @staticmethod
    def get() -> MockProvider:
        path = (EngineConfig.MOCK_FIXTURES_FILE or "").strip()
        if not path:
            raise RuntimeError("Mock provider requires AETHERDIALECT_MOCK_FIXTURES_FILE")
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        token = MockProvider._mock_cache_token(runtime)
        cache_key = (path, token)
        if runtime is not None:
            if runtime.mock_provider is None or runtime.mock_fixtures_path != path:
                provider = _mock_provider_instances.get(cache_key)
                if provider is None:
                    provider = MockProvider(path)
                    _mock_provider_instances[cache_key] = provider
                runtime.mock_provider = provider
                runtime.mock_fixtures_path = path
            else:
                _mock_provider_instances[cache_key] = cast(MockProvider, runtime.mock_provider)
            return cast(MockProvider, runtime.mock_provider)
        provider = _mock_provider_instances.get(cache_key)
        if provider is None:
            provider = MockProvider(path)
            _mock_provider_instances[cache_key] = provider
        MockProvider._instance = provider
        return provider

    @staticmethod
    def reset_mock_provider(*, clear_literals: bool = False) -> None:
        """Clear cached mock fixture index."""
        runtime = SandboxRuntimeState.current_sandbox_runtime()
        if runtime is not None:
            path = runtime.mock_fixtures_path or (EngineConfig.MOCK_FIXTURES_FILE or "").strip()
            token = MockProvider._mock_cache_token(runtime)
            if path:
                _mock_provider_instances.pop((path, token), None)
            runtime.mock_provider = None
            runtime.mock_fixtures_path = None
        MockProvider._instance = None
        _mock_provider_instances.clear()
        _handcrafted_entries_cache.clear()
        if clear_literals:
            MockProvider.clear_canonical_schema_literals_cache()


register_after_fork_callback(LLMProvider.clear_all_clients_after_fork)
