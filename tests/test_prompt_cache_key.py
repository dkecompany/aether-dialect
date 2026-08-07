"""prompt_cache_key length cap and stability."""

from __future__ import annotations

import hashlib

import pytest

from aetherdialect._core_utils import business_knowledge_scope, prompt_cache_schema_scope
from aetherdialect._llm_provider import LLMProvider


@pytest.mark.fast
def test_key_at_most_64_chars() -> None:
    long_hash = "h" * 80
    with prompt_cache_schema_scope(long_hash):
        key = LLMProvider.resolve_prompt_cache_key("intent_compose_task")
    assert key is not None
    assert len(key) <= 64
    raw = f"intent_compose_task:{long_hash}"
    assert key == f"intent_c:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@pytest.mark.fast
def test_key_stable_for_same_inputs() -> None:
    schema_hash = "abc123def456"
    digest = "digestvalue0123456789"
    with prompt_cache_schema_scope(schema_hash):
        with business_knowledge_scope((), digest):
            a = LLMProvider.resolve_prompt_cache_key("intent")
            b = LLMProvider.resolve_prompt_cache_key("intent")
    assert a == b
    assert a is not None
    assert len(a) <= 64
    assert a.startswith("intent:")
