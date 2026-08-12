"""prompt_cache_key length cap and stability."""

from __future__ import annotations

import hashlib

import pytest

from aetherdialect._llm_provider import LLMProvider
from aetherdialect._utils import domain_knowledge_scope, prompt_cache_schema_scope


@pytest.mark.fast
def test_key_at_most_64_chars() -> None:
    long_hash = "h" * 80
    long_family = "intent_compose_task_family_name_is_quite_long"
    with prompt_cache_schema_scope(long_hash):
        key = LLMProvider.resolve_prompt_cache_key(long_family)
    assert key is not None
    assert len(key) <= 64
    raw = f"{long_family}:{long_hash[:32]}"
    assert len(raw) > 64
    assert key == f"{long_family[:8]}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@pytest.mark.fast
def test_key_stable_for_same_inputs() -> None:
    schema_hash = "abc123def456"
    digest = "digestvalue0123456789"
    with prompt_cache_schema_scope(schema_hash):
        with domain_knowledge_scope((), digest):
            a = LLMProvider.resolve_prompt_cache_key("intent")
            b = LLMProvider.resolve_prompt_cache_key("intent")
    assert a == b
    assert a is not None
    assert len(a) <= 64
    assert a.startswith("intent:")
