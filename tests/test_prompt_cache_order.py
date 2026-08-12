"""Tests for static-first prompt serialization and prompt_cache_key routing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants_runtime import (
    FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER,
    META_SCHEMA_CATALOG_PROMPT_KEY_ORDER,
    SEMANTIC_REPAIR_PROMPT_KEY_ORDER,
)
from aetherdialect._contracts_base import DomainKnowledgeEntry
from aetherdialect._contracts_core import InterpretPlan
from aetherdialect._intent_loop import (
    build_intent_ground_prompt,
    build_intent_interpret_prompt,
    build_intent_semantic_repair_prompt,
)
from aetherdialect._llm_provider import LLMProvider, MockProvider
from aetherdialect._sql_gen import build_join_choice_prompt
from aetherdialect._utils import (
    domain_context_payload,
    prompt_cache_schema_scope,
    prompt_json,
    schema_prompt_cache_id,
    stable_json,
)


@pytest.fixture
def _non_mock_llm_provider():
    prev = EngineConfig.LLM_PROVIDER
    EngineConfig.LLM_PROVIDER = "openai"
    yield
    EngineConfig.LLM_PROVIDER = prev


def _key_index(payload: str, key: str) -> int:
    return payload.index(f'"{key}"')


class TestPromptJson:
    def test_static_keys_precede_question(self) -> None:
        body = {
            "question": "how many orders?",
            "schema_domain": {"tables": {"orders": {}}},
            "task": "interpret",
            "supported_capabilities": ["filter"],
        }
        ordered = prompt_json(
            body,
            ("task", "supported_capabilities", "schema_domain", "question"),
        )
        assert _key_index(ordered, "schema_domain") < _key_index(ordered, "question")
        assert _key_index(ordered, "task") < _key_index(ordered, "schema_domain")

    def test_differs_from_stable_json_when_alphabetical_puts_question_first(self) -> None:
        body = {
            "question": "q",
            "schema_summary": {"tables": {"t": {}}},
            "task": "parse",
        }
        ordered = prompt_json(body, ("task", "schema_summary", "question"))
        canonical = stable_json(body)
        assert ordered != canonical
        assert _key_index(ordered, "schema_summary") < _key_index(ordered, "question")

    def test_trailing_keys_preserve_body_insertion_order(self) -> None:
        body = {"task": "t", "extra_a": 1, "extra_b": 2}
        ordered = prompt_json(body, ("task",))
        assert ordered.index('"extra_a"') < ordered.index('"extra_b"')

    def test_fuzzy_reuse_prompt_places_param_slots_before_question(self) -> None:
        body = {
            "task": "extract",
            "extraction_rules": [],
            "output_format": {},
            "matched_question": "old",
            "matched_values": {},
            "param_keys": ["p1"],
            "param_slots": [{"param_key": "p1"}],
            "question": "new question",
        }
        ordered = prompt_json(body, FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER)
        assert "parameterized_sql" not in ordered
        assert _key_index(ordered, "param_slots") < _key_index(ordered, "question")


class TestIntentPromptOrdering:
    def test_interpret_prompt_schema_before_question(self) -> None:
        payload = build_intent_interpret_prompt(
            "count customers",
            '{"tables":{"customers":{}}}',
            "",
            (),
        )
        assert _key_index(payload, "schema_domain") < _key_index(payload, "question")

    def test_ground_prompt_schema_before_interpret_plan_and_question(self) -> None:
        plan = InterpretPlan(approach="count rows", tables=("customers",))
        payload = build_intent_ground_prompt(
            "count customers",
            plan,
            '{"customers":{"columns":{}}}',
            "",
            "[]",
            (),
            (),
        )
        assert _key_index(payload, "schema_literal_json") < _key_index(payload, "interpret_plan")
        assert _key_index(payload, "interpret_plan") < _key_index(payload, "question")


class TestJoinPromptOrdering:
    def test_join_choice_prompt_static_before_question(self) -> None:
        scopes = [
            {
                "scope": "main",
                "tables": ["a", "b"],
                "candidates": [{"candidate_id": "J01", "join_path_signature": []}],
            },
        ]
        _system, user = build_join_choice_prompt("how many?", "SELECT 1", scopes)
        assert _key_index(user, "output_format") < _key_index(user, "question")
        assert _key_index(user, "scopes") < _key_index(user, "question")

    def test_join_prior_feedback_in_user_not_system(self) -> None:
        scopes = [
            {
                "scope": "main",
                "tables": ["a", "b"],
                "candidates": [{"candidate_id": "J01", "join_path_signature": []}],
            },
        ]
        feedback = ["avoid orders-customers via billing"]
        system, user = build_join_choice_prompt(
            "how many?",
            "SELECT 1",
            scopes,
            prior_join_feedback=feedback,
        )
        assert "avoid orders-customers via billing" in user
        assert "avoid orders-customers via billing" not in system
        assert _key_index(user, "question") < _key_index(user, "prior_join_feedback")


class TestMetaSchemaCatalogOrdering:
    def test_meta_schema_catalog_schema_before_question(self) -> None:
        body = {"question": "how many tables?", "schema": {"tables": {"orders": {}}}}
        ordered = prompt_json(body, META_SCHEMA_CATALOG_PROMPT_KEY_ORDER)
        assert _key_index(ordered, "schema") < _key_index(ordered, "question")


class TestSemanticRepairOrdering:
    def test_semantic_repair_schema_before_errors_and_intent(self) -> None:
        payload = build_intent_semantic_repair_prompt(
            "count customers",
            '{"tables":["customers"]}',
            [],
            [],
            '{"tables":{"customers":{}}}',
        )
        assert _key_index(payload, "schema_info") < _key_index(payload, "errors_to_fix")
        assert _key_index(payload, "schema_info") < _key_index(payload, "current_intent")
        assert SEMANTIC_REPAIR_PROMPT_KEY_ORDER.index("schema_info") < SEMANTIC_REPAIR_PROMPT_KEY_ORDER.index(
            "errors_to_fix"
        )


class TestDomainContextOrdering:
    def test_domain_context_payload_sorted_by_key(self) -> None:
        entries = (
            DomainKnowledgeEntry(key="zebra", kind="note", text="z"),
            DomainKnowledgeEntry(key="alpha", kind="note", text="a"),
        )
        payload = domain_context_payload(entries)
        assert payload is not None
        assert [row["key"] for row in payload] == ["alpha", "zebra"]


class TestMockFixtureLookupUnchanged:
    def test_mock_fixture_user_key_still_order_independent(self) -> None:
        a = stable_json({"question": "q", "schema_summary": {"t": 1}, "task": "t"})
        b = prompt_json(
            {"task": "t", "schema_summary": {"t": 1}, "question": "q"},
            ("task", "schema_summary", "question"),
        )
        assert a != b
        assert MockProvider.mock_fixture_user_key(a) == MockProvider.mock_fixture_user_key(b)


class TestPromptCacheKey:
    def test_resolve_without_schema_hash_still_returns_family_key(self) -> None:
        assert LLMProvider.resolve_prompt_cache_key("intent") == "intent"

    def test_resolve_with_schema_hash(self) -> None:
        with prompt_cache_schema_scope("schema-graph-abc"):
            assert LLMProvider.resolve_prompt_cache_key("intent") == "intent:schema-graph-abc"

    @pytest.mark.usefixtures("_non_mock_llm_provider")
    def test_llm_chat_attaches_prompt_cache_key(self) -> None:
        mock_resp = MagicMock()
        mock_resp.output_text = "{}"
        client = MagicMock()
        client.responses.create.return_value = mock_resp

        with prompt_cache_schema_scope("graph-123"):
            with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
                with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
                    with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=client):
                        with patch("aetherdialect._utils.debug"):
                            with patch("aetherdialect._utils.pipeline_trace"):
                                LLMProvider.chat("sys", "usr", max_retries=1, task="intent")

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs.get("prompt_cache_key") == "intent:graph-123"

    @pytest.mark.usefixtures("_non_mock_llm_provider")
    def test_llm_chat_attaches_family_only_prompt_cache_key_without_schema_scope(self) -> None:
        mock_resp = MagicMock()
        mock_resp.output_text = "{}"
        client = MagicMock()
        client.responses.create.return_value = mock_resp

        with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
            with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
                with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=client):
                    with patch("aetherdialect._utils.debug"):
                        with patch("aetherdialect._utils.pipeline_trace"):
                            LLMProvider.chat("sys", "usr", max_retries=1, task="join")

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs.get("prompt_cache_key") == "join"

    def test_llm_request_kwargs_attaches_prompt_cache_key(self) -> None:
        with prompt_cache_schema_scope("batch-graph-9"):
            kwargs = LLMProvider._llm_request_kwargs("sys", "usr", task="intent", timeout=30.0)
        assert kwargs.get("prompt_cache_key") == "intent:batch-graph-9"

    @pytest.mark.usefixtures("_non_mock_llm_provider")
    def test_fuzzy_reuse_extraction_attaches_prompt_cache_key(self) -> None:
        from types import SimpleNamespace

        from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
        from aetherdialect._contracts_core import ConcreteIntent, Template, ValueHistory
        from aetherdialect._contracts_schema import SQLShape, TemplateStats
        from aetherdialect._pipeline_generate import extract_fuzzy_reuse_params

        intent_sig = ConcreteIntent(
            intent_id="t1",
            tables=["customers"],
            grain="many",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("customers.name"),
                        op="=",
                        value_type="string",
                        param_key="p1",
                    ),
                ]
            ),
        )
        template = Template(
            id="T0001",
            effective_structural_hash="eff1",
            intent_signature=intent_sig,
            intent_key="ik1",
            tables_used=["customers"],
            sql_param="SELECT 1",
            sql_fp="fp1",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="cm1",
            value_history=ValueHistory(
                param_values=[{"p1": "old"}],
                questions=["old question"],
                natural_language=[],
            ),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        schema = SimpleNamespace(schema_graph_id="reuse-graph-42", deny_columns={}, tables={})
        with prompt_cache_schema_scope(schema_prompt_cache_id(schema)):
            expected_cache_key = LLMProvider.resolve_prompt_cache_key("default")
        mock_resp = MagicMock()
        mock_resp.output_text = '{"param_values": {"p1": "new"}}'
        client = MagicMock()
        client.responses.create.return_value = mock_resp

        with patch("aetherdialect._llm_provider.LLMProvider._provider_order", return_value=["openai"]):
            with patch("aetherdialect._llm_provider.LLMProvider._provider_is_configured", return_value=True):
                with patch("aetherdialect._llm_provider.LLMProvider._build_client", return_value=client):
                    with patch("aetherdialect._utils.debug"):
                        with patch("aetherdialect._utils.pipeline_trace"):
                            extract_fuzzy_reuse_params(
                                "new question",
                                template,
                                history_index=0,
                                literal_structural_only=True,
                                schema=schema,
                            )

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs.get("prompt_cache_key") == expected_cache_key


def test_interpret_prompt_parses_as_json() -> None:
    payload = build_intent_interpret_prompt(
        "count customers",
        '{"tables":{"customers":{}}}',
        "",
        (),
    )
    parsed = json.loads(payload)
    assert parsed["question"] == "count customers"
