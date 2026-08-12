"""Question validation routes: analytical / schema_catalog / domain_knowledge / restricted / invalid."""

from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import pytest

from aetherdialect import _utils_intent
from aetherdialect._contracts_core import QuestionRoute, QuestionValidationResult
from aetherdialect._utils_intent import validate_question


def _llm_json(payload: dict[str, str]):
    return patch("aetherdialect._utils_intent.LLMProvider.json", return_value=payload)


@pytest.mark.fast
def test_analytical_route() -> None:
    with _llm_json(
        {
            "valid_database_question": "yes",
            "query_type": "analytical",
            "corrected": "count of orders",
        }
    ):
        result = validate_question("count of orders")
    assert isinstance(result, QuestionValidationResult)
    assert result.accepted is True
    assert result.route is QuestionRoute.ANALYTICAL
    assert result.corrected == "count of orders"


@pytest.mark.fast
def test_schema_catalog_route() -> None:
    with _llm_json(
        {
            "valid_database_question": "yes",
            "query_type": "schema_catalog",
            "corrected": "what tables exist",
        }
    ):
        result = validate_question("what tables exist")
    assert result.accepted is True
    assert result.route is QuestionRoute.SCHEMA_CATALOG


@pytest.mark.fast
def test_schema_catalog_count_question_fixture() -> None:
    with _llm_json(
        {
            "valid_database_question": "yes",
            "query_type": "schema_catalog",
            "corrected": "how many tables are in the schema",
        }
    ):
        result = validate_question("how many tables are in the schema")
    assert result.accepted is True
    assert result.route is QuestionRoute.SCHEMA_CATALOG


@pytest.mark.fast
def test_domain_knowledge_route() -> None:
    with _llm_json(
        {
            "valid_database_question": "yes",
            "query_type": "domain_knowledge",
            "corrected": "what does ARR mean",
        }
    ):
        result = validate_question("what does ARR mean")
    assert result.accepted is True
    assert result.route is QuestionRoute.DOMAIN_KNOWLEDGE


@pytest.mark.fast
def test_restricted_unchanged() -> None:
    with _llm_json(
        {
            "valid_database_question": "no",
            "query_type": "restricted",
            "corrected": "drop table orders",
        }
    ):
        result = validate_question("drop table orders")
    assert result.accepted is False
    assert result.route is QuestionRoute.RESTRICTED


@pytest.mark.fast
def test_allowed_alias_maps_to_analytical() -> None:
    with _llm_json(
        {
            "valid_database_question": "yes",
            "query_type": "allowed",
            "corrected": "list customers",
        }
    ):
        result = validate_question("list customers")
    assert result.accepted is True
    assert result.route is QuestionRoute.ANALYTICAL


@pytest.mark.fast
def test_no_regex_router_helper_exists() -> None:
    """Route selection must not use a regex/keyword classifier on question text."""
    forbidden_names = (
        "route_question",
        "classify_question_route",
        "question_route_from_text",
        "detect_meta_question",
        "is_schema_catalog_question",
        "is_domain_knowledge_question",
    )
    for name, obj in inspect.getmembers(_utils_intent):
        if name in forbidden_names:
            raise AssertionError(f"unexpected router helper {_utils_intent.__name__}.{name}")
        if callable(obj) and name.startswith("_") and "route" in name.lower() and "question" in name.lower():
            src = inspect.getsource(obj)
            if re.search(r"re\.(search|match|findall|compile)", src):
                raise AssertionError(f"question route helper uses regex: {name}")
    from aetherdialect._constants_runtime import QUESTION_VALIDATION_SYSTEM

    assert "analytical" in QUESTION_VALIDATION_SYSTEM
    assert "schema_catalog" in QUESTION_VALIDATION_SYSTEM
    assert "domain_knowledge" in QUESTION_VALIDATION_SYSTEM
    assert '"allowed" if' not in QUESTION_VALIDATION_SYSTEM
    assert 'query_type": "allowed"' not in QUESTION_VALIDATION_SYSTEM
