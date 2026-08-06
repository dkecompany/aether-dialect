"""SECURITY.md hygiene: provider inventory honesty and scoped determinism claims."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SECURITY = _REPO / "docs" / "SECURITY.md"


def _security_text() -> str:
    return _SECURITY.read_text(encoding="utf-8")


def _section_between(text: str, start_heading: str, end_heading: str | None = None) -> str:
    start = text.index(start_heading)
    if end_heading is None:
        return text[start:]
    end = text.index(end_heading, start + len(start_heading))
    return text[start:end]


@pytest.mark.fast
def test_security_section_57_not_federation_mapping_no_llm_contradiction() -> None:
    """Section 5.7 must not pair a no-provider heading with LLM disclosure bullets."""
    text = _security_text()
    section = _section_between(text, "### 5.7", "### 5.8")
    lowered = section.lower()
    assert "federation mapping suggester" not in lowered
    assert "no model call" not in lowered and "no llm" not in lowered
    assert "deterministic selection" in lowered or "deterministic context assembly" in lowered
    assert "configured provider" in lowered


@pytest.mark.fast
def test_security_section_57_join_selection_provider_assist_pattern() -> None:
    """Section 5.7 documents deterministic candidate selection then provider disambiguation."""
    section = _section_between(_security_text(), "### 5.7", "### 5.8")
    lowered = section.lower()
    assert "join" in lowered
    assert "candidate" in lowered
    assert "provider output" in lowered and "not deterministic" in lowered


@pytest.mark.fast
def test_security_scoped_determinism_preamble() -> None:
    """Section 5 states what is deterministic vs provider-mediated nondeterminism."""
    section = _section_between(_security_text(), "## 5. LLM context inventory", "## 6.")
    lowered = section.lower()
    assert "context assembly" in lowered and "deterministic" in lowered
    assert "refusal" in lowered and "deterministic" in lowered
    assert "validation" in lowered and "deterministic" in lowered
    assert "provider output" in lowered and "not deterministic" in lowered


@pytest.mark.fast
def test_security_federation_mapping_suggester_outside_llm_inventory() -> None:
    """Federation mapping suggestions are documented as having no provider call."""
    text = _security_text()
    inventory = _section_between(text, "## 5. LLM context inventory", "## 6.")
    assert "suggest_cross_source_mappings" not in inventory
    assert "suggest_cross_source_mappings" in text
    schema_area = _section_between(text, "## 3. Schema profiling", "## 4.")
    assert "suggest_cross_source_mappings" in schema_area
    assert "no provider call" in schema_area.lower()


@pytest.mark.fast
def test_security_provider_call_site_inventory_complete() -> None:
    """Every questioning-surface provider call site appears in the inventory."""
    inventory = _section_between(_security_text(), "### 5.9", "## 6.")
    lowered = inventory.lower()
    required_phrases = (
        "validate_question",
        "normalize_question_via_llm",
        "enriched_display_alias_map",
        "inspect_tabular_upload",
        "identifier naming",
        "refine_descriptions",
        "question feedback",
        "template reuse",
        "schema_base",
        "apply_column_roles_llm",
    )
    missing = [phrase for phrase in required_phrases if phrase.lower() not in lowered]
    assert not missing, f"missing provider call sites in 5.9 inventory: {missing}"


@pytest.mark.fast
def test_security_upload_inspection_identifier_call_site() -> None:
    """Upload inspection documents the identifier-naming provider call."""
    upload_section = _section_between(_security_text(), "### 5.8", "### 5.9")
    lowered = upload_section.lower()
    assert "identifier naming" in lowered
    assert "validate_upload_sources" in lowered or "inspect_tabular_upload" in lowered
    assert "label text only" in lowered or "label text" in lowered


@pytest.mark.fast
def test_security_no_unscoped_full_pipeline_determinism() -> None:
    """Threat model does not claim the full questioning pipeline is deterministic."""
    threat = _section_between(_security_text(), "## 1. Threat model", "## 2.")
    lowered = threat.lower()
    assert not re.search(
        r"full(?:y)?\s+deterministic\s+(?:question|pipeline|turn)",
        lowered,
    )
    assert "sql generation" in lowered or "deterministic sql" in lowered
    assert "provider output" in lowered or "provider-mediated" in lowered
