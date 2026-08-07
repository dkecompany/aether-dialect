"""Public-consumer hygiene for docs/SANDBOX_DATA_REFERENCE.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "docs" / "SANDBOX_DATA_REFERENCE.md"
_SANDBOX_GUIDE = _REPO / "docs" / "SANDBOX.md"
_QUESTIONS = _REPO / "scripts" / "data" / "sandbox_questions.txt"

_FORBIDDEN_INTERNAL_PATHS = (
    "live_tests/",
    "scripts/",
    "src/",
)

_FORBIDDEN_PHRASES = (
    "Offline versus live",
    "restricted consumer path",
    "activates the restricted consumer path",
    "**See also:**",
)


def _reference_text() -> str:
    return _REFERENCE.read_text(encoding="utf-8")


def _tier_heading(tier_name: str) -> str:
    display = "views_questions" if tier_name == "views questions" else tier_name
    return f"### `{display}` tier"


def _question_tiers() -> dict[str, list[str]]:
    tiers: dict[str, list[str]] = {}
    current: str | None = None
    for raw in _QUESTIONS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            current = line[2:].strip()
            tiers[current] = []
            continue
        if current is not None:
            tiers[current].append(line)
    return tiers


@pytest.mark.fast
def test_sandbox_data_reference_has_no_offline_versus_live_section() -> None:
    text = _reference_text()
    assert "## Offline versus live" not in text
    assert "[Offline versus live]" not in text


@pytest.mark.fast
def test_sandbox_data_reference_has_no_see_also_footer() -> None:
    text = _reference_text()
    assert "**See also:**" not in text


@pytest.mark.fast
def test_sandbox_data_reference_has_no_internal_repo_paths() -> None:
    text = _reference_text()
    for fragment in _FORBIDDEN_INTERNAL_PATHS:
        assert fragment not in text, f"SANDBOX_DATA_REFERENCE.md must not reference {fragment!r}"


@pytest.mark.fast
def test_sandbox_data_reference_has_no_restricted_consumer_path_language() -> None:
    text = _reference_text()
    for phrase in _FORBIDDEN_PHRASES[1:3]:
        assert phrase not in text, f"unexpected phrase {phrase!r}"


@pytest.mark.fast
def test_sandbox_guide_has_no_restricted_consumer_path_language() -> None:
    text = _SANDBOX_GUIDE.read_text(encoding="utf-8")
    for phrase in ("restricted consumer path", "activates the restricted consumer path"):
        assert phrase not in text, f"SANDBOX.md must not use {phrase!r}"


@pytest.mark.fast
def test_sandbox_data_reference_questions_are_individual_bullets() -> None:
    text = _reference_text()
    tiers = _question_tiers()
    for tier_name, questions in tiers.items():
        heading = _tier_heading(tier_name)
        assert heading in text, f"missing tier heading {heading!r}"
        section = text.split(heading, 1)[1].split("\n## ", 1)[0]
        for question in questions:
            assert f"- {question}" in section, f"tier {tier_name!r} question must be its own bullet: {question!r}"
        assert " · " not in section, f"tier {tier_name!r} must not use middot paragraphs"


@pytest.mark.fast
def test_sandbox_data_reference_publishes_bundled_note_fixtures() -> None:
    text = _reference_text()
    for fixture_name in ("rental_shop_notes.txt", "sandbox_space_catalog_notes.txt"):
        assert fixture_name in text
    notes_path = _REPO / "scripts" / "data" / "rental_shop_notes.txt"
    catalog_path = _REPO / "scripts" / "data" / "sandbox_space_catalog_notes.txt"
    assert notes_path.read_text(encoding="utf-8").strip() in text
    assert catalog_path.read_text(encoding="utf-8").strip() in text


@pytest.mark.fast
def test_sandbox_data_reference_publishes_demo_overrides_json() -> None:
    text = _reference_text()
    demo = (_REPO / "scripts" / "data" / "sandbox_overrides_demo.json").read_text(encoding="utf-8").strip()
    assert "sandbox_overrides_demo.json" in text
    assert demo in text


@pytest.mark.fast
def test_sandbox_data_reference_consumer_scope_uses_example_allow_list() -> None:
    text = _reference_text()
    consumer = text.split("## Consumer scopes", 1)[1].split("\n## ", 1)[0]
    assert 'role="consumer"' in consumer or "role='consumer'" in consumer
    assert "example" in consumer.lower()
    assert "customer" in consumer and "payment" in consumer and "rental" in consumer
    assert re.search(r"permission|schema", consumer, re.IGNORECASE)


@pytest.mark.fast
def test_sandbox_data_reference_documents_spaces_without_required_notes() -> None:
    text = _reference_text()
    assert "SpaceContext" in text
    lowered = text.lower()
    assert "without notes" in lowered or "notes are optional" in lowered or "optional" in lowered
