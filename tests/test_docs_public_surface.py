"""Docs and ``_PUBLIC_API`` alignment with the stable import surface."""

from __future__ import annotations

import re
from pathlib import Path

import aetherdialect
from aetherdialect import AsyncPipelineSession, PipelineSession
from aetherdialect.aetherdialect import _PUBLIC_API

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_API_REFERENCE = _DOCS / "API_REFERENCE.md"
_INTEGRATOR_GUIDE = _DOCS / "INTEGRATOR_GUIDE.md"
_USER_GUIDE = _DOCS / "USER_GUIDE.md"


def _api_reference_text() -> str:
    return _API_REFERENCE.read_text(encoding="utf-8")


def _session_step_section(md: str) -> str:
    return md.split("## SessionStep", 1)[1].split("\n## ", 1)[0]


def _public_api_names() -> set[str]:
    names: set[str] = set()
    for item in _PUBLIC_API:
        if isinstance(item, str):
            if item == aetherdialect.__version__:
                names.add("__version__")
            else:
                names.add(item)
        else:
            names.add(item.__name__)
    return names


def test_session_step_fields_documented() -> None:
    """SessionStep docs cover public fields; refusal codes live on diagnostics."""
    api = _api_reference_text()
    integrator = _INTEGRATOR_GUIDE.read_text(encoding="utf-8")
    user_guide = _USER_GUIDE.read_text(encoding="utf-8")
    step_section = _session_step_section(api)

    for field in ("retryable", "notices", "data_truncated"):
        assert f"`{field}`" in step_section, f"SessionStep table missing `{field}`"

    assert "`refusal_diagnostic_code`" not in step_section

    refusal_catalog = api.split("**Terminal refusals**", 1)[1].split("**Federation diagnostics**", 1)[0]
    assert "SessionStep.refusal_diagnostic_code" not in refusal_catalog
    assert "SessionStep.diagnostics" in refusal_catalog

    checklist = integrator.split("### SessionStep fields (embedding checklist)", 1)[1].split("\n### ", 1)[0]
    assert "`refusal_diagnostic_code`" not in checklist
    assert "`diagnostics`" in checklist

    assert "construction_phase_callback" in api
    assert "ask_phase_callback" in api
    assert "## PhaseProgressEvent" in api

    audit_catalog = api.split("#### Audit `event_type` catalog", 1)[1].split("### LLM usage", 1)[0]
    assert re.search(r"^\| `close` \|", audit_catalog, re.MULTILINE)
    assert re.search(r"^\| `federation_semijoin_key_transfer` \|", audit_catalog, re.MULTILINE)

    pipeline_section = api.split("## PipelineSession methods", 1)[1].split("## AsyncPipelineSession", 1)[0]
    assert "reuse_saved_question" in pipeline_section
    assert re.search(r"`cancel\(\)`", pipeline_section)
    assert "Deprecated" in pipeline_section and "cancel_active_federation_turn" in pipeline_section

    async_section = api.split("## AsyncPipelineSession methods", 1)[1].split("## Package helpers", 1)[0]
    assert re.search(r"`cancel\(\)`", async_section)
    assert "Deprecated" in async_section and "cancel_active_federation_turn" in async_section

    helpers_section = api.split("## Package helpers", 1)[1].split("\n## ", 1)[0]
    assert "cancel()" in helpers_section or "`cancel()`" in helpers_section
    assert "cancel_active_federation_turn" not in helpers_section.split("Federation cancellation", 1)[1][:120]

    assert "reuse_saved_question" in user_guide

    assert hasattr(PipelineSession, "reuse_saved_question")
    assert callable(PipelineSession.cancel)
    assert callable(PipelineSession.cancel_active_federation_turn)
    assert callable(AsyncPipelineSession.cancel)


def test_guarantees_section_present() -> None:
    """Integrator guide states closed-enough openness guarantees."""
    integrator = _INTEGRATOR_GUIDE.read_text(encoding="utf-8")
    assert "## Guarantees" in integrator

    guarantees = integrator.split("## Guarantees", 1)[1].split("\n## ", 1)[0]
    for token in (
        "ask rebuild",
        "template reuse",
        "join signature",
        "federation egress",
        "fail-open",
        "SUPPORT_MATRIX",
    ):
        assert token in guarantees, f"Guarantees section missing token: {token}"


def test_public_api_matches_all() -> None:
    """``_PUBLIC_API`` names match ``aetherdialect.__all__`` exactly."""
    public_names = _public_api_names()
    all_names = set(aetherdialect.__all__)
    missing_from_public = sorted(all_names - public_names)
    extra_in_public = sorted(public_names - all_names)
    assert missing_from_public == [], f"missing from _PUBLIC_API: {missing_from_public}"
    assert extra_in_public == [], f"extra in _PUBLIC_API: {extra_in_public}"
