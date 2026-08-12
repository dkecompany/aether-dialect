"""Docs and ``_PUBLIC_API`` alignment with the stable import surface."""

from __future__ import annotations

import re
from pathlib import Path

import aetherdialect
from aetherdialect import AsyncPipelineSession, PipelineSession
from aetherdialect.aetherdialect import _PUBLIC_API

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_API_REFERENCE = _DOCS / "API_REFERENCE.md"
_TROUBLESHOOTING = _DOCS / "TROUBLESHOOTING.md"
_INTEGRATOR_GUIDE = _DOCS / "INTEGRATOR_GUIDE.md"
_USER_GUIDE = _DOCS / "USER_GUIDE.md"


def _api_reference_text() -> str:
    return _API_REFERENCE.read_text(encoding="utf-8")


def _session_step_section(md: str) -> str:
    return md.split("### SessionStep", 1)[1].split("\n### ", 1)[0]


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
    """SessionStep docs cover the public session contract."""
    api = _api_reference_text()
    troubleshooting = _TROUBLESHOOTING.read_text(encoding="utf-8")
    step_section = _session_step_section(api)

    for field in ("data_truncated", "answer", "error", "diagnostics", "intent_summary"):
        assert f"`{field}`" in step_section, f"SessionStep table missing `{field}`"

    assert "## SessionOutcome" in api
    assert "## SessionError" in api
    assert "phase_callback" in api
    assert "diagnostic_sink" in api
    assert "## PhaseProgressEvent" in api
    assert "Troubleshooting" in api
    assert "## REFUSAL_CATALOGUE" in troubleshooting

    sessions_section = api.split("## Sessions", 1)[1].split("## Structure and knowledge documents", 1)[0]
    assert re.search(r"`cancel\(\)`", sessions_section)
    assert "cancel_active_federation_turn" not in sessions_section

    assert callable(PipelineSession.cancel)
    assert callable(AsyncPipelineSession.cancel)


def test_guarantees_section_present() -> None:
    """Integrator guide includes a Guarantees section."""
    integrator = _INTEGRATOR_GUIDE.read_text(encoding="utf-8")
    assert "## Guarantees" in integrator

    guarantees = integrator.split("## Guarantees", 1)[1].split("\n## ", 1)[0]
    for token in (
        "ask rebuild",
        "Template reuse",
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
    # ConfigSnapshot may appear in __all__ while remaining outside _PUBLIC_API.
    extra_in_public = [name for name in extra_in_public if name != "ConfigSnapshot"]
    assert extra_in_public == [], f"extra in _PUBLIC_API: {extra_in_public}"
