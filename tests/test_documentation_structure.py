"""Documentation structure and vocabulary consistency checks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"


def _github_slug(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def _heading_slug(markdown: str, *, prefix: str) -> str:
    match = re.search(rf"^###\s+({re.escape(prefix)}.+)$", markdown, re.MULTILINE)
    assert match is not None, f"missing heading starting with {prefix!r}"
    return _github_slug(match.group(1))


@pytest.mark.fast
def test_documentation_ownership_links_to_canonical_guides() -> None:
    getting_started = (_DOCS / "GETTING_STARTED.md").read_text(encoding="utf-8")
    how_it_works = (_DOCS / "HOW_IT_WORKS.md").read_text(encoding="utf-8")
    assert "SANDBOX.md" in getting_started or "Sandbox guide" in getting_started
    assert "Support matrix" in how_it_works or "SUPPORT_MATRIX.md" in how_it_works


@pytest.mark.fast
def test_pre_commit_enforces_coverage_floor() -> None:
    text = (_REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "--cov-fail-under=68" in text


@pytest.mark.fast
def test_support_matrix_is_canonical_for_refused_constructs() -> None:
    text = (_DOCS / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    assert "## Refused constructs" in text
    assert "LATERAL" in text
    assert "Unrelated table pairing" in text
    assert "compare an entity to itself" in text.lower()


@pytest.mark.fast
def test_sandbox_guide_leads_with_engine_session_mode() -> None:
    text = (_DOCS / "SANDBOX.md").read_text(encoding="utf-8")
    assert 'mode="writer"' in text or "mode='writer'" in text
    assert "logistics" in text
    assert "crm" in text


@pytest.mark.fast
def test_sandbox_guide_matches_closed_world_surface() -> None:
    text = (_DOCS / "SANDBOX.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "preset=" not in text
    assert "restricted_consumer" not in text
    assert "engine_context=scope" in text.replace(" ", "") or "offline_sandbox(engine_context=" in text
    assert "sandbox.engine(" in text
    assert "with Sandbox() as" in text
    assert "notes.txt" in text
    assert "init_notices" in text
    assert "resolved to bundled fixture" in text
    assert "maintainer_access" in text
    assert "closed-world" in lowered or "closed world" in lowered
    assert 'role="consumer"' in text or "role='consumer'" in text


@pytest.mark.fast
def test_api_reference_documents_federation_declaration_surface() -> None:
    text = (_DOCS / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "export_federation_declaration" in text
    assert "declaration_file" in text


@pytest.mark.fast
def test_how_it_works_mentions_federated_member_statements() -> None:
    text = (_DOCS / "HOW_IT_WORKS.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "byte-identical" in lowered or "same deterministic renderer" in lowered
    assert "arrow" in lowered or "frame" in lowered


@pytest.mark.fast
def test_readme_lists_documentation_reading_order() -> None:
    text = (_REPO / "README.md").read_text(encoding="utf-8")
    assert "Reading order" in text


@pytest.mark.fast
def test_security_documents_replica_column_projection() -> None:
    text = (_DOCS / "SECURITY.md").read_text(encoding="utf-8")
    assert "staff_id" in text
    assert "password" in text.lower()
    assert "cardinality" in text.lower()


@pytest.mark.fast
def test_getting_started_anchor_matches_api_reference_link() -> None:
    api = (_DOCS / "API_REFERENCE.md").read_text(encoding="utf-8")
    getting_started = (_DOCS / "GETTING_STARTED.md").read_text(encoding="utf-8")
    slug = _heading_slug(getting_started, prefix="Step 4")
    assert f"GETTING_STARTED.md#{slug}" in api


@pytest.mark.fast
def test_sandbox_data_reference_exists_with_key_sections() -> None:
    path = _DOCS / "SANDBOX_DATA_REFERENCE.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## Overview",
        "## Single-engine schema",
        "## Views",
        "## Bundled notes",
        "## Overrides and sensitivity fixtures",
        "## Consumer scopes",
        "## Federation topology",
        "## Offline versus live",
        "## Question corpus",
        "## Authoring checklist",
    ):
        assert heading in text
    assert "34" in text
    assert "active_customer_v" in text
    assert "sandbox_rental_shop" in text
    assert "views_questions" in text


@pytest.mark.fast
def test_sandbox_guide_links_to_data_reference() -> None:
    text = (_DOCS / "SANDBOX.md").read_text(encoding="utf-8")
    assert "SANDBOX_DATA_REFERENCE.md" in text


_INTEGRATOR_GUIDE = _DOCS / "INTEGRATOR_GUIDE.md"
_FORBIDDEN_ARTIFACT_EDIT_PATTERNS = (
    re.compile(r"(?<!never )edit\s+persisted\s+sidecars", re.IGNORECASE),
    re.compile(r"(?<!never )edit\s+.*sidecars?\s+under", re.IGNORECASE),
)
_EXPORT_APPLY_PAIRS = (
    ("export_schema_overrides", "apply_schema_overrides"),
    ("export_federation_declaration", "apply_federation_declaration"),
    ("export_aetherspace", "apply_aetherspace"),
    ("preview_migration_map", "apply_migration_map"),
)


_API_REFERENCE = _DOCS / "API_REFERENCE.md"
_FORBIDDEN_INTERNAL_EXPORT_PATTERNS = (
    re.compile(r"export_federation_manifest\b"),
    re.compile(r"export_federation_mappings\b"),
    re.compile(r"export_manifest\b"),
    re.compile(r"export_mappings\b"),
)
_FORBIDDEN_DERIVED_ROSTER_PATTERNS = (
    re.compile(r"derived roster", re.IGNORECASE),
    re.compile(r"member roster", re.IGNORECASE),
    re.compile(r"Derived at compose time"),
    re.compile(r"`sources` and `table_namespace` are derived"),
    re.compile(r"`sources\[\]`"),
)


def _assert_doc_has_no_internal_federation_exports(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_INTERNAL_EXPORT_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"{path.name} documents internal federation export {match.group(0)!r}" if match else None
    for pattern in _FORBIDDEN_DERIVED_ROSTER_PATTERNS:
        match = pattern.search(text)
        assert match is None, (
            f"{path.name} documents derived federation roster material: {match.group(0)!r}" if match else None
        )


@pytest.mark.fast
def test_api_reference_has_no_internal_federation_exports() -> None:
    _assert_doc_has_no_internal_federation_exports(_API_REFERENCE)


@pytest.mark.fast
def test_integrator_guide_has_no_internal_federation_exports() -> None:
    _assert_doc_has_no_internal_federation_exports(_INTEGRATOR_GUIDE)


@pytest.mark.fast
def test_integrator_guide_never_edit_artifacts_sidecars() -> None:
    text = _INTEGRATOR_GUIDE.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_ARTIFACT_EDIT_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"INTEGRATOR_GUIDE.md tells users to edit artifacts/sidecars directly: {match.group(0)!r}"
    lowered = text.lower()
    assert "never edit" in lowered or "library-owned" in lowered or "library owned" in lowered, (
        "INTEGRATOR_GUIDE.md must state the artifacts directory is library-owned and never edited by hand"
    )
    for export_fn, apply_fn in _EXPORT_APPLY_PAIRS:
        assert export_fn in text, f"INTEGRATOR_GUIDE.md must document {export_fn}"
        assert apply_fn in text, f"INTEGRATOR_GUIDE.md must document {apply_fn}"


# --- Shared vocabulary across documents ---

_UPLOAD_GUIDES = (
    _DOCS / "INTEGRATOR_GUIDE.md",
    _DOCS / "SUPPORT_MATRIX.md",
    _DOCS / "USER_GUIDE.md",
)
_CANONICAL_UPLOAD_PHRASE = "inspect first, then construct"
_SECURITY_UPLOAD_ANCHOR = "SECURITY.md#58-upload-inspection-csv-file-engine"
_ROLE_VOCABULARY_GUIDES = ("INTEGRATOR_GUIDE.md", "SANDBOX.md", "HOW_IT_WORKS.md")
_FEDERATION_DECLARATION_GUIDES = (
    "GETTING_STARTED.md",
    "INTEGRATOR_GUIDE.md",
    "SANDBOX.md",
    "USER_GUIDE.md",
)


@pytest.mark.fast
def test_upload_guides_share_canonical_phrase() -> None:
    for path in _UPLOAD_GUIDES:
        text = path.read_text(encoding="utf-8").lower()
        assert _CANONICAL_UPLOAD_PHRASE in text, f"{path.name} must use {_CANONICAL_UPLOAD_PHRASE!r}"


@pytest.mark.fast
def test_upload_construction_failures_raise_config_error() -> None:
    for path in _UPLOAD_GUIDES:
        text = path.read_text(encoding="utf-8")
        assert "ConfigError" in text, f"{path.name} must name ConfigError for upload failures"
        assert "construction refuses" not in text.lower(), (
            f"{path.name} must use raises ConfigError, not construction refuses, for upload path"
        )


@pytest.mark.fast
def test_security_upload_links_use_canonical_anchor() -> None:
    for name in ("USER_GUIDE.md", "INTEGRATOR_GUIDE.md"):
        text = (_DOCS / name).read_text(encoding="utf-8")
        assert _SECURITY_UPLOAD_ANCHOR in text, f"{name} must link to {_SECURITY_UPLOAD_ANCHOR}"
        assert "#56-upload-inspection" not in text, f"{name} must not use stale security anchor"


@pytest.mark.fast
def test_role_vocabulary_engine_role_and_session_mode() -> None:
    for name in _ROLE_VOCABULARY_GUIDES:
        text = (_DOCS / name).read_text(encoding="utf-8").lower()
        assert "engine role" in text, f"{name} must use engine role vocabulary"
        assert "session mode" in text, f"{name} must use session mode vocabulary"


@pytest.mark.fast
def test_federation_declaration_canonical_filename() -> None:
    for name in _FEDERATION_DECLARATION_GUIDES:
        text = (_DOCS / name).read_text(encoding="utf-8")
        assert "federation_declaration.json" in text, f"{name} must name federation_declaration.json"
        assert "declaration_file" in text, f"{name} must document declaration_file="


@pytest.mark.fast
def test_user_guide_federation_reentry_links_not_internal_exports() -> None:
    text = (_DOCS / "USER_GUIDE.md").read_text(encoding="utf-8")
    assert "export_federation_manifest" not in text
    assert "export_federation_mappings" not in text
    assert "INTEGRATOR_GUIDE.md#artifacts-are-library-owned" in text


@pytest.mark.fast
def test_user_guide_aetherfederation_uses_declaration_not_manifest() -> None:
    text = (_DOCS / "USER_GUIDE.md").read_text(encoding="utf-8")
    start = text.index("## AetherFederation")
    end = text.index("## FederationContext")
    section = text[start:end]
    assert "federation declaration" in section.lower()
    assert "in the manifest" not in section.lower()


@pytest.mark.fast
def test_support_matrix_links_user_guide_for_upload_detail() -> None:
    text = (_DOCS / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    assert "USER_GUIDE.md#csv-and-excel-uploads" in text
