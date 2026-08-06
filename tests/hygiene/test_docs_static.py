"""Static documentation structure and vocabulary hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DOCS = _REPO / "docs"

_INTERNAL_DOC_IMPORT_RE = re.compile(
    r"(?:from\s+aetherdialect\s+import\s+_|import\s+aetherdialect\._|aetherdialect\._[A-Za-z]\w*)"
)
_FORBIDDEN_DOC_FEDERATION_EXPORT_RE = re.compile(r"\bexport_federation_(?:manifest|mappings)\b")
_SUPPORT_MATRIX_INTENT_CARRIES_RE = re.compile(r"intent carry", re.IGNORECASE)
_FORBIDDEN_DELETED_DOC_RE = re.compile(
    r"docs/(?:TROUBLESHOOTING|CHANGELOG)\.md",
    re.IGNORECASE,
)
_DOC_META_HEADING_RE = re.compile(r"^##\s+(New|Updated|Changelog)\b", re.IGNORECASE)
_DOC_PLAN_TEST_FILE_RE = re.compile(r"test_(?:step\d+|phase_[a-z])", re.IGNORECASE)


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
        "## Bundled notes and named spaces",
        "## Overrides and sensitivity fixtures",
        "## Consumer scopes",
        "## Federation topology",
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
    ("export_overrides", "apply_overrides"),
    ("export_federation_declaration", "apply_federation_declaration"),
    ("export_aetherspace", "apply_aetherspace"),
    ("preview_migration_map", "apply_migration_map"),
)

_API_REFERENCE = _DOCS / "API_REFERENCE.md"
_DOCS_GUIDES = tuple(sorted(_DOCS.glob("*.md")) if _DOCS.is_dir() else ())

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
def test_sandbox_guide_no_internal_federation_exports() -> None:
    text = (_DOCS / "SANDBOX.md").read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_INTERNAL_EXPORT_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"SANDBOX.md documents internal federation export {match.group(0)!r}" if match else None
    assert "export_federation_declaration" in text
    assert "apply_federation_declaration" in text


@pytest.mark.fast
def test_integrator_guide_owner_vs_consumer_roles_anchor() -> None:
    text = _INTEGRATOR_GUIDE.read_text(encoding="utf-8")
    assert "SANDBOX.md#owner-vs-consumer-roles" in text
    assert "owner-vs-consumer-presets" not in text


@pytest.mark.fast
def test_api_reference_refusal_codes_use_refusal_wording() -> None:
    from aetherdialect._constants import REFUSAL_DIAGNOSTIC_CODES

    text = _API_REFERENCE.read_text(encoding="utf-8")
    start = text.index("**Terminal refusals**")
    section = text[start : text.index("**Federation diagnostics**", start)]
    assert "refusal" in section.lower()
    missing = sorted(code for code in REFUSAL_DIAGNOSTIC_CODES if code not in section)
    assert not missing, f"API_REFERENCE.md terminal refusals section missing codes: {missing}"


@pytest.mark.fast
def test_support_matrix_links_user_guide_for_upload_detail() -> None:
    text = (_DOCS / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    assert "USER_GUIDE.md#csv-and-excel-uploads" in text


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", _DOCS_GUIDES)
def test_all_docs_avoid_internal_federation_exports(doc_path: Path) -> None:
    text = doc_path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_INTERNAL_EXPORT_PATTERNS:
        match = pattern.search(text)
        assert match is None, (
            f"{doc_path.name} documents internal federation export {match.group(0)!r}" if match else None
        )


@pytest.mark.fast
def test_sandbox_data_reference_matches_shipped_sources() -> None:
    """Table and federation inventories in SANDBOX_DATA_REFERENCE.md must match scripts/data sources."""
    import json
    import re

    repo = Path(__file__).resolve().parents[2]
    reference = (repo / "docs" / "SANDBOX_DATA_REFERENCE.md").read_text(encoding="utf-8")
    sql_text = (repo / "scripts" / "data" / "rental_shop.sql").read_text(encoding="utf-8")
    partition = json.loads((repo / "scripts" / "data" / "federation_partition.json").read_text(encoding="utf-8"))
    declaration = json.loads((repo / "scripts" / "data" / "federation_declaration.json").read_text(encoding="utf-8"))

    doc_tables = {
        match.group(1)
        for match in re.finditer(r"^#### `([^`]+)`\s*$", reference, flags=re.MULTILINE)
        if match.group(1) not in {"active_customer_v", "store_revenue_v", "film_catalog_v"}
    }
    sql_tables = {match.group(1) for match in re.finditer(r"^CREATE TABLE\s+(\w+)\s*\(", sql_text, flags=re.MULTILINE)}
    assert doc_tables == sql_tables
    assert len(doc_tables) == 34

    for member, tables in partition.items():
        if not isinstance(tables, list):
            continue
        section = re.search(
            rf"\| `{re.escape(member)}` \| DuckDB in-memory \| (.+?) \|",
            reference,
        )
        assert section is not None, f"missing federation member row for {member!r}"
        listed = {name.strip().strip("`") for name in section.group(1).split(",")}
        assert listed == set(tables), f"federation member {member!r} table list drift"

    declared_members = {
        str(member.get("source"))
        for table in declaration.get("logical_tables", [])
        if isinstance(table, dict)
        for member in table.get("members", [])
        if isinstance(member, dict) and member.get("source")
    }
    assert declared_members.issubset(set(partition))


def _doc_body_sections(lines: list[str]) -> list[int]:
    """Return line indices of ``##`` headings that are not the front- matter ``## Sections``."""
    return [idx for idx, line in enumerate(lines) if line.startswith("## ") and not line.startswith("## Sections")]


def _doc_front_matter_violations(doc_path: Path) -> list[str]:
    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations: list[str] = []
    if not lines or not lines[0].startswith("# "):
        violations.append("first line must be an H1 title")
        return violations
    try:
        reading_idx = next(idx for idx, line in enumerate(lines) if line.startswith("**Reading order:**"))
    except StopIteration:
        violations.append("missing **Reading order:** line")
        return violations
    try:
        sections_idx = next(idx for idx, line in enumerate(lines) if line.strip() == "## Sections")
    except StopIteration:
        violations.append("missing ## Sections heading")
        return violations
    about_lines = [line for line in lines[1:reading_idx] if line.strip()]
    if not about_lines:
        violations.append("missing About paragraph before **Reading order:**")
    if not (reading_idx < sections_idx):
        violations.append("**Reading order:** must appear before ## Sections")
    separator_indices = [idx for idx, line in enumerate(lines) if line.strip() == "---"]
    if not separator_indices:
        violations.append("missing --- separator before body")
        return violations
    body_headings = _doc_body_sections(lines)
    if not body_headings:
        violations.append("missing body section heading (## …)")
        return violations
    first_body = body_headings[0]
    if not any(idx < first_body for idx in separator_indices):
        violations.append("--- must appear before the first body section")
    return violations


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", _DOCS_GUIDES)
def test_docs_avoid_internal_federation_export_names(doc_path: Path) -> None:
    """User docs must document declaration export/apply, not internal manifest/mappings exports."""
    text = doc_path.read_text(encoding="utf-8")
    match = _FORBIDDEN_DOC_FEDERATION_EXPORT_RE.search(text)
    if match:
        rel = doc_path.relative_to(_REPO)
        pytest.fail(f"{rel} documents removed federation export {match.group(0)!r}")


@pytest.mark.fast
def test_support_matrix_avoids_intent_carries_phrasing() -> None:
    """Dialect notes must describe user-visible filters, not internal intent phrasing."""
    text = (_DOCS / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    match = _SUPPORT_MATRIX_INTENT_CARRIES_RE.search(text)
    assert match is None, f"SUPPORT_MATRIX.md uses internal phrasing: {match.group(0)!r}" if match else None


@pytest.mark.fast
def test_tests_do_not_require_deleted_troubleshooting_or_changelog_docs() -> None:
    """Catalogue tests must point at API_REFERENCE or code constants, not deleted docs."""
    tests_dir = _REPO / "tests"
    hits: list[str] = []
    paths = sorted(tests_dir.glob("test_*.py")) + sorted((tests_dir / "hygiene").glob("test_*.py"))
    for path in paths:
        if path.name in {"test_static_core.py", "test_static_hygiene.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            if _FORBIDDEN_DELETED_DOC_RE.search(line):
                hits.append(f"{path.name}:{idx}: {line.strip()}")
    if hits:
        pytest.fail("Tests still reference deleted docs:\n" + "\n".join(hits))


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", _DOCS_GUIDES)
def test_docs_avoid_internal_import_paths(doc_path: Path) -> None:
    """User docs must not teach ``aetherdialect._*`` internal import paths."""
    text = doc_path.read_text(encoding="utf-8")
    hits = [
        f"line {idx}: {line.strip()}"
        for idx, line in enumerate(text.splitlines(), 1)
        if _INTERNAL_DOC_IMPORT_RE.search(line)
    ]
    if hits:
        rel = doc_path.relative_to(_REPO)
        pytest.fail(f"Internal import path(s) in {rel}:\n" + "\n".join(hits))


_DOC_META_HEADING_RE = re.compile(r"^##\s+(New|Updated|Changelog)\b", re.IGNORECASE)
_DOC_PLAN_TEST_FILE_RE = re.compile(r"test_(?:step\d+|phase_[a-z])", re.IGNORECASE)


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", _DOCS_GUIDES)
def test_doc_guides_follow_front_matter_template(doc_path: Path) -> None:
    """Each docs/*.md guide uses About → Reading order → Sections → --- → body."""
    violations = _doc_front_matter_violations(doc_path)
    if violations:
        rel = doc_path.relative_to(_REPO)
        pytest.fail(f"{rel} front-matter violations:\n" + "\n".join(violations))


@pytest.mark.fast
@pytest.mark.parametrize("doc_path", _DOCS_GUIDES)
def test_docs_avoid_plan_meta_headings(doc_path: Path) -> None:
    """User docs must not use changelog-style headings or reference plan-style test filenames."""
    text = doc_path.read_text(encoding="utf-8")
    hits: list[str] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if _DOC_META_HEADING_RE.search(line):
            hits.append(f"line {idx}: {line.strip()}")
        if _DOC_PLAN_TEST_FILE_RE.search(line):
            hits.append(f"line {idx}: plan-style test filename reference: {line.strip()}")
    if hits:
        rel = doc_path.relative_to(_REPO)
        pytest.fail(f"Plan meta heading(s) in {rel}:\n" + "\n".join(hits))
