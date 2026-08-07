"""Regression guard against plan-only vocabulary in library, script, and test code."""

from __future__ import annotations

import ast
import re
import tokenize
from io import BytesIO
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_LIBRARY_AND_SCRIPT_ROOTS = (
    _REPO_ROOT / "src" / "aetherdialect",
    _REPO_ROOT / "scripts",
)
_TEST_ROOT = _REPO_ROOT / "tests"
_DOCS_ROOT = _REPO_ROOT / "docs"
_LIVE_TEST_ROOT = _REPO_ROOT / "live_tests"

_PLAN_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-O]\d{1,2}|T\d{2})\b(?![:\d])")
_TIER_LETTER_RE = re.compile(r"\bTier [A-Z]\b")
_BRACKET_PLAN_RE = re.compile(r"\[P\d+\]")
_PHASE_PLAN_RE = re.compile(r"\bPhase [A-Z0-9]+\b")
_STEP_NUMBER_RE = re.compile(r"\bstep \d+\b", re.IGNORECASE)
_PLAN_P_RE = re.compile(r"\bplan p\d+\b", re.IGNORECASE)
_PLAN_ITEMS_RE = re.compile(r"\bfederation plan (?:items|gaps|completion)\b", re.IGNORECASE)
_AUDIT_GAP_RE = re.compile(r"\baudit gap\b", re.IGNORECASE)
_PLAN_SECTION_RE = re.compile(r"\bplan section \d+(?:\.\d+)?\b", re.IGNORECASE)
_PLAN_LETTER_SECTION_RE = re.compile(r"\bplan [A-Z]\.\d+\b")
_STEPS_RANGE_RE = re.compile(r"\bsteps \d+-\d+\b", re.IGNORECASE)
_TIER_HYPHEN_RE = re.compile(r"\bTier-[A-Z]\b")
_NOT_YET_IMPLEMENTED_RE = re.compile(r"\bnot yet implemented\b", re.IGNORECASE)
_FUTURE_WORK_RE = re.compile(r"\bfuture work\b", re.IGNORECASE)
_LATER_PHASE_RE = re.compile(r"\bin a later phase\b", re.IGNORECASE)
_PART_REFERENCE_RE = re.compile(r"\bPart [A-Z]\b")

_PATTERN_CHECKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_PLAN_ID_RE, "plan step id"),
    (_TIER_LETTER_RE, "tier letter"),
    (_TIER_HYPHEN_RE, "tier hyphen letter"),
    (_BRACKET_PLAN_RE, "bracket plan id"),
    (_PHASE_PLAN_RE, "phase plan id"),
    (_STEP_NUMBER_RE, "numbered step reference"),
    (_STEPS_RANGE_RE, "numbered steps range"),
    (_PLAN_P_RE, "plan p-id"),
    (_PLAN_SECTION_RE, "plan section id"),
    (_PLAN_LETTER_SECTION_RE, "plan letter section id"),
    (_PLAN_ITEMS_RE, "plan items/gaps"),
    (_AUDIT_GAP_RE, "audit gap"),
)

_DOCS_ONLY_PATTERN_CHECKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_NOT_YET_IMPLEMENTED_RE, "not yet implemented"),
    (_FUTURE_WORK_RE, "future work"),
    (_LATER_PHASE_RE, "later phase reference"),
    (_PART_REFERENCE_RE, "part reference"),
)

_DOCS_PATTERN_CHECKS: tuple[tuple[re.Pattern[str], str], ...] = (
    tuple(item for item in _PATTERN_CHECKS if item[0] not in (_STEP_NUMBER_RE, _STEPS_RANGE_RE))
    + _DOCS_ONLY_PATTERN_CHECKS
)

_ALLOWED_TIER_PATHS = frozenset(
    {
        _REPO_ROOT / "src" / "aetherdialect" / "_constants.py",
    }
)

_DOMAIN_PHASE_ALLOWLIST = frozenset(
    {
        "SCHEMA_BUILD_PHASE_A",
        "SCHEMA_BUILD_PHASE_B",
        "SCHEMA_BUILD_PHASE_C",
        "SCHEMA_BUILD_PHASE_D",
        "SCHEMA_BUILD_PHASE_E",
        "SCHEMA_BUILD_PHASE_F",
        "SCHEMA_BUILD_PHASE_G",
        "SCHEMA_BUILD_PHASE_H",
        "FEDERATION_COMPOSITION_PHASE_E",
        "FEDERATION_COMPOSITION_PHASE_F",
        "FEDERATION_COMPOSITION_PHASE_G",
        "FEDERATION_COMPOSITION_PHASE_H",
        "ASK_PHASE_A",
        "ASK_PHASE_B",
        "ASK_PHASE_C",
        "ASK_PHASE_D",
        "ASK_PHASE_E",
        "ASK_PHASE_F",
        "ASK_PHASE_G",
        "ASK_PHASE_H",
        "ASK_PHASE_I",
        "ASK_PHASE_J",
        "ASK_PHASE_K",
        "ASK_PHASE_L",
        "ASK_PHASE_M",
        "ASK_PHASE_N",
    }
)

_PLAN_STYLE_TEST_FILE_RE = re.compile(
    r"(?:test_(?:federation_(?:plan_|generic_plan)|federation_p\d+)|_[iltkjgp]\d+(?:_[iltkjgp]\d+)*(?:_|\.py$))",
    re.IGNORECASE,
)

_JOIN_CANDIDATE_ID_RE = re.compile(r"\bJ\d{2}\b")
_SPREADSHEET_RANGE_RE = re.compile(r"[A-Z]+\d+:[A-Z]+\d+")
_DOMAIN_TOKEN_ALLOWLIST = frozenset({"K8"})


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _iter_markdown_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.md"))


def _line_has_domain_phase_allowlist(text: str) -> bool:
    return any(token in text for token in _DOMAIN_PHASE_ALLOWLIST)


def _plan_id_match_is_domain_false_positive(match: str, line: str) -> bool:
    if match in _DOMAIN_TOKEN_ALLOWLIST:
        return True
    if _JOIN_CANDIDATE_ID_RE.fullmatch(match):
        return True
    for cell_range in _SPREADSHEET_RANGE_RE.findall(line):
        if match in cell_range:
            return True
    lowered = line.lower()
    if re.fullmatch(r"[A-Z]\d", match) and any(
        token in lowered for token in (" range", " grid", "spreadsheet", "excel")
    ):
        return True
    return False


def _module_docstring(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return None
    return ast.get_docstring(tree)


def _comment_surfaces(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, comment_text) pairs from a Python source file."""
    raw = path.read_bytes()
    surfaces: list[tuple[int, str]] = []
    try:
        for tok in tokenize.tokenize(BytesIO(raw).readline):
            if tok.type == tokenize.COMMENT:
                surfaces.append((tok.start[0], tok.string))
    except tokenize.TokenError:
        return surfaces
    return surfaces


def _docstring_surfaces(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, docstring_text) pairs from module, class, and function docstrings."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    surfaces: list[tuple[int, str]] = []
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        surfaces.append((tree.body[0].lineno, tree.body[0].value.value))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc:
            surfaces.append((node.lineno, doc))
    return surfaces


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _scan_text_surfaces(
    path: Path,
    surfaces: list[tuple[int, str]],
    *,
    skip_fixture_writes: bool = False,
    pattern_checks: tuple[tuple[re.Pattern[str], str], ...] = _PATTERN_CHECKS,
) -> list[str]:
    rel = _display_path(path)
    hits: list[str] = []
    for line_no, text in surfaces:
        for line in text.splitlines():
            if skip_fixture_writes and "f.write(" in line and _PHASE_PLAN_RE.search(line):
                continue
            if path.resolve() in _ALLOWED_TIER_PATHS and _TIER_LETTER_RE.search(line):
                continue
            if _line_has_domain_phase_allowlist(line):
                continue
            for pattern, label in pattern_checks:
                found = pattern.search(line)
                if not found:
                    continue
                if pattern is _PLAN_ID_RE and _plan_id_match_is_domain_false_positive(found.group(0), line):
                    continue
                hits.append(f"{rel}:{line_no}: {label}: {line.strip()}")
                break
    return hits


def _plan_metacommentary_hits(path: Path, *, skip_fixture_writes: bool = False) -> list[str]:
    surfaces = _comment_surfaces(path) + _docstring_surfaces(path)
    return _scan_text_surfaces(path, surfaces, skip_fixture_writes=skip_fixture_writes)


def _markdown_plan_metacommentary_hits(path: Path) -> list[str]:
    surfaces = [(idx, line) for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)]
    return _scan_text_surfaces(path, surfaces, pattern_checks=_DOCS_PATTERN_CHECKS)


@pytest.mark.fast
def test_no_plan_step_ids_in_library_or_scripts() -> None:
    """Library and maintainer scripts must not embed plan step identifiers."""
    hits: list[str] = []
    for root in _LIBRARY_AND_SCRIPT_ROOTS:
        for path in _iter_python_files(root):
            hits.extend(_plan_metacommentary_hits(path))
    assert not hits, "plan metacommentary found:\n" + "\n".join(hits)


@pytest.mark.fast
def test_no_tier_letter_metacommentary_outside_constants() -> None:
    """Docstrings and comments must not use ad-hoc tier-letter plan vocabulary."""
    hits: list[str] = []
    for root in _LIBRARY_AND_SCRIPT_ROOTS:
        for path in _iter_python_files(root):
            if path.resolve() in _ALLOWED_TIER_PATHS:
                continue
            for line_no, text in _comment_surfaces(path) + _docstring_surfaces(path):
                if _line_has_domain_phase_allowlist(text):
                    continue
                for match in _TIER_LETTER_RE.finditer(text):
                    hits.append(f"{path.relative_to(_REPO_ROOT)}:{line_no}:{match.group(0)}")
    assert not hits, "tier-letter metacommentary found:\n" + "\n".join(hits)


@pytest.mark.fast
def test_no_plan_metacommentary_in_tests() -> None:
    """Tests must not embed plan vocabulary in module docstrings or comments."""
    hits: list[str] = []
    for path in _iter_python_files(_TEST_ROOT):
        if path.name == "test_plan_meta.py":
            continue
        hits.extend(_plan_metacommentary_hits(path, skip_fixture_writes=True))
    assert not hits, "plan metacommentary in tests:\n" + "\n".join(hits)


@pytest.mark.fast
def test_no_plan_metacommentary_in_live_tests() -> None:
    """Live tests must not embed plan vocabulary in comments or docstrings."""
    hits: list[str] = []
    if not _LIVE_TEST_ROOT.is_dir():
        pytest.skip("live_tests directory not present")
    for path in _iter_python_files(_LIVE_TEST_ROOT):
        hits.extend(_plan_metacommentary_hits(path))
    assert not hits, "plan metacommentary in live_tests:\n" + "\n".join(hits)


@pytest.mark.fast
def test_docs_plan_phrase_in_markdown_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("This feature is not yet implemented.\n", encoding="utf-8")
    hits = _markdown_plan_metacommentary_hits(path)
    assert hits


@pytest.mark.fast
def test_docs_part_reference_in_markdown_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("See Part T for migration tiering.\n", encoding="utf-8")
    hits = _markdown_plan_metacommentary_hits(path)
    assert hits


@pytest.mark.fast
def test_no_plan_metacommentary_in_docs() -> None:
    """Published documentation must not embed plan vocabulary or roadmap phrasing."""
    hits: list[str] = []
    if not _DOCS_ROOT.is_dir():
        pytest.skip("docs directory not present")
    for path in _iter_markdown_files(_DOCS_ROOT):
        hits.extend(_markdown_plan_metacommentary_hits(path))
    assert not hits, "plan metacommentary in docs:\n" + "\n".join(hits)


@pytest.mark.fast
def test_no_plan_style_test_filenames() -> None:
    """Test modules must not encode plan ids in their filenames."""
    hits = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_python_files(_TEST_ROOT)
        if _PLAN_STYLE_TEST_FILE_RE.search(path.name)
    ]
    assert not hits, "plan-style test filenames:\n" + "\n".join(hits)


@pytest.mark.fast
def test_section_letter_identifier_in_comment_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("# fan-out guard from C24\nx = 1\n", encoding="utf-8")
    hits = _plan_metacommentary_hits(path)
    assert hits


@pytest.mark.fast
def test_section_letter_identifier_in_data_literal_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "allowed.py"
    path.write_text('LABEL = "C24"\n', encoding="utf-8")
    assert not _plan_metacommentary_hits(path)


@pytest.mark.fast
def test_domain_phase_allowlist_still_passes_guard() -> None:
    text = "# uses ASK_PHASE_J for pipeline tracing"
    assert not _scan_text_surfaces(Path("sample.py"), [(1, text)])
