"""Session diagnostic catalogue parity and federation attribution checks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from aetherdialect import _constants
from aetherdialect._constants import SOFT_DIAGNOSTIC_CODES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "aetherdialect"
_TROUBLESHOOTING = _REPO_ROOT / "docs" / "TROUBLESHOOTING.md"
_SESSION_CODE_RE = re.compile(r"\| `([^`]+)` \|")
_PIPELINE_TRACE_SECTION = "## Pipeline trace markers"


def _diagnostic_constant_map() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(_constants).items()
        if name.startswith("DIAGNOSTIC_CODE_") and isinstance(value, str)
    }


def _resolve_code(node: ast.expr | None, const_map: dict[str, str]) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in const_map:
        return const_map[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        key = f"{node.value.id}.{node.attr}"
        if key in const_map:
            return const_map[key]
    return None


def _details_pairs(node: ast.Call) -> dict[str, str]:
    for kw in node.keywords:
        if kw.arg != "details" or not isinstance(kw.value, ast.Tuple):
            continue
        out: dict[str, str] = {}
        for elt in kw.value.elts:
            if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                continue
            key_node, val_node = elt.elts
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if isinstance(val_node, ast.Constant):
                    out[key_node.value] = str(val_node.value)
                elif isinstance(val_node, ast.JoinedStr):
                    out[key_node.value] = "<dynamic>"
                else:
                    out[key_node.value] = "<expr>"
        return out
    return {}


def _keyword_str(node: ast.Call, name: str) -> str | None:
    for kw in node.keywords:
        if kw.arg != name:
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
        return "<expr>"
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _DiagnosticEmissionVisitor(ast.NodeVisitor):
    def __init__(self, const_map: dict[str, str]) -> None:
        self._const_map = const_map
        self.codes: set[str] = set()
        self._local_strings: dict[str, str] = {}
        self.federation_attribution_gaps: list[str] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            resolved = _resolve_code(node.value, self._const_map)
            if resolved:
                self._local_strings[node.targets[0].id] = resolved
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call) or _call_name(call) != "phase_timer":
                continue
            code = _resolve_code(_keyword("code", call), self._const_map)
            if code:
                self.codes.add(code)
            if code and code.startswith("FEDERATION_"):
                self._record_federation_call(call, label="phase_timer")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name == "notify":
            code = _resolve_code(_keyword("code", node), self._const_map)
            if code is None:
                code = self._local_strings.get(_keyword_name("code", node) or "", "")
            if code:
                self.codes.add(code)
                if code.startswith("FEDERATION_"):
                    self._record_federation_call(node, label="notify")
        elif name == "pipeline_trace" and node.args:
            code = _resolve_code(node.args[0], self._const_map)
            if code:
                self.codes.add(code)
        elif name == "Diagnostic":
            code = _resolve_code(_keyword("code", node), self._const_map)
            if code is None and "code" in {kw.arg for kw in node.keywords if kw.arg}:
                for kw in node.keywords:
                    if kw.arg == "code" and isinstance(kw.value, ast.Name):
                        code = self._local_strings.get(kw.value.id)
            if code:
                self.codes.add(code)
                if code.startswith("FEDERATION_") and _keyword("source_id", node) is None:
                    self.federation_attribution_gaps.append("Diagnostic missing source_id")
        self.generic_visit(node)

    def _record_federation_call(self, node: ast.Call, *, label: str) -> None:
        details = _details_pairs(node)
        has_source = _keyword("source_id", node) is not None or bool(details.get("source_id"))
        has_phase = _keyword("phase", node) is not None or bool(details.get("phase"))
        if not has_source or not has_phase:
            self.federation_attribution_gaps.append(f"{label} missing source_id or phase")


def _keyword(name: str, call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _keyword_name(name: str, call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Name):
            return kw.value.id
    return None


def discover_emitted_session_diagnostic_codes() -> set[str]:
    const_map = _diagnostic_constant_map()
    emitted: set[str] = set(SOFT_DIAGNOSTIC_CODES)
    for name, value in const_map.items():
        if name.startswith("DIAGNOSTIC_CODE_DATA_QUALITY_"):
            emitted.add(value)
    visitor = _DiagnosticEmissionVisitor(const_map)
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor.visit(tree)
    emitted.update(visitor.codes)
    return emitted


def declared_session_diagnostic_codes() -> set[str]:
    return set(_diagnostic_constant_map().values())


def parse_documented_session_diagnostic_codes() -> set[str]:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    body, _, _tail = text.partition(_PIPELINE_TRACE_SECTION)
    codes = set(_SESSION_CODE_RE.findall(body))
    return {code for code in codes if code}


def parse_documented_pipeline_trace_markers() -> set[str]:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    if _PIPELINE_TRACE_SECTION not in text:
        return set()
    tail = text.split(_PIPELINE_TRACE_SECTION, 1)[1]
    return set(_SESSION_CODE_RE.findall(tail))


def federation_attribution_gaps() -> list[str]:
    const_map = _diagnostic_constant_map()
    visitor = _DiagnosticEmissionVisitor(const_map)
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor.visit(tree)
    return visitor.federation_attribution_gaps


@pytest.mark.fast
def test_session_diagnostic_catalogue_matches_emitted_set() -> None:
    emitted = discover_emitted_session_diagnostic_codes()
    documented = parse_documented_session_diagnostic_codes()
    assert emitted == documented, (
        f"missing from docs: {sorted(emitted - documented)}; documented but not emitted: {sorted(documented - emitted)}"
    )


@pytest.mark.fast
def test_pipeline_trace_markers_are_documented_separately() -> None:
    documented = parse_documented_pipeline_trace_markers()
    assert documented == {
        "ENUM_PROMPT_TRUNCATED",
        "FEDERATION_JOIN_CANDIDATE_CAP",
        "FEDERATION_MAPPING_DRIFT",
        "JOIN_CANDIDATE_CAP",
    }


@pytest.mark.fast
def test_every_declared_diagnostic_code_is_emitted_or_removed() -> None:
    declared = declared_session_diagnostic_codes()
    emitted = discover_emitted_session_diagnostic_codes()
    orphaned = sorted(declared - emitted)
    assert not orphaned, f"declared but never emitted: {orphaned}"


@pytest.mark.fast
def test_federation_diagnostics_carry_source_id_and_phase() -> None:
    gaps = federation_attribution_gaps()
    assert gaps == []


@pytest.mark.fast
def test_refusal_diagnostic_codes_are_declared_emitted_and_documented() -> None:
    from aetherdialect._refusal_diagnostics import REFUSAL_DIAGNOSTIC_CODES

    declared = declared_session_diagnostic_codes()
    emitted = discover_emitted_session_diagnostic_codes()
    documented = parse_documented_session_diagnostic_codes()
    missing_declared = sorted(REFUSAL_DIAGNOSTIC_CODES - declared)
    missing_emitted = sorted(REFUSAL_DIAGNOSTIC_CODES - emitted)
    missing_docs = sorted(REFUSAL_DIAGNOSTIC_CODES - documented)
    assert not missing_declared, f"refusal codes missing from constants: {missing_declared}"
    assert not missing_emitted, f"refusal codes never emitted: {missing_emitted}"
    assert not missing_docs, f"refusal codes missing from TROUBLESHOOTING.md: {missing_docs}"
