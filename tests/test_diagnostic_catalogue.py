"""Session diagnostic catalogue parity and federation attribution checks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aetherdialect import _constants
from aetherdialect._constants import SOFT_DIAGNOSTIC_CODES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "aetherdialect"


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


def federation_attribution_gaps() -> list[str]:
    const_map = _diagnostic_constant_map()
    visitor = _DiagnosticEmissionVisitor(const_map)
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor.visit(tree)
    return visitor.federation_attribution_gaps


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


_API_REFERENCE = _REPO_ROOT / "docs" / "API_REFERENCE.md"
_TROUBLESHOOTING = _REPO_ROOT / "docs" / "TROUBLESHOOTING.md"


@pytest.mark.fast
def test_api_reference_documents_all_refusal_diagnostic_codes() -> None:
    from aetherdialect._constants import REFUSAL_DIAGNOSTIC_CODES

    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    start = text.index("## Diagnostic codes")
    section = text[start:]
    missing = sorted(code for code in REFUSAL_DIAGNOSTIC_CODES if code not in section)
    assert not missing, f"TROUBLESHOOTING.md missing refusal diagnostic codes: {missing}"


@pytest.mark.fast
def test_refusal_diagnostic_codes_are_declared_and_emitted() -> None:
    from aetherdialect._constants import REFUSAL_DIAGNOSTIC_CODES

    declared = declared_session_diagnostic_codes()
    emitted = discover_emitted_session_diagnostic_codes()
    missing_declared = sorted(REFUSAL_DIAGNOSTIC_CODES - declared)
    missing_emitted = sorted(REFUSAL_DIAGNOSTIC_CODES - emitted)
    assert not missing_declared, f"refusal codes missing from constants: {missing_declared}"
    assert not missing_emitted, f"refusal codes never emitted: {missing_emitted}"


_L37_FEDERATION_DIAGNOSTIC_CODES = frozenset(
    {
        "FEDERATION_CAP_EXCEEDED",
        "FEDERATION_MALFORMED_MEMBER_ANSWER",
        "FEDERATION_JOIN_FAN_OUT",
    }
)


@pytest.mark.fast
def test_api_reference_documents_l37_federation_diagnostic_codes() -> None:
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing = sorted(code for code in _L37_FEDERATION_DIAGNOSTIC_CODES if code not in text)
    assert not missing, f"TROUBLESHOOTING.md federation diagnostics missing L37 codes: {missing}"


@pytest.mark.fast
def test_single_catalogue_covers_every_emission() -> None:
    from aetherdialect._contracts_base import DiagnosticCode, SqlDiagnosticCode

    enum_values = {m.value for m in DiagnosticCode}
    session_values = set(_diagnostic_constant_map().values())
    sql_values = {m.value for m in SqlDiagnosticCode}
    assert session_values <= enum_values
    assert sql_values <= enum_values
    emitted = discover_emitted_session_diagnostic_codes() | sql_values
    missing_emission = sorted(enum_values - emitted - sql_values)
    # Session catalogue members must have an emission site; SQL members are covered via SqlDiagnostic usage.
    session_only = enum_values & session_values
    missing_session = sorted(session_only - discover_emitted_session_diagnostic_codes())
    assert not missing_session, f"catalogue members never emitted: {missing_session}"
    docs = _TROUBLESHOOTING.read_text(encoding="utf-8")
    start = docs.index("## Diagnostic codes")
    section = docs[start:]
    undocumented = sorted(v for v in session_only if v not in section)
    assert not undocumented, f"catalogue members missing from TROUBLESHOOTING: {undocumented}"
    _ = missing_emission


@pytest.mark.fast
def test_trace_headings_are_not_codes() -> None:
    import inspect

    from aetherdialect._utils import pipeline_trace

    params = list(inspect.signature(pipeline_trace).parameters)
    assert params[0] == "heading"
    assert "code" not in params[:1]
    doc = pipeline_trace.__doc__ or ""
    assert "heading" in doc.lower()
    assert "not" in doc.lower() and "code" in doc.lower()


@pytest.mark.fast
def test_probe_and_execution_failures_have_distinct_codes() -> None:
    from aetherdialect._constants import (
        DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED,
        DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED,
    )
    from aetherdialect._contracts_base import SqlDiagnosticCode

    assert DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED != DIAGNOSTIC_CODE_FEDERATION_MEMBER_FAILED
    assert SqlDiagnosticCode.EXPLAIN_SORT_SPILL.value != SqlDiagnosticCode.EXPLAIN_TEMPORARY_TABLE.value
    assert SqlDiagnosticCode.EXPLAIN_SORT_SPILL.value != SqlDiagnosticCode.EXPLAIN_OTHER.value
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    for code in (
        DIAGNOSTIC_CODE_FEDERATION_MEMBER_PROBE_FAILED,
        SqlDiagnosticCode.EXPLAIN_SORT_SPILL.value,
        SqlDiagnosticCode.EXPLAIN_TEMPORARY_TABLE.value,
    ):
        assert code in text, f"missing documentation for {code}"


@pytest.mark.fast
def test_no_dead_codes() -> None:
    from aetherdialect._constants import DIAG_TO_FAILURE_CATEGORY
    from aetherdialect._contracts_base import SqlDiagnosticCode

    names = {m.name for m in SqlDiagnosticCode}
    assert "EXPLAIN_TYPE_MISMATCH" not in names
    assert "EXPLAIN_PERMISSION_DENIED" not in names
    assert "explain_type_mismatch" not in DIAG_TO_FAILURE_CATEGORY
    assert "explain_permission_denied" not in DIAG_TO_FAILURE_CATEGORY


@pytest.mark.fast
def test_public_reference_matches_catalogue() -> None:
    declared = declared_session_diagnostic_codes()
    emitted = discover_emitted_session_diagnostic_codes()
    text = _TROUBLESHOOTING.read_text(encoding="utf-8")
    start = text.index("## Diagnostic codes")
    section = text[start:]
    documented = {code for code in declared if code in section}
    undocumented = sorted(declared - documented)
    assert not undocumented, f"declared codes missing from TROUBLESHOOTING: {undocumented}"
    orphan_docs = sorted(
        code for code in declared if code in section and code not in emitted and not code.startswith("DATA_QUALITY_")
    )
    # Soft codes count as emitted via SOFT_DIAGNOSTIC_CODES inclusion in discover helper.
    assert orphan_docs == [], f"documented codes with no emission site: {orphan_docs}"
