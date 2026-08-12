"""Hygiene tests must not bypass src package enforcement."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_HYGIENE = Path(__file__).resolve().parent
_SRC_MARKER = "src/aetherdialect"

# Environmental skips for optional trees are allowed; skipping src coverage is not.
_ALLOWED_SKIP_REASON_SUBSTR = (
    "live_tests directory not present",
    "docs directory not present",
)

_SKIP_CALL_NAMES = frozenset({"skip", "importorskip"})
_SKIP_ATTR_NAMES = frozenset({"skip", "skipif", "xfail", "importorskip"})


def _reason_allowed(reason: str) -> bool:
    text = str(reason or "")
    return any(token in text for token in _ALLOWED_SKIP_REASON_SUBSTR)


def _const_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_reason(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg in {"reason", "msg"}:
            got = _const_str(kw.value)
            if got is not None:
                return got
    if call.args:
        got = _const_str(call.args[0])
        if got is not None:
            return got
    return ""


def _is_pytest_skip_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in _SKIP_CALL_NAMES:
        return True
    if isinstance(func, ast.Attribute) and func.attr in _SKIP_ATTR_NAMES:
        return True
    return False


def _is_pytest_mark_skip(node: ast.AST) -> bool:
    # @pytest.mark.skip / skipif / xfail
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr not in {"skip", "skipif", "xfail"}:
        return False
    val = node.value
    return isinstance(val, ast.Attribute) and val.attr == "mark"


@pytest.mark.fast
def test_hygiene_tests_have_no_src_skips_or_hacks() -> None:
    """Fail if hygiene tests skip/xfail or otherwise dodge checking src/aetherdialect."""
    violations: list[str] = []
    for path in sorted(_HYGIENE.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_pytest_skip_call(node):
                reason = _call_reason(node)
                if not _reason_allowed(reason):
                    violations.append(f"{path.name}:{node.lineno}: skip/xfail {reason!r}")
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    if _is_pytest_mark_skip(target):
                        reason = _call_reason(dec) if isinstance(dec, ast.Call) else ""
                        if not _reason_allowed(reason):
                            violations.append(
                                f"{path.name}:{getattr(dec, 'lineno', node.lineno)}: mark.skip/xfail {reason!r}"
                            )
        # Textual hacks that disable hygiene scanners for src modules.
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if re.search(r"sys\.modules\[.aetherdialect", stripped):
                violations.append(f"{path.name}:{lineno}: sys.modules stub for aetherdialect")
            if (
                "pytest.skip" in stripped
                and _SRC_MARKER in stripped
                and not any(token in stripped for token in _ALLOWED_SKIP_REASON_SUBSTR)
            ):
                violations.append(f"{path.name}:{lineno}: skip mentioning src/aetherdialect")
    assert not violations, "hygiene tests must not skip/hack src coverage:\n" + "\n".join(violations)
