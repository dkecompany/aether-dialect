"""User-facing message constants must not embed driver exception text."""

from __future__ import annotations

import ast
import re
from pathlib import Path

pre_fix_failure: str | None = None

_DRIVER_INTERPOLATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{exc\b"),
    re.compile(r"\{e\b"),
    re.compile(r"\{err\b"),
    re.compile(r"\{error\b"),
    re.compile(r"str\s*\(\s*exc\b"),
    re.compile(r"str\s*\(\s*e\b"),
    re.compile(r"repr\s*\(\s*exc\b"),
    re.compile(r"%\s*\(\s*exc\b"),
    re.compile(r"%\s*\(\s*e\b"),
)

_MESSAGE_NAME_RE = re.compile(
    r"^(?:REPHRASE_HINT_MESSAGES|USER_REJECTED_RESULT_BUCKET_TIPS|REFUSAL_[A-Z0-9_]+|"
    r"[A-Z0-9_]+(?:_MESSAGE|_USER_MESSAGE|_LINE|_BODY|_PREFIX))$"
)


def _constants_message_assignments(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if not _MESSAGE_NAME_RE.match(name):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                out.append((name, node.value.value))
            elif isinstance(node.value, ast.Dict):
                for key, val in zip(node.value.keys, node.value.values, strict=False):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if isinstance(val, ast.Constant) and isinstance(val.value, str):
                            out.append((f"{name}[{key.value!r}]", val.value))
    return out


def test_no_driver_text_in_user_messages() -> None:
    global pre_fix_failure
    constants_path = Path(__file__).resolve().parents[1] / "src" / "aetherdialect" / "_constants.py"
    source = constants_path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern in _DRIVER_INTERPOLATION_PATTERNS:
        for match in pattern.finditer(source):
            violations.append(f"source pattern {pattern.pattern!r} at offset {match.start()}")
    for name, text in _constants_message_assignments(source):
        if _DRIVER_SECRET_TOKEN in text:
            violations.append(f"{name} embeds synthetic driver token")
    if violations:
        pre_fix_failure = "; ".join(violations[:5])
    assert not violations, pre_fix_failure


_DRIVER_SECRET_TOKEN = "OperationalError: secret driver payload"
