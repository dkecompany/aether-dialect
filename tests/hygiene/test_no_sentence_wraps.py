"""Adjacent string literals must not wrap mid-sentence."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "aetherdialect"
_TERM = (".", "!", "?", "\n")
_SQL_BOTH = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|ORDER BY|GROUP BY|INSERT|UPDATE|DELETE)\b",
    re.I,
)
_REGEXISH = re.compile(r"\\[bsdwnW]|\(\?:")


def _ok_between(toks: list[tokenize.TokenInfo], i1: int, i2: int) -> bool:
    for x in toks[i1 + 1 : i2]:
        if x.type in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        ):
            continue
        if x.type == tokenize.OP and x.string in "()[],":
            continue
        return False
    return True


def _is_prose_wrap(c1: str, c2: str) -> bool:
    if not c1 or not c2:
        return False
    if _REGEXISH.search(c1) or _REGEXISH.search(c2):
        return False
    if _SQL_BOTH.search(c1) and _SQL_BOTH.search(c2):
        return False
    body = c1.rstrip(" \t")
    if body.endswith(_TERM) or body.endswith((":", "{", "[", "(", ",", ";")):
        return False
    words = body.split()
    # Slash/pipe mid-enumeration inside English prose (not short tokens / regex).
    if body.endswith(("/", "|")) and len(words) >= 4 and c2[:1].islower():
        return True
    # Trailing-space wrap of one unfinished sentence into the next literal.
    if not c1.endswith((" ", "\t")) or len(words) < 5:
        return False
    if c2[:1].islower() or c2.startswith(
        ("tables", "columns", "predicate", "connecting", "and ", "or ", "the ", "a ", "an ")
    ):
        return True
    return False


def _mid_sentence_wraps(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    strings = [(i, t) for i, t in enumerate(toks) if t.type == tokenize.STRING]
    out: list[str] = []
    for j in range(len(strings) - 1):
        i1, t1 = strings[j]
        i2, t2 = strings[j + 1]
        if not _ok_between(toks, i1, i2) or t1.end[0] == t2.start[0]:
            continue
        try:
            c1 = ast.literal_eval(t1.string)
            c2 = ast.literal_eval(t2.string)
        except Exception:
            continue
        if not isinstance(c1, str) or not isinstance(c2, str):
            continue
        if not _is_prose_wrap(c1, c2):
            continue
        out.append(f"{path.name}:{t1.start[0]}->{t2.start[0]}: {c1[-60:]!r} + {c2[:40]!r}")
    return out


@pytest.mark.fast
def test_no_mid_sentence_string_literal_wraps() -> None:
    """English prose in adjacent string literals must not break one sentence across lines."""
    violations: list[str] = []
    for path in sorted(_SRC.glob("*.py")):
        violations.extend(_mid_sentence_wraps(path))
    assert not violations, "mid-sentence string wraps:\n" + "\n".join(violations[:40])
