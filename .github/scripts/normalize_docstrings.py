"""Normalize Google-style docstrings for the prepare → docformatter → finalize pipeline.

Prepare (before docformatter):
  - Unwrap the summary paragraph (undo agent line breaks).
  - Isolate each Args/Returns entry as its own paragraph so docformatter wraps independently.

Finalize (after docformatter):
  - Collapse arg paragraphs back to consecutive lines under each section header.
  - Re-apply four-space indent under section headers.

Run from repo root::

    python .github/scripts/normalize_docstrings.py --phase prepare
    docformatter -i -r src tests live_tests
    python .github/scripts/normalize_docstrings.py --phase finalize

Or use ``.github/scripts/format_docstrings.py`` for all three steps.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

Phase = Literal["prepare", "finalize"]

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_TARGETS = [
    _REPO / "src" / "aetherdialect",
    _REPO / "tests",
    _REPO / "live_tests",
]

_DOC_SECTION_HEADER_RE = re.compile(
    r"^(Args|Arguments|Returns|Raises|Yields|Note|Notes|Example|Examples|Attributes|Warnings?)\s*:\s*(.*)$",
    re.IGNORECASE,
)

_PARAM_LINE_RE = re.compile(r"^[A-Za-z_][\w]*\s*:")


def _is_section_header(line: str) -> bool:
    return bool(_DOC_SECTION_HEADER_RE.match(line.strip()))


def _is_param_line(line: str) -> bool:
    return bool(_PARAM_LINE_RE.match(line.strip()))


def _cleanup_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev_blank = False
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if out and not prev_blank:
                out.append("")
            prev_blank = True
            continue
        prev_blank = False
        out.append(stripped)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    while len(out) >= 2 and not out[-1] and not out[-2]:
        out.pop()
    return out


def _first_section_index(lines: list[str]) -> int | None:
    for i, ln in enumerate(lines):
        if _is_section_header(ln):
            return i
    return None


def unwrap_summary_block(lines: list[str]) -> list[str]:
    """Join summary lines (before the first section header) into one paragraph."""
    idx = _first_section_index(lines)
    if idx is None:
        if not lines:
            return []
        joined = " ".join(ln.strip() for ln in lines if ln.strip())
        return [joined] if joined else []
    summary = [ln.strip() for ln in lines[:idx] if ln.strip()]
    rest = lines[idx:]
    if not summary:
        return rest
    return [" ".join(summary), *rest]


def isolate_args_as_paragraphs(lines: list[str]) -> list[str]:
    """Insert a blank line before each param entry under section headers."""
    if not lines:
        return []
    out: list[str] = []
    in_section = False
    for ln in lines:
        s = ln.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if _is_section_header(s):
            in_section = True
            out.append(s)
            continue
        if in_section and _is_param_line(s):
            if out and out[-1] != "" and _is_param_line(out[-1]):
                out.append("")
            out.append(s)
            continue
        if in_section:
            out.append(s)
            continue
        out.append(s)
    return _cleanup_blank_lines(out)


def collapse_arg_paragraphs(lines: list[str]) -> list[str]:
    """Remove blank lines between param entries within a section block."""
    if not lines:
        return []
    out: list[str] = []
    in_section = False
    after_section_header = False
    for ln in lines:
        s = ln.strip()
        if not s:
            if not in_section:
                if out and out[-1] != "":
                    out.append("")
            elif after_section_header:
                out.append("")
                after_section_header = False
            continue
        if _is_section_header(s):
            if out and out[-1] != "":
                out.append("")
            in_section = True
            after_section_header = True
            out.append(s)
            continue
        if in_section:
            out.append(s)
            after_section_header = False
            continue
        if out and out[-1] != "":
            out.append("")
        out.append(s)
    return _cleanup_blank_lines(out)


def join_summary_when_no_sections(lines: list[str]) -> list[str]:
    """Collapse no-section docstrings to one line (any length; docformatter wraps later)."""
    if _first_section_index(lines) is not None:
        return lines
    joined = " ".join(ln.strip() for ln in lines if ln.strip())
    return [joined] if joined else []


def indent_under_doc_sections(lines: list[str]) -> list[str]:
    if not lines:
        return []
    out: list[str] = []
    in_block = False
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        s = ln.strip()
        if _is_section_header(s):
            in_block = True
            out.append(s)
            continue
        if in_block:
            out.append(f"    {s}")
        else:
            out.append(s)
    return out


def normalize_docstring_text(raw: str, phase: Phase) -> str:
    if not raw:
        return ""
    text = inspect.cleandoc(raw)
    if not text.strip():
        return ""
    lines = _cleanup_blank_lines([ln.rstrip() for ln in text.splitlines()])
    if phase == "prepare":
        lines = unwrap_summary_block(lines)
        lines = isolate_args_as_paragraphs(lines)
    else:
        lines = collapse_arg_paragraphs(lines)
        lines = indent_under_doc_sections(lines)
        lines = join_summary_when_no_sections(lines)
    return "\n".join(lines)


def _delimiter_for_body(body: str) -> str | None:
    has_dq = '"""' in body
    has_sq = "'''" in body
    if has_dq and has_sq:
        return None
    if has_dq:
        return "'''"
    return '"""'


def format_docstring_literal_slice(value: str, continuation_indent: str) -> str:
    if not value:
        return '""""""'
    q = _delimiter_for_body(value)
    if q is None:
        return ast.unparse(ast.Constant(value))
    body = value.replace("\\", "\\\\")
    lines = body.split("\n")
    if len(lines) == 1:
        return f"{q}{lines[0]}{q}"
    parts: list[str] = [q]
    for ln in lines:
        if not ln.strip():
            parts.append("")
        else:
            parts.append(f"{continuation_indent}{ln}")
    parts.append(f"{continuation_indent}{q}")
    return "\n".join(parts)


def _iter_docstring_exprs(tree: ast.Module) -> list[ast.Expr]:
    out: list[ast.Expr] = []
    if tree.body:
        first = tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            out.append(first)
        elif isinstance(first, ast.ImportFrom) and first.module == "__future__" and len(tree.body) > 1:
            second = tree.body[1]
            if (
                isinstance(second, ast.Expr)
                and isinstance(second.value, ast.Constant)
                and isinstance(second.value.value, str)
            ):
                out.append(second)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.body:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.append(first)
    return out


def _utf8_byte_prefix_len_chars(s: str, byte_len: int) -> int:
    raw = s.encode("utf-8")
    return len(raw[:byte_len].decode("utf-8"))


def _leading_whitespace_before_ast_start(source: str, lineno: int, col_offset: int | None) -> str:
    if col_offset is None or lineno < 1:
        return ""
    lines = source.splitlines(keepends=True)
    if lineno > len(lines):
        return ""
    line = lines[lineno - 1]
    max_b = len(line.encode("utf-8"))
    byte_len = min(int(col_offset), max_b)
    nchars = _utf8_byte_prefix_len_chars(line, byte_len)
    return line[:nchars]


def _line_col_to_offset(text: str, lineno: int, col_byte: int) -> int:
    lines = text.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        return len(text)
    line_start = sum(len(lines[i]) for i in range(lineno - 1))
    line = lines[lineno - 1]
    inner = _utf8_byte_prefix_len_chars(line, min(col_byte, len(line.encode("utf-8"))))
    return line_start + inner


def replace_docstrings(source: str, tree: ast.Module, phase: Phase) -> str:
    stmts = _iter_docstring_exprs(tree)
    pieces: list[tuple[int, int, str]] = []
    for stmt in stmts:
        val = stmt.value
        assert isinstance(val, ast.Constant) and isinstance(val.value, str)
        new_inner = normalize_docstring_text(val.value, phase)
        if new_inner == val.value:
            continue
        cont = _leading_whitespace_before_ast_start(source, stmt.lineno, stmt.col_offset)
        new_seg = format_docstring_literal_slice(new_inner, cont)
        end = stmt.end_lineno
        end_col = stmt.end_col_offset
        if end is None or end_col is None:
            continue
        begin_ch = _line_col_to_offset(source, stmt.lineno, stmt.col_offset)
        end_ch = _line_col_to_offset(source, end, end_col)
        pieces.append((begin_ch, end_ch, new_seg))
    pieces.sort(key=lambda x: x[0], reverse=True)
    out = source
    for a, b, repl in pieces:
        out = out[:a] + repl + out[b:]
    return out


def iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def write_source_file(path: Path, content: str, *, retries: int = 5) -> None:
    """Atomic write with retries (helps on OneDrive / locked files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
            )
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                tmp.write_text(content, encoding="utf-8", newline="\n")
                os.replace(tmp, path)
                return
            finally:
                tmp.unlink(missing_ok=True)
        except OSError as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    if last_err is not None:
        raise last_err


def process_file(path: Path, phase: Phase, check: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        print(f"skip (syntax): {path.relative_to(_REPO)}", file=sys.stderr)
        return False
    assert isinstance(tree, ast.Module)
    try:
        updated = replace_docstrings(raw, tree, phase)
        ast.parse(updated)
    except Exception as exc:
        print(f"skip ({exc}): {path.relative_to(_REPO)}", file=sys.stderr)
        return False
    if updated == raw:
        return False
    if not check:
        write_source_file(path, updated)
    print(path.relative_to(_REPO))
    return True


def run_on_targets(
    targets: list[Path],
    phase: Phase,
    check: bool,
) -> int:
    changed = 0
    for root in targets:
        for path in iter_py_files(root):
            if process_file(path, phase, check):
                changed += 1
    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "finalize"),
        required=True,
        help="prepare: before docformatter; finalize: after docformatter",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that would change; exit 1 if any would",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Roots to scan (default: src/aetherdialect, tests, live_tests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = [p.resolve() for p in args.paths] if args.paths else _DEFAULT_TARGETS
    changed = run_on_targets(targets, args.phase, args.check)
    if args.check and changed:
        print(f"would update {changed} file(s)", file=sys.stderr)
        return 1
    print(f"updated {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
