"""Tests for .github.scripts.normalize_docstrings."""

from __future__ import annotations

import pytest
from normalize_docstrings import (
    collapse_arg_paragraphs,
    indent_under_doc_sections,
    isolate_args_as_paragraphs,
    normalize_docstring_text,
    unwrap_summary_block,
)


def test_unwrap_summary_joins_bad_agent_wraps() -> None:
    lines = [
        "Create WhereParam from",
        "dictionary.",
        "",
        "Args:",
        "",
        "d: Dictionary.",
    ]
    out = unwrap_summary_block(lines)
    assert out[0] == "Create WhereParam from dictionary."
    assert out[1] == "Args:"


def test_isolate_args_inserts_blank_between_params() -> None:
    lines = [
        "Summary.",
        "",
        "Args:",
        "",
        "name1: First.",
        "name2: Second.",
    ]
    out = isolate_args_as_paragraphs(lines)
    assert "name1: First." in out
    assert "name2: Second." in out
    i1 = out.index("name1: First.")
    i2 = out.index("name2: Second.")
    assert i1 < i2
    assert out[i2 - 1] == ""


def test_collapse_args_removes_blank_between_params() -> None:
    lines = [
        "Summary.",
        "",
        "Args:",
        "",
        "name1: First.",
        "",
        "name2: Second.",
    ]
    out = collapse_arg_paragraphs(lines)
    i1 = out.index("name1: First.")
    i2 = out.index("name2: Second.")
    assert i2 == i1 + 1


def test_finalize_indents_under_args() -> None:
    lines = [
        "Summary.",
        "",
        "Args:",
        "",
        "d: Dictionary.",
    ]
    out = indent_under_doc_sections(lines)
    assert out[out.index("Args:") + 2] == "    d: Dictionary."


def test_prepare_phase_full() -> None:
    raw = """Create WhereParam from
    dictionary.

    Args:

        d: Dictionary with keys.
        other: Second key.
    """
    result = normalize_docstring_text(raw, "prepare")
    assert "Create WhereParam from" in result
    assert "Args:" in result
    assert "d: Dictionary with keys." in result
    assert "other: Second key." in result
    d_idx = result.index("d: Dictionary")
    o_idx = result.index("other: Second key.")
    assert result[o_idx - 1] == "\n" or "\n\nother:" in result
    assert o_idx > d_idx


def test_prepare_unwraps_summary_without_sections() -> None:
    raw = """Short summary split
    across lines."""
    result = normalize_docstring_text(raw, "prepare")
    assert result == "Short summary split across lines."


def test_finalize_joins_long_no_section_summary_to_one_line() -> None:
    words = ["word"] * 40
    wrapped = "\n".join(" ".join(words[i : i + 10]) for i in range(0, 40, 10))
    result = normalize_docstring_text(wrapped, "finalize")
    assert result == " ".join(words)
    assert len(result) > 100


def test_prepare_unwraps_summary_before_args() -> None:
    raw = """Line one of summary
    continues here.

    Args:

        x: Value.
    """
    result = normalize_docstring_text(raw, "prepare")
    assert result.startswith("Line one of summary continues here.")
    assert "Args:" in result


def test_finalize_phase_preserves_blank_after_args_header() -> None:
    raw = """Summary line.

Args:

    name1: First description.

    name2: Second description.

Returns:

    Populated instance.
"""
    result = normalize_docstring_text(raw, "finalize")
    assert "Args:\n\n    name1:" in result
    assert "    name1: First description.\n    name2: Second description." in result


@pytest.mark.parametrize(
    ("phase", "snippet"),
    [
        (
            "prepare",
            "Bad wrap\nsummary.\n\nArgs:\n\nx: One.\ny: Two.",
        ),
        (
            "finalize",
            "Summary.\n\nArgs:\n\nx: One.\n\ny: Two.",
        ),
    ],
)
def test_normalize_roundtrip_structure(phase: str, snippet: str) -> None:
    out = normalize_docstring_text(snippet, phase)
    assert out
    assert "Args:" in out
