"""SUPPORT_MATRIX.md hygiene: user-facing language, single refusal list, legend placement."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SUPPORT_MATRIX = _REPO / "docs" / "SUPPORT_MATRIX.md"

_INTERNAL_IR_TERMS = (
    "PredicateGroup",
    "preserve_tables",
    "semi_join",
    "anti_join",
    "distinct_on",
    "order_by_cols",
    "string_agg",
    "agg_sep_param_key",
    "agg_order_by",
    "window_registry",
    "numeric_argument",
    "What the intent carries",
)

_ORDINARY_SQL_CAPABILITY_ROWS = (
    "Nested boolean filters",
    "Explicit null placement",
    "Ordered string aggregation",
    "Statistical aggregates",
    "Extended window ranking",
)

_INTERNAL_VOCAB_TERMS = (
    "frequent_values",
    "value_overlap_sample",
    "JOIN_COMPARISON_SCOPE_MAX_HOPS",
    "sj_<table>",
    "inject_pruning_predicates",
)

_REFUSED_CONSTRUCT_HEADINGS = (
    "## Quick unsupported SQL",
    "## Refused constructs and reformulations",
    "## Refused constructs",
)


def _matrix_text() -> str:
    return _SUPPORT_MATRIX.read_text(encoding="utf-8")


def _section_between(text: str, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start + len(start_heading))
    return text[start:end]


def _engine_capabilities_table(text: str) -> str:
    return _section_between(text, "## Engine capabilities", "**See also:**")


def _legend_section(text: str) -> str:
    return _section_between(text, "## Legend", "## ")


@pytest.mark.fast
def test_support_matrix_no_internal_representation_column() -> None:
    """Supported constructs are user-facing; no internal representation column or field names."""
    text = _matrix_text()
    supported = _section_between(text, "## Supported intent constructs", "## Refused")
    assert "What the intent carries" not in supported
    found = [term for term in _INTERNAL_IR_TERMS if term in supported]
    assert not found, f"internal IR terms in supported constructs: {found}"


@pytest.mark.fast
def test_support_matrix_no_preserve_tables() -> None:
    """preserve_tables is internal and must not appear anywhere in the document."""
    assert "preserve_tables" not in _matrix_text()


@pytest.mark.fast
def test_support_matrix_no_ordinary_sql_capability_rows() -> None:
    """Ordinary SQL render paths are not listed as special capabilities."""
    supported = _section_between(
        _matrix_text(),
        "## Supported intent constructs",
        "## Refused",
    )
    found = [row for row in _ORDINARY_SQL_CAPABILITY_ROWS if row in supported]
    assert not found, f"ordinary SQL rows still listed as capabilities: {found}"


@pytest.mark.fast
def test_support_matrix_refused_constructs_listed_once() -> None:
    """Refused constructs appear in one canonical section, not repeated across the document."""
    text = _matrix_text()
    present = [heading for heading in _REFUSED_CONSTRUCT_HEADINGS if heading in text]
    assert present == ["## Refused constructs"], (
        f"expected exactly one refused-constructs section; found {present!r}"
    )


@pytest.mark.fast
def test_support_matrix_no_internal_vocabulary() -> None:
    """Dialect notes and guidance avoid internal profiling names and configuration constants."""
    text = _matrix_text()
    found = [term for term in _INTERNAL_VOCAB_TERMS if term in text]
    assert not found, f"internal vocabulary still present: {found}"


@pytest.mark.fast
def test_support_matrix_self_comparison_user_guidance() -> None:
    """Self-comparison and cross-table range guidance is one user-facing sentence."""
    text = _matrix_text()
    lowered = text.lower()
    assert "compare" in lowered and "itself" in lowered
    assert "range" in lowered or "between two tables" in lowered
    assert "sj_" not in text
    assert "JOIN_COMPARISON_SCOPE_MAX_HOPS" not in text
    # Count guidance sentences in refused constructs (exclude table header/separator rows).
    refused = _section_between(text, "## Refused constructs", "## Dialect-specific notes")
    guidance_lines = [
        line.strip()
        for line in refused.splitlines()
        if line.strip()
        and not line.strip().startswith("|")
        and not line.strip().startswith("**")
        and not line.strip().startswith("#")
        and not line.strip().startswith("Each refusal")
        and not line.strip().startswith("---")
    ]
    assert len(guidance_lines) == 1, (
        f"expected one user-guidance sentence after refused table; got {guidance_lines!r}"
    )


@pytest.mark.fast
def test_support_matrix_legend_immediately_before_engine_capabilities() -> None:
    """Legend explains engine-capability table vocabulary and sits directly above that table."""
    text = _matrix_text()
    legend_pos = text.index("## Legend")
    engine_pos = text.index("## Engine capabilities")
    assert legend_pos < engine_pos
    between = text[legend_pos:engine_pos]
    assert "## " not in between.replace("## Legend", "", 1), (
        "another heading sits between Legend and Engine capabilities"
    )


@pytest.mark.fast
def test_support_matrix_legend_matches_engine_capability_cells() -> None:
    """Every legend vocabulary item appears in the table; every distinct cell form is defined."""
    text = _matrix_text()
    table = _engine_capabilities_table(text)
    legend = _legend_section(text)

    legend_vocab = {
        "n/a",
        "inject",
        "no-op",
        "partial",
        "none",
        "per-source + glue",
        "member only",
        "DuckDB in-process",
    }
    for token in legend_vocab:
        assert token in legend, f"legend missing vocabulary for {token!r}"
        assert token in table, f"engine table missing cell value {token!r}"

    orphan_legend_terms = ("NoOpQueryLogSource", "foreign_keys_add", "inject_pruning_predicates")
    for term in orphan_legend_terms:
        assert term not in legend, f"orphan legend entry still documents {term!r}"

    distinct_cells = set()
    for line in table.splitlines():
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Capability"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        distinct_cells.update(cells[1:])

    def _cell_covered(cell: str) -> bool:
        normalized = cell.strip().strip("*").strip("`")
        if normalized in legend_vocab:
            return True
        if "->" in cell and "->" in legend:
            return True
        if "EXPLAIN" in cell or "dry-run" in cell:
            return "EXPLAIN" in legend or "dry-run" in legend
        if normalized == "pglast":
            return "pglast" in legend
        if normalized.startswith("sqlglot"):
            return "sqlglot" in legend
        if "full" in cell and "full" in legend:
            return True
        if "inject" in cell and "inject" in legend:
            return True
        if normalized in {"PRAGMA", "SQLAlchemy", "Unity Catalog"}:
            return normalized in legend
        if normalized == "none":
            return "`none`" in legend or "none" in legend
        if normalized in {
            "performance_schema",
            "pg_stat_statements",
            "svl_qlog",
            "system.query.history",
            "QUERY_HISTORY",
            "JOBS",
            "dm_exec_query_stats",
        }:
            return "Query-log source" in legend
        if "SHOWPLAN" in cell:
            return "SHOWPLAN" in legend
        return False

    undefined = [cell for cell in sorted(distinct_cells) if not _cell_covered(cell)]
    assert not undefined, f"engine table cells lack legend definitions: {undefined}"
