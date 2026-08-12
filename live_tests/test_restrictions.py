"""Restricted SQL form tests against seeded baseline templates."""

from __future__ import annotations

import re

from .live_support import seeded_runner

_FORBIDDEN_PATTERNS = (
    re.compile(r"\bUNION\b", re.IGNORECASE),
    re.compile(r"\bOVER\s*\(", re.IGNORECASE),
    re.compile(r"\bEXISTS\s*\(", re.IGNORECASE),
)


def test_seeded_baseline_templates_emit_no_forbidden_forms(schema, schema_terms, t2s) -> None:
    """Every SQL stored in the ``baseline_templates`` kit is free of UNION / OVER / EXISTS."""
    with seeded_runner(schema, schema_terms, t2s, label="restrict_seeded", kits=("baseline_templates",)) as runner:
        for tid in runner.seeded_ids["baseline_templates"].values():
            tmpl = runner.templates[tid]
            for pattern in _FORBIDDEN_PATTERNS:
                assert not pattern.search(tmpl.sql_param or ""), (
                    f"[RESTRICT-SEED] template {tid!r} sql contains forbidden pattern "
                    f"{pattern.pattern!r}: {tmpl.sql_param!r}"
                )
