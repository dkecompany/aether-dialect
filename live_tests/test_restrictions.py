"""
Restricted SQL form tests.

The forbidden-form behaviour is fundamentally a property of LLM output, but the deterministic half — that pre-approved templates emit SQL free of forbidden patterns — is exercised against the seeded ``baseline_templates`` kit. One live NL smoke remains.
"""

from __future__ import annotations

import re

import pytest

from aetherdialect._live_testing import run_and_assert

from ._seed_helpers import seeded_runner
from .mydb_scenarios import restrictions_scenarios

_scenarios = restrictions_scenarios()
_FORBIDDEN_PATTERNS = (
    re.compile(r"\bUNION\b", re.IGNORECASE),
    re.compile(r"\bOVER\s*\(", re.IGNORECASE),
    re.compile(r"\bEXISTS\s*\(", re.IGNORECASE),
)


@pytest.mark.live
@pytest.mark.parametrize("scenario", _scenarios[:1], ids=[s.id for s in _scenarios[:1]])
def test_restrictions_live_smoke(runner, scenario):
    """Single live NL smoke verifying the pipeline does not emit forbidden patterns."""

    run_and_assert(runner, scenario, header=f"[{scenario.id}] {scenario.question}")


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
