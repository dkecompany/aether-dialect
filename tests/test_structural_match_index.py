"""Structural template match narrows via union_family_index."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from aetherdialect._intent_process import body_similarity_key, collect_structural_match_templates, structural_compare
from tests.test_intent_process import TestMatchTemplateForUnion, _col, _runtime


@pytest.mark.fast
def test_match_uses_union_family_index() -> None:
    """collect_structural_match_templates scans only index candidates when provided."""
    intent = _runtime(["t"], [_col("t", "a")])
    bk = body_similarity_key(intent)

    tmpl_match = TestMatchTemplateForUnion._make_template("T0001", [_col("t", "a")], ["t"])
    templates = {"T0001": tmpl_match}
    for i in range(2, 12):
        tid = f"T{i:04d}"
        decoy = TestMatchTemplateForUnion._make_template(tid, [_col("t", f"c{i}")], ["t"])
        templates[tid] = replace(decoy, intent_key=f"ik_{tid}")

    indexed_calls: list[str] = []
    full_calls: list[str] = []
    real_compare = structural_compare

    def _spy(intent_obj, tmpl, *, mode="warmup_gold_store_check"):
        indexed_calls.append(tmpl.id)
        return real_compare(intent_obj, tmpl, mode=mode)

    ufi = {bk: ["T0001"]}
    with patch("aetherdialect._intent_process.structural_compare", side_effect=_spy):
        out = collect_structural_match_templates(intent, templates, union_family_index=ufi)
    assert [t.id for t in out] == ["T0001"]
    assert len(indexed_calls) == 1

    def _spy_full(intent_obj, tmpl, *, mode="warmup_gold_store_check"):
        full_calls.append(tmpl.id)
        return real_compare(intent_obj, tmpl, mode=mode)

    with patch("aetherdialect._intent_process.structural_compare", side_effect=_spy_full):
        collect_structural_match_templates(intent, templates)
    assert len(full_calls) == len(templates)
