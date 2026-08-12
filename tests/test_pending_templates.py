"""Pending template lifecycle and param-bound question expansion."""

from __future__ import annotations

from aetherdialect._contracts_base import ApprovalState
from aetherdialect._contracts_core import RuntimeIntent, Template, ValueHistory
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._templates_ops import TemplateOps


def _minimal_template(*, tid: str, question: str, pending: bool) -> Template:
    intent = RuntimeIntent(natural_language=question, tables=["a"])
    concrete = intent.to_concrete("")
    return Template(
        id=tid,
        schema_graph_id="g",
        effective_structural_hash="h",
        intent_signature=concrete,
        intent_key=f"k-{tid}",
        tables_used=["a"],
        sql_param="SELECT 1",
        sql_fp=f"fp-{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="",
        value_history=ValueHistory(param_values=[{}], questions=[question], natural_language=[question]),
        stats=TemplateStats(accept=0 if pending else 1, reject=0),
        approval_state=ApprovalState.PENDING if pending else ApprovalState.APPROVED,
    )


def test_template_is_pending_and_find() -> None:
    pending = _minimal_template(tid="t1", question="how many?", pending=True)
    approved = _minimal_template(tid="t2", question="how many?", pending=False)
    templates = {"t1": pending, "t2": approved}
    assert TemplateOps.template_is_pending(pending)
    assert not TemplateOps.template_is_pending(approved)
    found = TemplateOps.find_pending_template_for_question(templates, "how many?")
    assert found is not None and found.id == "t1"


def test_delete_pending_templates_for_question() -> None:
    pending = _minimal_template(tid="t1", question="how many?", pending=True)
    store: dict = {"templates": {}, "next_id": 2}
    templates = {"t1": pending}
    removed = TemplateOps.delete_pending_templates_for_question(store, templates, "how many?")
    assert removed == 1
    assert "t1" not in templates


def test_expand_param_bound_questions_literal() -> None:
    vh = ValueHistory(param_values=[], questions=[], natural_language=[])
    TemplateOps.expand_param_bound_questions(
        vh,
        old_params={"p1": "2024"},
        new_params={"p1": "2025"},
        donor_questions=["rentals in 2024"],
    )
    assert any("2025" in q for q in vh.questions)
    assert any("2024" in q for q in vh.questions)


def test_approve_pending_template() -> None:
    intent = RuntimeIntent(natural_language="how many?", tables=["a"])
    pending = _minimal_template(tid="t1", question="how many?", pending=True)
    store: dict = {"templates": {}, "next_id": 2, "schema_graph_id": "g"}
    templates = {"t1": pending}
    TemplateOps.approve_pending_template(
        store,
        templates,
        pending,
        intent=intent,
        q_norm="how many?",
    )
    assert pending.approval_state == ApprovalState.APPROVED
    assert pending.stats.accept >= 1
    assert not TemplateOps.template_is_pending(pending)
