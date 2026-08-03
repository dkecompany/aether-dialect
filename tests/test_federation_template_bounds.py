"""Federation plan templates must be bounded and persist only after user acceptance."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import FederationPlanTemplate
from aetherdialect._contracts_core import (
    FederatedPlan,
    FederatedPrepareOutcome,
    GenerationPath,
    RuntimeIntent,
    SourceStep,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    credit_federation_plan_accept,
    delete_federation_plan_template,
    load_federation_plan_templates,
    save_federation_plan_template,
)
from aetherdialect._pipeline import complete_user_feedback_reject, credit_federation_accept
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, source_id: str = "") -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


def _plan_template(plan_id: str, *, accepted: tuple[str, ...] = ()) -> FederationPlanTemplate:
    return FederationPlanTemplate(
        plan_id=plan_id,
        composite_schema_graph_id="cg",
        intent_key=plan_id,
        step_fingerprints=(),
        combine_hash="h",
        question="show orders",
        accepted_questions=accepted,
    )


@pytest.mark.fast
def test_execute_success_does_not_persist_plan_template() -> None:
    """Successful federation execute must defer plan template persistence until accept."""
    from aetherdialect._contracts_core import SqlGenerationOutcome
    from aetherdialect._main_execution import _run_sql_execution_for_gen_out

    sub_intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = FederatedPlan(steps=(SourceStep(source_id="a", sub_intent=sub_intent),))
    fed_prep = FederatedPrepareOutcome(success=True, plan=plan, display_sql="display")
    gen_out = SqlGenerationOutcome(
        "display",
        True,
        GenerationPath.FEDERATION_PLAN,
        None,
        federation_plan_id="plan1",
    )
    pending = _plan_template("plan1")
    owner = MagicMock()
    owner._federation_storage_dir = "/tmp/fed"
    session = MagicMock()
    session._owner = owner
    session._pending_federation_plan_template = pending
    with (
        patch("aetherdialect._main_execution.execute_federated_prepare") as mock_exec,
        patch("aetherdialect._federation.save_federation_plan_template") as save_plan,
    ):
        mock_exec.return_value = MagicMock(rows=[(1,)], bundle=MagicMock())
        _run_sql_execution_for_gen_out(
            intent=sub_intent,
            exec_schema=_graph("t", source_id="a"),
            exec_dialect=MagicMock(),
            tmpl_sd=None,
            gen_out=gen_out,
            owner=owner,
            choice_port=session,
            federated_prepare=fed_prep,
        )
    save_plan.assert_not_called()
    assert session._pending_federation_plan_template is pending


@pytest.mark.fast
def test_accept_persists_plan_with_credited_question() -> None:
    """User accept must persist the plan template and credit the accepted question."""
    with tempfile.TemporaryDirectory() as tmp:
        pending = _plan_template("plan1")
        credit_federation_accept(
            q_norm="show orders",
            federation_dir=tmp,
            plan_id="plan1",
            steps=(),
            stores_by_source={},
            pending_plan_template=pending,
        )
        loaded = load_federation_plan_templates(tmp)
        assert "plan1" in loaded
        assert "show orders" in loaded["plan1"].accepted_questions


@pytest.mark.fast
def test_reject_deletes_unaccepted_plan_template() -> None:
    """Rejecting a federation turn must remove a plan that was never accepted."""
    from aetherdialect._pipeline import UserFeedbackRejectSuspendContext

    with tempfile.TemporaryDirectory() as tmp:
        save_federation_plan_template(tmp, _plan_template("plan1"))
        assert "plan1" in load_federation_plan_templates(tmp)
        schema = _graph("t")
        ctx = UserFeedbackRejectSuspendContext(
            intent=RuntimeIntent(
                tables=["t"],
                grain="many",
                select_cols=[],
                group_by_cols=[],
                order_by_cols=[],
                where=None,
            ),
            sql="select 1",
            schema=schema,
            store={},
            templates={},
            rejected={},
            q_norm="show orders",
            generation_path=GenerationPath.FEDERATION_PLAN,
            matched_template=None,
            matched_rejected_template=None,
            dialect=None,
            structural_match_templates=None,
        )
        complete_user_feedback_reject(
            ctx,
            needs_reason=False,
            reject_reason="",
            federation_dir=tmp,
            federation_plan_id="plan1",
            cross_source_join_feedback=True,
            persist_template_learning=False,
        )
        assert "plan1" not in load_federation_plan_templates(tmp)


@pytest.mark.fast
def test_plan_template_file_cap_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan template file size must be capped."""
    monkeypatch.setattr(
        "aetherdialect._federation.FEDERATION_PLAN_TEMPLATE_FILE_CAP",
        2,
        raising=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_federation_plan_template(tmp, _plan_template("plan1", accepted=("q1",)))
        save_federation_plan_template(tmp, _plan_template("plan2", accepted=("q2",)))
        save_federation_plan_template(tmp, _plan_template("plan3", accepted=("q3",)))
        loaded = load_federation_plan_templates(tmp)
        assert len(loaded) == 2
        assert "plan3" in loaded


@pytest.mark.fast
def test_accepted_questions_cap_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted question list per plan must be capped."""
    monkeypatch.setattr(
        "aetherdialect._federation.FEDERATION_PLAN_ACCEPTED_QUESTIONS_CAP",
        2,
        raising=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_federation_plan_template(tmp, _plan_template("plan1", accepted=("q1",)))
        credit_federation_plan_accept(tmp, "plan1", "q2")
        credit_federation_plan_accept(tmp, "plan1", "q3")
        loaded = load_federation_plan_templates(tmp)["plan1"]
        assert len(loaded.accepted_questions) == 2
        assert loaded.accepted_questions[-1] == "q3"
        assert "q1" not in loaded.accepted_questions


@pytest.mark.fast
def test_delete_federation_plan_template_removes_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_federation_plan_template(tmp, _plan_template("plan1"))
        delete_federation_plan_template(tmp, "plan1")
        assert load_federation_plan_templates(tmp) == {}
