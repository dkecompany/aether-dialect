"""Semijoin key transfers must emit audit events."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aetherdialect._constants import AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import FederationExecutionContext, RuntimeIntent, SelectCol, SourceStep
from aetherdialect._core_utils import pop_federation_execution_context, push_federation_execution_context
from aetherdialect._federation import member_stage_for_source, plan_federated_intent
from aetherdialect._pipeline import _execute_federation_source_step
from tests.test_federation_combine_pushdown import _left_join_manifest, _left_join_schema


@pytest.mark.fast
def test_semijoin_key_transfer_emits_audit_event(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, schema, manifest)
    target_step = next(step for step in plan.steps if step.source_id == "b")
    member_stage = member_stage_for_source(plan, "b")
    assert member_stage is not None
    assert member_stage.reducing_edges
    target_intent = replace(
        target_step.sub_intent,
        tables=["tb"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("tb.a_id"))],
    )
    prep = type(
        "Prep",
        (),
        {
            "sub_intent": target_intent,
            "sql": "SELECT a_id FROM tb",
            "structural_defaults": None,
        },
    )()
    executed = {"a": pd.DataFrame({"id": [10, 20, 30]})}
    audit_events: list[dict[str, object]] = []

    def _audit_emit(
        event_type: str,
        *,
        question: str | None = None,
        schema_hash: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        audit_events.append(
            {
                "event_type": event_type,
                "question": question,
                "schema_hash": schema_hash,
                "details": details,
            }
        )

    fed_ctx = FederationExecutionContext(plan_id="plan_audit", audit_emit=_audit_emit)
    token = push_federation_execution_context(fed_ctx)
    monkeypatch.setattr(
        "aetherdialect._pipeline.execute_guarded_sql",
        lambda *a, **k: [(10,)],
    )
    monkeypatch.setattr(
        "aetherdialect._pipeline.generate_and_validate_sql",
        lambda *a, **k: type("Out", (), {"success": True, "sql": "SELECT a_id FROM tb WHERE a_id IN (10,20,30)"})(),
    )
    monkeypatch.setattr(
        "aetherdialect._pipeline.build_result_dataframe",
        lambda *a, **k: pd.DataFrame({"a_id": [10]}),
    )
    monkeypatch.setattr(
        "aetherdialect._validation_execute.validate_sql",
        lambda *a, **k: (True, None, None, None),
    )
    monkeypatch.setattr("aetherdialect._pipeline.validate_federated_sub_intent", lambda *_a, **_k: None)
    mock_dialect = MagicMock()
    mock_dialect.finalize_render.return_value = "SELECT a_id FROM tb"
    try:
        frame = _execute_federation_source_step(
            SourceStep(source_id="b", sub_intent=target_intent),
            prepared_by_source={"b": prep},
            composite_schema=schema,
            dialect_map={},
            dialect=mock_dialect,
            manifest=manifest,
            executed=executed,
            plan=plan,
            semijoin_cap=100,
            q_norm="q",
            join_candidates={},
            cmap={},
            store={},
            gate_kwargs={},
        )
    finally:
        pop_federation_execution_context(token)
    assert frame is not None
    assert len(audit_events) == 1
    event = audit_events[0]
    assert event["event_type"] == AUDIT_EVENT_FEDERATION_SEMIJOIN_KEY_TRANSFER
    detail_map = dict(event["details"])
    assert detail_map["source_member"] == "a"
    assert detail_map["target_member"] == "b"
    assert detail_map["column"] == "id"
    assert detail_map["key_count"] == "3"
