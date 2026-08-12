"""Ask and replay paths must surface the same refusal code and text for one condition."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
)
from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    AggregateJoinFanOutError,
    ClauseWidenedRowsetError,
    ConcreteIntent,
    GenerationPath,
    NoJoinPathError,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SchemaGraph, SQLShape, TemplateStats
from aetherdialect._pipeline_execute import (
    execute_reuse_with_params,
    prepare_federated_sql_plan,
)
from aetherdialect._pipeline_generate import federation_ineligible_refusal_outcome, sql_validation_refusal_outcome
from aetherdialect._utils import (
    refusal_diagnostic_code_for_exception,
    refusal_diagnostic_code_for_federation_reason,
    refusal_message_for_exception,
)
from tests.test_join_fan_out import _join_signature, _parent_child_schema


def _refusal_fields(outcome) -> tuple[str | None, str]:
    code = getattr(outcome, "refusal_diagnostic_code", None)
    message = getattr(outcome, "sql_validation_error", None) or ""
    return code, message


def _fan_out_intent(schema: SchemaGraph) -> RuntimeIntent:
    return RuntimeIntent(
        tables=["parent", "child"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("sum", "parent.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=_join_signature(schema),
    )


def _clause_widened_intent(schema: SchemaGraph) -> RuntimeIntent:
    return RuntimeIntent(
        tables=["parent", "child"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("parent.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        limit=5,
        chosen_join_path_signature=_join_signature(schema),
    )


def _fan_out_template(schema: SchemaGraph, intent: RuntimeIntent) -> Template:
    concrete = ConcreteIntent(
        intent_id="id",
        tables=list(intent.tables or []),
        grain=intent.grain,
        select_cols=list(intent.select_cols or []),
        group_by_cols=list(intent.group_by_cols or []),
        order_by_cols=list(intent.order_by_cols or []),
        where=intent.where,
        chosen_join_path_signature=list(intent.chosen_join_path_signature or []),
        limit=intent.limit,
        limit_param_key=intent.limit_param_key or "",
        distinct_select_index=intent.distinct_select_index,
        distinct_on=list(intent.distinct_on or []),
    )
    return Template(
        id="T_fan",
        effective_structural_hash=schema.effective_structural_hash,
        intent_signature=concrete,
        intent_key="ik_fan",
        tables_used=list(intent.tables or []),
        sql_param="SELECT 1",
        sql_fp="fp",
        shape=SQLShape(num_joins=1, has_group_by=False, has_agg=True),
        colmap_sig="c",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("condition", "exc_factory"),
    [
        (
            "aggregate_join_fan_out",
            lambda schema: AggregateJoinFanOutError("main query", "SUM(parent.amount) would duplicate rows"),
        ),
        (
            "clause_widened_rowset",
            lambda schema: ClauseWidenedRowsetError("main query", "LIMIT would apply after join widening"),
        ),
        (
            "join_path_unavailable",
            lambda _schema: NoJoinPathError("main query", ["alpha", "beta"]),
        ),
    ],
)
def test_same_condition_same_refusal_on_every_path(condition: str, exc_factory) -> None:
    schema = _parent_child_schema()
    exc = exc_factory(schema)
    expected_code = refusal_diagnostic_code_for_exception(exc)
    expected_message = refusal_message_for_exception(exc)

    ask_outcome = sql_validation_refusal_outcome(
        exc,
        generation_path=GenerationPath.FRESH,
        matched_template=None,
        structural_match_templates=(),
    )
    ask_code, ask_message = _refusal_fields(ask_outcome)
    assert ask_code == expected_code
    assert ask_message == expected_message

    if condition == "join_path_unavailable":
        return

    intent = _fan_out_intent(schema) if condition == "aggregate_join_fan_out" else _clause_widened_intent(schema)
    tmpl = _fan_out_template(schema, intent)
    dialect = MagicMock()
    dialect.finalize_render.return_value = "SELECT 1"

    with patch("aetherdialect._templates.TemplateRefs.template_is_live", return_value=(True, ())):
        replay_outcome = execute_reuse_with_params(
            "q",
            tmpl,
            {},
            dialect,
            {},
            {"T_fan": tmpl},
            {},
            schema,
            reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
            prompt=False,
            on_param_incomplete="return_none",
        )

    assert replay_outcome is not None
    replay_code, replay_message = _refusal_fields(replay_outcome)
    assert replay_code == ask_code
    assert replay_message == ask_message
    if expected_code == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT:
        assert replay_code == DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT


@pytest.mark.fast
def test_federation_ineligible_ask_and_prepare_share_refusal() -> None:
    from aetherdialect._contracts_core import FederatedPlan
    from aetherdialect._federation_execute import federation_user_facing_ineligible_message

    reason = "member capability: where operator 'ilike' is not supported by federation member 'sqlite'"
    expected_message = federation_user_facing_ineligible_message(reason)
    plan = FederatedPlan(steps=(), residual=None, ineligible_reason=reason)
    schema = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="fed")
    dialect = MagicMock()
    prepare_outcome = prepare_federated_sql_plan(
        "q",
        plan,
        schema,
        dialect=dialect,
        dialects_by_source={},
        join_candidates={},
        cmap={},
        store={},
    )
    replay_outcome = federation_ineligible_refusal_outcome(
        reason,
        generation_path=GenerationPath.FEDERATION_PLAN,
        matched_template=None,
    )
    assert prepare_outcome.sql_validation_error == expected_message
    replay_code, replay_message = _refusal_fields(replay_outcome)
    assert replay_message == expected_message
    assert replay_code == DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP
    assert refusal_diagnostic_code_for_federation_reason(reason) == replay_code
