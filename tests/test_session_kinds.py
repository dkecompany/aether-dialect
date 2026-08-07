"""Public session kind mapping for reuse vs post-exec SQL confirm."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import (
    PIPELINE_SUSPEND_ID_DIRECT_REUSE,
    PIPELINE_SUSPEND_ID_SQL,
    SESSION_KIND_AWAITING_REUSE_CONFIRM,
    SESSION_KIND_AWAITING_SQL_CONFIRM,
    SUSPEND_ID_TO_SESSION_KIND,
)
from aetherdialect._contracts_base import PipelineSuspended
from aetherdialect._contracts_core import (
    DirectReuseSuspendContext,
    GenerationPath,
    RuntimeIntent,
)
from aetherdialect._main_execution import PipelineSession


def _runtime_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=[],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_reuse_and_sql_confirm_kinds_differ() -> None:
    assert SUSPEND_ID_TO_SESSION_KIND[PIPELINE_SUSPEND_ID_DIRECT_REUSE] == SESSION_KIND_AWAITING_REUSE_CONFIRM
    assert SUSPEND_ID_TO_SESSION_KIND[PIPELINE_SUSPEND_ID_SQL] == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert (
        SUSPEND_ID_TO_SESSION_KIND[PIPELINE_SUSPEND_ID_DIRECT_REUSE]
        != SUSPEND_ID_TO_SESSION_KIND[PIPELINE_SUSPEND_ID_SQL]
    )

    owner = MagicMock()
    owner._audit_emit = MagicMock()
    owner._schema_graph = MagicMock(effective_structural_hash="h")
    sess = PipelineSession(owner)

    reuse_ctx = DirectReuseSuspendContext(
        q_norm="how many",
        ref_tmpl=MagicMock(),
        dialect=None,
        store={},
        templates={},
        rejected={},
        schema=None,
        intent=_runtime_intent(),
        sql="SELECT 1",
        rows=((1,),),
        display_sql="SELECT 1",
        headers=("n",),
        is_exact=True,
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        sd_reuse=None,
    )
    reuse_step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "reuse?", reuse_ctx))
    sql_step = sess._suspend_to_step(PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "sql?", None))

    assert reuse_step.kind == SESSION_KIND_AWAITING_REUSE_CONFIRM
    assert sql_step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
    assert reuse_step.kind != sql_step.kind
