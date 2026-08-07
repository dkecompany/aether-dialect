"""execute_template accepts p-param overrides and refuses pending/unknown keys."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import ApprovalState, ConfigError, TemplateExecutionResult
from aetherdialect._contracts_core import Template
from tests.test_aetherdialect import _make_aether_stub
from tests.test_reuse_saved_question import _minimal_template


def _engine(tmpl: Template) -> AetherEngine:
    dialect = MagicMock()
    dialect.sqlglot_dialect = "postgresql"
    dialect.name = "postgresql"
    dialect.config = MagicMock()
    return _make_aether_stub(_templates={tmpl.id: tmpl}, _dialect=dialect, _context_name="master", _store={})


@pytest.mark.fast
def test_pparam_override_executes() -> None:
    tmpl = _minimal_template()
    engine = _engine(tmpl)
    expected = TemplateExecutionResult(
        rows=((1,),),
        sql=tmpl.sql_param,
        display_sql=tmpl.sql_param,
        columns=("c",),
    )
    with patch("aetherdialect.aetherdialect.execute_stored_template_by_ref", return_value=expected) as run:
        result = engine.execute_template("T0001", {"p1": "y"})
    assert result is expected
    assert run.call_args.args[1] == {"p1": "y"}


@pytest.mark.fast
def test_unknown_param_key_raises() -> None:
    tmpl = _minimal_template()
    engine = _engine(tmpl)
    with pytest.raises(ConfigError, match="Unknown parameter|unknown parameter"):
        engine.execute_template("T0001", {"p99": "nope"})


@pytest.mark.fast
def test_pending_refused() -> None:
    tmpl = _minimal_template()
    tmpl.approval_state = ApprovalState.PENDING
    engine = _engine(tmpl)
    with pytest.raises(ConfigError, match="pending"):
        engine.execute_template("T0001", {"p1": "x"})
