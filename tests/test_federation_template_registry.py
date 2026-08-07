"""Federation exposes the same template registry bridge as AetherEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._contracts_base import TemplateExecutionResult
from tests.test_reuse_saved_question import _minimal_template


def _federation_stub(*, templates: dict | None = None) -> AetherFederation:
    fed = object.__new__(AetherFederation)
    tmpl = _minimal_template()
    fed._sandbox_closed = False
    fed._templates = templates if templates is not None else {tmpl.id: tmpl}
    fed._store = {}
    fed._rejected = {}
    fed._schema_graph = MagicMock()
    fed._schema_graph.tables = {}
    fed._dialect = MagicMock()
    fed._dialect.sqlglot_dialect = "postgresql"
    fed._dialect.name = "postgresql"
    fed._dialect.config = MagicMock()
    fed._runtime_config = MagicMock(llm_execution=MagicMock(), execution_context=None, engine_context=None)
    fed._consumer_visible_objects = None
    fed._schema_role = "owner"
    fed._engine_identity = None
    fed._limits = MagicMock()
    fed._pipeline_writer_lock = __import__("threading").Lock()
    return fed


@pytest.mark.fast
def test_list_fetch_execute_roundtrip() -> None:
    fed = _federation_stub()
    rows = fed.list_templates()
    assert len(rows) == 1
    assert rows[0].id == "T0001"
    detail = fed.fetch_template("T0001")
    assert detail.summary.id == "T0001"
    assert detail.parameters[0].handle == "p1"
    expected = TemplateExecutionResult(
        rows=((1,),),
        sql="SELECT 1 FROM t1 WHERE category = :p1",
        display_sql="SELECT 1 FROM t1 WHERE category = :p1",
        columns=("c",),
    )
    with patch("aetherdialect.aetherdialect.execute_stored_template_by_ref", return_value=expected) as run:
        result = fed.execute_template("T0001", {"p1": "x"})
    assert result is expected
    run.assert_called_once()
