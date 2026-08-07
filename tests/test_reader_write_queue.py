"""Reader sessions must not append durable write-queue events."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import WRITE_QUEUE_FILENAME
from aetherdialect._contracts_core import (
    GenerationPath,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import handle_user_feedback
from aetherdialect._templates import TemplateOps


def _matched_template() -> Template:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    return Template(
        id="tmpl_reader_accept",
        effective_structural_hash="eff_1",
        intent_signature=intent,
        intent_key="key_reader",
        tables_used=["orders"],
        sql_param="SELECT order_id FROM orders",
        sql_fp="fp_reader",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="sig_reader",
        value_history=ValueHistory(
            param_values=[{}], questions=["how many orders"], natural_language=["how many orders"]
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


@pytest.mark.fast
def test_reader_accept_does_not_grow_queue_file(tmp_path) -> None:
    store_dir = tmp_path / "intent_templates" / "spaces" / "master"
    store_dir.mkdir(parents=True)
    store = TemplateOps.empty_template_store("graph_1")
    store._store_dir = str(store_dir)
    templates = TemplateOps.store_to_templates(store)
    tmpl = _matched_template()
    templates[tmpl.id] = tmpl

    schema = MagicMock()
    schema.schema_graph_id = "graph_1"
    schema.effective_structural_hash = "eff_1"

    intent = tmpl.intent_signature

    choice_port = MagicMock()
    choice_port._session_mode = "reader"
    choice_port.has_pending_choice.return_value = False
    choice_port.note_turn_outcome = MagicMock()

    queue_path = tmp_path / WRITE_QUEUE_FILENAME
    assert not queue_path.is_file()

    with patch("aetherdialect._pipeline.notify", lambda *a, **k: None):
        with patch("aetherdialect._pipeline.print_rephrase_hint", lambda *a, **k: None):
            handle_user_feedback(
                "y",
                intent,
                "SELECT order_id FROM orders",
                schema,
                store,
                templates,
                {},
                "how many orders",
                GenerationPath.EXACT_QUESTION_REUSE,
                tmpl,
                None,
                choice_port=choice_port,
                persist_template_learning=False,
            )

    assert not queue_path.is_file()
