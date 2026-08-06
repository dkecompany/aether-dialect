"""Sandbox paraphrase catalog injection and direct-reuse behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import EngineConfig
from aetherdialect._constants import SESSION_KIND_AWAITING_SQL_CONFIRM
from aetherdialect._contracts_base import MockFixtureMissingError
from aetherdialect._contracts_core import QuestionFormStorage, RuntimeIntent, ValueHistory
from aetherdialect._llm_provider import MockProvider
from aetherdialect._templates import (
    TemplateOps,
)
from tests.test_reuse_saved_question import _minimal_template


@pytest.fixture(autouse=True)
def _reset_paraphrase_registry() -> None:
    TemplateOps.clear_sandbox_paraphrase_source()
    yield
    TemplateOps.clear_sandbox_paraphrase_source()
    MockProvider.reset_mock_provider()


@pytest.mark.fast
def test_append_runtime_paraphrase_variants_injects_bundled_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    canonical = "How many books do we have?"
    TemplateOps.set_sandbox_paraphrase_source(
        {
            canonical: [
                "What is the total number of books?",
                "Book count please",
            ],
        },
    )
    vh = ValueHistory(
        param_values=[{"s1": 10}],
        questions=[canonical],
        natural_language=["nl"],
        accept_counts=[1],
    )
    schema = MagicMock()
    TemplateOps._append_runtime_paraphrase_variants(
        vh,
        canonical,
        {"s1": 10},
        "nl",
        schema,
        ["book"],
    )
    assert "What is the total number of books?" in vh.questions
    assert "Book count please" in vh.questions
    TemplateOps._append_runtime_paraphrase_variants(
        vh,
        canonical,
        {"s1": 10},
        "nl",
        schema,
        ["book"],
    )
    assert vh.questions.count("Book count please") == 1


@pytest.mark.fast
def test_append_runtime_paraphrase_variants_noop_without_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    canonical = "How many books do we have?"
    vh = ValueHistory(
        param_values=[{"s1": 10}],
        questions=[canonical],
        natural_language=["nl"],
        accept_counts=[1],
    )
    schema = MagicMock()
    TemplateOps._append_runtime_paraphrase_variants(
        vh,
        canonical,
        {"s1": 10},
        "nl",
        schema,
        ["book"],
    )
    assert vh.questions == [canonical]


@pytest.mark.fast
def test_resolve_param_display_names_uses_fixture_under_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(EngineConfig, "MOCK_FIXTURES_FILE", "fixtures.json")
    tmpl = _minimal_template()
    tmpl.param_display_names = {}
    schema = MagicMock()
    schema.tables = {}
    with patch(
        "aetherdialect._templates.LLMProvider.chat",
        return_value='{"display_names":{"p1":"Item category"}}',
    ):
        names = TemplateOps.resolve_param_display_names(
            tmpl,
            {
                "p1": MagicMock(
                    handle="p1",
                    column_expr="t1.category",
                    op="=",
                    value_type="string",
                    upper_handle="",
                    unit_handle="",
                ),
            },
            {"p1": "x"},
            schema=schema,
            question_nl="count of item in category x",
            persist=False,
        )
    assert names["p1"] == "Item category"


@pytest.mark.fast
def test_resolve_param_display_names_falls_back_on_missing_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(EngineConfig, "MOCK_FIXTURES_FILE", "fixtures.json")
    tmpl = _minimal_template()
    tmpl.param_display_names = {}
    schema = MagicMock()
    schema.tables = {}
    slot_meta = MagicMock(
        handle="p1",
        column_expr="t1.category",
        op="=",
        value_type="string",
        upper_handle="",
        unit_handle="",
    )
    with patch(
        "aetherdialect._templates.LLMProvider.chat",
        side_effect=MockFixtureMissingError(task="default", system="s", user="u"),
    ):
        names = TemplateOps.resolve_param_display_names(
            tmpl,
            {"p1": slot_meta},
            {"p1": "x"},
            schema=schema,
            question_nl="count of item in category x",
            persist=False,
        )
    assert names["p1"] == "category"


@pytest.mark.needs_corpus
def test_catalog_paraphrase_hits_direct_reuse_after_accept() -> None:
    canonical = "How many items are in the catalog by item type?"
    pairs = AetherEngine.sandbox_paraphrase_pairs()
    row = next(item for item in pairs if item.get("canonical") == canonical)
    paraphrases = row.get("paraphrases")
    assert isinstance(paraphrases, list) and paraphrases
    paraphrase = str(paraphrases[0])

    with AetherEngine.offline_sandbox() as sb:
        with sb.engine.session() as session:
            accepted = session.accept_until_done(canonical)
            assert accepted.done
            assert accepted.sql

            step = session.ask(paraphrase)
            assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
            assert step.sql

            while not step.done:
                if step.reply_shape == "yes_no":
                    step = session.step("y")
                elif step.reply_shape == "free_text":
                    step = session.step("ok")
                else:
                    break

    assert step.done
    assert step.sql


@pytest.mark.fast
def test_record_value_history_on_accept_raises_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    canonical = "User facing question"
    normalized = "normalized question form"
    vh = ValueHistory(param_values=[], questions=[], natural_language=[])
    form_storage = QuestionFormStorage(corrected=canonical, normalized_optional=normalized)
    TemplateOps.record_value_history_on_accept(
        vh,
        param_values={"p1": "x"},
        natural_language="nl",
        form_storage=form_storage,
        q_norm_fallback=canonical,
    )
    assert vh.accept_counts[0] == 1
    assert vh.accept_counts[1] == 1
    assert vh.questions == [canonical, normalized]


@pytest.mark.fast
def test_insert_template_record_accept_primary_not_double_counted(
    monkeypatch: pytest.MonkeyPatch,
    schema_graph,
) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(EngineConfig, "MOCK_FIXTURES_FILE", "fixtures.json")
    canonical = "How many books do we have?"
    normalized = "book count"
    paraphrase = "What is the total number of books?"
    TemplateOps.set_sandbox_paraphrase_source({canonical: [paraphrase]})
    form_storage = QuestionFormStorage(corrected=canonical, normalized_optional=normalized)
    intent = RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        natural_language="nl",
    )
    store: dict[str, object] = {"next_id": 1}
    templates: dict[str, object] = {}
    tmpl = TemplateOps.insert_template(
        store,
        templates,
        schema_graph,
        canonical,
        intent,
        "SELECT COUNT(*) FROM orders",
        form_storage=form_storage,
        record_accept=True,
    )
    vh = tmpl.value_history
    assert vh.accept_counts.count(1) == 2
    assert vh.accept_counts[0] == 1
    assert vh.questions[0] == canonical
    norm_idx = vh.questions.index(normalized)
    assert vh.accept_counts[norm_idx] == 1
    para_idx = vh.questions.index(paraphrase)
    assert vh.accept_counts[para_idx] == 0


@pytest.mark.fast
def test_insert_template_record_accept_without_registry_raises_normalized_only(
    monkeypatch: pytest.MonkeyPatch,
    schema_graph,
) -> None:
    monkeypatch.setattr(EngineConfig, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(EngineConfig, "MOCK_FIXTURES_FILE", "fixtures.json")
    canonical = "How many books do we have?"
    normalized = "book count"
    form_storage = QuestionFormStorage(corrected=canonical, normalized_optional=normalized)
    intent = RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        natural_language="nl",
    )
    store: dict[str, object] = {"next_id": 1}
    templates: dict[str, object] = {}
    tmpl = TemplateOps.insert_template(
        store,
        templates,
        schema_graph,
        canonical,
        intent,
        "SELECT COUNT(*) FROM orders",
        form_storage=form_storage,
        record_accept=True,
    )
    vh = tmpl.value_history
    assert len(vh.questions) == 2
    assert vh.accept_counts == [1, 1]
    assert vh.questions == [canonical, normalized]
