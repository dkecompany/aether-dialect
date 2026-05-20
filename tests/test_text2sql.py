"""Tests for :mod:`aetherdialect.text2sql` facade helpers and validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import MIGRATION_MAP_FILENAME, load_runtime_config
from aetherdialect._templates import empty_template_store
from aetherdialect._contracts_base import (
    ConfigError,
    LLMConfig,
    QSimSummary,
    RuntimeConfig,
    SchemaContext,
)
from aetherdialect._contracts_core import (
    FilterParam,
    NormalizedExpr,
    OrderByCol,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect.text2sql import Text2SQL


def _sample_llm_execution():
    """Return a merged :class:`LlmExecutionConfig` for test doubles."""

    return load_runtime_config(merged_env=dict(os.environ))


def _make_t2s_stub(**overrides):
    """Build a minimal ``Text2SQL`` instance shell for unit tests."""

    defaults = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/t2s",
            schema_context=SchemaContext(),
            llm_execution=_sample_llm_execution(),
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=MagicMock(),
        _dialect=MagicMock(),
        _artifacts_dir=Path("/tmp/t2s"),
        _store=empty_template_store("unit_test_eff"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_stats={"table_count": 10, "total_filterable": 40},
    )
    defaults.update(overrides)

    obj = Text2SQL.__new__(Text2SQL)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


class TestValidateNumIntents:
    """Tests for ``Text2SQL._validate_num_intents``."""

    def test_in_range_passes(self) -> None:
        t = _make_t2s_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        lo, hi = t._compute_num_intents_range()
        t._validate_num_intents(lo)
        t._validate_num_intents(hi)

    def test_below_range_raises(self) -> None:
        t = _make_t2s_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_intents must be"):
            t._validate_num_intents(1)

    def test_above_range_raises(self) -> None:
        t = _make_t2s_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_intents must be"):
            t._validate_num_intents(999)


class TestValidateNumQuestions:
    """Tests for ``Text2SQL._validate_num_questions``."""

    def test_in_range_passes(self) -> None:
        t = _make_t2s_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        lo, hi = t._compute_num_questions_range()
        t._validate_num_questions(lo)
        t._validate_num_questions(hi)

    def test_below_range_raises(self) -> None:
        t = _make_t2s_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_questions must be"):
            t._validate_num_questions(1)


class TestGetQsimSummary:
    """Tests for ``Text2SQL.get_qsim_summary``."""

    def test_filters_versions_inclusive(self) -> None:
        t = _make_t2s_stub()
        summaries = [
            QSimSummary(version=1, num_intents=1, num_questions=1, seed=0),
            QSimSummary(version=2, num_intents=1, num_questions=1, seed=0),
            QSimSummary(version=5, num_intents=1, num_questions=1, seed=0),
        ]
        with patch("aetherdialect.text2sql.load_qsim_summaries", return_value=summaries):
            snap = t.get_qsim_summary(2, 4)
        text = snap.format_human()
        assert "v2" in text
        assert "Latest: v5" in text


class TestEnsureLlm:
    """Tests for ``Text2SQL._ensure_llm``."""

    @patch("aetherdialect.text2sql.llm_credentials_configured", return_value=False)
    def test_no_credentials_raises_config_error(self, _mock_lc: MagicMock) -> None:
        t = _make_t2s_stub()
        with pytest.raises(ConfigError, match="LLM is not configured"):
            t._ensure_llm()

    @patch("aetherdialect.text2sql.llm_credentials_configured", return_value=True)
    def test_configured_passes(self, _mock_lc: MagicMock) -> None:
        t = _make_t2s_stub()
        t._ensure_llm()


class TestInitPatches:
    """Construction smoke tests with heavy dependencies patched."""

    @patch("aetherdialect.text2sql.initialize_text2sql")
    def test_init_builds_schema_and_store(self, mock_init: MagicMock) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="C:/tmp/x/text2sql",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="C:/tmp/x/text2sql",
            store={"effective_structural_hash": "h"},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = Text2SQL(SchemaContext(), artifacts_dir="x", config_file=None)
        assert t._schema_graph is sg
        assert t._artifacts_dir == Path("C:/tmp/x/text2sql")

    @patch("aetherdialect.text2sql.initialize_text2sql")
    def test_init_calls_initialize_with_schema_context(self, mock_init: MagicMock) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="C:/tmp/x/aetherdialect/slug",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="C:/tmp/x/aetherdialect/slug",
            store={"effective_structural_hash": "h"},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        Text2SQL(SchemaContext(), artifacts_dir="x", config_file=None)
        assert mock_init.called
        assert mock_init.call_args[0][0] is not None


class TestClearLearningCaches:
    """Tests for template / simulation / full learning clear APIs."""

    @patch("aetherdialect.text2sql.initialize_text2sql")
    @patch("aetherdialect.text2sql.clear_template_store_only", return_value=True)
    def test_clear_template_store_reinitializes(
        self,
        mock_clear: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/t2s",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/t2s",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_t2s_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/t2s"),
        )
        out = t.clear_template_store()
        assert out is True
        mock_clear.assert_called_once()
        mock_init.assert_called_once()

    @patch("aetherdialect.text2sql.initialize_text2sql")
    @patch("aetherdialect.text2sql.clear_simulation_caches_only", return_value=3)
    def test_clear_simulation_caches_reinitializes(
        self,
        mock_clear: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/t2s",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/t2s",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_t2s_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/t2s"),
        )
        n = t.clear_simulation_caches()
        assert n == 3
        mock_clear.assert_called_once()
        mock_init.assert_called_once()

    @patch("aetherdialect.text2sql.initialize_text2sql")
    @patch("aetherdialect.text2sql._clear_persisted_overrides")
    @patch("aetherdialect.text2sql.clear_template_store_only")
    @patch("aetherdialect.text2sql.clear_simulation_caches_only")
    def test_clear_all_learning_with_overrides_only_clears_disks(
        self,
        mock_sim: MagicMock,
        mock_tmpl: MagicMock,
        mock_clear_over: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/t2s",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/t2s",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_t2s_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/t2s"),
        )
        t.clear_all_learning(keep_overrides=True)
        mock_tmpl.assert_called_once()
        mock_sim.assert_called_once()
        mock_clear_over.assert_not_called()
        mock_init.assert_called_once()

    @patch("aetherdialect.text2sql.initialize_text2sql")
    @patch("aetherdialect.text2sql._clear_persisted_overrides", return_value=True)
    @patch("aetherdialect.text2sql.clear_template_store_only")
    @patch("aetherdialect.text2sql.clear_simulation_caches_only")
    def test_clear_all_learning_drops_overrides(
        self,
        mock_sim: MagicMock,
        mock_tmpl: MagicMock,
        mock_clear_over: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._main_execution import Text2SQLInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = Text2SQLInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/t2s",
                schema_context=SchemaContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/t2s",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_t2s_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/t2s"),
        )
        t.clear_all_learning(keep_overrides=False)
        mock_tmpl.assert_called_once()
        mock_sim.assert_called_once()
        mock_clear_over.assert_called_once()
        mock_init.assert_called_once()


class TestApplyMigrationMap:
    """Tests for ``Text2SQL.apply_migration_map``."""

    def test_apply_migration_map_copies_to_cwd(self) -> None:
        captured: dict[str, object] = {}

        def _capture_init(self: Text2SQL, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "incoming_map.json"
            src.write_text('{"tables": []}', encoding="utf-8")
            old = os.getcwd()
            os.chdir(td)
            try:
                with patch.object(Text2SQL, "__init__", _capture_init):
                    out = Text2SQL.apply_migration_map(
                        str(src),
                        schema_context=SchemaContext(),
                        artifacts_dir="/tmp/art",
                    )
                assert isinstance(out, Text2SQL)
                dst = Path(td) / MIGRATION_MAP_FILENAME
                assert dst.is_file()
                assert dst.read_text(encoding="utf-8") == '{"tables": []}'
                args = captured.get("args", ())
                kwargs = captured.get("kwargs", {})
                ctx = kwargs.get("schema_context")
                if ctx is None and len(args) >= 1:
                    ctx = args[0]
                assert ctx == SchemaContext()
                assert kwargs.get("artifacts_dir") == "/tmp/art"
            finally:
                os.chdir(old)


class TestBuildIntentSummary:
    """Smoke test for session intent summary projection."""

    def test_build_intent_summary_fields(self) -> None:
        from aetherdialect._main_execution import _build_intent_summary

        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[NormalizedExpr.from_column("t1.region")],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("t1.id"), direction="desc")],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t1.status"),
                    op="=",
                    raw_value="open",
                )
            ],
            natural_language="  headline  ",
            limit=10,
        )
        s = _build_intent_summary(intent)
        assert s.tables == ("t1",)
        assert s.limit == 10
        assert s.natural_language == "headline"
        assert len(s.select_cols) == 1
        assert len(s.filters) == 1
        assert len(s.group_by) == 1
        assert len(s.order_by) == 1
        assert s.order_by[0].upper().endswith("DESC")
