"""Tests for :mod:`aetherdialect.aetherdialect` facade helpers and validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants import MIGRATION_MAP_FILENAME
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    LLMConfig,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    RuntimeConfig,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import (
    QSimSummary,
)
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._templates import TemplateOps


def _sample_llm_execution():
    """Return a merged :class:`LlmExecutionConfig` for test doubles."""
    return load_runtime_config(merged_env=dict(os.environ))


def _make_aether_stub(**overrides):
    """Build a minimal ``AetherEngine`` instance shell for unit tests."""
    defaults = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/aether",
            engine_context=EngineContext(),
            llm_execution=_sample_llm_execution(),
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=MagicMock(),
        _dialect=MagicMock(),
        _artifacts_dir=Path("/tmp/aether"),
        _store=TemplateOps.empty_template_store("unit_test_eff"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _construction_phase_callback=None,
        _ask_phase_callback=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_stats={"table_count": 10, "total_filterable": 40},
        _schema_role="owner",
        _consumer_visible_objects=None,
        _token_provider=None,
        _native_connection=None,
        _sandbox_closed=False,
        _tenant_slug=None,
    )
    defaults.update(overrides)

    obj = AetherEngine.__new__(AetherEngine)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


class TestValidateNumIntents:
    """Tests for ``AetherEngine._validate_num_intents``."""

    def test_in_range_passes(self) -> None:
        t = _make_aether_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        lo, hi = t._compute_num_intents_range()
        t._validate_num_intents(lo)
        t._validate_num_intents(hi)

    def test_below_range_raises(self) -> None:
        t = _make_aether_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_intents must be"):
            t._validate_num_intents(1)

    def test_above_range_raises(self) -> None:
        t = _make_aether_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_intents must be"):
            t._validate_num_intents(999)


class TestValidateNumQuestions:
    """Tests for ``AetherEngine._validate_num_questions``."""

    def test_in_range_passes(self) -> None:
        t = _make_aether_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        lo, hi = t._compute_num_questions_range()
        t._validate_num_questions(lo)
        t._validate_num_questions(hi)

    def test_below_range_raises(self) -> None:
        t = _make_aether_stub(_schema_stats={"table_count": 10, "total_filterable": 40})
        with pytest.raises(ValueError, match="num_questions must be"):
            t._validate_num_questions(1)


class TestGetQsimSummary:
    """Tests for ``AetherEngine.get_qsim_summary``."""

    def test_filters_versions_inclusive(self) -> None:
        t = _make_aether_stub()
        summaries = [
            QSimSummary(version=1, num_intents=1, num_questions=1, seed=0),
            QSimSummary(version=2, num_intents=1, num_questions=1, seed=0),
            QSimSummary(version=5, num_intents=1, num_questions=1, seed=0),
        ]
        with patch("aetherdialect.aetherdialect.load_qsim_summaries", return_value=summaries):
            snap = t.get_qsim_summary(2, 4)
        text = snap.format_human()
        assert "v2" in text
        assert "Latest: v5" in text


class TestEnsureLlm:
    """Tests for ``AetherEngine._ensure_llm``."""

    @patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=False)
    def test_no_credentials_raises_config_error(self, _mock_lc: MagicMock) -> None:
        t = _make_aether_stub()
        with pytest.raises(ConfigError, match="LLM is not configured"):
            t._ensure_llm()

    @patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True)
    def test_configured_passes(self, _mock_lc: MagicMock) -> None:
        t = _make_aether_stub()
        t._ensure_llm()


class TestInitPatches:
    """Construction smoke tests with heavy dependencies patched."""

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    def test_init_builds_schema_and_store(self, mock_init: MagicMock) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="C:/tmp/x/aether_artifacts",
                engine_context=EngineContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="C:/tmp/x/aether_artifacts",
            store={"effective_structural_hash": "h"},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = AetherEngine(EngineContext(), artifacts_dir="x", config_file=None)
        assert t._schema_graph is sg
        assert t._artifacts_dir == Path("C:/tmp/x/aether_artifacts")

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    def test_init_calls_initialize_with_engine_context(self, mock_init: MagicMock) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="C:/tmp/x/aetherdialect/slug",
                engine_context=EngineContext(),
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
        AetherEngine(EngineContext(), artifacts_dir="x", config_file=None)
        assert mock_init.called
        assert mock_init.call_args[0][0] is not None


class TestClearLearningCaches:
    """Tests for template / simulation / full learning clear APIs."""

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    @patch("aetherdialect.aetherdialect.clear_template_store_only", return_value=True)
    def test_clear_template_store_reinitializes(
        self,
        mock_clear: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/aether",
                engine_context=EngineContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/aether",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_aether_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/aether"),
        )
        out = t.clear_template_store()
        assert out is True
        mock_clear.assert_called_once()
        mock_init.assert_called_once()

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    @patch("aetherdialect.aetherdialect.clear_simulation_caches_only", return_value=3)
    def test_clear_simulation_caches_reinitializes(
        self,
        mock_clear: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/aether",
                engine_context=EngineContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/aether",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_aether_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/aether"),
        )
        n = t.clear_simulation_caches()
        assert n == 3
        mock_clear.assert_called_once()
        mock_init.assert_called_once()

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    @patch("aetherdialect.aetherdialect.clear_persisted_overrides")
    @patch("aetherdialect.aetherdialect.clear_template_store_only")
    @patch("aetherdialect.aetherdialect.clear_simulation_caches_only")
    def test_clear_all_learning_with_overrides_only_clears_disks(
        self,
        mock_sim: MagicMock,
        mock_tmpl: MagicMock,
        mock_clear_over: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/aether",
                engine_context=EngineContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/aether",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_aether_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/aether"),
        )
        t.clear_all_learning(keep_overrides=True)
        mock_tmpl.assert_called_once()
        mock_sim.assert_called_once()
        mock_clear_over.assert_not_called()
        mock_init.assert_called_once()

    @patch("aetherdialect.aetherdialect.initialize_aether_engine")
    @patch("aetherdialect.aetherdialect.clear_persisted_overrides", return_value=True)
    @patch("aetherdialect.aetherdialect.clear_template_store_only")
    @patch("aetherdialect.aetherdialect.clear_simulation_caches_only")
    def test_clear_all_learning_drops_overrides(
        self,
        mock_sim: MagicMock,
        mock_tmpl: MagicMock,
        mock_clear_over: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        from aetherdialect._contracts_base import AetherEngineInitResult

        sg = MagicMock(
            effective_structural_hash="eh",
            tables={},
            refresh_schema_stats=MagicMock(return_value={"table_count": 1, "total_filterable": 2}),
        )
        mock_init.return_value = AetherEngineInitResult(
            runtime_config=RuntimeConfig(
                engine="postgresql",
                artifacts_dir="/tmp/aether",
                engine_context=EngineContext(),
                llm_execution=_sample_llm_execution(),
            ),
            llm_config=LLMConfig(provider="openai"),
            schema_graph=sg,
            dialect=MagicMock(),
            artifacts_dir="/tmp/aether",
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            schema_stats={"table_count": 1, "total_filterable": 2},
        )
        t = _make_aether_stub(
            _schema_graph=sg,
            _artifacts_dir=Path("/tmp/aether"),
        )
        t.clear_all_learning(keep_overrides=False)
        mock_tmpl.assert_called_once()
        mock_sim.assert_called_once()
        mock_clear_over.assert_called_once()
        mock_init.assert_called_once()


class TestApplyMigrationMap:
    """Tests for ``AetherEngine.apply_migration_map``."""

    def test_apply_migration_map_copies_to_artifacts_dir(self) -> None:
        captured: dict[str, object] = {}

        def _capture_init(self: AetherEngine, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "incoming_map.json"
            src.write_text('{"tables": []}', encoding="utf-8")
            artifacts = Path(td) / "artifacts"
            artifacts.mkdir()
            wrong_cwd = Path(td) / "wrong_cwd"
            wrong_cwd.mkdir()
            old = os.getcwd()
            os.chdir(wrong_cwd)
            try:
                with patch.object(AetherEngine, "__init__", _capture_init):
                    out = AetherEngine.apply_migration_map(
                        str(src),
                        engine_context=EngineContext(),
                        artifacts_dir=str(artifacts),
                    )
                assert isinstance(out, AetherEngine)
                dst = artifacts / MIGRATION_MAP_FILENAME
                assert dst.is_file()
                assert dst.read_text(encoding="utf-8") == '{"tables": []}'
                assert not (wrong_cwd / MIGRATION_MAP_FILENAME).is_file()
                args = captured.get("args", ())
                kwargs = captured.get("kwargs", {})
                ctx = kwargs.get("engine_context")
                if ctx is None and len(args) >= 1:
                    ctx = args[0]
                assert ctx == EngineContext()
                assert kwargs.get("artifacts_dir") == str(artifacts)
            finally:
                os.chdir(old)

    def test_apply_migration_map_forwards_native_connection(self) -> None:
        captured: dict[str, object] = {}
        sentinel = object()

        def _capture_init(self: AetherEngine, *args: object, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "map.json"
            src.write_text('{"version": 1, "action": "remap"}', encoding="utf-8")
            artifacts = Path(td) / "artifacts"
            artifacts.mkdir()
            with patch.object(AetherEngine, "__init__", _capture_init):
                AetherEngine.apply_migration_map(
                    str(src),
                    engine_context=EngineContext(),
                    artifacts_dir=str(artifacts),
                    native_connection=sentinel,
                    execution_engine=sentinel,
                )
        kwargs = captured.get("kwargs", {})
        assert kwargs.get("native_connection") is sentinel
        assert kwargs.get("execution_engine") is sentinel


class TestBuildIntentSummary:
    """Smoke test for session intent summary projection."""

    def test_build_intent_summary_fields(self) -> None:
        from aetherdialect._main_execution import MainExecutionOps

        intent = RuntimeIntent(
            tables=["t1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t1.id"))],
            group_by_cols=[NormalizedExpr.from_column("t1.region")],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("t1.id"), direction="desc")],
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("t1.status"),
                        op="=",
                        raw_value="open",
                    )
                ]
            ),
            natural_language="  headline  ",
            limit=10,
        )
        s = MainExecutionOps._build_intent_summary(intent)
        assert s.tables == ("t1",)
        assert s.limit == 10
        assert s.natural_language == "headline"
        assert len(s.select_cols) == 1
        assert len(s.filters) == 1
        assert len(s.group_by) == 1
        assert len(s.order_by) == 1
        assert s.order_by[0].upper().endswith("DESC")
