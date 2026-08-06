"""Unit tests for aetherdialect._seed_warmup module (mirrors ``seed_warmup.py``)."""

import json
import os
import tempfile
import zipfile
from unittest.mock import patch

import pytest

from aetherdialect._config import SeedWarmupConfig
from aetherdialect._contracts_base import (
    HavingParam,
    LlmJsonExhausted,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
    WorkloadFamily,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    GenerationPath,
    RuntimeIntent,
    SeedWarmupIntent,
    SeedWarmupResult,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import (
    ExpansionMetadata,
    SchemaGraph,
    SQLShape,
    TemplateStats,
    ValueDomain,
)
from aetherdialect._core_utils import debug
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._seed_warmup import SeedWarmupCacheSession

_abstract_values = SeedWarmupCacheSession._abstract_values
_allocate_stratum_quotas = SeedWarmupCacheSession._allocate_stratum_quotas
_ambiguous_join_reuse_from_parent = SeedWarmupCacheSession._ambiguous_join_reuse_from_parent
_build_value_domains = SeedWarmupCacheSession._build_value_domains
_confirm_gold_intent = SeedWarmupCacheSession._confirm_gold_intent
_create_template_from_result = SeedWarmupCacheSession._create_template_from_result
_decompose_between_where_param = SeedWarmupCacheSession._decompose_between_where_param
_gold_failure_trace_text = SeedWarmupCacheSession._gold_failure_trace_text
_identify_range_pairs = SeedWarmupCacheSession._identify_range_pairs
_load_seed_questions = SeedWarmupCacheSession._load_seed_questions
_load_warmup_anchor_lattice = SeedWarmupCacheSession._load_warmup_anchor_lattice
_parse_gold_intent_strict = SeedWarmupCacheSession._parse_gold_intent_strict
_replay_gold_intent_parse_for_telemetry = SeedWarmupCacheSession._replay_gold_intent_parse_for_telemetry
_save_warmup_anchor_lattice = SeedWarmupCacheSession._save_warmup_anchor_lattice
_seed_warmup_intent_sort_key = SeedWarmupCacheSession._seed_warmup_intent_sort_key
_warmup_anchor_lattice_json_path = SeedWarmupCacheSession._warmup_anchor_lattice_json_path
_warmup_canonical_result_signature = SeedWarmupCacheSession._warmup_canonical_result_signature
_warmup_pack_execute = SeedWarmupCacheSession._warmup_pack_execute
_warmup_stratum_key = SeedWarmupCacheSession._warmup_stratum_key
_warmup_submodular_cover_select = SeedWarmupCacheSession._warmup_submodular_cover_select
_warmup_synthetic_store_path_blocks = SeedWarmupCacheSession._warmup_synthetic_store_path_blocks
accepted_template_instance_keys = SeedWarmupCacheSession.accepted_template_instance_keys
build_anchor_lattice = SeedWarmupCacheSession.build_anchor_lattice
check_intent_against_expectation = SeedWarmupCacheSession.check_intent_against_expectation
get_next_seed_warmup_version = SeedWarmupCacheSession.get_next_seed_warmup_version
instantiate_intent = SeedWarmupCacheSession.instantiate_intent
load_seed_warmup_cache_zip = SeedWarmupCacheSession.load_seed_warmup_cache_zip
open_seed_warmup_cache_session = SeedWarmupCacheSession.open_seed_warmup_cache_session
resolve_joins_for_table_set = SeedWarmupCacheSession.resolve_joins_for_table_set
run_gold_intent_generation = SeedWarmupCacheSession.run_gold_intent_generation
run_seed_question_normalization = SeedWarmupCacheSession.run_seed_question_normalization
run_seed_warmup_execution = SeedWarmupCacheSession.run_seed_warmup_execution
save_seed_warmup_cache_zip = SeedWarmupCacheSession.save_seed_warmup_cache_zip
save_seed_warmup_report = SeedWarmupCacheSession.save_seed_warmup_report
seed_warmup_drops_detail_jsonl_path_for_report = SeedWarmupCacheSession.seed_warmup_drops_detail_jsonl_path_for_report
seed_warmup_drops_jsonl_path_for_report = SeedWarmupCacheSession.seed_warmup_drops_jsonl_path_for_report
warmup_intent_fingerprint = SeedWarmupCacheSession.warmup_intent_fingerprint
warmup_pool_operator_feature_stats = SeedWarmupCacheSession.warmup_pool_operator_feature_stats


def _warmup_intent(**overrides) -> SeedWarmupIntent:
    """Build a minimal SeedWarmupIntent with optional overrides."""
    defaults = dict(
        intent_id="warm_001",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        param_values={},
        expansion_metadata=None,
        limit=None,
    )
    defaults.update(overrides)
    return SeedWarmupIntent(**defaults)


class TestWarmupIntentFingerprint:
    """Tests for warmup_intent_fingerprint."""

    def test_stable_for_same_intent(self):
        intent = _warmup_intent()
        assert SeedWarmupCacheSession.warmup_intent_fingerprint(
            intent
        ) == SeedWarmupCacheSession.warmup_intent_fingerprint(intent)

    def test_changes_when_intent_id_changes(self):
        a = _warmup_intent(intent_id="a")
        b = _warmup_intent(intent_id="b")
        assert SeedWarmupCacheSession.warmup_intent_fingerprint(a) != SeedWarmupCacheSession.warmup_intent_fingerprint(
            b
        )


class TestOperatorFeatureVectorFootprint:
    """Tests for operator_feature_vector_for_seed_intent and warmup_pool_operator_feature_stats."""

    def test_minimal_vector(self):
        intent = _warmup_intent()
        v = intent.operator_feature_vector()
        assert v.window_kind == "none"
        assert v.workload_family == WorkloadFamily.EXTRACT

    def test_rank_window_kind(self):
        from aetherdialect._contracts_schema import (
            WindowRegistryStep,
            WindowSpec,
        )

        intent = _warmup_intent(
            window_registry=[
                WindowRegistryStep(registry_id="w1", window_spec=WindowSpec(function="row_number")),
            ],
        )
        v = intent.operator_feature_vector()
        assert v.window_kind == "rank"

    def test_pool_stats_empty(self):
        out = SeedWarmupCacheSession.warmup_pool_operator_feature_stats([])
        assert out["warmup_queue_distinct_operator_vectors"] == 0
        assert out["warmup_queue_operator_feature_4bit_tuple_distinct"] == 0


class TestAbstractValues:
    """Tests for abstract_values."""

    def test_clears_param_values(self):
        """Abstracted intent has empty param_values."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            param_values={"p1": "shipped", "p2": "100"},
        )
        result = SeedWarmupCacheSession._abstract_values(intent)
        assert result.param_values == {}

    def test_preserves_tables(self):
        """Abstracted intent preserves tables."""
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            param_values={"p1": "val"},
        )
        result = SeedWarmupCacheSession._abstract_values(intent)
        assert result.tables == ["orders", "customers"]

    def test_preserves_grain(self):
        """Abstracted intent preserves grain."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            param_values={"p1": "x"},
        )
        result = SeedWarmupCacheSession._abstract_values(intent)
        assert result.grain == "scalar"

    def test_preserves_filters(self):
        """Abstracted intent preserves filter structure."""
        f = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="=",
            value_type="string",
            param_key="p1",
            raw_value="shipped",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=PredicateGroup.from_list([f]),
            param_values={"p1": "shipped"},
        )
        result = SeedWarmupCacheSession._abstract_values(intent)
        assert len(result.where.leaves() if result.where else []) == 1
        assert result.param_values == {}

    def test_returns_new_object(self):
        """Abstracted intent is a different object."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            param_values={"p1": "val"},
        )
        result = SeedWarmupCacheSession._abstract_values(intent)
        assert result is not intent


class TestInstantiateIntent:
    """Tests for instantiate_intent."""

    def test_empty_filters_returns_intent(self):
        """Intent with no filters instantiates successfully."""
        intent = _warmup_intent()
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert result.param_values == {}

    def test_populates_filter_value(self):
        """Filter value is sampled from domain."""
        intent = _warmup_intent(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.status"),
                        op="=",
                        value_type="string",
                        param_key="p1",
                    )
                ]
            ),
        )
        domains = {
            "orders.status": ValueDomain(
                values=["shipped", "pending"],
                min_val=None,
                max_val=None,
            ),
        }
        result = SeedWarmupCacheSession.instantiate_intent(intent, domains)
        assert result is not None

    def test_null_filter_passthrough(self):
        """Null/is-null filters pass through without value sampling."""
        intent = _warmup_intent(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.status"),
                        op="is not null",
                        value_type="null",
                        param_key="p1",
                    )
                ]
            ),
        )
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert (result.where.leaves() if result.where else [])[0].op == "is not null"

    def test_date_window_passthrough(self):
        """date_window filters pass through without value sampling."""
        intent = _warmup_intent(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.order_date"),
                        op=">=",
                        value_type="date_window",
                        param_key="",
                        raw_value={"unit": "day", "amount": 30},
                    )
                ]
            ),
        )
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert (result.where.leaves() if result.where else [])[0].value_type == "date_window"

    def test_having_value_populated(self):
        """HAVING params get deterministic values."""
        intent = _warmup_intent(
            grain="grouped",
            group_by_cols=[NormalizedExpr.from_column("orders.status")],
            having=PredicateGroup.from_list(
                [
                    HavingParam(
                        left_expr=NormalizedExpr.from_agg("count", "*"),
                        op=">",
                        value_type="number",
                        param_key="h1",
                    )
                ]
            ),
        )
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert "h1" in result.param_values

    def test_right_expr_filter_passthrough(self):
        """Column-to-column filters are copied without sampling."""
        intent = _warmup_intent(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.customer_id"),
                        op="=",
                        value_type="integer",
                        param_key="p1",
                        right_expr=NormalizedExpr.from_column("customers.customer_id"),
                    )
                ]
            ),
        )
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert (result.where.leaves() if result.where else [])[0].right_expr is not None
        assert "p1" not in result.param_values

    def test_missing_domain_skips_value(self):
        """No domain for a value filter leaves param_values empty for that key."""
        intent = _warmup_intent(
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("orders.unknown_col"),
                        op="=",
                        value_type="string",
                        param_key="px",
                    )
                ]
            ),
        )
        result = SeedWarmupCacheSession.instantiate_intent(intent, {})
        assert result is not None
        assert "px" not in result.param_values


class TestGetNextSeedWarmupVersion:
    """Tests for get_next_seed_warmup_version."""

    def test_empty_directory(self):
        """Returns 1 for empty directory."""
        with tempfile.TemporaryDirectory() as td:
            assert SeedWarmupCacheSession.get_next_seed_warmup_version(td) == 1

    def test_existing_versions(self):
        """Returns max + 1 for existing report or bundle files."""
        with tempfile.TemporaryDirectory() as td:
            for v in [1, 2, 3]:
                open(os.path.join(td, f"seed_warmup_report_v{v}.json"), "w").close()
            assert SeedWarmupCacheSession.get_next_seed_warmup_version(td) == 4

    def test_single_version(self):
        """Returns 2 when only v1 bundle exists."""
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "seed_warmup_v1.zip"), "w").close()
            assert SeedWarmupCacheSession.get_next_seed_warmup_version(td) == 2


class TestResolveJoinsForTableSet:
    """Tests for resolve_joins_for_table_set join cache behavior."""

    def test_single_table_returns_j00(self, schema_graph):
        cache: dict = {}
        jid, sig, cands = SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["orders"],
            schema_graph,
            "test",
            cache,
        )
        assert jid == "J00"
        assert (
            SeedWarmupCacheSession._warmup_join_cache_key(["orders"], "test", hint_is_natural_language=False) in cache
        )

    def test_cache_hit_avoids_recompute(self, schema_graph):
        cache: dict = {}
        SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["orders"],
            schema_graph,
            "test",
            cache,
        )
        result1 = cache[
            SeedWarmupCacheSession._warmup_join_cache_key(["orders"], "test", hint_is_natural_language=False)
        ]
        SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["orders"],
            schema_graph,
            "test",
            cache,
        )
        result2 = cache[
            SeedWarmupCacheSession._warmup_join_cache_key(["orders"], "test", hint_is_natural_language=False)
        ]
        assert result1 is result2

    def test_different_table_sets_cached_separately(self, schema_graph):
        cache: dict = {}
        SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["orders"],
            schema_graph,
            "test",
            cache,
        )
        SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["customers"],
            schema_graph,
            "test",
            cache,
        )
        assert len(cache) == 2
        assert (
            SeedWarmupCacheSession._warmup_join_cache_key(["orders"], "test", hint_is_natural_language=False) in cache
        )
        assert (
            SeedWarmupCacheSession._warmup_join_cache_key(["customers"], "test", hint_is_natural_language=False)
            in cache
        )


class TestLoadSeedWarmupCacheZip:
    """Tests for load_seed_warmup_cache_zip."""

    def test_missing_zip_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            man, wu = SeedWarmupCacheSession.load_seed_warmup_cache_zip(td)
            assert man == {}
            assert wu == {}

    def test_reads_manifest_and_work_units(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            manifest = {"schema_hash": "s", "seed_content_hash": "h"}
            wrec = {
                "work_unit_id": "wid-1",
                "intent_fingerprint": "fp1",
                "execute_result": {"ok": True},
            }
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(
                    SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
                    json.dumps(manifest),
                )
                zf.writestr(
                    f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}wid-1.json",
                    json.dumps(wrec),
                )
            man, wu = SeedWarmupCacheSession.load_seed_warmup_cache_zip(td)
            assert man.get("schema_hash") == "s"
            assert "wid-1" in wu
            assert wu["wid-1"]["intent_fingerprint"] == "fp1"

    def test_work_unit_id_fallback_to_basename(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            wrec = {"intent_fingerprint": "fpx", "execute_result": {}}
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(
                    f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}from_name.json",
                    json.dumps(wrec),
                )
            _m, wu = SeedWarmupCacheSession.load_seed_warmup_cache_zip(td)
            assert "from_name" in wu


class TestOpenSeedWarmupCacheSession:
    """Tests for open_seed_warmup_cache_session."""

    def test_hash_mismatch_clears_work_units(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(
                    SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
                    json.dumps({"schema_hash": "old", "seed_content_hash": "old"}),
                )
                zf.writestr(
                    f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}x.json",
                    json.dumps(
                        {
                            "work_unit_id": "x",
                            "intent_fingerprint": "f",
                            "execute_result": {},
                        },
                    ),
                )
            sg = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="new_schema")
            sess = SeedWarmupCacheSession.open_seed_warmup_cache_session(td, sg, "new_seed")
            assert sess.work_units == {}
            assert sess.fp_to_wid == {}

    def test_hash_match_loads_fp_map(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(
                    SeedWarmupConfig.WARMUP_CACHE_MANIFEST,
                    json.dumps(
                        {
                            "effective_structural_hash": "same",
                            "schema_hash": "same",
                            "schema_graph_id": "same",
                            "seed_content_hash": "same",
                            "profiling_hash": "p0",
                        },
                    ),
                )
                zf.writestr(
                    f"{SeedWarmupConfig.WARMUP_CACHE_WORK_PREFIX}u1.json",
                    json.dumps(
                        {
                            "work_unit_id": "u1",
                            "intent_fingerprint": "finger",
                            "execute_result": {"ok": True},
                        },
                    ),
                )
            sg = SchemaGraph(
                tables={},
                join_paths_multi={},
                effective_structural_hash="same",
                structural_hash="same",
                schema_graph_id="same",
                profiling_hash="p0",
            )
            sess = SeedWarmupCacheSession.open_seed_warmup_cache_session(td, sg, "same")
            assert sess.fp_to_wid["finger"] == "u1"
            assert "u1" in sess.work_units


class TestWarmupPackExecute:
    """Smoke test for _warmup_pack_execute dict shape."""

    def test_keys_present(self):
        rt = _warmup_intent().to_runtime_intent()
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            rt,
            ok=False,
            final_sql=None,
            failure_code="x",
            error="e",
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        assert pack["ok"] is False
        assert pack["failure_code"] == "x"
        assert pack["body_key"] == "b"
        assert "runtime" in pack


class TestRunSeedQuestionNormalization:
    """Tests for SeedWarmupCacheSession.run_seed_question_normalization (LLM mocked)."""

    @patch("aetherdialect._seed_warmup.LLMProvider.json")
    def test_normalizes_batch(self, mock_llm):
        mock_llm.return_value = {
            "lines": [
                {"index": 1, "normalized": "One"},
                {"index": 2, "normalized": "Two"},
            ],
        }
        seeds = [{"number": 1, "question": "  a  "}, {"number": 2, "question": "b"}]
        phrases, json_body, txt = SeedWarmupCacheSession.run_seed_question_normalization(seeds)
        assert phrases[1]["original"] == "a"
        assert phrases[1]["normalized"] == "One"
        assert phrases[2]["normalized"] == "Two"
        assert "1" in json_body and "One" in json_body
        assert "1. One" in txt

    @patch("aetherdialect._seed_warmup.LLMProvider.json")
    def test_falls_back_when_llm_returns_no_lines(self, mock_llm):
        mock_llm.return_value = {"lines": "not-a-list"}
        seeds = [{"number": 5, "question": "orig"}]
        phrases, _, _ = SeedWarmupCacheSession.run_seed_question_normalization(seeds)
        assert phrases[5]["normalized"] == "orig"

    @patch("aetherdialect._seed_warmup.LLMProvider.json")
    def test_falls_back_to_originals_when_llm_json_exhausted(self, mock_llm):
        mock_llm.side_effect = LlmJsonExhausted(task="default", attempts=3)
        seeds = [{"number": 1, "question": "alpha"}, {"number": 2, "question": "beta"}]
        phrases, _, _ = SeedWarmupCacheSession.run_seed_question_normalization(seeds)
        assert phrases[1]["normalized"] == "alpha"
        assert phrases[2]["normalized"] == "beta"


class TestLoadSeedQuestions:
    """Tests for load_seed_questions."""

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            SeedWarmupCacheSession._load_seed_questions("/nonexistent/path.txt")

    def test_numbered_format(self):
        """Parses numbered lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("1. What is the total revenue?\n")
            f.write("2. How many customers?\n")
            f.flush()
            path = f.name
        try:
            results = SeedWarmupCacheSession._load_seed_questions(path)
            assert len(results) == 2
            assert "revenue" in results[0]["question"].lower()
        finally:
            os.unlink(path)

    def test_phase_headers(self):
        """Parses phase headers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("# Phase 1\n")
            f.write("What tables exist?\n")
            f.write("# Phase 2\n")
            f.write("Show revenue.\n")
            f.flush()
            path = f.name
        try:
            results = SeedWarmupCacheSession._load_seed_questions(path)
            assert len(results) >= 2
        finally:
            os.unlink(path)

    def test_max_seed_questions_caps_file(self):
        """Very large seed files are truncated to SeedWarmupConfig.MAX_SEED_QUESTIONS."""
        n = SeedWarmupConfig.MAX_SEED_QUESTIONS + 25
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for i in range(n):
                f.write(f"{i}. Seed question {i}?\n")
            f.flush()
            path = f.name
        try:
            results = SeedWarmupCacheSession._load_seed_questions(path)
            assert len(results) == SeedWarmupConfig.MAX_SEED_QUESTIONS
        finally:
            os.unlink(path)


class TestDecomposeBetweenWhereParam:
    """Tests for _decompose_between_where_param."""

    def test_between_decomposed(self):
        """BETWEEN filter becomes >= and <= pair."""
        f = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="between",
            value_type="number",
            param_key="p1",
        )
        parts = SeedWarmupCacheSession._decompose_between_where_param(f)
        assert len(parts) == 2
        ops = {p.op for p in parts}
        assert ">=" in ops
        assert "<=" in ops

    def test_non_between_passthrough(self):
        """Non-between filter passes through unchanged."""
        f = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="=",
            value_type="string",
            param_key="p1",
        )
        parts = SeedWarmupCacheSession._decompose_between_where_param(f)
        assert len(parts) == 1
        assert parts[0].op == "="

    def test_between_empty_param_key_uses_none_keys(self):
        """BETWEEN with no param_key yields filters with param_key None."""
        f = WhereParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="between",
            value_type="number",
            param_key="",
        )
        parts = SeedWarmupCacheSession._decompose_between_where_param(f)
        assert len(parts) == 2
        assert parts[0].param_key is None
        assert parts[1].param_key is None


class TestIdentifyRangePairs:
    """Tests for _identify_range_pairs."""

    def test_identifies_range(self):
        """Detects paired >= / <= on the same column."""
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">=",
                value_type="number",
                param_key="p1",
            ),
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op="<=",
                value_type="number",
                param_key="p2",
            ),
        ]
        pairs = SeedWarmupCacheSession._identify_range_pairs(filters)
        assert "orders.amount" in pairs
        assert "lower_idx" in pairs["orders.amount"]
        assert "upper_idx" in pairs["orders.amount"]

    def test_no_range(self):
        """Single-sided filter produces no range pair."""
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">=",
                value_type="number",
                param_key="p1",
            ),
        ]
        pairs = SeedWarmupCacheSession._identify_range_pairs(filters)
        assert pairs == {}

    def test_expr_filters_excluded(self):
        """Filters with right_expr are excluded."""
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">=",
                value_type="number",
                param_key="p1",
                right_expr=NormalizedExpr.from_column("orders.order_id"),
            ),
        ]
        pairs = SeedWarmupCacheSession._identify_range_pairs(filters)
        assert pairs == {}

    def test_gt_lt_pair_detected(self):
        """> and < on the same column form a range pair."""
        filters = [
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op=">",
                value_type="number",
                param_key="a",
            ),
            WhereParam(
                left_expr=NormalizedExpr.from_column("orders.amount"),
                op="<",
                value_type="number",
                param_key="b",
            ),
        ]
        pairs = SeedWarmupCacheSession._identify_range_pairs(filters)
        assert "orders.amount" in pairs


class TestCreateTemplateFromResult:
    """Tests for create_template_from_result."""

    def test_unsuccessful_result_returns_none(self, schema_graph):
        """Unsuccessful result returns None."""
        result = SeedWarmupResult(
            intent=_warmup_intent(),
            sql=None,
            success=False,
            question="q",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(result, schema_graph, next_id=1)
        assert tmpl is None

    def test_successful_result_returns_template(self, schema_graph):
        """Successful result returns a Template."""
        intent = _warmup_intent()
        intent.expected_rows = "many"
        intent.natural_language = "show orders"
        intent.chosen_join_candidate_id = ""
        intent.chosen_join_path_signature = []
        result = SeedWarmupResult(
            intent=intent,
            sql="SELECT order_id FROM orders",
            success=True,
            question="show orders",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(result, schema_graph, next_id=42)
        assert tmpl is not None
        assert tmpl.id == "T0042"
        assert tmpl.source == "synthetic"
        assert tmpl.trust_level == 1

    def test_custom_source_and_trust(self, schema_graph):
        """Custom source and trust_level are respected."""
        intent = _warmup_intent()
        intent.expected_rows = "many"
        intent.natural_language = "show orders"
        intent.chosen_join_candidate_id = ""
        intent.chosen_join_path_signature = []
        result = SeedWarmupResult(
            intent=intent,
            sql="SELECT order_id FROM orders",
            success=True,
            question="show orders",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(
            result, schema_graph, next_id=1, source="gold", trust_level=2
        )
        assert tmpl.source == "gold"
        assert tmpl.trust_level == 2

    def test_single_value_history_row_without_seed_provenance(self, schema_graph):
        """No seed provenance intent → one primary ``ValueHistory`` row when normalized equals surface."""
        intent = _warmup_intent()
        intent.expected_rows = "many"
        intent.natural_language = "show orders"
        intent.chosen_join_candidate_id = ""
        intent.chosen_join_path_signature = []
        result = SeedWarmupResult(
            intent=intent,
            sql="SELECT order_id FROM orders",
            success=True,
            question="show orders",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(result, schema_graph, next_id=5)
        assert isinstance(tmpl.value_history, ValueHistory)
        assert len(tmpl.value_history) == 1
        assert tmpl.value_history.questions[0] == "show orders"

    def test_normalized_optional_second_row_without_seed_provenance(self, schema_graph):
        """Surface question differs from normalized form → second ``ValueHistory`` row with raised accept count."""
        intent = _warmup_intent()
        intent.expected_rows = "many"
        intent.natural_language = "show orders"
        intent.chosen_join_candidate_id = ""
        intent.chosen_join_path_signature = []
        result = SeedWarmupResult(
            intent=intent,
            sql="SELECT order_id FROM orders",
            success=True,
            question="User facing question",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(result, schema_graph, next_id=6)
        assert isinstance(tmpl.value_history, ValueHistory)
        assert len(tmpl.value_history) == 2
        assert tmpl.value_history.questions[0] == "User facing question"
        assert tmpl.value_history.accept_counts[1] == 1

    def test_value_history_rows_with_seed_provenance(self, schema_graph):
        """Seed original, normalized, and LLM question each get a `ValueHistory` row."""
        intent = _warmup_intent()
        intent.expected_rows = "many"
        intent.natural_language = "nl summary"
        intent.param_values = {"p": 1}
        intent.chosen_join_candidate_id = ""
        intent.chosen_join_path_signature = []
        provenance_intent = SeedWarmupIntent(
            intent_id="gold_1",
            tables=intent.tables,
            grain=intent.grain,
            select_cols=intent.select_cols,
            group_by_cols=intent.group_by_cols,
            order_by_cols=intent.order_by_cols,
            where=intent.where,
            having=intent.having,
            param_values=intent.param_values,
            seed_prompt_original="raw seed line",
            seed_prompt_normalized="Refined seed line",
            seed_index=1,
            source="gold",
        )
        result = SeedWarmupResult(
            intent=intent,
            sql="SELECT order_id FROM orders",
            success=True,
            question="LLM question text",
        )
        tmpl = SeedWarmupCacheSession._create_template_from_result(
            result,
            schema_graph,
            next_id=9,
            seed_warmup_intent=provenance_intent,
        )
        vh = tmpl.value_history
        assert len(vh) == 3
        assert vh.questions[0] == "raw seed line"
        assert vh.questions[1] == "Refined seed line"
        assert vh.questions[2] == "LLM question text"
        assert vh.natural_language[0] == vh.natural_language[1] == vh.natural_language[2] == "nl summary"
        assert vh.param_values[0] == vh.param_values[1] == vh.param_values[2] == {"p": 1}


class TestGoldFailureTraceText:
    """Tests for _gold_failure_trace_text."""

    def test_header_and_sections(self):
        body = SeedWarmupCacheSession._gold_failure_trace_text(7, ["block_a", "block_b"])
        assert "seed_warmup_version=7" in body
        assert "failed_seed_count=2" in body
        assert "interactive_gold=false" in body
        assert "block_a" in body and "block_b" in body


class TestParseGoldIntentStrict:
    """Tests for _parse_gold_intent_strict."""

    def test_strict_returns_intent_on_first_parse(self, schema_graph, monkeypatch):
        ok = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            natural_language="x",
        )

        def fake_parse(q, schema_inner, max_retries=3):
            return ok, [], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        intent, warns = SeedWarmupCacheSession._parse_gold_intent_strict("any", schema_graph)
        assert intent is ok
        assert warns == []

    def test_strict_retries_once(self, schema_graph, monkeypatch):
        calls = {"n": 0}
        ok = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )

        def fake_parse(q, schema_inner, max_retries=3):
            calls["n"] += 1
            if calls["n"] == 1:
                return None, ["first"], 0, None
            return ok, [], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        intent, warns = SeedWarmupCacheSession._parse_gold_intent_strict("q", schema_graph)
        assert intent is ok
        assert calls["n"] == 2

    def test_strict_none_after_two_failed_attempts(self, schema_graph, monkeypatch):
        monkeypatch.setattr(
            "aetherdialect._seed_warmup.full_intent_parse",
            lambda q, s, max_retries=3: (None, ["bad"], 0, None),
        )
        intent, warns = SeedWarmupCacheSession._parse_gold_intent_strict("q", schema_graph)
        assert intent is None
        assert "bad" in warns


class TestReplayGoldIntentParseForTelemetry:
    """Tests for _replay_gold_intent_parse_for_telemetry."""

    def test_calls_full_intent_parse_twice(self, schema_graph, monkeypatch):
        calls = {"n": 0}

        def fake_parse(q, schema_inner, max_retries=3):
            calls["n"] += 1
            return None, [], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        SeedWarmupCacheSession._replay_gold_intent_parse_for_telemetry("  Q  ", schema_graph)
        assert calls["n"] == 2


class TestConfirmGoldIntent:
    """Tests for _confirm_gold_intent."""

    def test_accepts_on_y(self, monkeypatch):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            natural_language="nl",
        )
        monkeypatch.setattr("aetherdialect._seed_warmup.ask_user_choice", lambda _p, _opts: "y")
        ok, out = SeedWarmupCacheSession._confirm_gold_intent("Q?", intent)
        assert ok is True
        assert out is intent

    def test_rejects_on_n(self, monkeypatch):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            natural_language="nl",
        )
        monkeypatch.setattr("aetherdialect._seed_warmup.ask_user_choice", lambda _p, _opts: "n")
        ok, out = SeedWarmupCacheSession._confirm_gold_intent("Q?", intent)
        assert ok is False
        assert out is None


class TestGoldFailureTrace:
    """Tests for non-interactive gold failure trace payload."""

    def test_returns_trace_body_when_parse_fails(self, tmp_path, schema_graph, monkeypatch):
        """``interactive=False`` and a failed seed yields a non-empty trace string."""
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("1) List all films\n", encoding="utf-8")
        phrases = {1: {"original": "List all films", "normalized": "List all films"}}

        def fake_parse(question, schema_graph_inner, max_retries=3):
            debug("fake_parse_invoked")
            return None, ["unit_warn"], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        _got, _stats, trace_body, _nb = SeedWarmupCacheSession.run_gold_intent_generation(
            schema_graph,
            str(seed_file),
            interactive=False,
            seed_phrases=phrases,
            seed_warmup_version=5,
        )
        assert trace_body is not None
        assert "seed_warmup_version=5" in trace_body
        assert "interactive_gold=false" in trace_body
        assert "seed_index=1" in trace_body
        assert "unit_warn" in trace_body
        assert trace_body.count("fake_parse_invoked") >= 2

    def test_no_trace_body_when_interactive(self, tmp_path, schema_graph, monkeypatch):
        """Failed parse with ``interactive=True`` does not build a trace body."""
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("1) List all films\n", encoding="utf-8")
        phrases = {1: {"original": "List all films", "normalized": "List all films"}}

        def fake_parse(question, schema_graph_inner, max_retries=3):
            return None, ["x"], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        _a, _b, trace_body, _c = SeedWarmupCacheSession.run_gold_intent_generation(
            schema_graph,
            str(seed_file),
            interactive=True,
            seed_phrases=phrases,
            seed_warmup_version=9,
        )
        assert trace_body is None

    def test_semantic_warnings_do_not_fail_gold(self, tmp_path, schema_graph, monkeypatch):
        """Non-interactive run saves gold when parse returns an intent with warnings only."""
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("1) List all films\n", encoding="utf-8")
        phrases = {1: {"original": "List all films", "normalized": "List all films"}}
        ok_intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.film_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            natural_language="films",
        )

        def fake_parse(question, schema_graph_inner, max_retries=3):
            return ok_intent, ["semantic_warn_only"], 0, None

        monkeypatch.setattr("aetherdialect._seed_warmup.full_intent_parse", fake_parse)
        gold, stats, trace_body, _nb = SeedWarmupCacheSession.run_gold_intent_generation(
            schema_graph,
            str(seed_file),
            interactive=False,
            seed_phrases=phrases,
            seed_warmup_version=3,
        )
        assert len(gold) == 1
        assert stats["gold_failed"] == 0
        assert stats["gold_new"] == 1
        assert trace_body is None


class TestSeedWarmupCacheAndReport:
    """Cache work units and extended warmup report fields."""

    def test_write_work_unit_then_get_cached_execute(self):
        si = _warmup_intent(intent_id="cache_01")
        fp = SeedWarmupCacheSession.warmup_intent_fingerprint(si)
        rt = si.to_runtime_intent()
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            rt,
            ok=True,
            final_sql="SELECT 1",
            failure_code=None,
            error=None,
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        sess.write_work_unit(fp, si, pack, report_version=1)
        got = sess.get_cached_execute(fp)
        assert got is not None
        assert got.get("ok") is True
        assert got.get("final_sql") == "SELECT 1"

    def test_save_report_failure_histogram_and_drops_jsonl(self, tmp_path):
        si = _warmup_intent()
        rt = si.to_runtime_intent()
        ok_row = SeedWarmupResult(rt, "", success=True, sql="x")
        bad_row = SeedWarmupResult(
            rt,
            "",
            success=False,
            failure_code="join_resolution_failed",
            error="e",
        )
        report_path = str(tmp_path / "seed_warmup_report_v2.json")
        SeedWarmupCacheSession.save_seed_warmup_report(
            [ok_row, bad_row],
            report_path,
            funnel={
                "warmup_drop_audit": [
                    {
                        "intent_index": 1,
                        "intent_id": "i1",
                        "failure_code": "stratum_quota_exceeded",
                        "origin": "synthetic",
                        "stratum_id": "s1",
                    }
                ],
            },
        )
        drops_path = SeedWarmupCacheSession.seed_warmup_drops_jsonl_path_for_report(report_path)
        assert drops_path and os.path.isfile(drops_path)
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["failure_histogram"]["success"] == 1
        assert data["failure_histogram"]["join_resolution_failed"] == 1
        assert data["drops_by_code"]["stratum_quota_exceeded"] == 1
        with open(drops_path, encoding="utf-8") as jf:
            lines = [ln for ln in jf if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["failure_code"] == "stratum_quota_exceeded"

    def test_save_report_includes_sampling_block(self, tmp_path):
        report_path = str(tmp_path / "seed_warmup_report_v3.json")
        si = _warmup_intent()
        rt = si.to_runtime_intent()
        row = SeedWarmupResult(rt, "", success=True, sql="x")
        ws = {
            "skipped_due_to_low_volume": True,
            "counts": {"eligible_total": 1},
            "sampling_drops_by_code": {},
            "sampling_drop_records": [],
        }
        SeedWarmupCacheSession.save_seed_warmup_report(
            [row],
            report_path,
            funnel={"warmup_sampling": ws},
        )
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["sampling"]["skipped_due_to_low_volume"] is True
        assert "sampling_drop_records" not in data["sampling"]

    def test_save_report_writes_drops_detail_jsonl(self, tmp_path):
        report_path = str(tmp_path / "seed_warmup_report_v9.json")
        si = _warmup_intent()
        rt = si.to_runtime_intent()
        row = SeedWarmupResult(rt, "", success=True, sql="x")
        drop_row = {
            "drop_phase": "sampling",
            "intent_index": 0,
            "failure_code": "stratum_quota_exceeded",
        }
        SeedWarmupCacheSession.save_seed_warmup_report(
            [row],
            report_path,
            funnel={
                "warmup_sampling": {
                    "sampling_drop_records": [drop_row],
                },
            },
        )
        detail = SeedWarmupCacheSession.seed_warmup_drops_detail_jsonl_path_for_report(report_path)
        assert detail and os.path.isfile(detail)
        with open(detail, encoding="utf-8") as df:
            assert json.loads(df.readline())["failure_code"] == "stratum_quota_exceeded"


class TestSeedWarmupDropsPathHelpers:
    """Tests for drops path helpers on unrecognized report names."""

    def test_unknown_basename_returns_none(self):
        assert SeedWarmupCacheSession.seed_warmup_drops_jsonl_path_for_report("/tmp/other.json") is None
        assert SeedWarmupCacheSession.seed_warmup_drops_detail_jsonl_path_for_report("/tmp/other.json") is None


class TestSeedWarmupCacheSessionEdgeCases:
    """Extra SeedWarmupCacheSession behavior."""

    def test_ensure_work_unit_id_stable(self):
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        a = sess.ensure_work_unit_id("fp1")
        b = sess.ensure_work_unit_id("fp1")
        assert a == b
        assert sess.ensure_work_unit_id("fp2") != a

    def test_get_cached_execute_mismatch_fingerprint(self):
        si = _warmup_intent(intent_id="x")
        fp = SeedWarmupCacheSession.warmup_intent_fingerprint(si)
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            si.to_runtime_intent(),
            ok=True,
            final_sql="SELECT 1",
            failure_code=None,
            error=None,
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        sess.write_work_unit(fp, si, pack, report_version=1)
        assert sess.get_cached_execute("wrong_fp") is None

    def test_mark_sampled_in_requires_ok_execute(self):
        si = _warmup_intent()
        fp = SeedWarmupCacheSession.warmup_intent_fingerprint(si)
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            si.to_runtime_intent(),
            ok=False,
            final_sql=None,
            failure_code="bad",
            error="e",
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        sess.write_work_unit(fp, si, pack, report_version=1)
        assert sess.mark_sampled_in(fp) is None

    def test_record_question_llm_updates_state(self):
        si = _warmup_intent()
        fp = SeedWarmupCacheSession.warmup_intent_fingerprint(si)
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            si.to_runtime_intent(),
            ok=True,
            final_sql="SELECT 1",
            failure_code=None,
            error=None,
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        sess.write_work_unit(fp, si, pack, report_version=1)
        sess.record_question_llm(fp, {"q": "?"}, ok=True)
        wid = sess.fp_to_wid[fp]
        assert sess.work_units[wid]["lifecycle_state"] == "llm_done"


class TestWarmupResultSignature:
    """Post-execute result-signature atoms for submodular cover."""

    def test_canonical_result_signature_stable(self):
        rows_a = [(1, "x", None), (2, "y", 3.14159265)]
        rows_b = [(2, "y", 3.14159265), (1, "x", None)]
        assert SeedWarmupCacheSession._warmup_canonical_result_signature(
            rows_a
        ) == SeedWarmupCacheSession._warmup_canonical_result_signature(rows_b)

    def test_rsig_atom_included_when_provided(self):
        from aetherdialect._seed_warmup import SeedWarmupCacheSession

        _warmup_submodular_atoms_for_row = SeedWarmupCacheSession._warmup_submodular_atoms_for_row

        ordered = [_warmup_intent(intent_id="i0")]
        rsig = "abc123"
        atoms = SeedWarmupCacheSession._warmup_submodular_atoms_for_row(ordered, 0, position_rsig={0: rsig})
        assert f"rsig:{rsig}" in atoms


class TestWarmupStratifiedSamplingDetail:
    """Stratified sampling metadata and ``global_cap_after_gold``."""

    def test_low_volume_skips_stratification(self, monkeypatch):
        monkeypatch.setattr("aetherdialect._config.SeedWarmupConfig.WARMUP_KEEP_ALL_BELOW", 10)
        ordered = [_warmup_intent(intent_id=f"i{k}") for k in range(3)]
        eligible = [0, 1, 2]
        sampled, gold_dropped, detail = SeedWarmupCacheSession._warmup_submodular_cover_select(ordered, eligible)
        assert detail["skipped_due_to_low_volume"] is True
        assert sampled <= {0, 1, 2}
        assert gold_dropped == {0, 1, 2} - sampled

    def test_global_cap_after_gold_when_budget_zero(self, monkeypatch):
        from aetherdialect._contracts_schema import ExpansionMetadata

        monkeypatch.setattr("aetherdialect._config.SeedWarmupConfig.WARMUP_STRATUM_MIN", 0)
        monkeypatch.setattr("aetherdialect._config.SeedWarmupConfig.WARMUP_TARGET_CAP", 2)
        monkeypatch.setattr("aetherdialect._config.SeedWarmupConfig.WARMUP_KEEP_ALL_BELOW", 1)
        ordered: list = []
        for i in range(3):
            ordered.append(
                _warmup_intent(
                    intent_id=f"g{i}",
                    source="gold",
                )
            )
        for i in range(2):
            ordered.append(
                _warmup_intent(
                    intent_id=f"s{i}",
                    source="synthetic",
                    expansion_metadata=ExpansionMetadata(operator="x", depth=1),
                )
            )
        eligible = list(range(5))
        _sampled, _gd, detail = SeedWarmupCacheSession._warmup_submodular_cover_select(ordered, eligible)
        codes = [r["failure_code"] for r in detail["sampling_drop_records"]]
        assert any(c in codes for c in ("gold_cap_exceeded", "gold_stratum_quota_exceeded"))
        assert "global_cap_after_gold" in codes
        assert detail["counts"]["synthetic_budget"] == 0


class TestSeedWarmupExecutionCache:
    """Execute cache reuse across repeated runs."""

    def test_second_run_hits_execute_cache(self, monkeypatch, schema_graph, tmp_path):
        calls = {"execute": 0}

        def fake_validate(_dialect, _sql, _params=None, **_kw):
            return True, None, None, []

        monkeypatch.setattr("aetherdialect._seed_warmup.validate_sql", fake_validate)

        d = PostgresDialect.__new__(PostgresDialect)

        def _prep(sql_param, params, schema, intent, **kw):
            return sql_param

        def _exec(_sql, _params=None):
            calls["execute"] += 1
            return []

        d.finalize_render = _prep
        d.execute = _exec
        d.can_explain = lambda: False

        join_cache: dict = {}
        SeedWarmupCacheSession.resolve_joins_for_table_set(
            ["orders"],
            schema_graph,
            "jc1",
            join_cache,
        )
        si = _warmup_intent(
            intent_id="cache_run",
            tables=["orders"],
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        lattice_root = str(tmp_path)
        SeedWarmupCacheSession.run_seed_warmup_execution(
            [si],
            schema_graph,
            d,
            1,
            join_cache=join_cache,
            warmup_cache=sess,
            warmup_report_version=1,
            warmup_lattice_root=lattice_root,
        )
        first_exec = calls["execute"]
        assert first_exec >= 1
        SeedWarmupCacheSession.run_seed_warmup_execution(
            [si],
            schema_graph,
            d,
            1,
            join_cache=join_cache,
            warmup_cache=sess,
            warmup_report_version=1,
            warmup_lattice_root=lattice_root,
        )
        assert calls["execute"] == first_exec
        assert sess.execute_hits >= 1


class TestWarmupAnchorLatticePersistence:
    """Disk-backed anchor lattice JSON next to warmup artifacts."""

    def test_roundtrip_load_save(self, tmp_path, schema_graph):
        root = str(tmp_path)
        path = SeedWarmupCacheSession._warmup_anchor_lattice_json_path(root, schema_graph)
        cells = {"k1": ["alpha", "beta"]}
        SeedWarmupCacheSession._save_warmup_anchor_lattice(path, schema_graph, cells)
        assert os.path.isfile(path)
        assert SeedWarmupCacheSession._load_warmup_anchor_lattice(path) == cells


class TestSeedWarmupCacheGoldSnapshotAndSampledIn:
    """Gold snapshot in cache zip and sampled_in lifecycle on work units."""

    def test_save_cache_zip_includes_gold_intents_json(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = {
                "schema_hash": "s1",
                "seed_content_hash": "h1",
                "policy_version": "1",
                "code_version": "1",
            }
            gold = [{"intent_id": "g1", "tables": ["orders"]}]
            SeedWarmupCacheSession.save_seed_warmup_cache_zip(td, manifest, {}, gold_intent_dicts=gold)
            zpath = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            with zipfile.ZipFile(zpath, "r") as zf:
                assert SeedWarmupConfig.WARMUP_CACHE_GOLD_INTENTS_JSON in zf.namelist()
                body = json.loads(zf.read(SeedWarmupConfig.WARMUP_CACHE_GOLD_INTENTS_JSON).decode())
                assert body == gold
                man = json.loads(zf.read(SeedWarmupConfig.WARMUP_CACHE_MANIFEST).decode())
            assert man.get("gold_intent_count") == 1

    def test_save_cache_zip_without_gold_omits_gold_blob(self):
        with tempfile.TemporaryDirectory() as td:
            SeedWarmupCacheSession.save_seed_warmup_cache_zip(td, {"k": "v"}, {})
            zpath = os.path.join(td, SeedWarmupConfig.SEED_WARMUP_CACHE_ZIP)
            with zipfile.ZipFile(zpath, "r") as zf:
                assert SeedWarmupConfig.WARMUP_CACHE_GOLD_INTENTS_JSON not in zf.namelist()
            man, wu = SeedWarmupCacheSession.load_seed_warmup_cache_zip(td)
            assert "gold_intent_count" not in man

    def test_mark_sampled_in_after_execute_recorded(self):
        si = _warmup_intent(intent_id="life_01")
        fp = SeedWarmupCacheSession.warmup_intent_fingerprint(si)
        rt = si.to_runtime_intent()
        pack = SeedWarmupCacheSession._warmup_pack_execute(
            rt,
            ok=True,
            final_sql="SELECT 1",
            failure_code=None,
            error=None,
            body_key="b",
            join_path_key="j",
            template_instance_key="t",
        )
        sess = SeedWarmupCacheSession(manifest={}, work_units={})
        sess.write_work_unit(fp, si, pack, report_version=1)
        wid = sess.fp_to_wid[fp]
        assert sess.work_units[wid]["lifecycle_state"] == "execute_recorded"
        assert sess.mark_sampled_in(fp) == wid
        assert sess.work_units[wid]["lifecycle_state"] == "sampled_in"


class TestSeedWarmupIntentSortKey:
    """Tests for _seed_warmup_intent_sort_key."""

    def test_depth_before_id(self):
        shallow = _warmup_intent(
            intent_id="z",
            expansion_metadata=ExpansionMetadata(operator="x", depth=0),
        )
        deep = _warmup_intent(
            intent_id="a",
            expansion_metadata=ExpansionMetadata(operator="x", depth=5),
        )
        assert SeedWarmupCacheSession._seed_warmup_intent_sort_key(
            shallow
        ) < SeedWarmupCacheSession._seed_warmup_intent_sort_key(deep)

    def test_same_depth_sorts_by_intent_id(self):
        a = _warmup_intent(intent_id="m", expansion_metadata=ExpansionMetadata(operator="o", depth=1))
        b = _warmup_intent(intent_id="n", expansion_metadata=ExpansionMetadata(operator="o", depth=1))
        assert SeedWarmupCacheSession._seed_warmup_intent_sort_key(
            a
        ) < SeedWarmupCacheSession._seed_warmup_intent_sort_key(b)


class TestAmbiguousJoinReuseFromParent:
    """Tests for _ambiguous_join_reuse_from_parent."""

    def test_returns_cached_entry_when_parent_matches_tables(self):
        parent = _warmup_intent(intent_id="p1", tables=["customers", "orders"])
        child = _warmup_intent(
            intent_id="c1",
            tables=["customers", "orders"],
            expansion_metadata=ExpansionMetadata(
                operator="join",
                parent_intent_id="p1",
                depth=1,
            ),
        )
        key = SeedWarmupCacheSession._warmup_join_cache_key(
            ["customers", "orders"], "p1", hint_is_natural_language=False
        )
        entry = ("JX", ["sig"], {"J00": []})
        cache = {key: entry}
        id_to = {"p1": parent, "c1": child}
        assert SeedWarmupCacheSession._ambiguous_join_reuse_from_parent(child, cache, id_to) is entry

    def test_none_without_parent_link(self):
        intent = _warmup_intent(expansion_metadata=None)
        assert SeedWarmupCacheSession._ambiguous_join_reuse_from_parent(intent, {}, {}) is None

    def test_none_when_parent_tables_differ(self):
        parent = _warmup_intent(intent_id="p", tables=["orders"])
        child = _warmup_intent(
            intent_id="c",
            tables=["customers", "orders"],
            expansion_metadata=ExpansionMetadata(operator="x", parent_intent_id="p", depth=1),
        )
        assert SeedWarmupCacheSession._ambiguous_join_reuse_from_parent(child, {}, {"p": parent, "c": child}) is None


class TestAllocateStratumQuotas:
    """Tests for _allocate_stratum_quotas."""

    def test_empty_counts(self):
        assert SeedWarmupCacheSession._allocate_stratum_quotas({}, 5, 1) == {}

    def test_respects_budget_and_floor(self):
        counts = {"a": 10, "b": 10}
        q = SeedWarmupCacheSession._allocate_stratum_quotas(counts, budget=5, floor=2)
        assert sum(q.values()) == 5
        assert q["a"] >= 2 and q["b"] >= 2


class TestWarmupStratumKey:
    """Tests for _warmup_stratum_key."""

    def test_gold_vs_synthetic_prefix(self):
        g = _warmup_intent(source="gold", tables=["orders"])
        s = _warmup_intent(
            source="synthetic",
            tables=["orders"],
            expansion_metadata=ExpansionMetadata(operator="op", depth=2),
        )
        assert SeedWarmupCacheSession._warmup_stratum_key(g).startswith("gold|")
        assert SeedWarmupCacheSession._warmup_stratum_key(s).startswith("synthetic|")


class TestAcceptedTemplateInstanceKeys:
    """Tests for accepted_template_instance_keys."""

    def test_collects_keys(self):
        conc = ConcreteIntent(
            intent_id="ik",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        tmpl = Template(
            id="T0001",
            effective_structural_hash="h",
            intent_signature=conc,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT order_id FROM orders",
            sql_fp="sfp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        keys = SeedWarmupCacheSession.accepted_template_instance_keys({"T0001": tmpl})
        assert len(keys) == 1
        assert next(iter(keys))


class TestBuildValueDomains:
    """Tests for _build_value_domains."""

    def test_builds_keys_per_column(self, schema_graph):
        dom = SeedWarmupCacheSession._build_value_domains(schema_graph)
        assert "orders.order_id" in dom
        assert isinstance(dom["orders.order_id"], ValueDomain)


class TestWarmupSyntheticStorePathBlocks:
    """Tests for _warmup_synthetic_store_path_blocks."""

    def test_gold_intent_never_blocked(self, monkeypatch):
        intent = _warmup_intent(source="gold")
        rt = intent.to_runtime_intent()

        class CR:
            union_eligible = True
            union_sql_path = GenerationPath.UNION_TEMPLATE_WIDEN

        monkeypatch.setattr("aetherdialect._seed_warmup.structural_compare", lambda _r, _t: CR())
        assert SeedWarmupCacheSession._warmup_synthetic_store_path_blocks(intent, rt, {}) is None

    def test_path41_blocked_for_synthetic(self, monkeypatch):
        intent = _warmup_intent(source="synthetic")
        rt = intent.to_runtime_intent()

        class CR:
            union_eligible = True
            union_sql_path = GenerationPath.UNION_TEMPLATE_WIDEN

        monkeypatch.setattr("aetherdialect._seed_warmup.structural_compare", lambda _r, _t: CR())
        conc = ConcreteIntent(
            intent_id="ik",
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        tmpl = Template(
            id="T1",
            effective_structural_hash="h",
            intent_signature=conc,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT 1",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["nl"]),
            stats=TemplateStats(accept=1, reject=0),
            trust_level=1,
        )
        assert (
            SeedWarmupCacheSession._warmup_synthetic_store_path_blocks(intent, rt, {"T1": tmpl})
            == "warmup_path41_not_allowed"
        )


class TestQuestionFromSqlIntegration:
    """``generate_question_from_sql`` with LLM mocked (lives in utils, used by warmup)."""

    @patch("aetherdialect._utils.LLMProvider.json")
    def test_realistic_question_returned(self, mock_llm, schema_graph):
        from aetherdialect._utils import generate_question_from_sql

        mock_llm.return_value = {
            "question": "How many orders exist?",
            "is_realistic": True,
            "drop_reason": None,
        }
        result = generate_question_from_sql(
            "SELECT COUNT(*) FROM orders",
            schema_graph,
            ["orders"],
        )
        assert result is not None
        assert result["is_realistic"] is True
        assert "orders" in result["question"].lower()

    @patch("aetherdialect._utils.LLMProvider.json")
    def test_unrealistic_dropped(self, mock_llm, schema_graph):
        from aetherdialect._utils import generate_question_from_sql

        mock_llm.return_value = {
            "question": "",
            "is_realistic": False,
            "drop_reason": "meaningless cross join",
        }
        result = generate_question_from_sql(
            "SELECT * FROM orders, customers",
            schema_graph,
            ["orders", "customers"],
        )
        assert result is not None
        assert result["is_realistic"] is False
        assert "meaningless" in result["drop_reason"]

    @patch("aetherdialect._utils.LLMProvider.json", side_effect=Exception("timeout"))
    def test_llm_failure_propagates(self, mock_llm, schema_graph):
        from aetherdialect._utils import generate_question_from_sql

        with pytest.raises(Exception, match="timeout"):
            generate_question_from_sql(
                "SELECT 1",
                schema_graph,
                ["orders"],
            )


class TestBuildAnchorLatticeDiskMerge:
    """Tests for eager ``build_anchor_lattice`` loading cached anchors."""

    def test_disk_hit_collapses_multiple_rows_per_cell(self, schema_graph):
        """Multiple synthetic survivors sharing a lattice key reuse one disk entry."""
        a = _warmup_intent(intent_id="wa")
        b = _warmup_intent(intent_id="wb")
        ka = a.anchor_lattice_key()
        kb = b.anchor_lattice_key()
        assert ka == kb
        schema_graph.schema_graph_id = schema_graph.effective_structural_hash
        sig = ka.signature(schema_graph.schema_graph_id)
        disk = {sig: ["Anchor one", "Anchor two"]}
        lattice = SeedWarmupCacheSession.build_anchor_lattice([(a, "SELECT 1"), (b, "SELECT 2")], schema_graph, disk)
        assert len(lattice.cells) == 1
        cell = lattice.cells[ka]
        assert cell.anchors == ("Anchor one", "Anchor two")


class TestCheckIntentAgainstExpectation:
    """Tests for gold expectation matching."""

    def test_must_where_matches_param_values(self):
        intent = {
            "tables": ["reservation"],
            "grain": "scalar",
            "where": [
                {
                    "left_expr": "reservation.status",
                    "op": "=",
                    "param_key": "p1",
                }
            ],
            "param_values": {"p1": "expired"},
        }
        failures = SeedWarmupCacheSession.check_intent_against_expectation(
            intent,
            {"grain": "scalar", "must_tables": ["reservation"], "must_where": ["expired"]},
        )
        assert failures == []

    def test_allowed_grains_accepts_either_shape(self):
        intent = {"tables": ["book", "publisher"], "grain": "grouped", "limit": 1}
        failures = SeedWarmupCacheSession.check_intent_against_expectation(
            intent,
            {
                "allowed_grains": ["row_level", "grouped"],
                "must_tables": ["publisher", "book"],
                "limit": 1,
            },
        )
        assert failures == []

    def test_cte_resolved_tables_satisfy_must_tables(self):
        intent = {
            "tables": ["cte1"],
            "grain": "scalar",
            "cte_steps": [{"name": "cte1", "tables": ["inventory", "store"]}],
        }
        failures = SeedWarmupCacheSession.check_intent_against_expectation(
            intent,
            {"grain": "scalar", "must_tables": ["inventory", "store"]},
        )
        assert failures == []
