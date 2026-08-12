"""Tests for live_testing module: soft assertions, scenario construction, and assertion logic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import (
    ConcreteIntent,
    Expected,
    FeedbackMode,
    GenerationPath,
    QuestionRoute,
    QuestionValidationResult,
    RuntimeCteStep,
    RuntimeIntent,
    Scenario,
    SelectCol,
    SequenceScenario,
    SoftAssert,
    SoftFailure,
    SqlGenerationOutcome,
    StepResult,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import SQLShape, TemplateStats
from aetherdialect._live_testing import (
    _assert_scenario,
    _assertion_table_names,
    _build_reuse_intent,
    _run_pipeline_core,
    run_and_assert,
    run_sequence_and_assert,
)
from aetherdialect._utils import _make_input_responder, _make_prompt_responders


class TestSoftAssert:
    """Tests for SoftAssert accumulation and reporting."""

    def test_no_failures_passes(self):
        """Empty SoftAssert should report as passed."""
        sa = SoftAssert()
        assert sa.passed is True

    def test_check_true_does_not_record(self):
        """True condition should not record a failure."""
        sa = SoftAssert()
        sa.check(True, "field", "expected", "actual")
        assert sa.passed is True

    def test_check_false_records_failure(self):
        """False condition should record a failure."""
        sa = SoftAssert()
        sa.check(False, "row_count", 10, 5)
        assert sa.passed is False
        assert len(sa.failures) == 1

    def test_multiple_failures_accumulated(self):
        """Multiple failures should all be recorded."""
        sa = SoftAssert()
        sa.check(False, "a", 1, 2)
        sa.check(False, "b", 3, 4)
        sa.check(True, "c", 5, 5)
        assert len(sa.failures) == 2

    def test_report_no_failures_does_nothing(self):
        """Report with no failures should not raise."""
        sa = SoftAssert()
        sa.report("header")

    def test_report_with_failures_raises(self):
        """Report with failures should raise AssertionError."""
        sa = SoftAssert()
        sa.check(False, "x", 1, 2)
        with pytest.raises(AssertionError, match="x"):
            sa.report("Test failed")

    def test_report_includes_header(self):
        """Error message should include the header."""
        sa = SoftAssert()
        sa.check(False, "f", "a", "b")
        with pytest.raises(AssertionError, match="MY HEADER"):
            sa.report("MY HEADER")

    def test_failure_message_default(self):
        """Default message should include field, expected, actual."""
        sa = SoftAssert()
        sa.check(False, "count", 10, 5)
        assert "count" in sa.failures[0].message
        assert "10" in sa.failures[0].message

    def test_failure_custom_message(self):
        """Custom message should be used when provided."""
        sa = SoftAssert()
        sa.check(False, "f", 1, 2, "custom msg here")
        assert sa.failures[0].message == "custom msg here"


class TestSoftFailure:
    """Tests for SoftFailure dataclass."""

    def test_fields_stored(self):
        """All fields should be stored correctly."""
        f = SoftFailure(field="rows", expected=10, actual=5, message="too few")
        assert f.field == "rows"
        assert f.expected == 10
        assert f.actual == 5
        assert f.message == "too few"


class TestExpected:
    """Tests for Expected dataclass construction."""

    def test_defaults_are_none(self):
        """All optional fields should default to None or False."""
        e = Expected()
        assert e.tables is None
        assert e.min_rows is None
        assert e.max_rows is None
        assert e.reuse_type is None
        assert e.should_fail_validation is False
        assert e.max_llm_calls is None

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        e = Expected(
            tables=["orders"],
            min_rows=1,
            max_rows=100,
            contains_join=True,
        )
        assert e.tables == ["orders"]
        assert e.min_rows == 1
        assert e.max_rows == 100
        assert e.contains_join is True


class TestScenario:
    """Tests for Scenario dataclass construction."""

    def test_defaults(self):
        """Default auto_responses and feedback should be set."""
        s = Scenario(id="S-001", question="How many orders?", expected=Expected())
        assert s.feedback == "y"
        assert s.auto_responses is None
        assert s.reject_reason == "incorrect results"

    def test_custom_values(self):
        """Custom values should override defaults."""
        s = Scenario(
            id="S-002",
            question="Count customers",
            expected=Expected(min_rows=1),
            category="aggregation",
            auto_responses=["y", "n"],
            feedback="n",
            reject_reason="wrong tables",
        )
        assert s.category == "aggregation"
        assert s.feedback == "n"
        assert s.auto_responses == ["y", "n"]


class TestSequenceScenario:
    """Tests for SequenceScenario dataclass."""

    def test_stores_steps(self):
        """Steps should be stored as a list."""
        s1 = Scenario(id="S1", question="Q1", expected=Expected())
        s2 = Scenario(id="S2", question="Q2", expected=Expected())
        seq = SequenceScenario(id="SEQ-001", steps=[s1, s2])
        assert len(seq.steps) == 2
        assert seq.steps[0].id == "S1"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_defaults(self):
        """Default fields should be sensible."""
        r = StepResult(scenario_id="S1", question="Q?")
        assert r.status == "unknown"
        assert r.intent is None
        assert r.sql is None
        assert r.rows is None
        assert r.validation_failed is False

    def test_captured_logs_default_empty(self):
        """captured_logs should default to empty list."""
        r = StepResult(scenario_id="S1", question="Q?")
        assert r.captured_logs == []


class TestMakePromptResponders:
    """Tests for _make_prompt_responders."""

    def test_ask_user_choice_drains_queue(self):
        ask_uc, _iyn = _make_prompt_responders(["y", "n", "y"])
        assert ask_uc("Prompt?", ["y", "n"]) == "y"
        assert ask_uc("Prompt?", ["y", "n"]) == "n"
        assert ask_uc("Prompt?", ["y", "n"]) == "y"

    def test_ask_user_choice_defaults_to_y_when_empty(self):
        ask_uc, _iyn = _make_prompt_responders([])
        assert ask_uc("Prompt?", ["y", "n"]) == "y"

    def test_shared_queue_between_ask_and_interactive_yes_no(self):
        ask_uc, iyn = _make_prompt_responders(["y", "n"])
        assert ask_uc("P?", ["y", "n"]) == "y"
        assert iyn("stage", "P?", ["y", "n"]) == "n"
        assert iyn("stage", "P?", ["y", "n"]) == "y"

    def test_interactive_yes_no_with_choice_port_drains_same_queue(self):
        _ask_uc, iyn = _make_prompt_responders(["n"])
        assert iyn("s", "p", ["y", "n"], choice_port=object()) == "n"


class TestMakeInputResponder:
    """Tests for _make_input_responder."""

    def test_first_call_returns_reason(self):
        """First call should return the rejection reason."""
        responder = _make_input_responder("wrong data")
        assert responder("Why?") == "wrong data"

    def test_subsequent_calls_return_n(self):
        """Subsequent calls should return 'n'."""
        responder = _make_input_responder("bad")
        responder("First?")
        assert responder("Second?") == "n"
        assert responder("Third?") == "n"

    def test_default_reason(self):
        """Default reason should be 'incorrect results'."""
        responder = _make_input_responder()
        assert responder("") == "incorrect results"


class TestAssertScenario:
    """Tests for assert_scenario."""

    def _make_result(self, **overrides):
        """Build a StepResult with defaults."""
        defaults = dict(
            scenario_id="S1",
            question="test?",
            status="ok",
            sql="SELECT * FROM orders",
            rows=[(1,), (2,), (3,)],
            reuse_type="none",
            validation_failed=False,
        )
        defaults.update(overrides)
        r = StepResult(**defaults)
        return r

    def test_status_check(self):
        """Status assertion should pass when matching."""
        result = self._make_result(status="ok")
        expected = Expected(status="ok")
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_status_mismatch(self):
        """Status mismatch should record failure."""
        result = self._make_result(status="error")
        expected = Expected(status="ok")
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_status_in_pass(self):
        """status_in should pass when result status in allowed tuple."""
        result = self._make_result(status="validation_failed")
        expected = Expected(status_in=("ok", "validation_failed"))
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_status_in_fail(self):
        """status_in should fail when result status not in tuple."""
        result = self._make_result(status="error")
        expected = Expected(status_in=("ok", "validation_failed"))
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_min_rows_pass(self):
        """min_rows within range should pass."""
        result = self._make_result(rows=[(1,), (2,)])
        expected = Expected(min_rows=1)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_min_rows_fail(self):
        """Too few rows should record failure."""
        result = self._make_result(rows=[])
        expected = Expected(min_rows=1)
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_max_rows_pass(self):
        """max_rows within range should pass."""
        result = self._make_result(rows=[(1,)])
        expected = Expected(max_rows=5)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_max_rows_fail(self):
        """Too many rows should record failure."""
        result = self._make_result(rows=[(i,) for i in range(100)])
        expected = Expected(max_rows=5)
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_max_llm_calls_pass(self):
        """llm_calls at or below max_llm_calls should pass."""
        result = self._make_result(llm_calls=2)
        expected = Expected(max_llm_calls=5)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_max_llm_calls_fail(self):
        """llm_calls above max_llm_calls should fail."""
        result = self._make_result(llm_calls=4)
        expected = Expected(max_llm_calls=1)
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_contains_join_true(self):
        """SQL with JOIN should pass when expected."""
        result = self._make_result(sql="SELECT * FROM a JOIN b ON a.id = b.id")
        expected = Expected(contains_join=True)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_contains_join_false(self):
        """SQL without JOIN should pass when expected False."""
        result = self._make_result(sql="SELECT * FROM orders")
        expected = Expected(contains_join=False)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_contains_group_by(self):
        """SQL with GROUP BY should pass when expected."""
        result = self._make_result(sql="SELECT status, COUNT(*) FROM orders GROUP BY status")
        expected = Expected(contains_group_by=True)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_contains_cte(self):
        """SQL starting with WITH should pass cte check."""
        result = self._make_result(sql="WITH x AS (SELECT 1) SELECT * FROM x")
        expected = Expected(contains_cte=True)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_sql_contains(self):
        """sql_contains substrings should all be found."""
        result = self._make_result(sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id")
        expected = Expected(sql_contains=["customer_id", "SUM"])
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_sql_contains_one_of_first_group(self):
        result = self._make_result(sql="SELECT a, AVG(b) OVER (PARTITION BY c) FROM t")
        expected = Expected(sql_contains_one_of=[["OVER (", "PARTITION BY"], ["GROUP BY", "AVG("]])
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_sql_contains_one_of_second_group(self):
        result = self._make_result(sql="SELECT a, AVG(b) FROM t GROUP BY a, c")
        expected = Expected(sql_contains_one_of=[["OVER (", "PARTITION BY"], ["GROUP BY", "AVG("]])
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_sql_contains_one_of_fail(self):
        result = self._make_result(sql="SELECT a, b FROM t")
        expected = Expected(sql_contains_one_of=[["OVER (", "PARTITION BY"], ["GROUP BY", "AVG("]])
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_sql_excludes(self):
        """sql_excludes substrings should not be found."""
        result = self._make_result(sql="SELECT * FROM orders")
        expected = Expected(sql_excludes=["DELETE", "UPDATE"])
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_sql_excludes_fail(self):
        """Found excluded substring should fail."""
        result = self._make_result(sql="SELECT * FROM orders WHERE customer_id = 1")
        expected = Expected(sql_excludes=["customer_id"])
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_should_fail_validation(self):
        """Validation failure assertion should pass when validation_failed is True."""
        result = self._make_result(validation_failed=True)
        expected = Expected(should_fail_validation=True)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_grain_tuple_pass(self):
        """Grain as tuple should pass when result grain in tuple."""
        from aetherdialect._contracts_core import RuntimeIntent

        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = self._make_result(intent=intent)
        expected = Expected(grain=("row_level", "grouped"))
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_grain_tuple_fail(self):
        """Grain as tuple should fail when result grain not in tuple."""
        from aetherdialect._contracts_core import RuntimeIntent

        intent = RuntimeIntent(
            tables=["orders"],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = self._make_result(intent=intent)
        expected = Expected(grain=("row_level", "grouped"))
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_reuse_type_tuple_pass(self):
        """reuse_type as tuple should pass when result in tuple."""
        result = self._make_result(reuse_type="direct_reuse")
        expected = Expected(reuse_type=("direct_reuse", "intent_direct_reuse"))
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_reuse_type_check(self):
        """Reuse type assertion should match."""
        result = self._make_result(reuse_type="direct_reuse")
        expected = Expected(reuse_type="direct_reuse")
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_column_names_pass(self):
        """Column names assertion should pass when matching."""
        from aetherdialect._contracts_base import NormalizedExpr
        from aetherdialect._contracts_core import (
            RuntimeIntent,
            SelectCol,
        )

        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("title")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = self._make_result(intent=intent, rows=[("Alien",)])
        expected = Expected(column_names_one_of=[["title"]])
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_column_names_fail(self):
        """Column names mismatch should record failure."""
        from aetherdialect._contracts_base import NormalizedExpr
        from aetherdialect._contracts_core import (
            RuntimeIntent,
            SelectCol,
        )

        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film_id")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = self._make_result(intent=intent, rows=[(1,)])
        expected = Expected(column_names_one_of=[["title"]])
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_column_names_one_of_multi_option_pass(self):
        """When multiple column sets allowed, matching one passes."""
        from aetherdialect._contracts_base import NormalizedExpr
        from aetherdialect._contracts_core import (
            RuntimeIntent,
            SelectCol,
        )

        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("first_name")),
                SelectCol(expr=NormalizedExpr.from_column("last_name")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = self._make_result(intent=intent, rows=[("a", "b")])
        expected = Expected(
            column_names_one_of=[
                ["customer_id", "first_name", "last_name"],
                ["first_name", "last_name"],
            ]
        )
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_row_value_check_pass(self):
        """Custom row value check should pass when returning True."""
        result = self._make_result(rows=[(1, "alice"), (2, "bob")])
        expected = Expected(row_value_check=lambda rows: len(rows) == 2)
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_row_value_check_fail(self):
        """Custom row value check should fail when returning False."""
        result = self._make_result(rows=[(1,)])
        expected = Expected(row_value_check=lambda rows: len(rows) > 5)
        soft = _assert_scenario(result, expected)
        assert not soft.passed

    def test_none_fields_skipped(self):
        """None expected fields should be silently skipped."""
        result = self._make_result()
        expected = Expected()
        soft = _assert_scenario(result, expected)
        assert soft.passed

    def test_existing_soft_reused(self):
        """Passing existing SoftAssert should append to it."""
        sa = SoftAssert()
        sa.check(False, "pre", 1, 2)
        result = self._make_result(status="ok")
        expected = Expected(status="ok")
        returned = _assert_scenario(result, expected, soft=sa)
        assert returned is sa
        assert len(returned.failures) == 1


class TestBuildReuseIntent:
    """Tests for _build_reuse_intent."""

    def test_builds_from_template(self):
        """Should construct RuntimeIntent from template intent_signature."""
        sig = SimpleNamespace(
            tables=["orders", "customers"],
            grain="grouped",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            having=None,
            column_map={},
        )
        tmpl = SimpleNamespace(intent_signature=sig)
        intent = _build_reuse_intent(tmpl)
        assert intent.tables == ["orders", "customers"]
        assert intent.grain == "grouped"

    def test_handles_none_fields(self):
        """None fields in signature should default to empty."""
        sig = SimpleNamespace(
            tables=None,
            grain=None,
            select_cols=None,
            group_by_cols=None,
            order_by_cols=None,
            where=None,
            having=None,
            column_map=None,
        )
        tmpl = SimpleNamespace(intent_signature=sig)
        intent = _build_reuse_intent(tmpl)
        assert intent.tables == []
        assert intent.grain == "row_level"


def _make_passing_result() -> StepResult:
    """Build a StepResult that passes basic assertions."""
    return StepResult(
        scenario_id="T1",
        question="test?",
        status="ok",
        sql="SELECT 1",
        rows=[(1,)],
    )


def _make_failing_result() -> StepResult:
    """Build a StepResult with zero rows to fail min_rows checks."""
    return StepResult(
        scenario_id="T1",
        question="test?",
        status="ok",
        sql="SELECT 1",
        rows=[],
    )


class _StepDeferredRunner:
    """Single-attempt runner returning scripted ``StepResult`` values from ``run_deferred``."""

    def __init__(self, step_results: list[StepResult]) -> None:
        self._step_results = list(step_results)
        self.run_deferred_calls: list[tuple[Scenario, int]] = []

    def run_deferred(self, scenario: Scenario, retries: int = 0) -> StepResult:
        self.run_deferred_calls.append((scenario, retries))
        idx = len(self.run_deferred_calls) - 1
        if idx >= len(self._step_results):
            return self._step_results[-1]
        return self._step_results[idx]


class _DeferredRetryRunner:
    """Root runner with ``clone`` / ``adopt_state_from`` matching ``run_and_assert`` expectations."""

    def __init__(self, attempts: list[list[StepResult]]) -> None:
        self._attempts = list(attempts)
        self.clone_count = 0
        self.last_attempt: _StepDeferredRunner | None = None

    def clone(self) -> _StepDeferredRunner:
        if self.clone_count >= len(self._attempts):
            seq = self._attempts[-1]
        else:
            seq = self._attempts[self.clone_count]
        self.clone_count += 1
        self.last_attempt = _StepDeferredRunner(seq)
        return self.last_attempt

    def adopt_state_from(self, _other: object) -> None:
        return None


class TestRunAndAssert:
    """Tests for the run_and_assert retry wrapper."""

    def test_passes_on_first_attempt(self):
        """No retry needed when assertions pass."""
        runner = _DeferredRetryRunner([[_make_passing_result()]])
        scenario = Scenario(id="T1", question="test?", expected=Expected(min_rows=1))
        run_and_assert(runner, scenario, header="[T1]")
        assert runner.clone_count == 1
        assert runner.last_attempt is not None
        assert len(runner.last_attempt.run_deferred_calls) == 1

    def test_retries_on_failure_then_passes(self):
        """Retry from scratch when first attempt fails assertions."""
        runner = _DeferredRetryRunner([[_make_failing_result()], [_make_passing_result()]])
        scenario = Scenario(id="T1", question="test?", expected=Expected(min_rows=1))
        run_and_assert(runner, scenario, header="[T1]", max_attempts=2)
        assert runner.clone_count == 2

    def test_raises_after_all_attempts_exhausted(self):
        """AssertionError raised when all retries fail."""
        runner = _DeferredRetryRunner([[_make_failing_result()], [_make_failing_result()]])
        scenario = Scenario(id="T1", question="test?", expected=Expected(min_rows=5))
        with pytest.raises(AssertionError, match="T1"):
            run_and_assert(runner, scenario, header="[T1]", max_attempts=2)
        assert runner.clone_count == 2

    def test_passes_retries_to_runner(self):
        """Per-attempt pipeline retries forwarded to clone.run_deferred."""
        runner = _DeferredRetryRunner([[_make_passing_result()]])
        scenario = Scenario(id="T1", question="test?", expected=Expected())
        run_and_assert(runner, scenario, header="[T1]", retries=3)
        assert runner.last_attempt is not None
        assert runner.last_attempt.run_deferred_calls == [(scenario, 3)]


class TestRunSequenceAndAssert:
    """Tests for the run_sequence_and_assert retry wrapper."""

    def test_passes_on_first_attempt(self):
        """No retry needed when all steps pass."""
        runner = _DeferredRetryRunner(
            [
                [
                    _make_passing_result(),
                ],
            ]
        )
        s1 = Scenario(id="S1", question="Q1", expected=Expected(min_rows=1))
        s2 = Scenario(id="S2", question="Q2", expected=Expected(min_rows=1))
        seq = SequenceScenario(id="SEQ", steps=[s1, s2])
        run_sequence_and_assert(runner, seq)
        assert runner.clone_count == 1
        assert runner.last_attempt is not None
        assert len(runner.last_attempt.run_deferred_calls) == 2

    def test_retries_entire_sequence_on_failure(self):
        """Full sequence re-runs when any step fails."""
        runner = _DeferredRetryRunner(
            [
                [_make_failing_result()],
                [
                    _make_passing_result(),
                ],
            ]
        )
        s1 = Scenario(id="S1", question="Q1", expected=Expected(min_rows=1))
        s2 = Scenario(id="S2", question="Q2", expected=Expected(min_rows=1))
        seq = SequenceScenario(id="SEQ", steps=[s1, s2])
        run_sequence_and_assert(runner, seq, max_attempts=2)
        assert runner.clone_count == 2

    def test_raises_after_all_attempts_exhausted(self):
        """AssertionError raised when all retries fail."""
        runner = _DeferredRetryRunner([[_make_failing_result()], [_make_failing_result()]])
        s1 = Scenario(id="S1", question="Q1", expected=Expected(min_rows=5))
        seq = SequenceScenario(id="SEQ", steps=[s1])
        with pytest.raises(AssertionError, match="SEQ"):
            run_sequence_and_assert(runner, seq, max_attempts=2)
        assert runner.clone_count == 2


class TestAssertScenarioPipelineError:
    """Tests for failing fast on uncaught pipeline errors."""

    def test_error_status_records_pipeline_error_when_not_expected(self):
        """Unexpected ``status=error`` should add a ``pipeline_error`` failure."""
        result = StepResult(
            scenario_id="X",
            question="q",
            status="error",
            error="TypeError: bad kwarg",
        )
        soft = _assert_scenario(result, Expected(min_rows=1))
        assert soft.passed is False
        assert any(f.field == "pipeline_error" for f in soft.failures)

    def test_error_status_skipped_when_status_in_allows_error(self):
        """When ``status_in`` includes ``error``, do not add ``pipeline_error``."""
        result = StepResult(
            scenario_id="X",
            question="q",
            status="error",
            error="boom",
        )
        soft = _assert_scenario(result, Expected(status_in=("ok", "error")))
        assert not any(f.field == "pipeline_error" for f in soft.failures)


class TestAssertScenarioInternalLogsAndImplicitStatus:
    """``Expected`` tightens status when row/SQL/column checks are set; internal log tokens fail ok runs."""

    def test_min_rows_implies_ok_status(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = StepResult(
            scenario_id="s",
            question="q",
            status="validation_failed",
            intent=intent,
            rows=[(1,)],
        )
        soft = _assert_scenario(result, Expected(min_rows=1))
        assert soft.passed is False
        assert any(f.field == "status" for f in soft.failures)

    def test_ok_status_rejects_enforce_select_only_failed_log(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        result = StepResult(
            scenario_id="s",
            question="q",
            status="ok",
            intent=intent,
            sql="SELECT 1",
            rows=[(1,)],
            captured_logs=["enforce_select_only FAILED: not_select"],
        )
        soft = _assert_scenario(result, Expected(min_rows=1))
        assert soft.passed is False
        assert any(f.field == "internal_failure_logs" for f in soft.failures)

    def test_live_testing_ops_run_and_assert_surfaces_internal_log_failure(self):
        class _Runner:
            def clone(self):
                return self

            def adopt_state_from(self, _other) -> None:
                return None

            def run_deferred(self, _scenario, retries: int = 1):
                intent = RuntimeIntent(
                    tables=["t"],
                    grain="row_level",
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.id"))],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                )
                return StepResult(
                    scenario_id="s",
                    question="q",
                    status="ok",
                    intent=intent,
                    sql="SELECT 1",
                    rows=[(1,)],
                    captured_logs=["enforce_select_only FAILED: x"],
                )

        scenario = Scenario(id="s", question="q", expected=Expected(min_rows=1))
        with pytest.raises(AssertionError, match="internal_failure_logs|logged failures"):
            run_and_assert(_Runner(), scenario, header="[meta]")


class TestRunPipelineCoreUnionPreview:
    """``_run_pipeline_core`` uses ``match_template_for_union`` before intent confirm (aligned with interactive)."""

    def test_confirm_intent_receives_union_flags_from_match_template_for_union(self):
        sc1 = SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))
        sc2 = SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))
        conc = ConcreteIntent(
            intent_id="tmpl",
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1, sc2],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            chosen_join_candidate_id="J00",
            chosen_join_path_signature=[],
        )
        tmpl = Template(
            id="TAlign",
            effective_structural_hash="h",
            intent_signature=conc,
            intent_key="ik",
            tables_used=["orders"],
            sql_param="SELECT 1 FROM orders",
            sql_fp="fp",
            shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
            colmap_sig="c",
            value_history=ValueHistory(param_values=[{}], questions=[], natural_language=[]),
            stats=TemplateStats(),
            trust_level=1,
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[sc1],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            natural_language="list order id and customer id",
        )
        templates = {"TAlign": tmpl}
        store: dict = {"next_id": 1, "templates": templates, "rejected_templates": {}}
        schema = MagicMock(schema_hash="sh")
        dialect = MagicMock()
        dialect.finalize_render.return_value = "EXEC"
        dialect.execute.return_value = []

        captured: dict = {}

        def capture_confirm(*args, **kwargs):
            captured.update(kwargs)
            return True

        gen_out = SqlGenerationOutcome(
            "SELECT 1",
            True,
            GenerationPath.UNION_TEMPLATE_WIDEN,
            tmpl,
            (),
        )

        def prep(*a, **k):
            return (
                tmpl,
                [sc1, sc2],
                True,
                GenerationPath.UNION_TEMPLATE_WIDEN,
                True,
                {"candidates": []},
                {"J00": []},
                {},
            )

        no_reuse = MagicMock(
            reuse_type="none",
            intent=None,
            reuse_candidate_normalized=None,
            best_template=None,
            similarity_score=0.0,
        )

        with (
            patch(
                "aetherdialect._live_testing.load_pipeline_resources",
                return_value=(dialect, schema, store, templates, {}, set()),
            ),
            patch(
                "aetherdialect._live_testing.handle_direct_sql_reuse",
                return_value=None,
            ),
            patch(
                "aetherdialect._live_testing.match_question_level_template_reuse",
                return_value=no_reuse,
            ),
            patch(
                "aetherdialect._live_testing.validate_question",
                return_value=QuestionValidationResult(accepted=True, route=QuestionRoute.ANALYTICAL, corrected="q"),
            ),
            patch(
                "aetherdialect._live_testing.normalize_question_via_llm",
                side_effect=lambda corrected, raw_original=None: corrected,
            ),
            patch(
                "aetherdialect._live_testing.parse_intent_via_llm",
                return_value=(intent, [], 1, None),
            ),
            patch(
                "aetherdialect._live_testing.match_template_for_union",
                return_value=(
                    tmpl,
                    [sc1, sc2],
                    True,
                    GenerationPath.UNION_TEMPLATE_WIDEN,
                ),
            ) as mock_mtf,
            patch(
                "aetherdialect._live_testing.collect_structural_match_templates",
                return_value=[],
            ),
            patch(
                "aetherdialect._live_testing.confirm_intent_with_user",
                side_effect=capture_confirm,
            ),
            patch(
                "aetherdialect._live_testing.prepare_union_match_join_phase",
                side_effect=prep,
            ),
            patch(
                "aetherdialect._live_testing.generate_and_validate_sql",
                return_value=gen_out,
            ),
            patch(
                "aetherdialect._live_testing.build_result_dataframe",
                return_value=None,
            ),
            patch("aetherdialect._live_testing.stamp_sql_shape"),
            patch(
                "aetherdialect._live_testing.display_final_results_to_stdout",
                return_value="ux",
            ),
            patch("aetherdialect._live_testing.handle_user_feedback"),
            patch("aetherdialect._live_testing.save_result_csv"),
        ):
            _run_pipeline_core(
                "list order id and customer id for orders",
                schema,
                store,
                templates,
                {},
                set(),
                "y",
                [],
                feedback_mode=FeedbackMode.DEFERRED_TEST,
            )

        mock_mtf.assert_called_once()
        assert captured.get("has_union_match") is True
        assert captured.get("cols_changed") is True


class TestAssertionTableNames:
    """CTE table assertions exclude normalized CTE alias names."""

    def test_excludes_cte_aliases_and_keeps_base_tables(self):
        intent = RuntimeIntent(
            tables=["cte2"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte2.line_count"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
            cte_steps=[
                RuntimeCteStep(
                    cte_name="cte1",
                    tables=["tbl_a", "tbl_b"],
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column("tbl_a.id"))],
                ),
                RuntimeCteStep(
                    cte_name="cte2",
                    tables=["cte1"],
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.id"))],
                ),
            ],
            interpret_cte_names=["cte1", "cte2"],
        )
        sql = (
            'WITH cte1 AS (SELECT "tbl_a"."id" FROM "tbl_a" INNER JOIN "tbl_b" ON "tbl_a"."id" = "tbl_b"."id") '
            'SELECT "cte1"."id" FROM cte1'
        )
        names = _assertion_table_names(intent, sql)
        assert "cte1" not in names
        assert "cte2" not in names
        assert "tbl_a" in names
        assert "tbl_b" in names

    def test_without_cte_steps_uses_intent_tables(self):
        intent = RuntimeIntent(
            tables=["tbl_a", "tbl_b"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("tbl_a.id"))],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        assert _assertion_table_names(intent, None) == ["tbl_a", "tbl_b"]


def test_run_pipeline_core_refuses_federated_composite_schema() -> None:
    from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
    from aetherdialect._schema_graph import recompute_join_paths_multi
    from aetherdialect._templates_ops import TemplateOps

    composite = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi=recompute_join_paths_multi({}),
        schema_graph_id="sg_fed",
    )
    store = TemplateOps.empty_template_store("sg_fed")
    result = _run_pipeline_core(
        question="join left and right",
        schema=composite,
        store=store,
        templates={},
        rejected={},
        schema_terms=set(),
        feedback="y",
        captured_logs=[],
        feedback_mode=FeedbackMode.DEFERRED_TEST,
    )
    assert result.status == "error"
    assert result.error is not None
    assert "LiveTestRunner does not support federated composite schemas" in result.error
