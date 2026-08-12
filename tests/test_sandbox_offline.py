"""Offline mock sandbox integration tests."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from aetherdialect import AetherEngine, EngineContext, Sandbox
from aetherdialect._config import DuckDBRuntimeConfig
from aetherdialect._constants import SESSION_KIND_AWAITING_SQL_CONFIRM
from aetherdialect._constants_runtime import (
    SANDBOX_MEMBER_SPACE_TABLES,
    SANDBOX_TOUR_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL,
    SANDBOX_VALIDATION_FAILURE_QUESTIONS,
)
from aetherdialect._contracts_base import OwnerOnlyOperationError, SandboxBuildSection, SessionActiveError
from aetherdialect._llm_provider import MockProvider
from aetherdialect._sql_gen import join_hints_multi
from tests._sandbox_structure import apply_demo_schema_structure_from_bundle

_sandbox_build_section = Sandbox._sandbox_build_section
assert_sandbox_complete = Sandbox.assert_sandbox_complete
data_zip_path = Sandbox.data_zip_path
fixtures_corpus_text = Sandbox.fixtures_corpus_text
sandbox_doctor = Sandbox.sandbox_doctor
sandbox_feedback_demo = Sandbox._sandbox_feedback_demo
sandbox_questions = Sandbox._sandbox_questions

_NARROW_ALLOW_OBJECTS = frozenset({"customer", "payment", "rental", "address", "city", "country"})

_TOUR_EXPECT_NO_SQL = SANDBOX_TOUR_EXPECT_NO_SQL
_VALIDATION_FAILURE_QUESTIONS = SANDBOX_VALIDATION_FAILURE_QUESTIONS
_VALIDATION_FAILURE_EXPECT_NO_SQL = SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL
_ZIP_REQUIRED_MEMBERS = (
    "rental_shop.sql",
    "rental_shop_views.sql",
    "rental_shop_notes.txt",
    "questions.txt",
    "fixtures/rental_shop_mock.json",
    "artifacts_baseline/owner/schema_graph.json.gz",
    "schema_literals.json",
    "schema_structure_demo.json",
    "sandbox_catalog.json",
    "sandbox_expectations.json",
    "sandbox_scenarios.json",
    "sandbox_handcrafted_fixtures.json",
    "federation_storefront_notes.txt",
    "federation_catalog_notes.txt",
    "federation_logistics_notes.txt",
    "federation_crm_notes.txt",
    "migration_demo/schema_migration_map.json",
)


@pytest.fixture(autouse=True)
def _reset_mock() -> None:
    MockProvider.reset_mock_provider()
    yield
    MockProvider.reset_mock_provider()


@pytest.fixture(scope="module")
def _fixture_corpus() -> str:
    raw = Sandbox.fixtures_corpus_text()
    assert '"fixtures": []' not in raw and '"fixtures":[]' not in raw, "fixture corpus is empty"
    return raw


@pytest.fixture(scope="module")
def owner_sandbox() -> Iterator[object]:
    """Module-scoped owner sandbox for expensive smoke paths."""
    sb = Sandbox.create_offline_sandbox(AetherEngine)
    yield sb
    sb.close()


duckdb = pytest.importorskip("duckdb")


def _zip_member_set() -> set[str]:
    with zipfile.ZipFile(Sandbox.data_zip_path()) as zf:
        return set(zf.namelist())


def _zip_members_without_locks() -> set[str]:
    return {n for n in _zip_member_set() if not n.endswith(".lock") and ".__write.lock" not in n}


def _sandbox_practice_questions() -> list[str]:
    if not Sandbox.data_zip_path().is_file():
        return []
    return sandbox_questions()


def _sandbox_build_questions(section: str) -> list[str]:
    if not Sandbox.data_zip_path().is_file():
        return []
    return Sandbox._sandbox_build_section(cast(SandboxBuildSection, section))


_QUESTION_SOURCE_BY_TEST: dict[str, str] = {
    "test_sandbox_question": "practice",
    "test_validation_failure_question": "validation_failures",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "feedback" in metafunc.fixturenames:
        if not Sandbox.data_zip_path().is_file():
            metafunc.parametrize("feedback", [])
            return
        demo = sandbox_feedback_demo()
        rejection = str(demo.get("allowed_rejection_text", "")).strip()
        metafunc.parametrize("feedback", [rejection] if rejection else [])
        return
    if "question" in metafunc.fixturenames:
        source = _QUESTION_SOURCE_BY_TEST.get(metafunc.function.__name__)
        if source == "practice":
            metafunc.parametrize("question", _sandbox_practice_questions())
        elif source is not None:
            metafunc.parametrize("question", _sandbox_build_questions(source))


class TestSandboxBundle:
    def test_sandbox_doctor_passes(self) -> None:
        assert Sandbox.sandbox_doctor() == []

    def test_data_zip_required_members(self) -> None:
        names = _zip_member_set()
        for leaf in _ZIP_REQUIRED_MEMBERS:
            assert any(n.endswith(leaf) or n == leaf for n in names), f"missing zip member {leaf!r}"
        assert any("migration_demo/artifacts_v1" in n for n in names)

    def test_data_zip_is_canonical_source(self) -> None:
        """Bundle invariants: zip is the only shipped source (no sandbox/data/ on disk)."""
        data_dir = Sandbox.data_zip_path().parent / "data"
        assert not data_dir.is_dir(), "sandbox/data/ must not exist; use data.zip only"
        names = _zip_members_without_locks()
        assert len(names) >= len(_ZIP_REQUIRED_MEMBERS)
        for leaf in _ZIP_REQUIRED_MEMBERS:
            assert any(n.endswith(leaf) or n == leaf for n in names), f"missing zip member {leaf!r}"
        assert any("migration_demo/artifacts_v1" in n for n in names)

    def test_fixture_corpus_populated(self, _fixture_corpus: str) -> None:
        data = json.loads(_fixture_corpus)
        assert len(data.get("fixtures", [])) >= 100

    def test_baseline_schema_graph_in_zip_has_category_pk(self) -> None:
        with zipfile.ZipFile(Sandbox.data_zip_path()) as zf:
            graph_name = next(n for n in zf.namelist() if n.endswith("artifacts_baseline/owner/schema_graph.json.gz"))
            payload = json.loads(gzip.decompress(zf.read(graph_name)))
        category = payload.get("tables", {}).get("category", {})
        assert category.get("primary_key") == ["category_id"]


class TestSandboxHandle:
    def test_sandbox_engine_isolated_from_live_duckdb_globals(self) -> None:
        """Sandbox must not attach to or reuse live-test DuckDB globals."""
        live_conn = duckdb.connect(":memory:")
        live_conn.execute("CREATE TABLE live_marker (id INTEGER)")
        live_path = "scripts/duckdb/rental_shop.duckdb"
        orig_connection = DuckDBRuntimeConfig.NATIVE_CONNECTION
        orig_path = DuckDBRuntimeConfig.DATABASE_PATH
        try:
            DuckDBRuntimeConfig.NATIVE_CONNECTION = live_conn
            DuckDBRuntimeConfig.DATABASE_PATH = live_path
            os.environ["DUCKDB_PATH"] = live_path
            with Sandbox.create_offline_sandbox(AetherEngine) as sb:
                sandbox_conn = sb.engine._native_connection
                assert sandbox_conn is not live_conn
                assert DuckDBRuntimeConfig.NATIVE_CONNECTION is live_conn
                tables = {str(row[0]).lower() for row in sandbox_conn.execute("SHOW TABLES").fetchall()}
                assert "film" in tables
                assert "live_marker" not in tables
                assert DuckDBRuntimeConfig.DATABASE_PATH == ":memory:"
                ctx = sb.engine._runtime_config.engine_context
                assert ctx.sql_file
                assert "aetherdialect_sandbox_extract" in str(ctx.sql_file).replace("\\", "/")
            assert DuckDBRuntimeConfig.NATIVE_CONNECTION is live_conn
            assert DuckDBRuntimeConfig.DATABASE_PATH == live_path
            assert live_conn.execute("SELECT COUNT(*) FROM live_marker").fetchone()[0] == 0
        finally:
            DuckDBRuntimeConfig.NATIVE_CONNECTION = orig_connection
            DuckDBRuntimeConfig.DATABASE_PATH = orig_path
            os.environ.pop("DUCKDB_PATH", None)
            live_conn.close()

    def test_offline_sandbox_smoke_init(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            assert sb.engine.dialect == "duckdb"
            assert sb.engine._sandbox_mode is True
            meta = sb.engine.export_structure()
            assert int(meta.get("table_count", 0)) == 34

    def test_sandbox_schema_init_uses_baseline_cache(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            col = sb.engine._schema_graph.get_column("item_feature", "feature_name")
            assert col is not None
            assert (col.data_type or "").lower().startswith("varchar") or (col.data_type or "").lower() == "text"
            meta = sb.engine.export_structure()
            assert int(meta.get("table_count", 0)) >= 34

    def test_handle_close_removes_owned_artifacts(self) -> None:
        sb = Sandbox.create_offline_sandbox(AetherEngine)
        artifacts = Path(sb.artifacts_dir)
        assert artifacts.is_dir()
        sb.close()
        assert not artifacts.exists()
        with pytest.raises(RuntimeError, match="closed"):
            with sb.engine.session():
                pass

    def test_fresh_sandbox_after_close(self) -> None:
        sb1 = Sandbox.create_offline_sandbox(AetherEngine)
        sb1.close()
        with Sandbox.create_offline_sandbox(AetherEngine) as sb2:
            assert Path(sb2.artifacts_dir) != Path(sb1.artifacts_dir)

    def test_context_manager_closes_handle(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            artifacts = Path(sb.artifacts_dir)
            assert artifacts.is_dir()
        assert not artifacts.exists()

    def test_cleanup_artifacts_false_preserves_dir(self) -> None:
        sb = Sandbox.create_offline_sandbox(AetherEngine, cleanup_artifacts=False)
        artifacts = Path(sb.artifacts_dir)
        sb.close()
        assert artifacts.is_dir()
        import shutil

        shutil.rmtree(artifacts, ignore_errors=True)

    def test_shared_artifacts_dir_refcount(self) -> None:
        shared = tempfile.mkdtemp(prefix="sandbox_refcount_")
        sb1 = Sandbox.create_offline_sandbox(AetherEngine, artifacts_dir=shared, cleanup_artifacts=False)
        sb2 = Sandbox.create_offline_sandbox(AetherEngine, artifacts_dir=shared, cleanup_artifacts=False)
        sb1.close()
        assert Path(shared).is_dir()
        sb2.close()
        assert Path(shared).is_dir()
        import shutil

        shutil.rmtree(shared, ignore_errors=True)

    def test_register_cwd_sidecar_cleanup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        sb = Sandbox.create_offline_sandbox(AetherEngine)
        marker = tmp_path / "sandbox_transient.json"
        marker.write_text("{}", encoding="utf-8")
        sb.register_cwd_sidecar(marker)
        apply_demo_schema_structure_from_bundle(sb.engine, handle=sb)
        source = tmp_path / "schema_structure.json"
        applied = tmp_path / "schema_overrides.applied.json"
        assert applied.is_file()
        assert not source.is_file()
        sb.close()
        assert not marker.exists()
        assert applied.is_file()

    @pytest.mark.not_fast
    def test_consumer_reader_preset_first_question(self, _fixture_corpus: str) -> None:
        practice = sandbox_questions()
        assert practice
        with Sandbox() as sandbox:
            engine = sandbox.engine(role="consumer")
            with engine.session(mode="reader") as session:
                step = session.accept_until_done(practice[0])
        assert step.done
        assert step.sql

    def test_custom_seed_sql(self) -> None:
        custom = tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False, encoding="utf-8")
        try:
            custom.write(
                "\n".join(
                    (
                        "CREATE TABLE probe (id INTEGER);",
                        "INSERT INTO probe VALUES (42);",
                    ),
                ),
            )
            custom.close()
            with Sandbox.create_offline_sandbox(AetherEngine, seed_sql=custom.name, maintainer_access=True) as sb:
                row = sb.engine._native_connection.execute("SELECT id FROM probe").fetchone()
                assert row is not None and int(row[0]) == 42
        finally:
            Path(custom.name).unlink(missing_ok=True)


class TestSandboxQuestions:
    def test_sandbox_questions_returns_practice_list(self) -> None:
        practice = sandbox_questions()
        assert len(practice) >= 40
        assert "How many books do we have?" in practice
        assert "How many rentals happened in 2025?" in practice

    @pytest.mark.not_fast
    def test_assert_sandbox_complete_with_corpus(self, _fixture_corpus: str) -> None:
        Sandbox.assert_sandbox_complete(AetherEngine)

    @pytest.mark.not_fast
    def test_fixture_corpus_covers_consumer_reader_practice(self, _fixture_corpus: str) -> None:
        practice = sandbox_questions()
        with Sandbox() as sandbox:
            engine = sandbox.engine(role="consumer")
            with engine.session(mode="reader") as session:
                for question in practice:
                    step = session.accept_until_done(question)
                    assert step.done, f"consumer reader failed for {question!r}: {step.error}"
                    if step.error and step.error.detail_code and "no mock fixture" in step.error.detail_code.lower():
                        pytest.fail(f"consumer reader missing fixture for {question!r}")
                    if question not in _TOUR_EXPECT_NO_SQL:
                        assert step.sql or step.error, f"consumer reader empty result for {question!r}"
                    session.reset()

    @pytest.mark.not_fast
    def test_feedback_samples_rejection_flow(self, feedback: str) -> None:
        demo = sandbox_feedback_demo()
        anchor = str(demo.get("anchor_question", "")).strip()
        assert anchor
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.ask(anchor)
                if not step.done and step.reply_shape == "yes_no":
                    step = session.step("n")
                if not step.done and step.reply_shape == "free_text":
                    step = session.step(feedback)
                while not step.done and step.reply_shape == "yes_no":
                    step = session.step("y")
        assert step.done

    @pytest.mark.not_fast
    def test_rank_films_question(self, _fixture_corpus: str) -> None:
        practice = sandbox_questions()
        rank_q = next((q for q in practice if "rank" in q.lower() and "category" in q.lower()), None)
        assert rank_q is not None, "rank-films question missing from corpus"
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.accept_until_done(rank_q)
        assert step.done
        assert step.sql
        err = step.error or ""
        assert "no mock fixture" not in err.lower(), err

    @pytest.mark.not_fast
    def test_practice_questions_have_mock_fixture_coverage(self, _fixture_corpus: str) -> None:
        """Every practice question must complete the owner sandbox accept path without missing mock fixtures."""
        practice = sandbox_questions()
        assert practice
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                for question in practice:
                    step = session.accept_until_done(question)
                    assert step.done, question
                    err = step.error or ""
                    assert "no mock fixture" not in err.lower(), f"{question!r}: {err}"
                    session.reset()

    @pytest.mark.not_fast
    def test_sandbox_question(self, question: str) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.accept_until_done(question)
        assert step.done
        if question in _TOUR_EXPECT_NO_SQL or step.error is not None and step.error.code.value == "not_a_question":
            assert step.sql is None or step.error
        else:
            assert step.sql

    @pytest.mark.not_fast
    def test_validation_failure_question(self, question: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert question in _VALIDATION_FAILURE_QUESTIONS
        if question == "Show payroll deductions by employee SSN.":
            monkeypatch.chdir(tmp_path)
            with Sandbox.create_offline_sandbox(AetherEngine) as sb:
                apply_demo_schema_structure_from_bundle(sb.engine, handle=sb)
                with sb.engine.session() as session:
                    step = session.accept_until_done(question)
        elif question == "Show me all staff salaries.":
            with Sandbox() as sandbox:
                engine = sandbox.engine(
                    EngineContext(allow_objects=_NARROW_ALLOW_OBJECTS),
                    role="consumer",
                )
                with engine.session(mode="reader") as session:
                    step = session.accept_until_done(question)
        else:
            with Sandbox.create_offline_sandbox(AetherEngine) as sb:
                with sb.engine.session() as session:
                    step = session.accept_until_done(question)
        assert step.done
        if question in _VALIDATION_FAILURE_EXPECT_NO_SQL:
            assert step.sql is None or step.error

    @pytest.mark.not_fast
    def test_direct_reuse_2025_2026_pair(self, _fixture_corpus: str) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                session.accept_until_done("How many rentals happened in 2025?")
                step = session.ask("How many rentals happened in 2026?")
                assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM
                while not step.done:
                    if step.reply_shape == "yes_no":
                        step = session.step("y")
                    elif step.reply_shape == "free_text":
                        step = session.step("ok")
                    else:
                        break
        assert step.done
        assert step.sql


class TestSandboxSessionWorkflows:
    @pytest.mark.not_fast
    def test_recipe_chat_basics_produces_sql(self) -> None:
        practice = sandbox_questions()
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.accept_until_done(practice[0])
        assert step.sql

    @pytest.mark.not_fast
    def test_recipe_rejections_completes(self) -> None:
        demo = sandbox_feedback_demo()
        anchor = str(demo.get("anchor_question", "")).strip()
        rejection = str(demo.get("allowed_rejection_text", "")).strip()
        assert anchor and rejection
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.ask(anchor)
                if not step.done and step.reply_shape == "yes_no":
                    step = session.step("n")
                if not step.done and step.reply_shape == "free_text":
                    step = session.step(rejection)
                while not step.done and step.reply_shape == "yes_no":
                    step = session.step("y")
        assert step.done

    @pytest.mark.not_fast
    def test_template_reuse_second_question(self, _fixture_corpus: str) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                session.accept_until_done("How many rentals happened in 2025?")
                step = session.ask("How many rentals happened in 2026?")
                assert step.kind == SESSION_KIND_AWAITING_SQL_CONFIRM

    @pytest.mark.not_fast
    def test_reader_writer_queue(
        self,
        _fixture_corpus: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        shared = tempfile.mkdtemp(prefix="sandbox_rw_test_")
        with Sandbox(artifacts_dir=shared, cleanup=False) as reader_sandbox:
            reader = reader_sandbox.engine(role="consumer")
            Sandbox._stage_demo_schema_structure(reader, reader_sandbox._extract_path)
            queue = reader._write_queue_path
            assert queue.is_file()
            with queue.open(encoding="utf-8") as fh:
                before = sum(1 for _ in fh)
            assert before > 0
        with Sandbox(artifacts_dir=shared, cleanup=False) as writer_sandbox:
            writer = writer_sandbox.engine(role="owner")
            with writer.session(mode="writer") as session:
                session.ask(sandbox_questions()[0])
            if queue.is_file():
                with queue.open(encoding="utf-8") as fh:
                    after = sum(1 for _ in fh)
            else:
                after = 0
            assert after < before

    def test_recipe_overrides_writes_applied_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            apply_demo_schema_structure_from_bundle(sb.engine, handle=sb)
        assert (tmp_path / "schema_overrides.applied.json").is_file()

    def test_migration_demo_bundle(self) -> None:
        with zipfile.ZipFile(Sandbox.data_zip_path()) as zf:
            names = zf.namelist()
            assert any("migration_demo/artifacts_v1" in n for n in names)
            assert any(n.endswith("schema_migration_map.json") for n in names)

    @pytest.mark.not_fast
    def test_recipe_validation_failures(self) -> None:
        fails = Sandbox._sandbox_build_section("validation_failures")
        with Sandbox.create_offline_sandbox(AetherEngine) as owner:
            with owner.engine.session() as session:
                step = session.accept_until_done(fails[0])
            assert step.done
        with Sandbox() as sandbox:
            consumer = sandbox.engine(
                EngineContext(allow_objects=_NARROW_ALLOW_OBJECTS),
                role="consumer",
            )
            with consumer.session(mode="reader") as session:
                step = session.accept_until_done("Show me all staff salaries.")
            assert step.done
            assert step.sql is None
            assert step.error is not None
            assert step.error.code.value == "forbidden"

    def test_recipe_maintenance(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            meta = sb.engine.export_structure()
            assert int(meta.get("table_count", 0)) == 34

    @pytest.mark.not_fast
    def test_recipe_full_session_completes(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.ask(sandbox_questions()[1])
                while not step.done:
                    if step.reply_shape == "yes_no":
                        step = session.step("y")
                    elif step.reply_shape == "free_text":
                        step = session.step("ok")
                    else:
                        break
            assert step.done

    @pytest.mark.not_fast
    def test_aetherspace_demo(self, _fixture_corpus: str) -> None:
        from aetherdialect._contracts_base import SpaceContext

        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            catalog = SpaceContext(
                tables=frozenset({"item", "film", "category", "item_category"}),
            )
            sb.engine.aetherspace("catalog", space_context=catalog)
            with sb.engine.session(space="catalog") as session:
                step = session.accept_until_done("Which films are in the Horror category?")
                assert step.done
                assert step.sql
                assert "film" in step.sql.lower()
            with sb.engine.session(space="catalog") as session:
                step = session.accept_until_done("What is total revenue by store?")
                assert step.done
                assert step.sql is None or step.error or step.error is not None and step.error.code.value == "forbidden"


class TestSandboxSecurity:
    def test_mock_fixture_missing(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.ask("totally unknown corpus question xyz123")
                assert step.done is True
                assert step.error and "No mock fixture" in step.error

    def test_consumer_writer_raises(self) -> None:
        with Sandbox() as sandbox:
            engine = sandbox.engine(role="consumer")
            with pytest.raises(OwnerOnlyOperationError):
                with engine.session(mode="writer"):
                    pass

    @pytest.mark.not_fast
    def test_session_active_error(self) -> None:
        practice = sandbox_questions()
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                step = session.ask(practice[0])
                assert not step.done
                with pytest.raises(SessionActiveError):
                    session.ask("another question while busy")

    @pytest.mark.not_fast
    def test_restricted_consumer_permission_denied(self) -> None:
        member = "catalog"
        with Sandbox() as sandbox:
            engine = sandbox.engine(
                EngineContext(allow_objects=SANDBOX_MEMBER_SPACE_TABLES[member]),
                role="consumer",
            )
            with engine.session(mode="reader") as session:
                step = session.accept_until_done("Show me all staff salaries.")
        assert step.done
        assert step.sql is None
        err = (step.error and str(step.error.detail_code or step.error.code.value) or "").lower()
        assert (
            step.error is not None
            and step.error.code.value == "forbidden"
            or "schema_invalid" in err
            or "permission" in err
        )

    @pytest.mark.not_fast
    def test_deny_columns_column_security(self) -> None:
        deny = frozenset({"customer.email"})
        with Sandbox.create_offline_sandbox(AetherEngine, deny_columns=deny) as sb:
            with sb.engine.session() as session:
                step = session.accept_until_done("Who are our top 5 customers by total payment?")
        assert step.done
        codes = {d.code for d in step.diagnostics}
        assert "denied_reference" in codes or step.error is not None

    def test_warmup_blocked_in_sandbox(self) -> None:
        from aetherdialect._contracts_base import ConfigError

        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with pytest.raises(ConfigError, match="sandbox"):
                sb.engine.run_seed_warmup("questions.txt")

    def test_qsim_blocked_in_sandbox(self) -> None:
        from aetherdialect._contracts_base import ConfigError

        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with pytest.raises(ConfigError, match="sandbox"):
                sb.engine.run_qsim()


class TestSandboxSchemaOverlay:
    def test_schema_graph_ddl_overlay_primary_keys(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            sg = sb.engine._schema_graph
            category = sg.tables["category"]
            assert category.primary_key == ["category_id"]
            assert category.columns["category_id"].is_visible is True
            film = sg.tables["film"]
            assert film.primary_key == ["item_id"]

    def test_schema_graph_ddl_overlay_not_null_from_sql_file(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            sg = sb.engine._schema_graph
            staff = sg.tables.get("staff")
            assert staff is not None, "staff table missing from sandbox schema graph"
            username = staff.columns.get("username")
            assert username is not None
            assert username.is_nullable is False

    def test_join_hints_multi_rank_films_tables(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            hints = join_hints_multi(sb.engine._schema_graph, ["category", "film", "rental"])
        substantive = [c for c in hints.get("candidates", []) if c.get("join_path_signature")]
        assert substantive, "expected join path candidates for rank-films scope"


class TestSandboxPublicApi:
    def test_sandbox_entry_points(self) -> None:
        assert callable(Sandbox.create_offline_sandbox)
        assert callable(Sandbox.sandbox_doctor)
        assert callable(Sandbox.assert_sandbox_complete)
        assert not hasattr(AetherEngine, "offline_sandbox")
        assert not hasattr(AetherEngine, "sandbox_questions")

    def test_session_accept_until_done_on_pipeline_session(self) -> None:
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            with sb.engine.session() as session:
                assert callable(session.accept_until_done)
                assert "accept_until_done" in dir(session)

    @pytest.mark.not_fast
    def test_owner_bundled_overrides(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with Sandbox.create_offline_sandbox(AetherEngine) as sb:
            apply_demo_schema_structure_from_bundle(sb.engine, handle=sb)
            assert (tmp_path / "schema_overrides.applied.json").is_file()

    @pytest.mark.not_fast
    def test_consumer_override_proposal_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with Sandbox() as sandbox:
            engine = sandbox.engine(role="consumer")
            Sandbox._stage_demo_schema_structure(engine, sandbox._extract_path)
            assert engine._write_queue_path.is_file()
