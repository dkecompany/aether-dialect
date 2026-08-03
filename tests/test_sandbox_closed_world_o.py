"""Sandbox closed-world scope, fixture resolution, and per-engine trust baseline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import EngineContext, Sandbox
from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import ConfigError, RuntimeConfig
from aetherdialect._llm_provider import (
    SandboxRuntimeState,
    bind_sandbox_runtime,
    current_sandbox_runtime,
    reset_sandbox_runtime,
    set_sandbox_recorded_corpus_question_count,
)
from aetherdialect._main_execution import load_runtime_config
from aetherdialect._sandbox import (
    SandboxFaithfulnessExpectation,
    _apply_bundled_federation_mappings,
    _apply_bundled_schema_overrides,
    _apply_sandbox_consumer_execution_scope,
    _resolve_sandbox_fixture_path,
    _resolve_sandbox_notes_and_sql,
    _sandbox_trusts_bundled_baseline,
)
from aetherdialect._templates import clear_sandbox_paraphrase_source, set_sandbox_paraphrase_source


@pytest.mark.fast
def test_sandbox_trust_baseline_matches_for_engine_and_preset_paths() -> None:
    narrowed = EngineContext(allow_objects=frozenset({"film", "customer"}))
    kwargs = dict(
        preset="owner_writer",
        schema_context=narrowed,
        bundled_notes="/bundle/rental_shop_notes.txt",
        bundled_sql="/bundle/rental_shop.sql",
        deny_columns=None,
        restricted_consumer=False,
        include="tables",
        engine_context=narrowed,
        notes_file=None,
        sql_file=None,
    )
    assert _sandbox_trusts_bundled_baseline(**kwargs) is False


@pytest.mark.fast
def test_per_engine_trust_baseline_leaves_global_policy_unchanged() -> None:
    prev = PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE
    PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE = False
    try:
        from aetherdialect.aetherdialect import AetherEngine

        engine = MagicMock(spec=AetherEngine)
        engine._trust_bundled_baseline = True
        assert PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE is False
    finally:
        PolicyConfig.SANDBOX_TRUST_SCHEMA_BASELINE = prev


@pytest.mark.fast
def test_consumer_scope_honours_user_subset() -> None:
    owner = MagicMock()
    owner._runtime_config = RuntimeConfig(
        engine="postgresql",
        artifacts_dir="/tmp/sandbox",
        engine_context=EngineContext(allow_objects=frozenset({"film", "customer"})),
        llm_execution=load_runtime_config(merged_env={}),
    )
    _apply_sandbox_consumer_execution_scope(owner, restricted=False)
    assert owner._runtime_config.execution_context.allow_objects == frozenset({"film", "customer"})
    assert owner._consumer_visible_objects == frozenset({"film", "customer"})


@pytest.mark.fast
def test_consumer_scope_refuses_outside_owner() -> None:
    owner = MagicMock()
    owner._runtime_config = RuntimeConfig(
        engine="postgresql",
        artifacts_dir="/tmp/sandbox",
        engine_context=EngineContext(allow_objects=frozenset({"not_a_table"})),
        llm_execution=load_runtime_config(merged_env={}),
    )
    with pytest.raises(ConfigError, match="exceed sandbox owner scope"):
        _apply_sandbox_consumer_execution_scope(owner, restricted=False)


@pytest.mark.fast
def test_resolve_bundled_fixture_alias_notes(tmp_path: Path) -> None:
    notes = tmp_path / "rental_shop_notes.txt"
    notes.write_text("catalog notes", encoding="utf-8")
    resolved, notice = _resolve_sandbox_fixture_path(tmp_path, "notes.txt")
    assert resolved == str(notes)
    assert notice is not None
    assert "rental_shop_notes.txt" in notice


@pytest.mark.fast
def test_resolve_bundled_fixture_existing_path_unchanged(tmp_path: Path) -> None:
    custom = tmp_path / "custom_notes.txt"
    custom.write_text("custom", encoding="utf-8")
    resolved, notice = _resolve_sandbox_fixture_path(tmp_path, str(custom))
    assert resolved == str(custom)
    assert notice is None


@pytest.mark.fast
def test_resolve_sandbox_notes_and_sql_applies_fixture_aliases(tmp_path: Path) -> None:
    notes = tmp_path / "rental_shop_notes.txt"
    notes.write_text("catalog notes", encoding="utf-8")
    sql = tmp_path / "rental_shop.sql"
    sql.write_text("select 1", encoding="utf-8")
    resolved_notes, resolved_sql, notices = _resolve_sandbox_notes_and_sql(
        engine_context=None,
        notes_file="notes.txt",
        sql_file="schema.sql",
        bundled_notes=str(notes),
        bundled_sql=str(sql),
        extract_path=tmp_path,
    )
    assert resolved_notes == str(notes)
    assert resolved_sql == str(sql)
    assert notices


def _write_minimal_bundle(root: Path) -> None:
    (root / "rental_shop_seed.sql").write_text(
        "\n".join(
            (
                "CREATE TABLE customer (customer_id INTEGER PRIMARY KEY);",
                "CREATE TABLE film (film_id INTEGER PRIMARY KEY);",
            ),
        ),
        encoding="utf-8",
    )
    (root / "rental_shop.sql").write_text("SELECT 1;", encoding="utf-8")
    (root / "rental_shop_notes.txt").write_text("catalog notes", encoding="utf-8")
    fixtures = root / "fixtures"
    fixtures.mkdir()
    (fixtures / "rental_shop_mock.json").write_text('{"fixtures": []}', encoding="utf-8")


@pytest.mark.fast
def test_offline_sandbox_rejects_preset_kwarg() -> None:
    from aetherdialect import AetherEngine

    with pytest.raises(TypeError, match="preset"):
        AetherEngine.offline_sandbox(preset="consumer_reader")


@pytest.mark.fast
def test_create_offline_sandbox_rejects_preset_kwarg() -> None:
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import create_offline_sandbox

    with pytest.raises(TypeError, match="preset"):
        create_offline_sandbox(AetherEngine, preset="consumer_reader")


@pytest.mark.fast
def test_offline_sandbox_rejects_restricted_consumer_kwarg() -> None:
    from aetherdialect import AetherEngine

    with pytest.raises(TypeError, match="restricted_consumer"):
        AetherEngine.offline_sandbox(restricted_consumer=True)


@pytest.mark.fast
def test_create_offline_sandbox_rejects_custom_seed_without_maintainer_access(tmp_path: Path) -> None:
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import create_offline_sandbox

    seed_path = tmp_path / "custom.sql"
    seed_path.write_text("CREATE TABLE t (id INTEGER);", encoding="utf-8")
    with pytest.raises(ConfigError, match="seed_sql"):
        create_offline_sandbox(AetherEngine, seed_sql=str(seed_path), maintainer_access=False)


@pytest.mark.fast
def test_load_dataset_rejects_custom_seed_on_closed_world_sandbox(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    seed_path = tmp_path / "custom.sql"
    seed_path.write_text("CREATE TABLE t (id INTEGER);", encoding="utf-8")
    sandbox = Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False)
    try:
        sandbox._maintainer_access = False
        with pytest.raises(ConfigError, match="maintainer_access"):
            sandbox.load_dataset("extra", seed_sql=str(seed_path))
    finally:
        sandbox.close()


@pytest.mark.fast
def test_load_dataset_accepts_bundled_dataset_name(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    sandbox = Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False)
    try:
        sandbox.load_dataset("main")
        tables = {str(row[0]).lower() for row in sandbox.connection("main").execute("SHOW TABLES").fetchall()}
        assert "customer" in tables
    finally:
        sandbox.close()


@pytest.mark.fast
def test_sandbox_rejects_bundle_dir_without_maintainer_access(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    with pytest.raises(ConfigError, match="bundle_dir"):
        Sandbox(bundle_dir=str(bundle), maintainer_access=False, auto_seed=False)


@pytest.mark.fast
def test_sandbox_rejects_data_zip_override_without_maintainer_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    monkeypatch.setenv("AETHERDIALECT_SANDBOX_DATA_ZIP", str(bundle))
    with pytest.raises(ConfigError, match="AETHERDIALECT_SANDBOX_DATA_ZIP"):
        Sandbox(maintainer_access=False, auto_seed=False)


@pytest.mark.fast
def test_sandbox_default_llm_mode_is_mock(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
        assert sandbox.llm_mode == "mock"
        assert sandbox.uses_network is False


@pytest.mark.fast
def test_sandbox_custom_llm_config_declares_network_mode(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    config = tmp_path / "live.toml"
    config.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
    with Sandbox(
        maintainer_access=True,
        bundle_dir=str(bundle),
        auto_seed=False,
        llm_config=str(config),
    ) as sandbox:
        assert sandbox.llm_mode == "network"
        assert sandbox.uses_network is True


@pytest.mark.fast
def test_sandbox_runtime_isolates_recorded_corpus_counts() -> None:
    left = SandboxRuntimeState()
    right = SandboxRuntimeState()
    left_token = bind_sandbox_runtime(left)
    try:
        set_sandbox_recorded_corpus_question_count(3)
        assert current_sandbox_runtime() is left
        assert current_sandbox_runtime().recorded_corpus_question_count == 3
    finally:
        reset_sandbox_runtime(left_token)
    right_token = bind_sandbox_runtime(right)
    try:
        assert current_sandbox_runtime().recorded_corpus_question_count is None
        set_sandbox_recorded_corpus_question_count(9)
        assert current_sandbox_runtime().recorded_corpus_question_count == 9
    finally:
        reset_sandbox_runtime(right_token)


@pytest.mark.fast
def test_sandbox_runtime_isolates_paraphrase_sources() -> None:
    left = SandboxRuntimeState()
    right = SandboxRuntimeState()
    left_token = bind_sandbox_runtime(left)
    try:
        set_sandbox_paraphrase_source({"q1": ["variant-a"]})
    finally:
        reset_sandbox_runtime(left_token)
    right_token = bind_sandbox_runtime(right)
    try:
        set_sandbox_paraphrase_source({"q2": ["variant-b"]})
        assert current_sandbox_runtime().paraphrase_source == {"q2": ["variant-b"]}
    finally:
        reset_sandbox_runtime(right_token)
    left_token = bind_sandbox_runtime(left)
    try:
        assert current_sandbox_runtime().paraphrase_source == {"q1": ["variant-a"]}
    finally:
        reset_sandbox_runtime(left_token)
        clear_sandbox_paraphrase_source()


@pytest.mark.fast
def test_sandbox_close_clears_faithfulness_index(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    sandbox = Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False)
    sandbox._runtime.faithfulness_by_question["cached"] = SandboxFaithfulnessExpectation()
    sandbox._runtime.faithfulness_loaded = True
    sandbox.close()
    assert "cached" not in sandbox._runtime.faithfulness_by_question
    assert sandbox._runtime.faithfulness_loaded is False


@pytest.mark.fast
def test_single_sandbox_builds_two_scoped_engines_without_reextracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    contexts_seen: list[frozenset[str]] = []

    class FakeEngine:
        def __init__(self, schema_context: EngineContext, **kwargs: object) -> None:
            del kwargs
            allow = frozenset(schema_context.allow_objects or ())
            contexts_seen.append(allow)
            self._schema_graph = MagicMock()
            self._schema_graph.tables = {name: None for name in allow}
            self._schema_graph.schema_literal_json = "{}"

    monkeypatch.setattr("aetherdialect._sandbox._aether_engine_cls", lambda: FakeEngine)
    with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
        extract_path = sandbox._extract_path
        sandbox.load_dataset("main")
        owner = sandbox.engine(EngineContext(allow_objects=frozenset({"customer"})))
        film_only = sandbox.engine(EngineContext(allow_objects=frozenset({"film"})))
        assert sandbox._extract_path is extract_path
        assert contexts_seen == [frozenset({"customer"}), frozenset({"film"})]
        assert "customer" in owner._schema_graph.tables
        assert "film" in film_only._schema_graph.tables
        assert "film" not in owner._schema_graph.tables
        assert "customer" not in film_only._schema_graph.tables


@pytest.mark.fast
def test_sandbox_context_manager_closes_and_cleans_up(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    artifacts: Path | None = None
    with Sandbox(maintainer_access=True, bundle_dir=str(bundle), auto_seed=False) as sandbox:
        sandbox.load_dataset("main")
        artifacts = Path(sandbox.artifacts_dir)
        assert artifacts.is_dir()
        assert "main" in sandbox.datasets
    assert artifacts is not None
    assert not artifacts.exists()


@pytest.mark.fast
def test_apply_bundled_schema_overrides_leaves_cwd_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aetherdialect._constants import SCHEMA_OVERRIDES_DEFAULT_FILENAME

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "schema_overrides_demo.json").write_text("{}", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = {p.name for p in work.iterdir()}
    engine = MagicMock()
    engine._artifacts_dir = str(tmp_path / "artifacts")
    Path(engine._artifacts_dir).mkdir()
    monkeypatch.setattr(
        "aetherdialect._sandbox._open_data_bundle",
        lambda **kwargs: type("Access", (), {"path": bundle, "owns_cleanup": False})(),
    )
    _apply_bundled_schema_overrides(engine, None)
    after = {p.name for p in work.iterdir()}
    applied = Path(engine._artifacts_dir) / SCHEMA_OVERRIDES_DEFAULT_FILENAME
    assert applied.is_file()
    assert after == before


@pytest.mark.fast
def test_federation_mapping_demo_leaves_cwd_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aetherdialect._constants import FEDERATION_DECLARATION_FILENAME

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / FEDERATION_DECLARATION_FILENAME).write_text("{}", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = {p.name for p in work.iterdir()}
    engine = MagicMock()
    engine._artifacts_dir = str(tmp_path / "artifacts")
    Path(engine._artifacts_dir).mkdir()
    handle = MagicMock()
    _apply_bundled_federation_mappings(engine, bundle, handle=handle)
    after = {p.name for p in work.iterdir()}
    assert (Path(engine._artifacts_dir) / FEDERATION_DECLARATION_FILENAME).is_file()
    assert after == before
