"""Tests for the Sandbox authoring environment entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aetherdialect import EngineContext, FederationContext, Sandbox
from aetherdialect._constants import (
    SANDBOX_BUNDLED_MEMBER_SEEDS,
    SCHEMA_OVERRIDES_DEFAULT_FILENAME,
    SCHEMA_OVERRIDES_VERSION,
)
from aetherdialect._contracts_base import FederationConfigError, FederationDeclarationError, MigrationPendingError

data_zip_path = Sandbox.data_zip_path


def _require_bundled_data() -> None:
    if not Sandbox.data_zip_path().is_file():
        pytest.skip("needs_corpus: bundled sandbox data.zip absent")


def _bundled_federation_member_paths() -> dict[str, str]:
    with Sandbox() as sandbox:
        root = sandbox._extract_path
        paths = {
            member_name: str(root / seed_name)
            for member_name, seed_name in SANDBOX_BUNDLED_MEMBER_SEEDS
            if (root / seed_name).is_file()
        }
        if len(paths) < 2:
            pytest.skip("federation partition seeds are not present in the bundle")
        return paths


def _write_federation_declaration(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _minimal_payment_declaration(*, semantics: str = "union") -> dict[str, object]:
    table_entry: dict[str, object] = {
        "logical": "payment",
        "semantics": semantics,
        "members": [
            {"source": "storefront", "table": "payment", "columns": {}},
            {"source": "catalog", "table": "payment", "columns": {}},
        ],
    }
    if semantics == "replica":
        table_entry["authoritative_source"] = "storefront"
    return {
        "version": 1,
        "federation_id": "sandbox_rental_shop",
        "aliases": {},
        "coordinator": {"row_cap": 500000},
        "cross_source_joins": [],
        "logical_columns": [],
        "logical_tables": [table_entry],
    }


@pytest.mark.fast
def test_sandbox_engine_honours_caller_engine_context() -> None:
    _require_bundled_data()
    scope = EngineContext(allow_objects=frozenset({"customer", "rental"}))
    with Sandbox() as sandbox:
        engine = sandbox.engine(scope)
        assert engine._runtime_config.engine_context.allow_objects == frozenset({"customer", "rental"})
        assert engine._sandbox_mode is True


@pytest.mark.fast
def test_load_dataset_seeds_additional_database(tmp_path: Path) -> None:
    _require_bundled_data()
    seed_path = tmp_path / "extra_seed.sql"
    seed_path.write_text(
        "\n".join(
            (
                "CREATE TABLE extra_widget (id INTEGER PRIMARY KEY);",
                "INSERT INTO extra_widget VALUES (7);",
            ),
        ),
        encoding="utf-8",
    )
    with Sandbox(maintainer_access=True) as sandbox:
        before = len(sandbox.datasets)
        sandbox.load_dataset("extra", seed_sql=str(seed_path))
        row = sandbox.connection("extra").execute("SELECT id FROM extra_widget").fetchone()
        assert row is not None and int(row[0]) == 7
        assert len(sandbox.datasets) == before + 1


@pytest.mark.fast
def test_datasets_reports_main_and_loaded_names() -> None:
    _require_bundled_data()
    with Sandbox() as sandbox:
        assert "main" in sandbox.datasets
        sandbox.load_dataset("storefront")
        assert "main" in sandbox.datasets
        assert "storefront" in sandbox.datasets


@pytest.mark.fast
def test_offline_sandbox_returns_sandbox_handle_with_built_engine() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import SandboxHandle

    with AetherEngine.offline_sandbox() as handle:
        assert isinstance(handle, SandboxHandle)
        assert handle.engine is not None
        assert handle._sandbox is not None
        assert callable(handle.session)


@pytest.mark.fast
def test_unadopted_sandbox_connected_engine_rejects_session() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine, EngineContext
    from aetherdialect._contracts_base import ConfigError

    with Sandbox() as sandbox:
        engine = AetherEngine(
            EngineContext(allow_objects=frozenset({"customer", "rental"})),
            native_connection=sandbox.connection(),
            artifacts_dir=sandbox.artifacts_dir,
            config_file=sandbox.config_file,
        )
        with pytest.raises(ConfigError, match="adopt"):
            engine.session()


@pytest.mark.fast
def test_adopting_manually_built_engine_suppresses_warmup() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine, EngineContext
    from aetherdialect._contracts_base import ConfigError

    with Sandbox() as sandbox:
        engine = AetherEngine(
            EngineContext(allow_objects=frozenset({"customer", "rental"})),
            native_connection=sandbox.connection(),
            artifacts_dir=sandbox.artifacts_dir,
            config_file=sandbox.config_file,
        )
        sandbox.adopt(engine)
        with pytest.raises(ConfigError, match="sandbox"):
            engine.run_seed_warmup("questions.txt")
        with engine.session() as session:
            assert session is not None


@pytest.mark.fast
def test_offline_sandbox_honours_supplied_engine_context() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    scope = EngineContext(allow_objects=frozenset({"customer", "rental"}))
    with AetherEngine.offline_sandbox(engine_context=scope) as handle:
        assert handle.engine._runtime_config.engine_context.allow_objects == frozenset({"customer", "rental"})


@pytest.mark.fast
def test_offline_sandbox_notes_file_override_beats_context_and_bundle(tmp_path: Path) -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    bundled_notes = tmp_path / "bundled_notes.txt"
    bundled_notes.write_text("bundled", encoding="utf-8")
    context_notes = tmp_path / "context_notes.txt"
    context_notes.write_text("context", encoding="utf-8")
    override_notes = tmp_path / "override_notes.txt"
    override_notes.write_text("override", encoding="utf-8")
    scope = EngineContext(notes_file=str(context_notes))
    with AetherEngine.offline_sandbox(engine_context=scope, notes_file=str(override_notes)) as handle:
        assert handle.engine._runtime_config.engine_context.notes_file == str(override_notes)


@pytest.mark.fast
def test_offline_sandbox_sql_file_override_beats_context_and_bundle(tmp_path: Path) -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    context_sql = tmp_path / "context.sql"
    context_sql.write_text("SELECT 1;", encoding="utf-8")
    override_sql = tmp_path / "override.sql"
    override_sql.write_text("SELECT 2;", encoding="utf-8")
    scope = EngineContext(sql_file=str(context_sql))
    with AetherEngine.offline_sandbox(engine_context=scope, sql_file=str(override_sql)) as handle:
        assert handle.engine._runtime_config.engine_context.sql_file == str(override_sql)


@pytest.mark.fast
def test_offline_sandbox_and_authoring_sandbox_share_bundle_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import Sandbox

    calls = 0
    real_open = Sandbox._open_data_bundle

    def counting_open(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr("aetherdialect._sandbox.Sandbox._open_data_bundle", counting_open)
    monkeypatch.setattr(Sandbox, "_open_data_bundle", staticmethod(counting_open))
    with AetherEngine.offline_sandbox() as handle:
        assert handle._sandbox is not None
        assert handle._sandbox._extract_path == handle._sandbox._extract_path
    assert calls == 1
    calls = 0
    with Sandbox() as sandbox:
        assert sandbox._extract_path.is_dir()
    assert calls == 1


@pytest.mark.fast
def test_narrow_engine_context_profiles_live_without_baseline_tables() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    scope = EngineContext(allow_objects=frozenset({"customer"}))
    with AetherEngine.offline_sandbox(engine_context=scope) as handle:
        tables = set(handle.engine._schema_graph.tables)
        assert "customer" in tables
        assert "film" not in tables


@pytest.mark.fast
def test_matching_engine_context_keeps_full_baseline_graph() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import Sandbox

    with AetherEngine.offline_sandbox() as default_handle:
        default_tables = set(default_handle.engine._schema_graph.tables)
    bundled_scope = Sandbox._owner_writer_schema_context(notes_file=None, sql_file=None)
    with AetherEngine.offline_sandbox(engine_context=bundled_scope) as handle:
        assert set(handle.engine._schema_graph.tables) == default_tables


@pytest.mark.fast
def test_sandbox_handle_exposes_connection_property() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    with AetherEngine.offline_sandbox() as handle:
        assert handle.connection is not None
        assert handle.connection is handle.engine._native_connection


@pytest.mark.fast
def test_sandbox_handle_adopt_delegates_to_sandbox() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine, EngineContext

    with AetherEngine.offline_sandbox() as handle:
        assert handle._sandbox is not None
        engine = AetherEngine(
            EngineContext(allow_objects=frozenset({"customer"})),
            native_connection=handle.connection,
            artifacts_dir=handle.artifacts_dir,
            config_file=handle._sandbox.config_file,
        )
        handle.adopt(engine)
        assert engine._sandbox_mode is True


@pytest.mark.fast
def test_federation_offline_sandbox_honours_federation_context() -> None:
    _require_bundled_data()
    from aetherdialect import Sandbox

    scope = FederationContext(allow_objects=frozenset({"payment"}))
    with Sandbox() as sandbox:
        federation = sandbox.federation("sandbox_rental_shop", context=scope)
        assert federation._master_context is scope


@pytest.mark.fast
def test_custom_federation_declaration_rejects_replica_without_authoritative_source(tmp_path: Path) -> None:
    _require_bundled_data()

    payload = _minimal_payment_declaration(semantics="replica")
    del payload["logical_tables"][0]["authoritative_source"]
    decl_path = _write_federation_declaration(tmp_path / "bad_replica.json", payload)
    members = _bundled_federation_member_paths()
    with pytest.raises(FederationConfigError, match="authoritative_source"):
        with Sandbox(maintainer_access=True) as sandbox:
            sandbox.federation(
                "sandbox_rental_shop",
                declaration_file=decl_path,
                members=members,
            )


@pytest.mark.fast
def test_custom_federation_declaration_rejects_unknown_mapping_source(tmp_path: Path) -> None:
    _require_bundled_data()

    payload = _minimal_payment_declaration()
    payload["logical_tables"][0]["members"] = [
        {"source": "ghost", "table": "payment", "columns": {}},
    ]
    decl_path = _write_federation_declaration(tmp_path / "unknown_source.json", payload)
    members = _bundled_federation_member_paths()
    with pytest.raises(FederationDeclarationError, match="unresolved source"):
        with Sandbox(maintainer_access=True) as sandbox:
            sandbox.federation(
                "sandbox_rental_shop",
                declaration_file=decl_path,
                members=members,
            )


@pytest.mark.fast
def test_custom_federation_declaration_rejects_unresolved_mapping_table(tmp_path: Path) -> None:
    _require_bundled_data()

    payload = _minimal_payment_declaration()
    payload["logical_tables"][0]["members"] = [
        {"source": "storefront", "table": "missing_payment", "columns": {}},
    ]
    decl_path = _write_federation_declaration(tmp_path / "missing_table.json", payload)
    members = _bundled_federation_member_paths()
    with pytest.raises(FederationDeclarationError, match="unresolved table"):
        with Sandbox(maintainer_access=True) as sandbox:
            sandbox.federation(
                "sandbox_rental_shop",
                declaration_file=decl_path,
                members=members,
            )


@pytest.mark.fast
def test_custom_federation_declaration_rejects_union_with_authoritative_source(tmp_path: Path) -> None:
    _require_bundled_data()

    payload = _minimal_payment_declaration()
    payload["logical_tables"][0]["authoritative_source"] = "storefront"
    decl_path = _write_federation_declaration(tmp_path / "union_with_authority.json", payload)
    members = _bundled_federation_member_paths()
    with pytest.raises(FederationConfigError, match="must not set authoritative_source"):
        with Sandbox(maintainer_access=True) as sandbox:
            sandbox.federation(
                "sandbox_rental_shop",
                declaration_file=decl_path,
                members=members,
            )


@pytest.mark.fast
def test_custom_federation_declaration_rejects_cross_source_join_on_unowned_table(tmp_path: Path) -> None:
    _require_bundled_data()

    payload = _minimal_payment_declaration()
    payload["cross_source_joins"] = [
        {
            "left": "catalog.payment.payment_id",
            "right": "storefront.payment.payment_id",
            "kind": "inner",
            "logical_key": "payment_id",
        },
    ]
    decl_path = _write_federation_declaration(tmp_path / "bad_join_owner.json", payload)
    members = _bundled_federation_member_paths()
    with pytest.raises(FederationDeclarationError, match="is not owned by"):
        with Sandbox(maintainer_access=True) as sandbox:
            sandbox.federation(
                "sandbox_rental_shop",
                declaration_file=decl_path,
                members=members,
            )


@pytest.mark.fast
def test_sandbox_apply_overrides_rejects_unknown_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine
    from aetherdialect._contracts_base import ConfigError

    monkeypatch.chdir(tmp_path)
    overrides = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {"customer": {"columns": {"ghost_col": {"description": "missing"}}}},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
    }
    (tmp_path / SCHEMA_OVERRIDES_DEFAULT_FILENAME).write_text(json.dumps(overrides), encoding="utf-8")
    with AetherEngine.offline_sandbox() as handle:
        with pytest.raises(ConfigError, match="unknown column"):
            handle.engine.apply_overrides()


@pytest.mark.fast
def test_sandbox_apply_migration_map_rejects_invalid_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine
    from aetherdialect._sandbox import Sandbox

    monkeypatch.chdir(tmp_path)
    bad_map = {
        "version": 1,
        "action": "remap",
        "table_renames": [{"from": "customer", "to": "ghost_customer"}],
    }
    map_path = tmp_path / "schema_migration_map.json"
    map_path.write_text(json.dumps(bad_map), encoding="utf-8")
    with AetherEngine.offline_sandbox() as handle:
        with pytest.raises(MigrationPendingError, match="validation failed"):
            AetherEngine.apply_migration_map(
                str(map_path),
                engine_context=Sandbox._owner_writer_schema_context(notes_file=None, sql_file=None),
                artifacts_dir=handle.artifacts_dir,
                config_file=handle._sandbox.config_file,
                execution_engine=handle.engine._execution_engine,
                native_connection=handle.connection,
                role="owner",
            )


@pytest.mark.fast
def test_sandbox_aetherspace_rejects_unknown_table() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine, SpaceContext
    from aetherdialect._contracts_base import ConfigError

    with AetherEngine.offline_sandbox() as handle:
        with pytest.raises(ConfigError, match="not in the schema graph"):
            handle.engine.aetherspace(
                "narrow",
                space_context=SpaceContext(tables=frozenset({"ghost_table"})),
            )


@pytest.mark.fast
def test_federation_aetherspace_rejects_partial_union_deny() -> None:
    _require_bundled_data()
    from aetherdialect import SpaceContext
    from aetherdialect._contracts_base import ConfigError

    with Sandbox() as sandbox:
        federation = sandbox.federation("sandbox_rental_shop")
        with pytest.raises(ConfigError, match="partially denies"):
            federation.aetherspace(
                "payments_only",
                space_context=SpaceContext(deny_objects=frozenset({"payment"})),
            )


@pytest.mark.fast
def test_unrecorded_question_in_recorded_corpus_mode_names_mode(tmp_path: Path) -> None:
    from aetherdialect._llm_provider import (
        MockFixtureMissingError,
        MockProvider,
        SandboxRuntimeState,
    )

    fixtures = tmp_path / "mock.json"
    fixtures.write_text(json.dumps({"fixtures": []}), encoding="utf-8")
    runtime = SandboxRuntimeState()
    token = SandboxRuntimeState.bind_sandbox_runtime(runtime)
    SandboxRuntimeState.set_sandbox_recorded_corpus_question_count(17)
    try:
        provider = MockProvider(str(fixtures))
        with pytest.raises(MockFixtureMissingError, match="recorded-corpus mode") as exc_info:
            provider.chat_text("sys", "Who invented SQL?", task="default", max_retries=1, timeout=1.0)
        message = str(exc_info.value)
        assert "17" in message
        assert "Sandbox.sandbox_questions()" in message
        assert "malformed" not in message.lower()
    finally:
        SandboxRuntimeState.set_sandbox_recorded_corpus_question_count(None)
        SandboxRuntimeState.reset_sandbox_runtime(token)


@pytest.mark.fast
def test_offline_sandbox_honours_supplied_llm_config(tmp_path: Path) -> None:
    _require_bundled_data()
    custom = tmp_path / "custom_llm.toml"
    custom.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
    with Sandbox(llm_config=str(custom)) as sandbox:
        assert sandbox.config_file == str(custom.resolve())


@pytest.mark.fast
def test_offline_sandbox_default_engine_is_owner_writer() -> None:
    _require_bundled_data()
    from aetherdialect import AetherEngine

    with AetherEngine.offline_sandbox() as handle:
        assert handle.engine._schema_role == "owner"


@pytest.mark.fast
def test_sandbox_guide_documents_engine_session_mode_as_primary_entry() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "SANDBOX.md").read_text(encoding="utf-8")
    assert 'sb.engine.session(mode="writer")' in text
    assert "maintainer_access" in text


@pytest.mark.fast
def test_api_reference_documents_sandbox_authoring_surface() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "sandbox.federation" in text
    assert "member_connections" in text
    assert "seed_sql" in text
    assert "cleanup_artifacts" in text
    assert "sandbox.adopt" in text
