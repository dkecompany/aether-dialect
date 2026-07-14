"""Focused live-test matrix for RBAC scope: deny_columns, allow_objects, pguser2 grants, AetherSpace, and named EngineContexts."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import ConfigError
from aetherdialect._constants import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import (
    AccessError,
    EngineContext,
    OwnerOnlyOperationError,
    SpaceContext,
)
from live_tests.conftest import (
    _RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS,
    _build_rbac_consumer_engine,
    _consumer_engine_context,
)


def _ask_until_settled(session, question: str):
    step = session.ask(question)
    while not step.done:
        if step.prompt:
            step = session.step("y")
        else:
            break
    return step


class TestDenyColumnsOwner:
    def test_staff_ssn_question_blocked(self, t2s_deny_columns_owner: AetherEngine) -> None:
        """Owner with deny_columns cannot query hidden staff.ssn."""
        with t2s_deny_columns_owner.session(mode="reader") as session:
            step = _ask_until_settled(session, "List staff social security numbers")
        assert step.status in ("restricted", "permission_denied")
        if step.status == "permission_denied":
            assert PERMISSION_DENIED_USER_MESSAGE in (step.message or "")

    def test_film_query_allowed_under_deny_columns(self, t2s_deny_columns_owner: AetherEngine) -> None:
        """deny_columns on staff does not block unrelated tables."""
        with t2s_deny_columns_owner.session(mode="reader") as session:
            step = _ask_until_settled(session, "How many films are in the catalog?")
        assert step.status != "permission_denied"


class TestAllowObjectsOwner:
    def test_subset_context_blocks_staff(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
    ) -> None:
        """Named allow_objects subset blocks staff even for owner execution scope."""
        subset = EngineContext(
            name="consumer_mirror",
            allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS,
        )
        narrowed = AetherEngine(
            subset,
            artifacts_dir=str(t2s_rbac_owner._artifacts_dir),
            config_file=rbac_postgres_config_path,
            role="owner",
        )
        with narrowed.session(mode="reader") as session:
            step = _ask_until_settled(session, "Show payroll for all employees including SSN")
        assert step.status == "permission_denied"

    def test_subset_context_allows_film(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
    ) -> None:
        """Named allow_objects subset still permits in-scope tables."""
        subset = EngineContext(
            name="films_only_ctx",
            allow_objects=frozenset({"film", "item"}),
        )
        narrowed = AetherEngine(
            subset,
            artifacts_dir=str(t2s_rbac_owner._artifacts_dir),
            config_file=rbac_postgres_config_path,
            role="owner",
        )
        with narrowed.session(mode="reader") as session:
            step = _ask_until_settled(session, "How many film titles exist?")
        assert step.status != "permission_denied"


class TestConsumerPguser2Grants:
    def test_staff_table_permission_denied(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Pguser2 without staff SELECT gets permission_denied on staff questions."""
        with t2s_consumer_pguser2.session(mode="reader") as session:
            step = _ask_until_settled(session, "List all staff first and last names")
        assert step.status == "permission_denied"
        assert step.sql is None

    def test_film_table_allowed(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Pguser2 with film grant can ask about film."""
        with t2s_consumer_pguser2.session(mode="reader") as session:
            step = _ask_until_settled(session, "How many films are in the catalog?")
        assert step.status != "permission_denied"

    def test_item_table_in_consumer_allow_set(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer allow_objects includes item for join paths."""
        visible = t2s_consumer_pguser2._consumer_visible_objects or frozenset()
        assert "item" in visible
        assert "item" in _RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS

    def test_execute_sql_staff_blocked(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Direct execute_sql on a forbidden table is blocked for consumer."""
        with pytest.raises(AccessError, match="administrator"):
            t2s_consumer_pguser2.execute_sql("SELECT staff_id FROM staff LIMIT 1")


class TestAetherSpacePostgres:
    def test_owner_creates_and_uses_scoped_space(self, t2s_rbac_owner: AetherEngine) -> None:
        """AetherSpace scoped session limits intent on Postgres."""
        t2s_rbac_owner.aetherspace(
            "films_only",
            SpaceContext(tables=frozenset({"film"}), columns=frozenset({"film.rating"})),
        )
        with t2s_rbac_owner.session(mode="reader", space="films_only") as session:
            step = _ask_until_settled(session, "List customer email addresses")
        assert step.status in ("failed", "permission_denied") or step.error is not None

    def test_space_descriptor_lists_tables(self, t2s_rbac_owner: AetherEngine) -> None:
        """Saved aetherspace snapshot exposes scoped tables."""
        t2s_rbac_owner.aetherspace("rentals_only", SpaceContext(tables=frozenset({"rental"})))
        desc = t2s_rbac_owner.aetherspace("rentals_only")
        scope = desc.list_scope()
        assert "rental" in scope.get("tables", [])

    def test_unknown_space_raises(self, t2s_rbac_owner: AetherEngine) -> None:
        """session(space=...) on unknown name raises ConfigError."""
        with pytest.raises(ConfigError, match="unknown aetherspace"):
            with t2s_rbac_owner.session(space="missing_space"):
                pass


class TestNamedEngineContext:
    def test_owner_persists_named_context(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
    ) -> None:
        """Owner creates a named EngineContext subset spec."""
        named = EngineContext(
            name="team_retail_persist",
            allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS,
        )
        AetherEngine(
            named,
            artifacts_dir=str(t2s_rbac_owner._artifacts_dir),
            config_file=rbac_postgres_config_path,
            role="owner",
        )
        names = t2s_rbac_owner.list_aetherengines()
        assert "team_retail_persist" in names

    def test_consumer_consumes_named_context_by_name(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
        rbac_consumer_config_path: str,
    ) -> None:
        """Consumer loads a persisted named context by string name."""
        named = EngineContext(
            name="team_retail_consume",
            allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS,
        )
        AetherEngine(
            named,
            artifacts_dir=str(t2s_rbac_owner._artifacts_dir),
            config_file=rbac_postgres_config_path,
            role="owner",
        )
        consumer = _build_rbac_consumer_engine(
            t2s_rbac_owner,
            engine_context="team_retail_consume",
            config_file=rbac_consumer_config_path,
        )
        assert consumer._context_name == "team_retail_consume"

    def test_consumer_engine_context_object_raises(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_consumer_config_path: str,
    ) -> None:
        """Consumer passing an EngineContext object raises OwnerOnlyOperationError."""
        with pytest.raises(OwnerOnlyOperationError):
            _build_rbac_consumer_engine(
                t2s_rbac_owner,
                engine_context=EngineContext(name="inline", allow_objects=frozenset({"film"})),
                config_file=rbac_consumer_config_path,
            )

    def test_export_named_context_round_trip(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
    ) -> None:
        """export_aetherengine dumps a persisted named context."""
        named = EngineContext(name="export_probe", allow_objects=frozenset({"film"}))
        AetherEngine(
            named,
            artifacts_dir=str(t2s_rbac_owner._artifacts_dir),
            config_file=rbac_postgres_config_path,
            role="owner",
        )
        path = t2s_rbac_owner.export_aetherengine("export_probe")
        assert path.is_file()


class TestConsumerWriterBlocked:
    def test_consumer_writer_session_raises(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer writer mode raises OwnerOnlyOperationError."""
        with pytest.raises(OwnerOnlyOperationError, match="writer"):
            with t2s_consumer_pguser2.session(mode="writer"):
                pass

    def test_consumer_aetherspace_write_raises(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer cannot define aetherspace snapshots."""
        with pytest.raises(OwnerOnlyOperationError):
            t2s_consumer_pguser2.aetherspace(
                "blocked",
                SpaceContext(tables=frozenset({"film"})),
            )


class TestRbacScopeComparisons:
    def test_owner_engine_context_has_full_schema(self, t2s_rbac_owner: AetherEngine) -> None:
        """Owner master context reflects staff (not consumer- restricted)."""
        assert "staff" in t2s_rbac_owner._schema_graph.tables

    def test_consumer_visible_objects_subset_of_owner(
        self,
        t2s_rbac_owner: AetherEngine,
        t2s_consumer_pguser2: AetherEngine,
    ) -> None:
        """Consumer visible_objects is a strict subset of owner tables."""
        owner_tables = set(t2s_rbac_owner._schema_graph.tables)
        consumer_visible = t2s_consumer_pguser2._consumer_visible_objects or frozenset()
        assert consumer_visible
        assert consumer_visible <= owner_tables
        assert "staff" not in consumer_visible

    def test_consumer_context_matches_grant_script_tables(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer allow_objects aligns with scripts/sql/grant_pguser2_rental_shop.sql."""
        ctx = _consumer_engine_context()
        assert ctx.allow_objects == _RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS
