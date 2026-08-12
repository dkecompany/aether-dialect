"""Focused live-test matrix for RBAC scope: deny_columns, allow_objects, pguser2 grants, AetherSpace, and named EngineContexts."""

from __future__ import annotations

import pytest

from aetherdialect import AetherEngine
from aetherdialect._constants_runtime import PERMISSION_DENIED_USER_MESSAGE
from aetherdialect._contracts_base import (
    ConfigError,
    EngineContext,
    OwnerOnlyOperationError,
    SpaceContext,
)
from live_tests.conftest import (
    _build_rbac_consumer_engine,
    _consumer_engine_context,
)
from live_tests.mydb_profile import PROFILE_CONSUMER_ALLOW_OBJECTS as _RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS


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
        # validate_question may classify SSN wording as restricted before intent parse;
        # deny_columns scope may also yield permission_denied later in the pipeline.
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
        t2s_rbac_owner.engine_context(
            "consumer_mirror",
            EngineContext(allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS),
        )
        narrowed = AetherEngine(
            "consumer_mirror",
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
        t2s_rbac_owner.engine_context(
            "films_only_ctx",
            EngineContext(allow_objects=frozenset({"film", "item"})),
        )
        narrowed = AetherEngine(
            "films_only_ctx",
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

    def test_execute_sql_removed_from_public_api(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Raw execute_sql is not part of the public engine surface."""
        assert not hasattr(t2s_consumer_pguser2, "execute_sql")


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
        scope = desc.tables
        assert "rental" in scope

    def test_unknown_space_raises(self, t2s_rbac_owner: AetherEngine) -> None:
        """session(space=...) on unknown name raises ConfigError."""
        with pytest.raises(ConfigError, match="unknown aetherspace"):
            with t2s_rbac_owner.session(space="missing_space"):
                pass


class TestNamedEngineContext:
    def test_owner_persists_named_context(
        self,
        t2s_rbac_owner: AetherEngine,
    ) -> None:
        """Owner registers a named EngineContext subset spec."""
        t2s_rbac_owner.engine_context(
            "team_retail_persist",
            EngineContext(allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS),
        )
        names = t2s_rbac_owner.list_contexts()
        assert "team_retail_persist" in names

    def test_consumer_consumes_named_context_by_name(
        self,
        t2s_rbac_owner: AetherEngine,
        rbac_postgres_config_path: str,
        rbac_consumer_config_path: str,
    ) -> None:
        """Consumer loads a persisted named context by string name."""
        t2s_rbac_owner.engine_context(
            "team_retail_consume",
            EngineContext(allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS),
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
                engine_context=EngineContext(allow_objects=frozenset({"film"})),
                config_file=rbac_consumer_config_path,
            )

    def test_export_named_context_round_trip(
        self,
        t2s_rbac_owner: AetherEngine,
    ) -> None:
        """export_context dumps a persisted named context."""
        t2s_rbac_owner.engine_context(
            "export_probe",
            EngineContext(allow_objects=frozenset({"film"})),
        )
        path = t2s_rbac_owner.export_context("export_probe")
        assert path.is_file()


class TestConsumerWriterBlocked:
    def test_consumer_writer_session_raises(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer writer mode raises OwnerOnlyOperationError."""
        with pytest.raises(OwnerOnlyOperationError, match="writer"):
            with t2s_consumer_pguser2.session(mode="writer"):
                pass

    def test_consumer_aetherspace_within_visibility(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer may define an aetherspace over tables in their effective visibility."""
        visible = t2s_consumer_pguser2._consumer_visible_objects or frozenset()
        table_keys = sorted(v for v in visible if "." not in v)
        assert table_keys, "consumer must have at least one visible table"
        name = "consumer_visible_space"
        desc = t2s_consumer_pguser2.aetherspace(
            name,
            SpaceContext(tables=frozenset({table_keys[0]})),
        )
        assert desc is not None
        assert desc.name == name
        assert any(s.uid == desc.uid for s in t2s_consumer_pguser2.list_aetherspaces())

    def test_consumer_aetherspace_outside_visibility_raises(self, t2s_consumer_pguser2: AetherEngine) -> None:
        """Consumer cannot define an aetherspace naming tables outside their visibility."""
        visible = t2s_consumer_pguser2._consumer_visible_objects or frozenset()
        table_keys = {v for v in visible if "." not in v}
        # staff is typically owner-only under the pguser2 grant script; fall back to any
        # owner-graph table the consumer cannot see.
        forbidden = "staff" if "staff" not in table_keys else None
        if forbidden is None:
            owner_tables = set(t2s_consumer_pguser2._schema_graph.tables)
            candidates = sorted(owner_tables - table_keys)
            assert candidates, "need at least one table outside consumer visibility"
            forbidden = candidates[0]
        with pytest.raises(ConfigError, match="outside visible scope"):
            t2s_consumer_pguser2.aetherspace(
                "blocked_outside",
                SpaceContext(tables=frozenset({forbidden})),
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
