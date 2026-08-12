"""Main execution hub composing space, interactive, init, and session mixins."""

from __future__ import annotations

from ._main_init import MainInitOps
from ._main_interactive import MainInteractiveOps
from ._main_session import MainSessionSerdeOps
from ._main_spaces import MainSpaceOps
from ._utils_artifacts import register_knowledge_migration_handler, register_structural_migration_handler


class MainExecutionOps(MainInteractiveOps, MainSpaceOps, MainInitOps, MainSessionSerdeOps):
    """Public main-execution surface composed from role mixins."""


register_structural_migration_handler(MainExecutionOps.apply_structural_migration_to_persisted_scopes)
register_knowledge_migration_handler(MainExecutionOps.migrate_engine_knowledge_artifacts)
