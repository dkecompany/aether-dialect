"""Stable import surface for the Text2SQL package."""

from importlib.metadata import PackageNotFoundError, version

from ._config import LlmExecutionConfig
from ._contracts_base import (
    AuditEvent,
    ConfigError,
    ConnectionError,
    DatabasePingFailed,
    Diagnostic,
    LlmTransientFailure,
    MigrationPendingError,
    MigrationPreview,
    RetryableError,
    RuntimeConfig,
    SchemaAccessError,
    SchemaContext,
    SessionActiveError,
    SessionStep,
    StatementTimeoutError,
)
from ._main_execution import PipelineSession
from .text2sql import (
    AsyncPipelineSession,
    ConfigSnapshot,
    QSimSummarySnapshot,
    SchemaStatsSnapshot,
    SeedWarmupSummarySnapshot,
    Text2SQL,
)

try:
    __version__ = version("aetherdialect")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "AsyncPipelineSession",
    "AuditEvent",
    "PipelineSession",
    "ConfigError",
    "ConnectionError",
    "ConfigSnapshot",
    "DatabasePingFailed",
    "Diagnostic",
    "LlmExecutionConfig",
    "LlmTransientFailure",
    "MigrationPendingError",
    "MigrationPreview",
    "QSimSummarySnapshot",
    "RetryableError",
    "RuntimeConfig",
    "SchemaAccessError",
    "SchemaContext",
    "SchemaStatsSnapshot",
    "SeedWarmupSummarySnapshot",
    "SessionActiveError",
    "SessionStep",
    "StatementTimeoutError",
    "Text2SQL",
    "__version__",
]
