# API reference

Lookup for the `aetherdialect` package: exported types and signatures, TOML flattening keys, database connection keys, JSON file schemas, diagnostic codes, and exceptions. Requires Python 3.10 or newer. Embedding flow and suspend `kind` values: [Integrator guide](INTEGRATOR_GUIDE.md). Operator semantics: [User guide](USER_GUIDE.md). End-to-end setup: [Getting started](GETTING_STARTED.md). This file does not narrate deployment patterns.

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [User guide](USER_GUIDE.md) -> [Integrator guide](INTEGRATOR_GUIDE.md) -> [Sandbox guide](SANDBOX.md) -> this document -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Exported symbols](#exported-symbols) | Package `__all__` surface |
| [EngineContext](#enginecontext) | Single-engine scope fields |
| [AetherEngine](#aetherengine) | Single-engine constructor and methods |
| [FederationContext](#federationcontext) | Composite scope fields |
| [AetherFederation](#aetherfederation) | Composite constructor and methods |
| [SpaceContext](#spacecontext) | AetherSpace allow/deny fields |
| [AetherSpace](#aetherspace) | Read-only space descriptor |
| [SessionStep](#sessionstep) | Turn return type fields |
| [Configuration](#configuration) | Merge order, TOML flattening, connection keys |
| [PipelineSession methods](#pipelinesession-methods) | Sync session API |
| [AsyncPipelineSession methods](#asyncpipelinesession-methods) | Async session API |
| [Package helpers](#package-helpers) | Top-level constants |
| [Schema overrides JSON (`schema_overrides.json`)](#schema-overrides-json-schema_overridesjson) | Analyst override editor shape |
| [Federation and migration JSON](#federation-and-migration-json) | Manifest, mappings, migration maps |
| [Observability](#observability) | Audit, diagnostics, LLM cost, code catalogs |
| [Offline sandbox](#offline-sandbox) | Sandbox handle surface |
| [Exceptions](#exceptions) | Raised error types |

---

## Exported symbols

Import from `aetherdialect`. Authoritative list: `aetherdialect.__all__` (aligned with `_PUBLIC_API` in `aetherdialect.py`).

| Symbol | Kind |
| --- | --- |
| `AetherEngine` | Facade |
| `AetherFederation` | Facade |
| `AetherSpace` | Descriptor |
| `EngineContext` | Dataclass |
| `FederationContext` | Dataclass |
| `SpaceContext` | Dataclass |
| `PipelineSession` | Session |
| `AsyncPipelineSession` | Session |
| `SessionStep` | Dataclass |
| `AuditEvent` | Dataclass |
| `Diagnostic` | Dataclass |
| `DataQualityReport` | Dataclass |
| `ConfigSnapshot` | Dataclass |
| `SchemaStatsSnapshot` | Dataclass |
| `SeedWarmupSummarySnapshot` | Dataclass |
| `QSimSummarySnapshot` | Dataclass |
| `MigrationPreview` | Dataclass |
| `ConfigError` | Exception |
| `ConnectionError` | Exception |
| `DatabasePingFailed` | Exception |
| `LlmTransientFailure` | Exception |
| `MigrationPendingError` | Exception |
| `MockFixtureMissingError` | Exception |
| `OwnerOnlyOperationError` | Exception |
| `RetryableError` | Exception |
| `SchemaAccessError` | Exception |
| `SessionActiveError` | Exception |
| `StatementTimeoutError` | Exception |
| `FederationConfigError` | Exception |
| `FederationDeclarationError` | Exception |
| `FederationIneligibleError` | Exception |
| `FederationInvariantError` | Exception |
| `FederationPartialFailureError` | Exception |
| `FederationRuntimeError` | Exception |
| `FederationMemberExecutionError` | Exception |
| `FederationCapExceededError` | Exception |
| `FederationJoinFanOutError` | Exception |
| `FederationMalformedMemberAnswerError` | Exception |
| `FederationMemberUnprofilableError` | Exception |
| `FederationTurnCancelledError` | Exception |
| `PersistedFederationInspection` | Dataclass |
| `BusinessKnowledgeEntry` | Dataclass |
| `PhaseProgressEvent` | Dataclass |
| `TablePreviewResult` | Dataclass |
| `PlanPreviewResult` | Dataclass |
| `UploadIngestResult` | Dataclass |
| `Sandbox` | Authoring environment |
| `inspect_tabular_upload` | Function |
| `PERMISSION_DENIED_USER_MESSAGE` | `str` constant |
| `__version__` | `str` |

## EngineContext

Frozen scope input to `AetherEngine`. **Setup:** [Getting started - EngineContext](GETTING_STARTED.md#step-4-wire-enginecontext-and-construct-the-engine). **Operator semantics:** [User guide - EngineContext](USER_GUIDE.md#enginecontext). **Build pipeline:** [How it works - Schema compilation](HOW_IT_WORKS.md#2-schema-compilation).

| Field | Type | Meaning |
| --- | --- | --- |
| `include` | `"tables" \| "views"` | Reflect base tables or views (default `"tables"`). `"both"` is rejected - run separate passes when both kinds are needed. |
| `allow_objects` / `deny_objects` | `frozenset[str]` | Table allow/deny lists. |
| `allow_columns` / `deny_columns` | `frozenset[str]` | Qualified `table.column` (or `*.column`) allow/deny lists. |
| `notes_file` | `str \| None` | Path to domain notes. |
| `sql_file` | `str \| None` | Path to guidance DDL. |

`EngineContext` is registered by constructing `AetherEngine(EngineContext(...))` for master scope, or by name via saved scope presets. The implicit master scope is selected by constructing with an `EngineContext` object whose `name` is `master`. Saved scope presets are registered by constructing `AetherEngine(EngineContext(name="...", ...))` with a non-`master` name and later selected by passing that preset name string to `AetherEngine(...)`.

`notes_file` and `sql_file` are set only on `EngineContext` at construction. They are **not** environment variables and are **not** flattened from `config_file` TOML.

`deny_columns` entries are absent from the built schema graph while the deny remains effective - unlike **restricted** or **hidden** sensitivity, which keep the column in the graph ([User guide - Sensitivity classification](USER_GUIDE.md#sensitivity-classification)).

## AetherEngine

Public facade for one database connection. The engine **name** is the connection identity (artifacts slug).

### Constructor

```python
AetherEngine(
    engine_context: EngineContext | str | None = None,
    *,
    artifacts_dir: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    connection: str | None = None,
    execution_engine: Any = None,
    native_connection: Any = None,
    source_selections: Mapping[str, Mapping[str, Any]] | None = None,
    audit_sink: Callable[[AuditEvent], None] | None = None,
    role: SchemaRole = "owner",
) -> None
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `engine_context` | `EngineContext \| str \| None` | Pass an `EngineContext` to define master scope (owner). Pass a `str` name to bind a saved scope preset (consumers must use a preset name only). When omitted, load persisted master from artifacts or raise `ConfigError`. |
| `artifacts_dir` | `str \| None` | Optional root; engine files under `<root>/aetherdialect/<slug>/` |
| `config_file` | `str \| os.PathLike[str] \| None` | TOML config path; when omitted, settings come from `os.environ` only |
| `connection` | `str \| None` | Named TOML connection sub-block when multiple databases share one engine type |
| `execution_engine` | `Any` | Optional SQLAlchemy engine (caller-owned pool or read replica) |
| `native_connection` | `Any` | Optional native DuckDB or SQLite connection for embedded in-memory DBs |
| `source_selections` | `Mapping[str, Mapping[str, Any]] \| None` | CSV file engine only: per-filename interpretation (`header_row`, `table_range`, `append_regions`, etc.) |
| `audit_sink` | `Callable[[AuditEvent], None] \| None` | Optional lifecycle audit callback |
| `role` | `"owner" \| "consumer"` | `owner` may mutate shared artifacts; `consumer` pins owner snapshot id |

Federation is configured through `AetherFederation`, not `AetherEngine`. Raises `ConfigError`, `ConnectionError`, `MigrationPendingError`, or other failures under [Exceptions](#exceptions).

### Methods

Methods marked **(master context only)** require the instance to be bound to the master engine context (typically `role="owner"` with a full `EngineContext`).

| Method | Returns | Description |
| --- | --- | --- |
| `apply_migration_map(path="schema_migration_map.json", *, config_file=None, engine_context, artifacts_dir, execution_engine=None, native_connection=None, role="owner")` classmethod | `AetherEngine` | Copies the editor map into `schema_migration_map.json`, then constructs `AetherEngine`. Pair with [User guide - Migration](USER_GUIDE.md#migration). |
| `preview_migration_map()` | `MigrationPreview` | Read-only preview of schema migration impact against stored artifacts. |
| `data_quality_report` (property) | `DataQualityReport \| None` | Upload validation report from successful file-engine construction. |
| `inspect_tabular_upload(path)` (module) | `DataQualityReport` | Inspect one CSV/Excel upload without constructing an engine. Raises `ConfigError` on fatal file failures. |
| `aetherspace(name, space_context=None)` **(master context only)** | `AetherSpace` | With `space_context`: owner define/overwrite snapshot (notes via `SpaceContext.notes_file`). Without: existence check. Cannot redefine `master`. |
| `export_aetherspace(name)` **(master context only)** | `Path` | JSON export of one named space (or implicit `master`) for review or apply. |
| `apply_aetherspace(name, *, source=None)` **(master context only)** | `AetherSpace` | Owner: persist one named space from the default export file or an explicit `source` path. Version mismatch is fatal with no migration. |
| `delete_aetherspace(name)` **(master context only)** | `bool` | Owner: remove one persisted named space snapshot (`master` cannot be deleted). |
| `export_engine_context(name)` **(master context only)** | `Path` | Read-only JSON dump of one saved scope preset (or implicit `master`). |
| `list_aetherspaces()` **(master context only)** | `tuple[str, ...]` | Saved space names plus implicit `master`. |
| `list_engine_contexts()` **(master context only)** | `tuple[str, ...]` | Saved scope-preset names plus implicit `master`. |
| `execute_sql(sql, params=None, *, as_dataframe=False)` | `list[tuple]` or `DataFrame` | Standalone validated `SELECT` through the active dialect and **engine context** scope (not aetherspace). |
| `export_schema_overrides()` | `Path` | Writes `./schema_overrides.json` atomically from the live graph. |
| `apply_schema_overrides()` | `None` | Validates `./schema_overrides.json`, mutates graph, persists cache and archives editor files. |
| `show_config()` | `ConfigSnapshot` | Redacted snapshot of engine, schema scope, database, and LLM settings. |
| `session(*, mode="writer", space="master")` | `PipelineSession` | Context manager; exit calls `reset()`. |
| `asession(*, mode="writer", space="master")` | `AsyncPipelineSession` | Same as `session` on worker threads. |
| `run_interactive(*, space="master")` | `None` | Prints to stdout; one question per call. Prefer `session` for services. |
| `run_seed_warmup(seed_filepath, interactive_gold=True, *, abort_on_gold_failure=False, max_kept_intents=2000)` | `None` | Full seed warmup; `max_kept_intents=None` keeps every intent that passes quality/dedup. |
| `run_seed_warmup_from_history(sql_history_filepath, *, expand=False, max_kept_intents=2000)` | `None` | SQL-history warmup. |
| `run_seed_warmup_from_query_log(lookback_days=730, max_queries=5000, *, expand=False, max_kept_intents=2000, min_runs=1, user_filter=None)` | `None` | Warehouse query-log warmup. |
| `get_schema_stats()` | `SchemaStatsSnapshot` | Copy of internal graph counters. |
| `write_queue_path` (property) | `Path` | Absolute path to `write_queue.jsonl`. |
| `get_seed_warmup_summary()` | `SeedWarmupSummarySnapshot` | Newest seed-warmup summary file if present. |
| `get_qsim_summary(start, end)` | `QSimSummarySnapshot` | QSim summary index for inclusive version range. |
| `get_questions_only(version)` | `None` | Prints numbered questions and writes `qsim_v{version}_questions.txt`. |
| `run_qsim(num_intents=20, num_questions=100, seed=None)` | `None` | QSim generator; prints summary. |
| `clear_persisted_overrides()` | `bool` | Removes overrides sidecar and schema cache when present; rebuilds. |
| `clear_template_store()` | `bool` | Removes template tree; rebuilds. |
| `clear_simulation_caches()` | `int` | Deletes QSim and seed-warmup artifacts; returns removed file count. |
| `clear_all_learning(*, keep_overrides=True)` | `None` | Clears templates, simulation caches, and optionally overrides. |
| `offline_sandbox(...)` classmethod | `SandboxHandle` | Offline practice handle ([Offline sandbox](#offline-sandbox)). |
| `sandbox_questions()` classmethod | `list` | Curated offline practice questions. |
| `sandbox_paraphrase_pairs()` classmethod | `list` | Canonical->paraphrase pairs from the bundled catalog. |
| `sandbox_validation_failure_demo()` classmethod | object | Questions that should end in terminal validation errors. |
| `sandbox_feedback_demo()` classmethod | object | Anchor question + allowed rejection text. |

`AetherEngine` has no `close()` method - release sessions via context managers and let your application dispose database connections. Only `AetherFederation.close()` and `SandboxHandle.close()` tear down federation-owned runtimes or sandbox temp directories.

## FederationContext

Frozen composite scope for `AetherFederation(..., context=...)`. **Operator semantics:** [User guide - FederationContext](USER_GUIDE.md#federationcontext).

| Field | Type | Meaning |
| --- | --- | --- |
| `include` | `"tables" \| "views"` | Include mode (default `"tables"`). `"both"` is rejected. |
| `allow_objects` / `deny_objects` | `frozenset[str]` | Composite table allow/deny lists. |
| `allow_columns` / `deny_columns` | `frozenset[str]` | Qualified `table.column`, `source.table.column`, or `*.column`. |
| `notes_file` | `str \| None` | Composite domain notes (master federation context). |

## AetherFederation

Composite facade over named member `AetherEngine` instances. The federation **name** must match `federation_id` in the manifest; artifact tree is `fed_<federation_id>/`.

### Constructor

```python
AetherFederation(
    name: str,
    *,
    members: Mapping[str, AetherEngine],
    declaration_file: str,
    context: FederationContext | None = None,
    artifacts_dir: str | None = None,
    role: SchemaRole = "owner",
    audit_sink: Callable[[AuditEvent], None] | None = None,
) -> None
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Stable federation name; must match `federation_id` in the declaration JSON; artifact tree is `fed_<federation_id>/` |
| `members` | `Mapping[str, AetherEngine]` | Member engines keyed by federation ``source_id`` (the name used in joins, logical tables/columns, aliases, and diagnostics) |
| `declaration_file` | `str` | Required path to authored `federation_declaration.json` (joins, aliases, coordinator caps, optional logical mappings) |
| `context` | `FederationContext \| None` | Optional composite scope |
| `artifacts_dir` | `str \| None` | Root for member trees and the federation tree |
| `role` | `"owner" \| "consumer"` | Owner may change declarations; consumer reads the composite |
| `audit_sink` | `Callable[[AuditEvent], None] \| None` | Optional lifecycle audit callback |

Declaration format and annotated example: [Sandbox - Federation declaration format](SANDBOX.md#federation-declaration-format). Persisted sidecar shapes: [Federation and migration JSON](#federation-and-migration-json).

The `members` argument is a **mapping from federation ``source_id`` to member engine**. Keys are the names you use in the declaration, in `add_engine` / `remove_engine`, and in `export_schema_overrides(source_id)`; they are not the TOML ``connection=`` sub-block on each engine. When an engine's federation handle (`_connection`) is set, it must equal its registration key. Values are fully constructed `AetherEngine` instances (each with its own database connection and member artifact tree under the shared `artifacts_dir` root).

```python
from aetherdialect import AetherEngine, AetherFederation, EngineContext

storefront = AetherEngine(
    EngineContext(notes_file="rental_shop_notes.txt"),
    artifacts_dir="/data/artifacts",
    config_file="aetherdialect.toml",
    connection="storefront_pg",
)
catalog = AetherEngine(
    EngineContext(notes_file="rental_shop_notes.txt"),
    artifacts_dir="/data/artifacts",
    config_file="aetherdialect.toml",
    connection="catalog_mysql",
)

fed = AetherFederation(
    "sandbox_rental_shop",
    members={
        "storefront": storefront,
        "catalog": catalog,
    },
    declaration_file="/path/to/federation_declaration.json",
    artifacts_dir="/data/artifacts",
)
```

Raises `FederationConfigError`, `FederationDeclarationError`, `FederationInvariantError`, or `MigrationPendingError` on misconfiguration. Embedding narrative: [Integrator guide - Embedding a federation](INTEGRATOR_GUIDE.md#embedding-a-federation).

### Methods

| Method | Returns | Description |
| --- | --- | --- |
| `add_engine(connection_name, engine)` | `None` | Owner: register member, recompose, persist. |
| `remove_engine(connection_name)` | `None` | Owner: remove member, prune plan templates, recompose. |
| `export_federation_declaration()` | `Path` | Owner: write authored `federation_declaration.json` to the working directory. |
| `apply_federation_declaration()` | `None` | Owner: apply the full authored declaration from working-directory `federation_declaration.json` and recompose. |
| `apply_migration_map(path="federation_migration_map.json")` | `None` | Owner: copy a federation migration map into the working directory and recompose. |
| `aetherspace(name, space_context=None)` | `AetherSpace` | Define or check a space on the composite graph (notes via `SpaceContext.notes_file`). |
| `session(*, mode="writer", space="master")` | `PipelineSession` | Same session contract as `AetherEngine`. |
| `asession(*, mode="writer", space="master")` | `AsyncPipelineSession` | Async session. |
| `run_interactive(*, space="master")` | `None` | Stdout interactive turn. |
| `show_config()` | `ConfigSnapshot` | Redacted federation topology snapshot. |
| `get_schema_stats()` | `SchemaStatsSnapshot` | Composite graph counters. |
| `export_schema_overrides(connection_name)` | `Path` | Member-scoped override export. |
| `apply_schema_overrides(connection_name)` | `None` | Member-scoped override apply, then recompose. |
| `clear_persisted_overrides(connection_name)` | `bool` | Member-scoped clear, then recompose. |
| `clear_template_store()` | `bool` | Clear composite, plan-record, and member templates. |
| `clear_simulation_caches()` | `int` | Clear federation and member QSim / seed-warmup artifacts. |
| `clear_all_learning(*, keep_overrides=True)` | `None` | Drain queues and clear federation + member learning. |
| `run_qsim(...)` | `None` | QSim routed through federation decomposition checks. |
| `run_seed_warmup(seed_filepath, interactive_gold=True, *, abort_on_gold_failure=False, max_kept_intents=2000)` | `None` | Routes seed-question warmup through federation decompose/combine: per-member SQL uses each member dialect; learning lands in owning member stores plus a composite plan template. |
| `run_seed_warmup_from_history(...)` / `run_seed_warmup_from_query_log(...)` | - | Raise `FederationConfigError` - SQL history and query logs are per-engine artifacts; run them on each member `AetherEngine`. |
| `execute_sql(...)` | - | Raise `ConfigError` (unsupported at federation scope). |
| `close()` | `None` | Dispose federation-owned source runtimes and release member connection handles the federation created. Idempotent. Does not call `close()` on member `AetherEngine` instances you passed in - only runtimes owned by the federation object. Call when tearing down a long-lived service. |

## SpaceContext

Frozen knowledge scope for [AetherSpace](#aetherspace) definitions. **Conceptual guide:** [User guide - AetherSpace](USER_GUIDE.md#aetherspace). Parallel to `EngineContext` and `FederationContext`: allow/deny lists plus optional `notes_file`.

| Field | Type | Meaning |
| --- | --- | --- |
| `tables` | `frozenset[str]` | Allowed table/view names (empty means no extra table filter beyond master). |
| `columns` | `frozenset[str]` | Qualified `table.column` allow list for the space. |
| `deny_objects` | `frozenset[str]` | Tables/views excluded from space knowledge. |
| `deny_columns` | `frozenset[str]` | Qualified `table.column` deny list for the space. |
| `notes_file` | `str \| None` | Optional path to domain notes. Content (text plus hash) is baked into the aetherspace snapshot; it does **not** enter a catalog-rebuild fingerprint. Defining a space does not rebuild engine artifacts. |

Every table/column must exist on the master graph at write time. There is no TOML block for spaces - snapshots persist under the engine or federation storage directory.

## AetherSpace

Read-only descriptor returned by `AetherEngine.aetherspace(name)` / `AetherFederation.aetherspace(name)` (existence check) or after a successful define/overwrite. The space **name** identifies this object. No `session()` method - select the space via `session(..., space=name)`.

| Member / method | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Normalised space name |
| `list_scope()` | `dict[str, tuple[str, ...]]` | Keys `"tables"` and `"columns"` with tuple values |
| `notes` | `str \| None` | Optional merged notes text when `SpaceContext.notes_file` was set at define time |

## SessionStep

Return type of `PipelineSession.ask` / `step` and the async equivalents. Suspend `kind` values and the embedding loop: [Integrator guide - Session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

| Field | Type | Meaning |
| --- | --- | --- |
| `done` | `bool` | `True` when the turn finished (success or terminal error). |
| `prompt` | `str \| None` | Short line to show before collecting input. |
| `kind` | `str` | Public stage id (`awaiting_intent_confirm`, `result`, `error`, ...). |
| `sql` | `str \| None` | SQL under discussion or display glue. |
| `data` | `DataFrame \| None` | Preview or full result rows. |
| `message` | `str \| None` | Optional multi-line body. |
| `error` | `str \| None` | Terminal failure text when `done` and failed. |
| `intent_summary` | object \| `None` | Compact intent headline when present. |
| `diagnostics` | `tuple[Diagnostic, ...]` | Turn-level tracing rows. |
| `status` | `str \| None` | Coarse failure category on terminal error steps. |
| `reply_shape` | `"yes_no" \| "free_text" \| None` | Input shape while suspended. |
| `semantic_warnings` | `tuple[str, ...]` | Intent-confirmation warnings. |
| `interpretation` | object \| `None` | Structured interpretation when present. |
| `parameters` | `tuple` | Parameter bindings when present. |
| `federated_bundle` | object \| `None` | Per-member statements on federated turns. |
| `federation_source_id` | `str \| None` | Member `source_id` on federation terminal errors that name a failing member. |
| `federation_phase` | `str \| None` | Federation stage (`member` or `coordinator`) when a terminal error is federation-scoped. |
| `federation_limit_key` | `str \| None` | Cap name (for example `row_cap`) when `FederationCapExceededError` surfaces on the step. |
| `federation_succeeded` | `tuple[tuple[str, int, str], ...]` | On partial failure, tuples describing members that completed before the failing member (`source_id`, row count, phase). Empty on other errors. |

## Configuration

### Merge order

1. Start from a string copy of `os.environ`.
2. When **config_file** is omitted, that copy is the effective environment for `AetherEngine`.
3. When **config_file** is provided, the parsed TOML is the single source of truth for every field listed in the flattening table that appears in the file: non-empty values replace `os.environ`, and fields present with empty values remove the corresponding key from the effective mapping so shell-inherited secrets cannot override an explicit empty assignment. Keys for sections or fields absent from the file keep their `os.environ` values.

The library never mutates `os.environ` during reads. When a TOML key shadows an environment value, one diagnostic per key is emitted with code `CONFIG_FILE_VALUE_APPLIED`.

### `config_file` TOML flattening

Every value is coerced with `str(...)`. Absent sections are skipped. Fields present with empty strings suppress inherited environment values for the mapped key. Parse failures raise `ConfigError`. The **Flattened environment key** column is the key written by TOML flattening; aliases are accepted when reading the effective environment.

| TOML section.key | Flattened environment key | Accepted aliases | Default (if unset) | Required |
| --- | --- | --- | --- | --- |
| `[openai].api_key` | `OPENAI_API_KEY` | - | - | yes when OpenAI is the active LLM stack |
| `[openai].base_url` | `OPENAI_BASE_URL` | - | - | no |
| `[azure_openai].endpoint` | `AZURE_OPENAI_ENDPOINT` | - | - | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].api_key` | `AZURE_OPENAI_API_KEY` | - | - | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].api_version` | `AZURE_OPENAI_API_VERSION` | - | - | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].base_url` | `AZURE_OPENAI_BASE_URL` | - | - | no |
| `[azure_openai.deployments].light` | `AZURE_OPENAI_DEPLOYMENT_LIGHT` | - | - | yes when Azure OpenAI is the active LLM stack; provision a `gpt-5-mini` deployment |
| `[azure_openai.deployments].heavy` | `AZURE_OPENAI_DEPLOYMENT_HEAVY` | - | - | yes when Azure OpenAI is the active LLM stack; provision a `gpt-5.4-mini` deployment |
| `[sqlite].path` | `SQLITE_PATH` | `SQLITE_DATABASE`, `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE` | `:memory:` | yes when SQLite is selected (path key for selection) |
| `[sqlite].database` | `SQLITE_DATABASE` | `SQLITE_PATH`, `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE` | `:memory:` | alias of `path` |
| `[duckdb].path` | `DUCKDB_PATH` | `DUCKDB_DATABASE`, `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN` | `:memory:` | yes when DuckDB is selected (path key for selection) |
| `[duckdb].database` | `DUCKDB_DATABASE` | `DUCKDB_PATH`, `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN` | `:memory:` | alias of `path` |
| `[duckdb].schema` | `DUCKDB_SCHEMA` | `DUCKDB_DEFAULT_SCHEMA` | `main` | no |
| `[csv].directory` | `CSV_DIRECTORY` | - | - | yes when CSV is selected (mutually exclusive with `files`) |
| `[csv].files` | `CSV_FILES` | - | - | yes when CSV is selected (comma-separated paths or TOML array; mutually exclusive with `directory`) |
| `[excel].directory` | `CSV_DIRECTORY` | - | - | alias of `[csv]`; selects the CSV file engine |
| `[excel].files` | `CSV_FILES` | - | - | alias of `[csv].files` |
| `[mysql].host` | `MYSQL_HOST` | `MYSQL_SERVER`, `MYSQL_HOSTNAME` | `localhost` | no |
| `[mysql].port` | `MYSQL_PORT` | `MYSQL_TCP_PORT` | `3306` | no |
| `[mysql].user` | `MYSQL_USER` | `MYSQL_USERNAME` | `root` | yes when MySQL is selected |
| `[mysql].password` | `MYSQL_PASSWORD` | `MYSQL_PWD` | - | yes when MySQL is selected |
| `[mysql].database` | `MYSQL_DATABASE` | `MYSQL_DB` | - | yes when MySQL is selected |
| `[mariadb].host` | `MARIADB_HOST` | `MARIADB_SERVER`, `MARIADB_HOSTNAME` | `localhost` | no |
| `[mariadb].port` | `MARIADB_PORT` | `MARIADB_TCP_PORT` | `3306` | no |
| `[mariadb].user` | `MARIADB_USER` | `MARIADB_USERNAME` | `root` | yes when MariaDB is selected |
| `[mariadb].password` | `MARIADB_PASSWORD` | `MARIADB_PWD` | - | yes when MariaDB is selected |
| `[mariadb].database` | `MARIADB_DATABASE` | `MARIADB_DB` | - | yes when MariaDB is selected |
| `[sqlserver].host` | `SQLSERVER_HOST` | `SQLSERVER_SERVER`, `MSSQL_HOST`, `MSSQL_SERVER` | `localhost` | no |
| `[sqlserver].port` | `SQLSERVER_PORT` | `MSSQL_PORT` | `1433` | no |
| `[sqlserver].user` | `SQLSERVER_USER` | `SQLSERVER_USERNAME`, `MSSQL_USER` | - | yes when SQL auth is used |
| `[sqlserver].password` | `SQLSERVER_PASSWORD` | `SQLSERVER_PWD`, `MSSQL_SA_PASSWORD`, `MSSQL_PASSWORD` | - | yes when SQL auth is used |
| `[sqlserver].database` | `SQLSERVER_DATABASE` | `SQLSERVER_DB`, `MSSQL_DATABASE`, `MSSQL_DB` | - | yes when SQL Server is selected |
| `[sqlserver].schema` | `SQLSERVER_SCHEMA` | `MSSQL_SCHEMA`, `SQLSERVER_DEFAULT_SCHEMA` | `dbo` | no |
| `[sqlserver].driver` | `SQLSERVER_DRIVER` | `MSSQL_DRIVER`, `ODBC_DRIVER` | `ODBC Driver 18 for SQL Server` | no |
| `[sqlserver].auth_mode` | `SQLSERVER_AUTH_MODE` | `MSSQL_AUTH_MODE` | `sql` | no |
| `[sqlserver].tenant_id` | `SQLSERVER_TENANT_ID` | `MSSQL_TENANT_ID`, `AZURE_TENANT_ID` | - | yes when Azure AD password auth is used |
| `[sqlserver].client_id` | `SQLSERVER_CLIENT_ID` | `MSSQL_CLIENT_ID`, `AZURE_CLIENT_ID` | - | yes when Azure AD service principal auth is used |
| `[sqlserver].client_secret` | `SQLSERVER_CLIENT_SECRET` | `MSSQL_CLIENT_SECRET`, `AZURE_CLIENT_SECRET` | - | yes when Azure AD service principal auth is used |
| `[postgresql].host` | `POSTGRES_HOST` | `POSTGRES_SERVER`, `POSTGRES_HOSTNAME`, `PGHOST`, `PGHOSTADDR` | `localhost` | no |
| `[postgresql].port` | `POSTGRES_PORT` | `PGPORT` | `5432` | no |
| `[postgresql].database` | `POSTGRES_DB` | `POSTGRES_DATABASE`, `PGDATABASE` | - | yes when PostgreSQL is selected |
| `[postgresql].schema` | `POSTGRES_SCHEMA` | `PGSCHEMA` | `public` | no |
| `[postgresql].user` | `POSTGRES_USER` | `POSTGRES_USERNAME`, `PGUSER` | `postgres` | yes when PostgreSQL is selected |
| `[postgresql].password` | `POSTGRES_PASSWORD` | `POSTGRES_PWD`, `PGPASSWORD` | - | yes when PostgreSQL is selected |
| `[redshift].host` | `REDSHIFT_HOST` | `REDSHIFT_SERVER` | `localhost` | no |
| `[redshift].port` | `REDSHIFT_PORT` | `REDSHIFT_TCP_PORT` | `5439` | no |
| `[redshift].user` | `REDSHIFT_USER` | `REDSHIFT_USERNAME` | `awsuser` | yes when Redshift is selected |
| `[redshift].password` | `REDSHIFT_PASSWORD` | `REDSHIFT_PWD` | - | yes when password auth is used |
| `[redshift].database` | `REDSHIFT_DATABASE` | `REDSHIFT_DB` | `dev` | yes when password auth is used |
| `[redshift].schema` | `REDSHIFT_SCHEMA` | - | `public` | no |
| `[redshift].use_iam` | `REDSHIFT_USE_IAM` | `REDSHIFT_IAM` | `false` | no |
| `[redshift].cluster_identifier` | `REDSHIFT_CLUSTER_IDENTIFIER` | `REDSHIFT_CLUSTER_ID` | - | yes when IAM auth is used (cluster) |
| `[redshift].workgroup` | `REDSHIFT_WORKGROUP` | `REDSHIFT_SERVERLESS_WORKGROUP` | - | yes when IAM auth is used (serverless) |
| `[redshift].region` | `REDSHIFT_REGION` | `REDSHIFT_AWS_REGION` | - | no |
| `[databricks].host` | `DATABRICKS_HOST` | `DATABRICKS_SERVER`, `DATABRICKS_SERVER_HOSTNAME` | - | yes when Databricks is selected |
| `[databricks].http_path` | `DATABRICKS_HTTP_PATH` | `DATABRICKS_SQL_HTTP_PATH`, `DATABRICKS_WAREHOUSE_HTTP_PATH` | - | yes when Databricks is selected |
| `[databricks].access_token` | `DATABRICKS_ACCESS_TOKEN` | `DATABRICKS_TOKEN`, `DATABRICKS_PAT`, `ACCESS_TOKEN` | - | yes when Databricks is selected |
| `[databricks].catalog` | `DATABRICKS_CATALOG` | `SPARK_DEFAULT_CATALOG` | - | no |
| `[databricks].schema` | `DATABRICKS_SCHEMA` | `DATABRICKS_DEFAULT_SCHEMA`, `SPARK_DEFAULT_SCHEMA` | - | no |
| `[snowflake].account` | `SNOWFLAKE_ACCOUNT` | `SNOWSQL_ACCOUNT`, `SF_ACCOUNT` | - | yes when Snowflake is selected |
| `[snowflake].user` | `SNOWFLAKE_USER` | `SNOWFLAKE_USERNAME`, `SNOWSQL_USER` | - | yes when Snowflake is selected |
| `[snowflake].password` | `SNOWFLAKE_PASSWORD` | `SNOWFLAKE_PWD`, `SNOWSQL_PWD` | - | yes when password auth is used |
| `[snowflake].database` | `SNOWFLAKE_DATABASE` | `SNOWFLAKE_DB`, `SNOWSQL_DATABASE` | - (optional; slug uses `db` when unset) | no |
| `[snowflake].schema` | `SNOWFLAKE_SCHEMA` | `SNOWSQL_SCHEMA`, `SNOWFLAKE_DEFAULT_SCHEMA` | `PUBLIC` | no |
| `[snowflake].warehouse` | `SNOWFLAKE_WAREHOUSE` | `SNOWSQL_WAREHOUSE` | - | no |
| `[snowflake].role` | `SNOWFLAKE_ROLE` | `SNOWSQL_ROLE` | - | no |
| `[snowflake].private_key_path` | `SNOWFLAKE_PRIVATE_KEY_PATH` | `SNOWFLAKE_PRIVATE_KEY`, `SNOWSQL_PRIVATE_KEY_PATH` | - | yes when key-pair auth is used |
| `[snowflake].private_key_passphrase` | `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE` | `SNOWSQL_PRIVATE_KEY_PASSPHRASE` | - | no |
| `[snowflake].authenticator` | `SNOWFLAKE_AUTHENTICATOR` | `SNOWSQL_AUTHENTICATOR` | - | no |
| `[snowflake].oauth_token` | `SNOWFLAKE_OAUTH_TOKEN` | `SNOWFLAKE_OAUTH`, `SNOWSQL_OAUTH_TOKEN` | - | yes when OAuth auth is used |
| `[bigquery].project` | `BIGQUERY_PROJECT` | `GOOGLE_CLOUD_PROJECT`, `GCP_PROJECT` | - | yes when BigQuery is selected |
| `[bigquery].dataset` | `BIGQUERY_DATASET` | `BIGQUERY_DB`, `GCP_DATASET`, `BIGQUERY_SCHEMA`, `BQ_DATASET` | - | yes when BigQuery is selected |
| `[bigquery].credentials_path` | `BIGQUERY_CREDENTIALS_PATH` | `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_CREDENTIALS_PATH`, `BQ_CREDENTIALS_PATH` | - | no (ADC used when unset) |
| `[bigquery].location` | `BIGQUERY_LOCATION` | `GCP_LOCATION`, `BQ_LOCATION`, `GOOGLE_CLOUD_LOCATION` | `US` | no |
| `[engine].selected` | `AETHERDIALECT_ENGINE` | - | - | yes when multiple database engines are configured |
| `[engine].connection` | `AETHERDIALECT_CONNECTION` | - | - | yes when multiple named connection sub-tables exist for the selected engine |
| `[llm].provider` | `AETHERDIALECT_LLM_PROVIDER` | - | - | yes when both LLM stacks are configured |
| `[mock].fixtures_file` | `AETHERDIALECT_MOCK_FIXTURES_FILE` | - | - | yes when `[llm] provider = "mock"` |
| `[execution].max_query_cost_rows` | `AETHERDIALECT_MAX_QUERY_COST_ROWS` | - | `50000000` | no |
| `[execution].max_query_cost_bytes` | `AETHERDIALECT_MAX_QUERY_COST_BYTES` | - | `50000000000` | no |
| `[execution].statement_timeout_ms` | `AETHERDIALECT_STATEMENT_TIMEOUT_MS` | - | `30000` | no |
| `[execution].llm_timeout_ms` | `AETHERDIALECT_LLM_TIMEOUT_MS` | - | `60000` | no |
| `[execution].profile_timeout_ms` | `AETHERDIALECT_PROFILE_TIMEOUT_MS` | - | `120000` | no |
| `[execution].explain_timeout_ms` | `AETHERDIALECT_EXPLAIN_TIMEOUT_MS` | - | falls back to `statement_timeout_ms` | no |
| - | `AETHERDIALECT_LLM_BATCH_ENABLED` | - | `false` | no; OpenAI-only offline corpus batching |

On **BigQuery**, when `AETHERDIALECT_MAX_QUERY_COST_BYTES` is active, execution sets `maximum_bytes_billed` on query jobs from that cap (in addition to optional `job_timeout_ms` from `statement_timeout_ms`).

`AETHERDIALECT_ENGINE` / `[engine] selected` accepts: `sqlite`, `duckdb`, `csv`, `mysql`, `mariadb`, `sqlserver`, `postgresql`, `redshift`, `databricks`, `snowflake`, `bigquery`. The TOML `[excel]` section aliases to the `csv` engine; do not set `AETHERDIALECT_ENGINE=excel`. When more than one engine block is complete and selectable, `[engine] selected` is required.

When Snowflake `database` is unset, `apply_environment` leaves it empty and the connection slug uses the placeholder segment `db`.

### Database connection reference

Optional DDL for join hints is supplied only via `EngineContext.sql_file` at construction (not env/TOML). Connection slug shapes: [How it works - Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-fingerprints).

| Engine | SQLAlchemy URL form (conceptual) | Primary env keys | Notes |
| --- | --- | --- | --- |
| SQLite | `sqlite:///{path}` (`sqlite:///:memory:` in-memory) | `SQLITE_PATH` or `SQLITE_DATABASE` | Path aliases: `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE`. Schema is always `main`. Stdlib `pysqlite`. |
| DuckDB | `duckdb:///{path}` (`duckdb:///:memory:` in-memory) | `DUCKDB_PATH` or `DUCKDB_DATABASE` | Path aliases: `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN`. Optional `DUCKDB_SCHEMA` / `DUCKDB_DEFAULT_SCHEMA` (default `main`). |
| CSV | In-memory DuckDB backend (no persistent URL) | `CSV_DIRECTORY` or `CSV_FILES` | Mutually exclusive. Accepts `.csv` and `.xlsx` (not `.xls`). `[excel]` TOML aliases here. Reflection rebuilds the graph per session. |
| MySQL | `mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4` | `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | Host aliases: `MYSQL_SERVER`, `MYSQL_HOSTNAME`. Port: `MYSQL_TCP_PORT`. User: `MYSQL_USERNAME`. Password: `MYSQL_PWD`. Database: `MYSQL_DB`. |
| MariaDB | `mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4` | `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE` | **MARIADB_* only** - no `MYSQL_*` fallback. Same sqlglot read=`mysql` backend as MySQL. |
| SQL Server | `mssql+pyodbc://...` (auth-mode dependent) | `SQLSERVER_HOST`, `SQLSERVER_PORT`, `SQLSERVER_DATABASE`, `SQLSERVER_USER`, `SQLSERVER_PASSWORD` | `MSSQL_*` and `AZURE_*` aliases in the flattening table above. `SQLSERVER_AUTH_MODE`: `sql` (default), `windows`, `aad_password`, `aad_sp`. |
| PostgreSQL | `postgresql+psycopg://{user}:{password}@{host}:{port}/{database}` | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE` / `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | `PG*` and `POSTGRES_*` aliases in the flattening table above. Optional `POSTGRES_SCHEMA` (default `public`). |
| Redshift | `redshift+redshift_connector://...` | `REDSHIFT_HOST`, `REDSHIFT_PORT`, `REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `REDSHIFT_DATABASE` | **Redshift-native aliases only** (no `PGHOST` / `PGUSER`). IAM: `REDSHIFT_USE_IAM`, `REDSHIFT_CLUSTER_IDENTIFIER` or `REDSHIFT_WORKGROUP`, optional `REDSHIFT_REGION`. |
| Databricks | SQL warehouse via `databricks-sql-connector`; optional SQLAlchemy URL | `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_ACCESS_TOKEN` | Token aliases: `DATABRICKS_TOKEN`, `DATABRICKS_PAT`, `ACCESS_TOKEN`. Optional `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`. **Fallback:** Databricks connector -> SQLAlchemy -> PySpark / `DatabricksSession` when the warehouse trio is absent or connect fails. |
| Snowflake | `snowflake://{auth}@{account}/?...` | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER` | `database` optional (slug uses `db` when unset). Password, key-pair, OAuth, or external-browser auth per env. **Fallback:** Snowflake connector (Arrow) -> SQLAlchemy -> Snowpark active session. |
| BigQuery | `bigquery://{project}/{dataset}?location=...` | `BIGQUERY_PROJECT`, `BIGQUERY_DATASET` | Dataset aliases: `BIGQUERY_DB`, `GCP_DATASET`, `BQ_DATASET`. ADC when `BIGQUERY_CREDENTIALS_PATH` unset. **Result fetch:** `bq_storage` -> `bq_client` -> SQLAlchemy. |

## PipelineSession methods

Sync session API on `AetherEngine.session()` and `AetherFederation.session()`. One **turn** is a question from `ask` through terminal `done` (success or error). While `step.done` is `False`, the engine is **suspended** and expects your next input via `step(...)`.

| Pattern | When to use |
| --- | --- |
| `ask(question)` + `step(response)` loop | Production UIs that show each suspend prompt and collect real user input. |
| `ask_until_done(question, on_confirm="y")` | Scripts that auto-confirm yes/no suspends only; raises if a free-text suspend appears. |
| `accept_until_done(question)` | Offline sandbox and smoke tests: auto-confirms yes/no and free-text suspends until the turn ends. |

| Method | Returns | Contract |
| --- | --- | --- |
| `ask(question: str)` | `SessionStep` | Starts a turn; raises `SessionActiveError` if busy. |
| `step(response=None)` | `SessionStep` | Supplies user text for a suspend (`"y"` / `"n"` for yes/no, or free text for feedback). |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep` | Auto-answers yes-or-no suspends; `on_confirm` is `"y"` or `"n"`; raises on free-text suspends. |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | `SessionStep` | Auto-answers yes-or-no and free-text suspends until the turn ends. |
| `awaiting_prompt()` | `bool` | `True` when the next input must go to `step`. |
| `reset()` | `None` | Clears suspend state and partial turn state. |
| `cancel_active_federation_turn()` | `bool` | Cooperative cancel between member stages/batches; does **not** interrupt an already-running DB statement. Returns `True` only when this session has an active turn. |
| `__enter__` / `__exit__` | `PipelineSession` / `None` | Exit calls `reset()`. |

`mode="reader"` skips durable template and feedback writes. `mode="writer"` is the default and serialises writer turns with a per-instance lock. Suspend `kind` values and the embedding loop: [Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).

## AsyncPipelineSession methods

| Method | Returns | Contract |
| --- | --- | --- |
| `ask(question)` | `SessionStep` | Delegates to `PipelineSession.ask` on a worker thread. |
| `step(response=None)` | `SessionStep` | Delegates to `PipelineSession.step`. |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep` | Delegates to `PipelineSession.ask_until_done`. |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | `SessionStep` | Delegates to `PipelineSession.accept_until_done`. |
| `reset()` | `None` | Delegates via `asyncio.to_thread`. |
| `awaiting_prompt()` | `bool` | Delegates via `asyncio.to_thread`. |
| `cancel_active_federation_turn()` | `bool` | Cooperative cancel between member stages/batches; does **not** interrupt an already-running DB statement. Returns `True` only when that session has an active turn. |
| `__aenter__` / `__aexit__` | `AsyncPipelineSession` / `bool` | Async context manager forwarding to the inner session. |

## Package helpers

| Symbol | Signature / value | Notes |
| --- | --- | --- |
| `PERMISSION_DENIED_USER_MESSAGE` | `str` | Stable user-facing text when execution is blocked by scope. |
| `__version__` | `str` | Package version string (currently `0.2.0`). |
| `MigrationPreview` | dataclass | `tier` (`compatible` / `remap` / `destructive`), `affected_tables`, `affected_columns`, `skeleton_path`. |

Federation cancellation lives on `PipelineSession` / `AsyncPipelineSession.cancel_active_federation_turn`, not as a package-level helper.

## Schema overrides JSON (`schema_overrides.json`)

Version `2` in the bundled sandbox demo; export writes `version: 1` for hand-edited production files. Export writes `./schema_overrides.json` in the process working directory. Apply reads the same path. Workflow: [User guide - Schema overrides](USER_GUIDE.md#schema-overrides).

Hand-edited files for `apply_schema_overrides` should use **plain strings** for descriptions and roles and `null` or a string for `sensitivity`. Do **not** add `owner` keys; the engine treats missing owner as analyst content.

| Location | Editable fields |
| --- | --- |
| Each table | `description` (string), `role` (string) |
| Each column | `description` (string), `role` (string), `sensitivity` (`null` or string), `usable` (`false` only - marks column unusable; cannot re-enable profiler-omitted columns) |
| Graph | `foreign_keys_add[]`, `foreign_keys_remove[]` |
| Primary keys | `primary_keys_add[]`, `primary_keys_remove[]` |
| Internal persistence | `_internal` (block lists; engine-maintained, not hand-authored) |

```json
{
  "version": 2,
  "tables": {
    "staff": {
      "columns": {
        "ssn": {
          "sensitivity": "hidden"
        },
        "password": {
          "sensitivity": "hidden"
        }
      }
    },
    "film": {
      "description": "Rental Shop film catalog.",
      "role": null
    }
  },
  "foreign_keys_add": [],
  "foreign_keys_remove": [],
  "primary_keys_add": [],
  "primary_keys_remove": []
}
```

Bundled sandbox demo (`sandbox_overrides_demo.json` in the corpus) matches the shape above: `staff.ssn` and `staff.password` are **hidden** for sensitivity exercises; `film` gets an analyst description.

`_readonly` is regenerated on export and ignored on apply. Successful apply persists `applied_overrides.json` beside the gzip schema cache and archives the editor JSON to `*.applied.json`.

## Federation and migration JSON

### `federation_declaration.json`

Single authored input document (passed to `declaration_file=`). Combines manifest fields and optional logical mapping sections. Complete annotated example: [Sandbox - Federation declaration format](SANDBOX.md#federation-declaration-format).

### `federation_manifest.json` (persisted sidecar)

Persisted federation tree sidecar storing `federation_id`, joins, optional aliases, and coordinator caps.

| Field | Type | Meaning |
| --- | --- | --- |
| `federation_id` | `str` | Stable federation name; artifact tree is `fed_<federation_id>/` |
| `cross_source_joins[]` | array | `left`, `right` (`table.column`), `kind`, `logical_key` |
| `aliases` | object | Optional map of logical alias -> `{source, table}` |
| `coordinator` | object | Cap settings only: `row_cap`, `default_source_row_cap`, `default_source_timeout_ms`, `semijoin_key_cap`, `spill_row_threshold`, `max_parallel_members`, `total_input_byte_cap` (coordinator is always in-process DuckDB; no `engine` key) |

An `engine` key inside `coordinator` is rejected - the coordinator is always in-process DuckDB.

### `federation_mappings.json` (persisted sidecar)

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `int` | Sidecar format version |
| `logical_columns[]` | array | `logical`, `members` (`table.column` per source), `role`, `unify_in_graph` |
| `logical_tables[]` | array | `logical`, `semantics` (`union` or `replica`), `members[]` with `source`, `table`, `columns` |

For `replica` semantics, designate one member as the authoritative source in the mapping (required for `replica`; rejected for `union`).

### `schema_migration_map.json`

Written in the working directory when migration requires operator action; consumed on the next successful `AetherEngine(...)` construction and renamed to `schema_migration_map.applied.json` (timestamped variant when a prior applied file exists). Workflow: [User guide - Migration](USER_GUIDE.md#migration).

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `int` | Map format version (`1` today) |
| `action` | `str` | `remap`, `destructive`, or `abort` |
| `table_renames` | array | Objects with `from` / `to` table names |
| `column_renames` | array | Objects with `table`, `from`, `to` column names |
| `dropped_tables` | array of `str` | Tables removed from the warehouse |
| `dropped_columns` | array | Objects with `table` and `column` |
| `added_tables` | array of `str` | Tables added to the warehouse |
| `added_columns` | array | Objects with `table` and `column` |
| `refresh_existing_descriptions_on_addition` | `bool` | Default `false`; may refresh descriptions when tables are added |

### `federation_migration_map.json`

Consumed on the next successful federation construction when present in the working directory; archived to `federation_migration_map.applied.json` on success. Per-source drift still uses `schema_migration_map.json` on each member tree.

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `int` | Map format version (`1`) |
| `action` | `str` | `remap`, `destructive`, or `abort` |
| `qualified_column_renames` | array | Objects with `from` / `to` qualified `table.column` refs |
| `namespace_renames` | array | Objects with `from` / `to` logical table names in `table_namespace` |
| `dropped_cross_source_joins` | array | Objects with `left` / `right` join endpoints to remove |

Complete example (`schema_migration_map.json`):

```json
{
  "version": 1,
  "action": "remap",
  "table_renames": [],
  "column_renames": [
    {
      "table": "item",
      "from": "title",
      "to": "item_title"
    }
  ],
  "dropped_tables": [],
  "dropped_columns": [],
  "added_tables": [],
  "added_columns": [],
  "refresh_existing_descriptions_on_addition": false
}
```

The bundled sandbox migration demo uses the same `item.title` -> `item_title` remap ([Sandbox guide - Migration demo](SANDBOX.md#migration-demo)).

## Observability

Three channels: [Integrator guide - Observability](INTEGRATOR_GUIDE.md#observability). Types and catalogs only below.

### `audit_sink`

Optional `Callable[[AuditEvent], None] \| None` on `AetherEngine` / `AetherFederation` construction. Not a boolean flag and does not print by itself.

### `Diagnostic`

| Field | Type | Meaning |
| --- | --- | --- |
| `stage` | `str` | Pipeline stage that emitted the row |
| `level` | `str` | `info`, `warn`, or `error` |
| `code` | `str` | Stable code string (see catalog below) |
| `message` | `str` | Human-readable text |
| `details` | `tuple[tuple[str, str], ...]` | Optional key-value metadata |
| `duration_ms` | `int \| None` | Wall-clock duration when applicable |
| `source_id` | `str \| None` | Federation member id when applicable |

### `DataQualityReport`

Frozen outcome of CSV upload validation during `AetherEngine` construction (file engine only). Blocking issues raise `ConfigError` with `narrative` as the message before the engine instance exists.

| Field | Type | Meaning |
| --- | --- | --- |
| `ok` | `bool` | `True` when no blocking or review upload issues remain |
| `issues` | `tuple[Diagnostic, ...]` | Deterministic findings with exact `file!sheet!cell` locations |
| `narrative` | `str` | Plain-language summary (LLM when available) |
| `suggested_selections` | `dict[str, dict]` | Per-filename interpretation map for `source_selections` |
| `confirmed_selections` | `dict[str, dict]` | Caller-accepted interpretation (empty until supplied) |
| `requires_review` (property) | `bool` | `True` when any issue needs caller confirmation before construction |

`to_json_dict()` returns `{ok, narrative, issues, suggested_selections, confirmed_selections, requires_review}`. `to_dict()` returns `{heading: detail}` for UI consumers. On successful construction, `audit_sink` receives `event_type="data_quality"` with `ok` and `issue_count` in `details`.

### `AuditEvent`

| Field | Type | Meaning |
| --- | --- | --- |
| `event_type` | `str` | Discriminator (see catalog below) |
| `timestamp_iso` | `str` | UTC timestamp when emitted |
| `question` | `str \| None` | Active question when relevant |
| `schema_hash` | `str \| None` | Schema fingerprint when relevant |
| `provider` | `"openai" \| "azure" \| "mock"` | LLM provider for the event |
| `details` | `tuple[tuple[str, str], ...]` | Lightweight metadata |

#### Audit `event_type` catalog

**Session lifecycle**

| `event_type` | When emitted |
| --- | --- |
| `init` | After successful `AetherEngine` / `AetherFederation` construction |
| `data_quality` | After successful construction when the file engine validated uploads |
| `ask_begin` | Start of `session.ask(...)` |
| `ask_done` | Turn completed (`details` includes `outcome`, `kind`) |
| `ask_error` | Terminal failure or fatal guard error |
| `ask_blocked` | `ask` rejected before raise (non-`str` question or `SessionActiveError`) |

**Admin operations**

| `event_type` | When emitted |
| --- | --- |
| `apply_schema_overrides` | After `apply_schema_overrides` persists |
| `clear_persisted_overrides` | After overrides sidecar removal and rebuild |
| `clear_template_store` | After template tree removal and reload |
| `clear_simulation_caches` | After QSim or seed-warmup cache deletion |
| `clear_all_learning` | After combined learning clears |

**Write queue drain** (writer session only)

| `event_type` | When emitted |
| --- | --- |
| `write_queue_feedback_record` | Writer applied a queued `feedback_record` |
| `write_queue_template_reject` | Writer applied a queued `template_reject` |
| `write_queue_template_accept` | Writer applied a queued `template_accept` |
| `write_queue_override_proposal` | Writer materialised a queued `override_proposal` |

**LLM usage** (emitted at turn end when the turn made model calls)

| `event_type` | When emitted |
| --- | --- |
| `llm_call` | Once per model request in the turn. `details` include `task`, `logical_model`, `api_model`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `attempt`, `elapsed_ms`. When `provider` is `openai` and the logical model is in the shipped price table, `cost_usd` is present; unknown OpenAI models carry `unpriced` instead. Azure and mock providers emit token counts only - no `cost_usd`. |
| `llm_turn` | Once per completed turn summarising all `llm_call` rows for that turn. OpenAI summaries include `cost_usd` (sum of priced calls), `price_table_as_of`, and `unpriced_models` when any call lacked a rate. |

### LLM usage and cost

Turn totals also appear on `SessionStep.diagnostics` as code **`LLM_TURN_COST`** (`stage="llm"`, `level="info"`). `message` and `details` carry `requests`, `input_tokens`, `cached_input_tokens`, `output_tokens`, and - for `provider="openai"` only - `cost_usd` and `price_table_as_of`.

| Provider | `cost_usd` on diagnostic / audit | Notes |
| --- | --- | --- |
| `openai` | Yes when the logical model is in the built-in table | List-price estimate from input, cached-input, and output token counts |
| `azure` | No | Token counts only - deployment rates are tenant-specific |
| `mock` | No | Offline sandbox |

**Shipped OpenAI price table** (USD per million tokens; as-of date in `price_table_as_of` detail field):

| Logical model | Input | Cached input | Output |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | 0.40 | 0.10 | 1.60 |
| `gpt-5.4-nano` | 0.10 | 0.025 | 0.40 |
| `gpt-5-mini` | 0.25 | 0.0625 | 2.00 |
| `gpt-5-nano` | 0.05 | 0.0125 | 0.40 |
| `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 |
| `gpt-4.1-nano` | 0.10 | 0.025 | 0.40 |

The as-of date is **`2026-07-26`** for this build. Figures are indicative for operator dashboards, not billing authority.

Per-phase token breakdown may also appear as `LLM_TURN_COST` or `ENGINE_INFO` rows from `emit_llm_usage_summary_diagnostics` when enabled.

### Diagnostic code catalog

**Pipeline semantics**

| Code | Typical use |
| --- | --- |
| `REUSE_HIT` | Template reuse matched |
| `REUSE_MISS` | No template reuse hit |
| `LOW_CONFIDENCE` | Low-confidence execution path |
| `LARGE_RESULT_WARNING` | Large result-set warning |
| `SENSITIVITY_GATE_HIT` | Sensitive field gate during repair |
| `INTERPRET_GROUND_RETRY` | Logical intent retry |
| `COMPOSE_REPAIR` | Schema or semantic repair pass |
| `FALLBACK_FRESH_RESTART` | Fresh restart after repair exhaustion |
| `SCHEMA_OVERRIDE_SKIP` | Override entry skipped during apply |

**LLM usage**

| Code | Typical use |
| --- | --- |
| `LLM_TURN_COST` | Turn-total token (and OpenAI cost) summary on `SessionStep.diagnostics` |

**Configuration**

| Code | Typical use |
| --- | --- |
| `CONFIG_FILE_VALUE_APPLIED` | TOML value overrode `os.environ` for a key |

**Upload validation (file engine)**

| Code | Typical use |
| --- | --- |
| `DATA_QUALITY_BLOCKING` | Structural issue that blocks engine construction |
| `DATA_QUALITY_ADVISORY` | Issue reported but not blocking |
| `DATA_QUALITY_AUTO_READ` | Encoding/newline/whitespace read normalization |
| `DATA_QUALITY_AUTO_CORRECTED` | Lossless blank-border trim applied in pipeline |

**Federation diagnostics**

| Code | Typical use |
| --- | --- |
| `FEDERATION_INELIGIBLE` | Intent cannot be decomposed into a federated plan |
| `FEDERATION_PARTIAL_FAILURE` | One member failed after others succeeded |
| `FEDERATION_MEMBER_FAILED` | Member generation or execution failed |
| `FEDERATION_MEMBER_GENERATED` | Per-member SQL generated during prepare |
| `FEDERATION_MEMBER_EXECUTED` | Member statement executed |
| `FEDERATION_COORDINATOR_EXECUTED` | Coordinator combine finished |
| `FEDERATION_SOURCES_QUERIED` | Audit summary of sources touched in a turn |
| `FEDERATION_PLAN_REPLAY` | Question-level reuse replayed a stored plan template |
| `FEDERATION_SEMIJOIN_SKIPPED` | Semi-join reduction skipped (disabled or ineligible) |

**Catch-all**

| Code | Typical use |
| --- | --- |
| `ENGINE_INFO` | Progress, CLI echo, schema summaries |

**Validation (lowercase)** - `SessionStep.diagnostics` may also carry EXPLAIN- and validator-derived codes (for example `explain_seq_scan_indexed`). Treat unknown codes as opaque.

## Offline sandbox

Walkthrough: [Sandbox guide](SANDBOX.md).

### `Sandbox` authoring environment

`Sandbox` is exported from `aetherdialect`. It unpacks the bundled corpus, seeds in-memory DuckDB datasets, writes (or accepts) LLM configuration, and exposes production-shaped constructors.

| Member | Returns | Purpose |
| --- | --- | --- |
| `Sandbox(*, llm_config=None, artifacts_dir=None, bundle_dir=None, cleanup=True, auto_seed=True)` | context manager | Enter the environment |
| `sandbox.engine(engine_context=None, *, role="owner", include="tables")` | `AetherEngine` | Build an engine on the default `main` dataset |
| `sandbox.federation(federation_id, *, declaration_file=None, members=None, context=None)` | `AetherFederation` | Build a federation over named datasets |
| `sandbox.load_dataset(name, *, seed_sql=None, sql_file=None)` | `str` | Seed an additional in-memory database |
| `sandbox.datasets` | `tuple[str, ...]` | Currently loaded dataset names |
| `sandbox.connection(name="main")` | driver connection | DuckDB connection for manual `AetherEngine(...)` construction |
| `sandbox.artifacts_dir` | `str` | Shared artifacts root |
| `sandbox.config_file` | `str \| None` | TOML path (temporary mock config or caller `llm_config`) |
| `sandbox.adopt(engine)` | `None` | Apply sandbox mode to a caller-built engine |
| `sandbox.questions()` | `tuple[str, ...]` | Recorded corpus questions when mock mode is active |
| `sandbox.close()` | `None` | Release connections, temp extract dir, and owned artifacts |

### `AetherEngine.offline_sandbox` quick path

`AetherEngine.offline_sandbox(**kwargs)` forwards to `create_offline_sandbox` and returns a **`SandboxHandle`** (not a top-level `__all__` export). Use it when you want a working engine immediately; use `Sandbox()` when you want to construct engines yourself.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `engine_context` | `EngineContext \| None` | none | Scope, notes, and SQL overrides |
| `notes_file` / `sql_file` | `str \| None` | none | Override bundled notes/SQL paths |
| `llm_config` | `str \| os.PathLike \| None` | none | TOML path replacing bundled mock provider |
| `maintainer_access` | `bool` | `False` | Maintainer corpus access for bundled sandbox fixtures |
| `artifacts_dir` | `str \| None` | temp dir owned by handle | Share learning between sandbox instances |
| `cleanup_artifacts` | `bool` | `True` | Remove owned temp artifacts on `close()` |
| `seed_sql` | `str \| None` | none | Maintainer: alternate seed SQL for the main dataset |
| `deny_columns` | `frozenset[str] \| None` | none | Column-security practice |
| `include` | `"tables" \| "views"` | `"tables"` | Reflection target (`"views"` applies bundled view DDL) |
| `bundle_dir` | `str \| None` | none | Maintainer: read bundle files from a directory instead of extracting `data.zip` |
| `connection` | `Any \| None` | none | Reuse an existing DuckDB connection |
| `owns_connection` | `bool \| None` | inferred | Whether the handle disposes `connection` on `close()` |

| `SandboxHandle` member | Role |
| --- | --- |
| `engine` | The `AetherEngine` or `AetherFederation` instance |
| `connection` | Primary DuckDB connection (main dataset or federation coordinator view) |
| `member_connections` | `tuple` of per-member connections for federation handles, else `None` |
| `artifacts_dir` | Artifacts directory string |
| `adopt(engine)` | Delegate to the backing `Sandbox.adopt` when present |
| `session(...)` | Alias for `engine.session(...)` with default writer mode |
| `apply_bundled_schema_overrides()` | Copy bundled override JSON and apply or enqueue |
| `close()` | Release temp extract dir, connections, and owned artifacts |

| Symbol | Role |
| --- | --- |
| `AetherEngine.sandbox_questions()` | Curated offline practice question list |
| `AetherEngine.sandbox_paraphrase_pairs()` | Canonical->paraphrase pairs |
| `AetherEngine.sandbox_validation_failure_demo()` | Questions that should end in terminal validation errors |
| `AetherEngine.sandbox_feedback_demo()` | Anchor question + allowed rejection text |
| `MockFixtureMissingError` | Mock LLM has no recorded answer (exported from `aetherdialect`) |

Warmup and QSim raise `ConfigError` on sandbox instances. Always use `with AetherEngine.offline_sandbox() as sb:` or call `sb.close()`.

## Exceptions

| Exception | Bases | When raised | Constructor kwargs |
| --- | --- | --- | --- |
| `ConfigError` | `ValueError` | Missing or invalid configuration, ambiguous engine/LLM, or unreadable TOML. | message |
| `ConnectionError` | `OSError` | Driver-level connection failures after construction. | message |
| `DatabasePingFailed` | `ConnectionError`, `RetryableError` | Retriable connectivity failures. | message |
| `LlmTransientFailure` | `RuntimeError`, `RetryableError` | Transient LLM HTTP failures. | message |
| `MigrationPendingError` | `ValueError` | Migration map missing, invalid, or `action="abort"`. | message |
| `MockFixtureMissingError` | `RuntimeError` | Mock LLM has no recorded answer for the requested turn. | message |
| `OwnerOnlyOperationError` | `ConfigError` | Consumer-role instance attempted an owner-only mutation. | `operation: str` |
| `SchemaAccessError` | `ValueError` | Unreadable scope, empty visible graph, or ambiguous allow-list entries. | message |
| `SessionActiveError` | `RuntimeError` | `ask` while a turn is already active. | message |
| `StatementTimeoutError` | `RuntimeError`, `RetryableError` | Statement timeout from the database engine / member fetch. | message |
| `FederationConfigError` | `ConfigError` | Invalid federation manifest, mappings, migration sidecar, or corrupt plan template file. | message |
| `FederationDeclarationError` | `FederationConfigError` | Manifest or mapping declaration rejected at build/registration. | message |
| `FederationIneligibleError` | `ConfigError` | Valid intent cannot be decomposed or executed as a federated plan. | message |
| `FederationInvariantError` | `FederationConfigError` | Composite or plan replay invariants violated. | message |
| `FederationRuntimeError` | `ConfigError` | Federated execution failed after planning succeeded. | message |
| `FederationPartialFailureError` | `FederationRuntimeError` | One member failed after others succeeded. Turn outcome is `federation_partial_failure`. | `source_id`, `phase`, `succeeded=()`, `retryable=False` |
| `FederationMemberExecutionError` | `FederationRuntimeError` | One federation member's query failed during execution. Catch by type: attribute the failure to that database; the turn may be retryable when the cause is a `RetryableError`. | `source_id`, `phase` |
| `FederationCapExceededError` | `FederationRuntimeError` | A federated row, byte, or timeout cap was exceeded. Catch by type: narrow or re-scope the question - retrying the same turn will not help. | `limit_key`, `source_id=""` |

Federation configuration is rejected at registration when every member is a file (`csv`) engine - load uploads into one CSV engine instead. Relative date-window filters on federated questions are anchored once at turn start so each member statement uses the same instant.

Catch `RetryableError` to branch on transient failures.

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
