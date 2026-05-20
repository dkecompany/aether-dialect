# API reference

Stable types, configuration keys, constructor contract, session APIs, JSON artefacts, observability catalogs, and exceptions exported from `aetherdialect`.

**Reading order:** [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → this file → [How it works](HOW_IT_WORKS.md) → [Offline testing and mock LLM (design)](OFFLINE_AND_MOCK_LLM.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

Requires Python 3.10 or newer. Install the `aetherdialect` distribution from PyPI; import symbols from the `aetherdialect` package. The authoritative export list is `aetherdialect.__all__`.

## Document map

| Section | Contents |
| ------- | -------- |
| [SchemaContext](#schemacontext) | Scope fields for construction |
| [Engine storage layout](#engine-storage-layout) | Artifact directory path |
| [Configuration](#configuration) | Merge order and TOML flattening table |
| [Observability](#observability) | `Diagnostic`, `AuditEvent`, codes, `audit_sink` |
| [Text2SQL constructor](#text2sql-constructor) | Parameters and types |
| [schema_migration_map.json](#schema_migration_mapjson) | Migration editor shape |
| [Schema overrides JSON](#schema-overrides-json-schema_overridesjson) | Overrides editor shape |
| [Text2SQL methods](#text2sql-methods) | Façade method table |
| [PipelineSession methods](#pipelinesession-methods) | Sync session API |
| [AsyncPipelineSession methods](#asyncpipelinesession-methods) | Async session API |
| [Public symbols](#public-symbols-aetherdialect__all__) | `__all__` list |
| [Exceptions](#exceptions) | Raised error types |

Embedding flow, suspend `kind` values, and worked examples are in the [Integrator guide](INTEGRATOR_GUIDE.md).

## SchemaContext

Frozen scope input to `Text2SQL`. Operator semantics and sensitivity classification are in the [User guide](USER_GUIDE.md).

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `include` | `"tables"` \| `"views"` \| `"both"` | Which relation kinds enter the graph |
| `allow_objects` | `tuple[str, ...]` | Optional allow-list of relation names |
| `deny_columns` | `tuple[str, ...]` | Qualified `table.column` or `*.column` denies |
| `allow_columns` | `tuple[str, ...]` | Qualified allow-list when non-empty |
| `notes_file` | `str \| None` | Optional analyst notes path |
| `sql_file` | `str \| None` | Optional DDL or annotated SQL path |

`deny_columns` entries are absent from the built `SchemaGraph` while the deny remains effective. `allow_columns` and `deny_columns` must use qualified names only.

## Engine storage layout

Resolved storage root:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

where `<artifacts_parent>` is the expanded `artifacts_dir` when provided, otherwise the platform user-data directory for the package, and `<connection_slug>` is derived from database connection parameters. Template shards, fingerprints, and lifecycle are in [How it works — Engine storage](HOW_IT_WORKS.md).

## Configuration

### Merge order

1. Start from a string copy of `os.environ`.
2. When **`config_file` is omitted**, that copy is the effective environment for `Text2SQL` / `initialize_text2sql`.
3. When **`config_file` is provided**, the parsed TOML is the single source of truth for every field listed in the flattening table that appears in the file: non-empty values replace `os.environ`, and fields present with empty values remove the corresponding key from the effective mapping so shell-inherited secrets cannot override an explicit empty assignment. Keys for sections or fields absent from the file keep their `os.environ` values.

The library never mutates `os.environ` during reads. When a TOML key shadows an environment value, one diagnostic per key is emitted with code `CONFIG_FILE_VALUE_APPLIED`.

### `config_file` TOML flattening

Parse rules follow `_load_config_file` in `aetherdialect._main_execution`. Every value is coerced with `str(...)`. Absent sections are skipped. Fields present with empty strings suppress inherited environment values for the mapped key. Parse failures raise `ConfigError`.

| TOML section.key | Flattened environment key | Accepted aliases | Default (if unset) | Required |
| ---------------- | ------------------------- | ---------------- | ------------------ | -------- |
| `[openai].api_key` | `OPENAI_API_KEY` | — | — | yes when OpenAI is the active LLM stack |
| `[openai].base_url` | `OPENAI_BASE_URL` | — | — | no |
| `[azure_openai].endpoint` | `AZURE_OPENAI_ENDPOINT` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].api_key` | `AZURE_OPENAI_API_KEY` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].api_version` | `AZURE_OPENAI_API_VERSION` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai].base_url` | `AZURE_OPENAI_BASE_URL` | — | — | no |
| `[azure_openai.deployments].light` | `AZURE_OPENAI_DEPLOYMENT_LIGHT` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai.deployments].medium` | `AZURE_OPENAI_DEPLOYMENT_MEDIUM` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[azure_openai.deployments].heavy` | `AZURE_OPENAI_DEPLOYMENT_HEAVY` | — | — | yes when Azure OpenAI is the active LLM stack |
| `[postgresql].host` | `POSTGRES_HOST` | `PGHOST`, `PGHOSTADDR`, `POSTGRES_HOST` | — | yes when PostgreSQL is selected |
| `[postgresql].port` | `POSTGRES_PORT` | `PGPORT`, `POSTGRES_PORT` | — | yes when PostgreSQL is selected |
| `[postgresql].database` | `POSTGRES_DB` | `PGDATABASE`, `POSTGRES_DB` | — | yes when PostgreSQL is selected |
| `[postgresql].schema` | `POSTGRES_SCHEMA` | `PGSCHEMA`, `POSTGRES_SCHEMA` | — | no |
| `[postgresql].user` | `POSTGRES_USER` | `PGUSER`, `POSTGRES_USER` | — | yes when PostgreSQL is selected |
| `[postgresql].password` | `POSTGRES_PASSWORD` | `PGPASSWORD`, `POSTGRES_PASSWORD` | — | yes when PostgreSQL is selected |
| `[databricks].host` | `DATABRICKS_HOST` | `DATABRICKS_HOST`, `DATABRICKS_SERVER_HOSTNAME` | — | yes when Databricks is selected |
| `[databricks].http_path` | `DATABRICKS_HTTP_PATH` | `DATABRICKS_HTTP_PATH`, `DATABRICKS_SQL_HTTP_PATH`, `DATABRICKS_WAREHOUSE_HTTP_PATH` | — | yes when Databricks is selected |
| `[databricks].access_token` | `DATABRICKS_ACCESS_TOKEN` | `DATABRICKS_TOKEN`, `DATABRICKS_ACCESS_TOKEN`, `DATABRICKS_PAT`, `ACCESS_TOKEN` | — | yes when Databricks is selected |
| `[databricks].catalog` | `DATABRICKS_CATALOG` | `DATABRICKS_CATALOG`, `SPARK_DEFAULT_CATALOG` | — | no |
| `[databricks].schema` | `DATABRICKS_SCHEMA` | `DATABRICKS_SCHEMA`, `DATABRICKS_DEFAULT_SCHEMA`, `SPARK_DEFAULT_SCHEMA` | — | no |
| `[databricks].cluster_id` | `DATABRICKS_CLUSTER_ID` | — | — | no |
| `[engine].selected` | `AETHERDIALECT_ENGINE` | — | — | yes when both database engines are configured |
| `[llm].provider` | `AETHERDIALECT_LLM_PROVIDER` | — | — | yes when both LLM stacks are configured |
| `[execution].max_query_cost_rows` | `AETHERDIALECT_MAX_QUERY_COST_ROWS` | — | `50000000` | no |
| `[execution].max_query_cost_bytes` | `AETHERDIALECT_MAX_QUERY_COST_BYTES` | — | `50000000000` | no |
| `[execution].statement_timeout_ms` | `AETHERDIALECT_STATEMENT_TIMEOUT_MS` | — | `30000` | no |
| `[execution].llm_timeout_ms` | `AETHERDIALECT_LLM_TIMEOUT_MS` | — | `60000` | no |
| `[execution].profile_timeout_ms` | `AETHERDIALECT_PROFILE_TIMEOUT_MS` | — | `120000` | no |
| `[execution].explain_timeout_ms` | `AETHERDIALECT_EXPLAIN_TIMEOUT_MS` | — | falls back to `statement_timeout_ms` | no |

## Observability

Three channels are documented in the [Integrator guide — Observability](INTEGRATOR_GUIDE.md#observability). This section lists types and catalogs only.

### `audit_sink`

Optional **`Callable[[AuditEvent], None] | None`** on `Text2SQL` construction. It is **not** a boolean flag and does **not** print anything by itself. When provided, the engine invokes your function with an `AuditEvent` at coarse lifecycle boundaries.

### `Diagnostic`

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `stage` | `str` | Pipeline stage that emitted the row |
| `level` | `str` | `info`, `warn`, or `error` |
| `code` | `str` | Stable code string (see catalog below) |
| `message` | `str` | Human-readable text |
| `details` | `tuple[tuple[str, str], ...]` | Optional key-value metadata |
| `duration_ms` | `int \| None` | Optional timing |

Each `SessionStep` from `ask` / `step` includes a `diagnostics` tuple produced during that step.

### `AuditEvent`

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `event_type` | `str` | Discriminator (see catalog below) |
| `timestamp_iso` | `str` | UTC timestamp when emitted |
| `question` | `str \| None` | Natural-language question when applicable |
| `schema_hash` | `str \| None` | Effective structural hash when applicable |
| `provider` | `"openai"` \| `"azure"` | Active LLM provider |
| `details` | `tuple[tuple[str, str], ...]` | Lightweight metadata |

SQL and per-step detail belong on the terminal `SessionStep`, not on audit rows.

#### Audit `event_type` catalog

**Session lifecycle**

| `event_type` | When emitted |
| ------------ | ------------ |
| `init` | After successful `Text2SQL` construction |
| `ask_begin` | Start of `session.ask(...)` |
| `ask_done` | Turn completed (`details` includes `outcome`, `kind`) |
| `ask_error` | Terminal failure or fatal guard error |
| `ask_blocked` | `ask` rejected before raise (non-`str` question or `SessionActiveError`) |

**Admin operations**

| `event_type` | When emitted |
| ------------ | ------------ |
| `apply_schema_overrides` | After `apply_schema_overrides` persists |
| `clear_persisted_overrides` | After overrides sidecar removal and rebuild |
| `clear_template_store` | After template tree removal and reload |
| `clear_simulation_caches` | After QSim or seed-warmup cache deletion |
| `clear_all_learning` | After combined learning clears |

**Write queue drain** (writer session only)

| `event_type` | When emitted |
| ------------ | ------------ |
| `write_queue_feedback_record` | Writer applied a queued `feedback_record` |
| `write_queue_template_reject` | Writer applied a queued `template_reject` |
| `write_queue_template_accept` | Writer applied a queued `template_accept` |
| `write_queue_override_proposal` | Writer materialised a queued `override_proposal` |

### Diagnostic code catalog

**Pipeline semantics**

| Code | Typical use |
| ---- | ----------- |
| `REUSE_HIT` | Template reuse matched |
| `REUSE_MISS` | No template reuse hit |
| `LOW_CONFIDENCE` | Low-confidence execution path |
| `LARGE_RESULT_WARNING` | Large result-set warning |
| `PII_GATE_HIT` | PII gate during repair |
| `STAGE_A_RETRY` | Logical intent retry |
| `STAGE_B_REPAIR` | Schema or semantic repair pass |
| `FALLBACK_FRESH_RESTART` | Fresh restart after repair exhaustion |
| `SCHEMA_OVERRIDE_SKIP` | Override entry skipped during apply |

**Configuration**

| Code | Typical use |
| ---- | ----------- |
| `CONFIG_FILE_VALUE_APPLIED` | TOML value overrode `os.environ` for a key |

**Catch-all**

| Code | Typical use |
| ---- | ----------- |
| `ENGINE_INFO` | Progress, CLI echo, schema summaries |

**Validation (lowercase)** — `SessionStep.diagnostics` may also carry EXPLAIN- and validator-derived codes (for example `explain_seq_scan_indexed`). Treat unknown codes as opaque. Soft versus hard EXPLAIN behaviour is in [Security](SECURITY.md).

## `Text2SQL` constructor

```python
Text2SQL(
    schema_context: SchemaContext | None = None,
    *,
    artifacts_dir: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    execution_engine: Any | None = None,
    audit_sink: Callable[[AuditEvent], None] | None = None,
) -> None
```

| Parameter | Type | Meaning |
| --------- | ---- | ------- |
| `schema_context` | `SchemaContext \| None` | Frozen scope. When omitted, a persisted context is loaded from engine storage when present; if none exists, `ConfigError` is raised. |
| `artifacts_dir` | `str \| None` | Parent directory; resolved storage is `<artifacts_parent>/aetherdialect/<connection_slug>/`. When omitted, a platform user-data parent is used. |
| `config_file` | `str \| os.PathLike[str] \| None` | Optional TOML path; see [Configuration](#configuration). |
| `execution_engine` | `Any \| None` | Optional caller-owned SQLAlchemy engine for execution. |
| `audit_sink` | `Callable[[AuditEvent], None] \| None` | Optional lifecycle callback; see [Observability](#observability). |

Raises `ConfigError`, `ConnectionError`, `MigrationPendingError`, or other failures documented under [Exceptions](#exceptions).

## `schema_migration_map.json`

Written in the working directory when migration requires operator action; consumed on the next successful `Text2SQL(...)` construction and renamed to `schema_migration_map.applied.json` (timestamped variant when a prior applied file exists). Workflow is in the [User guide — Migration](USER_GUIDE.md).

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `version` | `int` | Map format version (`1` today) |
| `action` | `str` | `remap`, `destructive`, or `abort` |
| `table_renames` | array | Objects with `from` / `to` table names |
| `column_renames` | array | Objects with `table`, `from`, `to` column names |
| `dropped_tables` | array of `str` | Tables removed from the warehouse |
| `dropped_columns` | array | Objects with `table` and `column` |
| `added_tables` | array of `str` | Tables added to the warehouse |
| `added_columns` | array | Objects with `table` and `column` |
| `refresh_existing_descriptions_on_addition` | `bool` | Default `false`; may refresh descriptions when tables are added |

## Schema overrides JSON (`schema_overrides.json`)

Version `4`. Export writes `./schema_overrides.json` in the process working directory. Apply reads the same path.

### Editable analyst and integrator surface

Hand-edited files for `apply_schema_overrides` should use **plain strings** for descriptions and roles and `null` or a string for `sensitivity`. Do **not** add `owner` keys; the engine treats missing owner as analyst content and records it under the internal `user_override` provenance during apply.

| Location | Editable fields |
| -------- | --------------- |
| Each table | `description` (string), `role` (string) |
| Each column | `description` (string), `role` (string), `sensitivity` (`null` or string) |
| Graph | `foreign_keys_add[]`, `foreign_keys_remove[]` |
| Primary keys | `primary_keys_add[]`, `primary_keys_remove[]` |
| Internal persistence | `_internal` (block lists; engine-maintained, not hand-authored) |

### Export merge envelopes (read-only lineage)

`export_schema_overrides` may rewrite analyst strings into merge envelopes (`{"value": "...", "owner": "..."}`) for sidecar provenance. That shape is for inspection and round-tripping after export, not for hand authoring.

### Full shape (illustrative hand-edited file)

```jsonc
{
  "version": 4,
  "tables": {
    "public.orders": {
      "description": "Orders placed by customers.",
      "role": "fact",
      "columns": {
        "order_id": {
          "description": "Surrogate primary key.",
          "role": "identifier",
          "sensitivity": null,
        },
      },
    },
  },
  "foreign_keys_add": [
    {
      "from": "public.orders.customer_id",
      "to": "public.customers.customer_id",
      "kind": "structural",
    },
  ],
  "foreign_keys_remove": [
    { "from": "public.bad.from_col", "to": "public.bad.to_col" },
  ],
  "primary_keys_add": [{ "table": "public.orders", "column": "order_id" }],
  "primary_keys_remove": [{ "table": "public.child", "column": "id" }],
  "_readonly": {
    "foreign_keys_current": [],
    "primary_keys_current": [],
    "tables_current": [],
    "columns_current": [],
  },
}
```

`_readonly` is regenerated on export and ignored on apply. Successful apply persists `applied_overrides.json` beside the gzip schema cache and archives the editor JSON to `*.applied.json`.

## `Text2SQL` methods

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `apply_migration_map(path="schema_migration_map.json", *, config_file=None, schema_context, artifacts_dir)` classmethod | `Text2SQL` | Copies the editor map into the working directory filename `schema_migration_map.json`, then constructs `Text2SQL`. Pair with the [User guide — Migration](USER_GUIDE.md). |
| `export_schema_overrides()` | `Path` | Writes `./schema_overrides.json` atomically from the live graph. |
| `apply_schema_overrides()` | `None` | Validates `./schema_overrides.json`, mutates graph, persists gzip schema cache and `applied_overrides.json`, prints summary via notify, renames editor files to `schema_overrides.applied.json` and `schema_overrides.applied.schema.json` (timestamped archives when prior files exist). |
| `show_config()` | `ConfigSnapshot` | Redacted snapshot of engine, schema scope, database, and LLM settings. Active engine name is available here and on `RuntimeConfig.engine`. |
| `session(*, mode="writer")` | `PipelineSession` | Context manager; exit calls `reset()`. No stdout. |
| `asession(*, mode="writer")` | `AsyncPipelineSession` | Same as `session` on worker threads. |
| `run_interactive()` | `None` | Prints to stdout; one question per call. Prefer `session` for services. |
| `run_seed_warmup(seed_filepath, interactive_gold=True)` | `None` | Warmup run with template writes; prints summary. |
| `run_seed_warmup_from_history(sql_history_filepath)` | `None` | SQL-history warmup; prints summary. |
| `run_seed_warmup_from_query_log(lookback_days=730, max_queries=5000)` | `None` | Warehouse query-log warmup; prints summary. |
| `dry_run_warmup(seed_filepath, interactive_gold=True)` | `None` | Same seed validation path as `run_seed_warmup`, but skips question-side LLM calls and does not persist new templates. |
| `get_schema_stats()` | `SchemaStatsSnapshot` | Copy of internal graph counters (for example `table_count`) for dashboards or health checks; not required for normal Q&A. |
| `get_seed_warmup_summary()` | `SeedWarmupSummarySnapshot` | Reads newest seed-warmup summary file if present. |
| `get_qsim_summary(start, end)` | `QSimSummarySnapshot` | Reads QSim summary index for inclusive version range. |
| `get_questions_only(version)` | `None` | Prints numbered questions and writes `qsim_v{version}_questions.txt` in the working directory. |
| `run_qsim(num_intents=20, num_questions=100, seed=None)` | `None` | QSim generator; prints summary. |
| `emit_otel_span(name, **attrs)` | context manager | Starts an OpenTelemetry span when `opentelemetry` is installed; otherwise no-op so callers can wrap `ask` / `step` without conditional imports. |
| `clear_persisted_overrides()` | `bool` | Removes overrides sidecar and schema cache when present; rebuilds init bundle; returns whether a sidecar existed. |
| `clear_template_store()` | `bool` | Removes `intent_templates/` (header and shards) and legacy `intent_templates.json.gz` when present; rebuilds init bundle. |
| `clear_simulation_caches()` | `int` | Deletes QSim and seed-warmup artefacts; returns removed file count; rebuilds init bundle. |
| `clear_all_learning(*, keep_overrides=True)` | `None` | Clears templates, simulation caches, and optionally overrides; rebuilds init bundle. |

## `PipelineSession` methods

| Method | Returns | Contract |
| ------ | ------- | -------- |
| `ask(question: str)` | `SessionStep` | Starts a turn; raises `SessionActiveError` if busy. |
| `step(response=None)` | `SessionStep` | Supplies user text for a suspend. |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep` | Auto-answers yes-or-no suspends; raises on free-text suspends. |
| `awaiting_prompt()` | `bool` | `True` when the next input must go to `step`. |
| `reset()` | `None` | Clears suspend state, choice queues, and partial turn state. |
| `__enter__` / `__exit__` | `PipelineSession` / `None` | Exit calls `reset()`. |

`mode="reader"` skips durable template and feedback writes. `mode="writer"` is the default and serialises writer turns with a per-instance lock. Suspend `kind` values and the state machine are in the [Integrator guide](INTEGRATOR_GUIDE.md).

## `AsyncPipelineSession` methods

| Method | Returns | Contract |
| ------ | ------- | -------- |
| `ask(question)` | `SessionStep` | Delegates to `PipelineSession.ask` on a worker thread. |
| `step(response=None)` | `SessionStep` | Delegates to `PipelineSession.step`. |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep` | Delegates to `PipelineSession.ask_until_done`. |
| `reset()` | `None` | Delegates to `PipelineSession.reset` via `asyncio.to_thread`. |
| `awaiting_prompt()` | `bool` | Delegates to `PipelineSession.awaiting_prompt`. |
| `__aenter__` / `__aexit__` | `AsyncPipelineSession` / `bool` | Async context manager forwarding to the inner session. |

## Public symbols (`aetherdialect.__all__`)

| Symbol | One-line role |
| ------ | ------------- |
| `AsyncPipelineSession` | Async façade over `PipelineSession` (`asyncio.to_thread`). |
| `AuditEvent` | Structured audit payload passed to `audit_sink`. |
| `ConfigError` | Invalid configuration or constructor arguments (`ValueError` subclass). |
| `ConfigSnapshot` | Redacted configuration text from `show_config`. |
| `ConnectionError` | Database connectivity failures (`OSError` subclass). |
| `DatabasePingFailed` | `ConnectionError` plus `RetryableError` marker. |
| `Diagnostic` | One structured diagnostic row on `SessionStep.diagnostics`. |
| `LlmExecutionConfig` | Per-call LLM timeout and retry limits. |
| `LlmTransientFailure` | Transient LLM HTTP failures (`RetryableError`). |
| `MigrationPendingError` | Migration map required or invalid; construction stops. |
| `MigrationPreview` | Tier and affected identifiers without applying a map. |
| `PipelineSession` | Synchronous suspend-or-resume session from `Text2SQL.session`. |
| `QSimSummarySnapshot` | Tuple of summary lines for QSim versions. |
| `RetryableError` | Marker mixin for transient failures. |
| `RuntimeConfig` | Frozen `engine`, `artifacts_dir`, `schema_context`, `llm_execution`. |
| `SchemaAccessError` | Scope, allow-list, or deny-list problems. |
| `SchemaContext` | Frozen schema scope input to `Text2SQL`. |
| `SchemaStatsSnapshot` | Table and column statistics dict. |
| `SeedWarmupSummarySnapshot` | Latest seed-warmup summary text. |
| `SessionActiveError` | `ask` while a turn is in progress. |
| `SessionStep` | One suspend or terminal payload from a session. |
| `StatementTimeoutError` | Engine statement timeout (`RetryableError`). |
| `Text2SQL` | Engine façade. |
| `__version__` | Package version string. |

## Exceptions

| Exception | Bases | When raised |
| --------- | ----- | ----------- |
| `ConfigError` | `ValueError` | Missing or invalid configuration, incomplete LLM or database blocks, invalid `SchemaContext`, ambiguous engine or LLM without `AETHERDIALECT_ENGINE` / `AETHERDIALECT_LLM_PROVIDER`, or unreadable `config_file`. |
| `ConnectionError` | `OSError` | Driver-level connection failures after construction. |
| `DatabasePingFailed` | `ConnectionError`, `RetryableError` | Retriable connectivity failures. |
| `LlmTransientFailure` | `RuntimeError`, `RetryableError` | Transient LLM HTTP failures. |
| `MigrationPendingError` | `Text2SQLError` | Migration map missing, invalid, or `action="abort"`. |
| `SchemaAccessError` | `ValueError` | Unreadable scope, empty visible graph, ambiguous allow-list entries. |
| `SessionActiveError` | `RuntimeError` | `ask` while a turn is active. |
| `StatementTimeoutError` | `RuntimeError`, `RetryableError` | Statement timeout from the engine. |

Catch `RetryableError` to branch on transient failures. `Text2SQLError` is an internal base for migration errors and is not exported.

---

**See also:** [README](../README.md) · [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [How it works](HOW_IT_WORKS.md) · [Offline testing and mock LLM (design)](OFFLINE_AND_MOCK_LLM.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md)
