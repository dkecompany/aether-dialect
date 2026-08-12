# API reference

Lookup for the `aetherdialect` package: exported types, constructor parameters, method contracts, document shapes, and exceptions. Requires Python 3.11 or newer. Embedding flow: [Integrator guide](INTEGRATOR_GUIDE.md). Operator semantics: [User guide](USER_GUIDE.md). Diagnostic and refusal catalogues: [Troubleshooting](TROUBLESHOOTING.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → this document → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Exported symbols](#exported-symbols) | Package `__all__` surface |
| [Configuration](#configuration) | Merge order, TOML flattening, limits |
| [EngineContext](#enginecontext) | Single-engine scope fields |
| [AetherEngine](#aetherengine) | Single-engine constructor and methods |
| [FederationContext](#federationcontext) | Composite scope fields |
| [AetherFederation](#aetherfederation) | Composite constructor and methods |
| [Shared lifecycle](#shared-lifecycle) | `refresh`, `close`, `RefreshReport` |
| [SpaceContext](#spacecontext) | AetherSpace allow/deny fields |
| [AetherSpace](#aetherspace) | Read-only space descriptor |
| [Sessions](#sessions) | `PipelineSession`, `SessionStep`, outcomes |
| [Structure and knowledge documents](#structure-and-knowledge-documents) | Dict exchange formats |
| [Tabular upload](#tabular-upload) | `inspect_tabular_upload`, ingest |
| [Federation documents](#federation-documents) | Declaration and migration maps |
| [Observability](#observability) | Audit, phase, diagnostic callbacks |
| [Sandbox](#sandbox) | Offline sandbox entry point |
| [Exceptions](#exceptions) | Raised error types |

---

## Exported symbols

Import from `aetherdialect`. Authoritative list: `aetherdialect.__all__`.

| Symbol | Kind |
| --- | --- |
| `AccessError` | Exception |
| `AggregateJoinFanOutError` | Exception |
| `AetherEngine` | Facade |
| `AetherError` | Exception |
| `AetherFederation` | Facade |
| `AetherSpace` | Dataclass |
| `AmbiguousDateLiteralError` | Exception |
| `ArtifactLockTimeoutError` | Exception |
| `AsyncPipelineSession` | Facade |
| `AuditEvent` | Dataclass |
| `DomainKnowledgeEntry` | Dataclass |
| `ClauseWidenedRowsetError` | Exception |
| `ComparisonJoinScopeExceededError` | Exception |
| `ConfigError` | Exception |
| `ConfigSnapshot` | Dataclass |
| `DatabaseConnectionError` | Exception |
| `DatabaseExecutionError` | Exception |
| `DatabasePingFailed` | Type |
| `DataQualityReport` | Dataclass |
| `Diagnostic` | Dataclass |
| `EngineContext` | Dataclass |
| `EngineLimits` | Dataclass |
| `FederationCapExceededError` | Exception |
| `FederationConfigError` | Exception |
| `FederationContext` | Dataclass |
| `FederationDeclarationError` | Exception |
| `FederationIneligibleError` | Exception |
| `FederationInvariantError` | Exception |
| `FederationJoinFanOutError` | Exception |
| `FederationMalformedMemberAnswerError` | Exception |
| `FederationMappingsAppliedSidecarError` | Exception |
| `FederationMemberExecutionError` | Exception |
| `FederationMemberProbeError` | Exception |
| `FederationMemberUnprofilableError` | Exception |
| `FederationPartialFailureError` | Exception |
| `FederationRuntimeError` | Exception |
| `FederationTurnCancelledError` | Exception |
| `FederationLimits` | Dataclass |
| `JoinCandidateCapExceededError` | Exception |
| `JoinColumnCountMismatchError` | Exception |
| `JoinInjectionAlignmentError` | Exception |
| `JoinInjectionFailedError` | Exception |
| `JoinPathKeyTypeError` | Exception |
| `JoinPathTieCapExceededError` | Exception |
| `JoinProbeEdgeKindMismatchError` | Exception |
| `LlmJsonExhausted` | Type |
| `LlmTransientFailure` | Type |
| `MigrationPendingError` | Exception |
| `MigrationPreview` | Dataclass |
| `MockFixtureMissingError` | Exception |
| `NoJoinPathError` | Exception |
| `NullInNegatedListError` | Exception |
| `OwnerOnlyOperationError` | Exception |
| `PersistedFederationInspection` | Dataclass |
| `PhaseProgressEvent` | Dataclass |
| `PipelineSession` | Facade |
| `PipelineSuspended` | Exception |
| `ProbeCtePlacementError` | Exception |
| `QSimSummarySnapshot` | Type |
| `RefinementRetry` | Exception |
| `ResultCapExceededError` | Exception |
| `RetryableDatabaseExecutionError` | Exception |
| `RetryableError` | Exception |
| `RetryableFederationPartialFailureError` | Exception |
| `RegistryRenderError` | Exception |
| `Sandbox` | Facade |
| `SchemaAccessError` | Exception |
| `SchemaInvariantError` | Exception |
| `SchemaStatsSnapshot` | Type |
| `SeedWarmupSummarySnapshot` | Type |
| `SessionActiveError` | Exception |
| `SessionError` | Dataclass |
| `SessionOutcome` | Enum |
| `SessionStep` | Dataclass |
| `SessionTurnCancelledError` | Exception |
| `SpaceContext` | Dataclass |
| `StatementTimeoutError` | Exception |
| `SubdayDateWindowOnDateColumnError` | Exception |
| `SuspendedSessionExpiredError` | Exception |
| `UploadIngestResult` | Dataclass |
| `inspect_tabular_upload` | Function |
| `__version__` | str |

## Configuration

### Merge order

1. Start from a string copy of `os.environ`.
2. When **`connection=`** is a mapping, merge those keys into the effective environment for this instance only (values are never written to `os.environ`). Validate every key against the active engine's accepted connection key set; unknown keys raise `ConfigError`. The reserved `name` key sets federation member identity when present.
3. When **`config_file`** is provided, parsed TOML is authoritative for every flattened key that appears in the file: non-empty values replace the effective mapping, and fields present with empty values remove the corresponding key so shell-inherited secrets cannot override an explicit empty assignment. Keys for sections or fields absent from the file keep their values from steps 1–2.
4. When **`config_file`** is omitted and **`connection=`** is not a mapping, step 1 is the effective environment.
5. Pass **`limits=`** on the constructor, or load `[limits]` / `[federation_limits]` via `EngineLimits.from_config_file` / `FederationLimits.from_config_file`. Limits are not flattened into the environment.

The library never mutates `os.environ` during reads. When a TOML key shadows an environment value, one diagnostic per key is emitted with code `CONFIG_FILE_VALUE_APPLIED`.

Runtime URL builders percent-encode user, password, and database name components when the library constructs SQLAlchemy URLs. Hand-built URL strings must encode special characters in credentials.

### Artifact storage slug

Engine artifacts live under `<artifacts_root>/aetherdialect/<connection_slug>/`. Federation artifacts live under `<artifacts_root>/aetherdialect/fed_<federation_id>/`. The slug derives from database location keys only (never credentials). Hosts serving multiple customers pass distinct `artifacts_dir` roots.

| Engine | Slug key order |
| --- | --- |
| `postgresql` | `host`, `port`, `database`, `schema` |
| `redshift` | `host`, `port`, `database`, `schema` |
| `mysql` | `host`, `port`, `database`, `schema` |
| `mariadb` | `host`, `port`, `database`, `schema` |
| `sqlserver` | `host`, `port`, `database`, `schema` |
| `oracle` | `host`, `port`, `service_name`, `sid`, `schema` |
| `snowflake` | `account`, `database`, `schema` |
| `bigquery` | `project`, `dataset` |
| `databricks` | `host`, `catalog`, `schema` |
| `duckdb` | `database`, `schema` |
| `sqlite` | `database`, `schema` |
| `csv` | `source`, `schema` |

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
| `[oracle].host` | `ORACLE_HOST` | `ORACLE_SERVER` | `localhost` | no |
| `[oracle].port` | `ORACLE_PORT` | - | `1521` | no |
| `[oracle].user` | `ORACLE_USER` | `ORACLE_USERNAME` | - | yes when password auth is used |
| `[oracle].password` | `ORACLE_PASSWORD` | `ORACLE_PWD` | - | yes when password auth is used |
| `[oracle].service_name` | `ORACLE_SERVICE_NAME` | `ORACLE_SERVICE` | - | yes when `sid` is unset |
| `[oracle].sid` | `ORACLE_SID` | - | - | yes when `service_name` is unset |
| `[oracle].schema` | `ORACLE_SCHEMA` | `ORACLE_DEFAULT_SCHEMA` | uppercased user | no |
| `[oracle].auth_mode` | `ORACLE_AUTH_MODE` | - | `password` | no |
| `[oracle].wallet_location` | `ORACLE_WALLET_LOCATION` | `ORACLE_WALLET` | - | yes when wallet auth is used |
| `[oracle].config_dir` | `ORACLE_CONFIG_DIR` | - | - | yes when wallet auth needs `tnsnames`/`sqlnet` |
| `[oracle].token` | `ORACLE_TOKEN` | `ORACLE_ACCESS_TOKEN` | - | yes when token auth is used |
| `[oracle].thick_mode` | `ORACLE_THICK_MODE` | - | `false` | no |
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
| `[sandbox].fixtures_file` | `AETHERDIALECT_SANDBOX_FIXTURES_FILE` | `AETHERDIALECT_MOCK_FIXTURES_FILE` | - | yes when `[llm] provider = "sandbox"` |
| `[mock].fixtures_file` | `AETHERDIALECT_MOCK_FIXTURES_FILE` | - | - | synonym for `[sandbox].fixtures_file` |





### EngineLimits

Pass `limits: EngineLimits | None = None` to [`AetherEngine`](#aetherengine) (default `EngineLimits()`). Property: `engine.limits`. Optional fields that accept `None` mean unlimited.

| Field | Default | Notes |
| --- | --- | --- |
| `pool_size` | `1` | SQLAlchemy pool size (≥ 1) |
| `pool_max_overflow` | `4` | |
| `pool_recycle_seconds` | `1800` | |
| `pool_pre_ping` | `True` | |
| `pool_timeout_seconds` | `30` | |
| `statement_timeout_ms` | `30000` | `None` = unlimited |
| `profile_timeout_ms` | `120000` | `None` = unlimited |
| `profiling_total_budget_seconds` | `None` | |
| `max_result_rows` | `100000` | `None` = unlimited |
| `max_result_bytes` | `268435456` | `None` = unlimited |
| `max_upload_bytes` | `268435456` | |
| `result_fetch_batch_rows` | `10000` | must not exceed `max_result_rows` when both set |
| `prompt_payload_max_bytes` | `262144` | |
| `write_queue_max_record_bytes` | `1048576` | |
| `write_queue_max_file_bytes` | `None` | |
| `template_store_max_count` | `None` | |
| `template_store_max_disk_bytes` | `None` | |
| `template_value_history_depth` | `64` | |
| `feedback_rows_per_question` | `8` | |
| `template_partition_cache_size` | `32` | |
| `artifact_lock_timeout_seconds` | `30` | |
| `applied_map_archive_count` | `3` | |
| `suspended_session_ttl_seconds` | `None` | |

`EngineLimits.from_config_file(path)` reads the `[limits]` table only.

### FederationLimits

Pass `limits: FederationLimits | None = None` to [`AetherFederation`](#aetherfederation) (default `FederationLimits()`). Property: `federation.limits`. When `member_defaults` is set, member engines constructed without explicit `limits=` receive those defaults.

| Field | Default | Notes |
| --- | --- | --- |
| `member_defaults` | `None` | nested `EngineLimits` applied to members without explicit limits |
| `max_members` | `8` | |
| `max_parallel_members` | `4` | ≤ `max_members` |
| `member_row_cap` | `100000` | `None` = unlimited |
| `member_bytes_cap` | `268435456` | |
| `member_statement_timeout_ms` | `None` | |
| `member_probe_timeout_seconds` | `10` | |
| `transfer_max_bytes` | `536870912` | |
| `reduction_key_max_count` | `10000` | |
| `plan_step_count_max` | `32` | |
| `coordinator_memory_limit_bytes` | `2147483648` | |
| `coordinator_threads` | `4` | |
| `coordinator_temp_dir` | `None` | |
| `coordinator_spill_max_bytes` | `None` | |
| `federation_plan_template_count` | `None` | |

`FederationLimits.from_config_file(path)` reads the `[federation_limits]` table only.

### Limits behaviour

Behavioural limits arrive through `limits=` or the `[limits]` / `[federation_limits]` tables. They are not part of environment flattening. Cost ceilings `AETHERDIALECT_MAX_QUERY_COST_ROWS` and `AETHERDIALECT_MAX_QUERY_COST_BYTES` remain policy class variables. `AETHERDIALECT_LLM_TIMEOUT_MS` and `AETHERDIALECT_LLM_BATCH_ENABLED` remain LLM policy keys.

Profiling statement timeouts apply on engines that expose portable session timeout SQL (PostgreSQL `statement_timeout`, MySQL `MAX_EXECUTION_TIME`, DuckDB `statement_timeout`, and similar). **Databricks, Spark, SQL Server, and Oracle** do not implement `profile_statement_timeout_sql`, so per-member `profile_timeout_ms` still resolves for federation members but does not bound profiling queries on those engines.

On **BigQuery**, when `AETHERDIALECT_MAX_QUERY_COST_BYTES` is active, execution sets `maximum_bytes_billed` on query jobs from that cap (in addition to optional `job_timeout_ms` from `EngineLimits.statement_timeout_ms`).

`AETHERDIALECT_ENGINE` / `[engine] selected` accepts: `sqlite`, `duckdb`, `csv`, `mysql`, `mariadb`, `sqlserver`, `postgresql`, `redshift`, `databricks`, `snowflake`, `bigquery`, `oracle`. The TOML `[excel]` section aliases to the `csv` engine. When more than one engine block is complete and selectable, `[engine] selected` is required.

When Snowflake `database` is unset, the connection slug uses the placeholder segment `db`.

Documents passed to `apply_structure`, `apply_knowledge`, `apply_federation`, `apply_migration_map`, `export_context`, and nested `connection=` / `source_selections` bodies are validated in full before any mutation. Unknown keys raise `ConfigError`.

---

## EngineContext

Frozen scope input to [`AetherEngine`](#aetherengine). **Setup:** [Getting started — EngineContext](GETTING_STARTED.md#step-4-wire-enginecontext-and-construct-the-engine). **Operator semantics:** [User guide — EngineContext](USER_GUIDE.md#enginecontext).

| Field | Type | Meaning |
| --- | --- | --- |
| `include` | `"tables"` \| `"views"` | Kind selector when `allow_objects` and `deny_objects` are both empty (default `"tables"`). Ignored when `allow_objects` is non-empty. |
| `allow_objects` / `deny_objects` | `frozenset[str]` | Allow-list or deny-after-both-kinds lists. See reflection mode below. |
| `allow_columns` / `deny_columns` | `frozenset[str]` | Qualified `table.column` or `*.column` allow/deny lists. |
| `notes_file` | `str` \| `None` | Path to domain notes. Mutually exclusive with `notes`. |
| `notes` | `str` \| `None` | Inline domain notes. Mutually exclusive with `notes_file`. |
| `sql_file` | `str` \| `None` | Path to guidance DDL. |

Register master scope by constructing with an [`EngineContext`](#enginecontext) object. Register named presets on an owner instance; consumers bind by preset name string.

`notes`, `notes_file`, and `sql_file` are set only on `EngineContext` at construction. They are not environment variables and are not flattened from TOML.

**Reflection mode**

| `allow_objects` | `deny_objects` | Effective mode | Behaviour |
| --- | --- | --- | --- |
| non-empty | any | `allow_list` | Resolve each allow name to its catalog kind; `include` is ignored. Apply `deny_objects` after. |
| empty | non-empty | `both_then_deny` | Reflect tables and views, then remove denied names. |
| empty | empty | `single_kind` | Reflect only the kind named by `include`. |

---

## AetherEngine

Public facade for one database connection. Artifact identity is the connection slug under `artifacts_dir`.

### Constructor

```python
AetherEngine(
    engine_context: EngineContext | str | None = None,
    *,
    artifacts_dir: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    connection: str | Mapping[str, Any] | None = None,
    execution_engine: Any = None,
    native_connection: Any = None,
    source_selections: Mapping[str, Mapping[str, Any]] | None = None,
    audit_sink: Callable[[AuditEvent], None] | None = None,
    phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
    diagnostic_sink: Callable[[Diagnostic], None] | None = None,
    role: SchemaRole = "owner",
    token_provider: Callable[[], str | Mapping[str, str]] | None = None,
    limits: EngineLimits | None = None,
) -> None
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `engine_context` | `EngineContext` \| `str` \| `None` | `EngineContext` defines master scope (owner). `str` binds a saved preset (consumers must use a preset name). When omitted, load persisted master from artifacts or raise `ConfigError`. |
| `artifacts_dir` | `str` \| `None` | Root directory; engine files under `<root>/aetherdialect/<connection_slug>/` |
| `config_file` | path \| `None` | TOML config path |
| `connection` | `str` \| mapping \| `None` | **String:** named TOML sub-block. **Mapping:** per-instance credentials (validated key set per engine). String form selects credentials and the artifact slug. |
| `execution_engine` | `Any` | Optional SQLAlchemy engine (caller-owned pool) for engines that accept it |
| `native_connection` | `Any` | Optional native DuckDB or SQLite connection (`duckdb`, `sqlite` only; raises `ConfigError` for other engines) |
| `source_selections` | mapping \| `None` | `csv` engine only: per-filename interpretation after [`inspect_tabular_upload`](#tabular-upload) |
| `audit_sink` | callable \| `None` | Optional [`AuditEvent`](#auditevent) callback |
| `phase_callback` | callable \| `None` | Optional [`PhaseProgressEvent`](#phaseprogressevent) callback during construction and ask turns |
| `diagnostic_sink` | callable \| `None` | Optional [`Diagnostic`](#diagnostic) callback |
| `role` | `"owner"` \| `"consumer"` | `owner` may mutate shared artifacts; `consumer` pins owner snapshot id |
| `token_provider` | callable \| `None` | Returns a fresh secret or credential-field mapping for [`refresh`](#shared-lifecycle) |
| `limits` | [`EngineLimits`](#enginelimits) \| `None` | Behavioural limits (default `EngineLimits()`) |

Raises `ConfigError`, `DatabaseConnectionError`, `MigrationPendingError`, or failures listed under [Exceptions](#exceptions).

**Properties**

| Property | Type | Meaning |
| --- | --- | --- |
| `dialect` | `str` | Selected SQL engine identity |
| `limits` | [`EngineLimits`](#enginelimits) | Effective behavioural limits |
| `data_quality_report` | [`DataQualityReport`](#dataqualityreport) \| `None` | Confirmed upload state after file-engine construction or ingestion |
| `default_space_uid` | `str` | Default aetherspace uid for sessions and exports |

**Concurrency.** One instance supports concurrent reader-mode sessions. Mutating methods serialize on an instance writer lock. Database connections are not fork-safe; construct a new engine in each child process after `fork()`.

Owner-only methods require `role="owner"`. Methods marked **(owner)** refuse with `ConfigError` when `role="consumer"` before mutating shared artifacts. `list_aetherspaces` is visibility-scoped for every role. `export_knowledge` for a named visible space is available to consumers; `apply_knowledge` and all structure writes are owner-only.

`space=None` on session, export, and apply methods resolves to [`default_space_uid`](#aetherengine).

Mutating facade methods take the instance writer lock and run serially. Guarded names are listed in the exported `MUTATING_ENGINE_METHODS` and `MUTATING_FEDERATION_METHODS` tuples.

### Methods

Lifecycle methods [`refresh`](#shared-lifecycle) and [`close`](#shared-lifecycle) are documented under [Shared lifecycle](#shared-lifecycle).

| Method | Returns | Description |
| --- | --- | --- |
| `preview_migration_map()` | [`MigrationPreview`](#migrationpreview) | Read-only preview of schema migration impact against stored artifacts. |
| `data_quality_report` (property) | [`DataQualityReport`](#dataqualityreport) \| `None` | Upload validation report after successful file-engine construction or ingestion, including `confirmed_selections`. |
| `ingest_upload_sources(paths, *, source_selections=None, relation_names=None, log_sink=None)` **(owner)** | [`UploadIngestResult`](#uploadingestresult) | Validate uploads and materialise accepted relations into this embedded member (`csv` or `duckdb` engines). |
| `aetherspace(name=None, space_context=None, *, uid=None, notes_file=None, notes=None)` **(owner)** for writes | [`AetherSpace`](#aetherspace) | **Create:** `name` + `space_context` mints a new `S####` uid (duplicate names raise). **Update:** `uid` + `space_context` overwrites that space (optional `name=` renames the label). **Read:** `uid=` or unique visible `name=` (ambiguous names raise). Define and update require scope ⊆ effective visibility. Cannot redefine the default master space. |
| `delete_aetherspace(name=None, *, uid=None, persist_learning=True)` **(owner)** | `AetherspaceDeleteResult` | Delete one persisted space and its learning partition. Prefer `uid=`; `name=` only when unique among visible spaces. The default master space cannot be deleted. |
| `list_aetherspaces(*, include_system=False)` | `tuple[AetherSpace, ...]` | Return visible [`AetherSpace`](#aetherspace) descriptors. Owners include the implicit master space. Consumers omit master. Credential-default system spaces are omitted unless `include_system=True`. |
| `default_space_uid` (property) | `str` | Default aetherspace uid for this instance: master for an owner, visibility-keyed default for a consumer. |
| `export_context(name)` **(owner)** | `dict` | Read-only export document for one saved scope preset (or implicit `master`). |
| `list_contexts()` **(owner)** | `tuple[str, ...]` | Saved scope-preset names plus implicit `master`. |
| `list_templates(*, space=None)` | `tuple[StoredTemplateSummary, ...]` | Caller-visible template summaries for the resolved space. |
| `fetch_template(template_ref, *, space=None)` | [`StoredTemplateDetail`](#storedtemplatedetail) | Detail for one template by `id`, including parameterized SQL and bind slots. |
| `execute_template(template_id, params=None, *, space=None, as_dataframe=False)` | rows or `DataFrame` | Re-run a stored template by `id` with a p-param dict. Raises `ConfigError` when `approval_state` is `pending`. Deterministic replay; does not learn. |
| `export_structure(space=None)` | `dict` | Structural inventory merged with editable structure for the default or one named space. See [Structure document](#structure-document). Suggested persistence name: `schema_structure.json`. |
| `apply_structure(document)` **(owner)** | `None` | Apply a structural document declaratively; the document becomes the truth. Deleting an entry on round-trip removes it. |
| `export_knowledge(space=None)` | `dict` | Knowledge payload for the default or one named space. See [Knowledge document](#knowledge-document). |
| `apply_knowledge(space, document)` **(owner)** | `None` | Replace space domain knowledge and description overlays from one exported document. |
| `apply_migration_map(document)` **(owner)** | `None` | Validate and persist a schema migration map document, then call `refresh()`. |
| `session(*, mode="writer", space=None, ephemeral_scope=None, data_row_cap=None)` | [`PipelineSession`](#pipelinesession) | Programmatic session sharing this instance's schema graph and template store. `writer` persists space-partition learning; `reader` is read-only. Consumers may open `writer` for learning only — structural APIs stay owner-only. |
| `asession(*, mode="writer", space=None)` | [`AsyncPipelineSession`](#asyncpipelinesession) | Async wrapper around [`session`](#pipelinesession) using worker threads. |
| `run_interactive(*, space=None)` | `None` | Stdout interactive loop; prefer `session` for services. |
| `run_seed_warmup(seed_filepath, interactive_gold=True, *, abort_on_gold_failure=False, max_kept_intents=2000)` **(owner)** | `None` | Full seed warmup; `max_kept_intents=None` keeps every intent that passes quality checks. |
| `run_seed_warmup_from_history(sql_history_filepath, *, expand=False, max_kept_intents=2000)` **(owner)** | `None` | SQL-history warmup. |
| `run_seed_warmup_from_query_log(lookback_days=730, max_queries=5000, *, expand=False, max_kept_intents=2000, min_runs=1, user_filter=None)` **(owner)** | `None` | Warehouse query-log warmup. |
| `get_seed_warmup_summary()` | [`SeedWarmupSummarySnapshot`](#seedwarmupsummarysnapshot) | Newest seed-warmup summary text if present. |
| `get_qsim_summary(start, end)` | [`QSimSummarySnapshot`](#qsimsummarysnapshot) | QSim summary lines for inclusive version range. |
| `get_questions_only(version)` | `None` | Print numbered questions and write `qsim_v{version}_questions.txt`. |
| `run_qsim(num_intents=20, num_questions=100, seed=None)` **(owner)** | `None` | QSim generator; writes per-run summary files under `qsim/`. |
| `clear_template_store(*, space=None)` **(owner)** | `bool` | Remove template learning then reload. `space=None` clears every partition; otherwise one space uid. |
| `clear_simulation_caches()` **(owner)** | `int` | Delete QSim and seed-warmup artifacts; return removed file count. |
| `clear_all_learning(*, keep_structure=True, space=None)` **(owner)** | `None` | With `space` set, clear only that template partition. Without `space`, clear all templates, simulation caches, and optionally applied structure. Domain knowledge is cleared via `export_knowledge` / `apply_knowledge`. |

---

## FederationContext

Frozen composite scope for [`AetherFederation`](#aetherfederation). **Operator semantics:** [User guide — FederationContext](USER_GUIDE.md#federationcontext).

| Field | Type | Meaning |
| --- | --- | --- |
| `include` | `"tables"` \| `"views"` | Include mode (default `"tables"`). |
| `allow_objects` / `deny_objects` | `frozenset[str]` | Composite table allow/deny lists (logical names for collapsed tables). |
| `allow_columns` / `deny_columns` | `frozenset[str]` | Qualified `table.column`, `source.table.column`, or `*.column`. |
| `notes_file` | `str` \| `None` | Composite domain notes path. Mutually exclusive with `notes`. |
| `notes` | `str` \| `None` | Inline composite domain notes. Mutually exclusive with `notes_file`. |

No `sql_file` field on federation scope.

---

## AetherFederation

Composite facade over member [`AetherEngine`](#aetherengine) instances. Artifact tree: `fed_<federation_id>/`.

### Constructor

```python
AetherFederation(
    name: str,
    *,
    members: Sequence[AetherEngine],
    declaration: str | os.PathLike[str] | Mapping[str, Any],
    context: FederationContext | None = None,
    artifacts_dir: str | None = None,
    role: SchemaRole = "owner",
    limits: FederationLimits | None = None,
    audit_sink: Callable[[AuditEvent], None] | None = None,
    phase_callback: Callable[[PhaseProgressEvent], None] | None = None,
    diagnostic_sink: Callable[[Diagnostic], None] | None = None,
) -> None
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Stable federation name matching `federation_id` in the declaration |
| `members` | `Sequence[AetherEngine]` | At least two distinct members. Each member's `source_id` is its connection name (TOML sub-block or `name` key in a `connection=` mapping). Duplicate connection names raise `FederationConfigError`. |
| `declaration` | path \| `dict` | Federation declaration document or path to any JSON file |
| `context` | [`FederationContext`](#federationcontext) \| `None` | Optional composite scope |
| `artifacts_dir` | `str` \| `None` | Root for member trees and federation tree |
| `role` | `"owner"` \| `"consumer"` | Owner may change declarations; consumer reads the composite |
| `limits` | [`FederationLimits`](#federationlimits) \| `None` | Federation coordination limits |
| `audit_sink` | callable \| `None` | Optional [`AuditEvent`](#auditevent) callback |
| `phase_callback` | callable \| `None` | Optional [`PhaseProgressEvent`](#phaseprogressevent) callback |
| `diagnostic_sink` | callable \| `None` | Optional [`Diagnostic`](#diagnostic) callback |

Engine-only constructor parameters (`config_file`, `connection`, `execution_engine`, `native_connection`, `source_selections`, `token_provider`) belong on each member [`AetherEngine`](#aetherengine).

```python
storefront = AetherEngine(..., connection="storefront_pg")
catalog = AetherEngine(..., connection="catalog_mysql")
fed = AetherFederation(
    "sandbox_rental_shop",
    members=(storefront, catalog),
    declaration=declaration_dict,
    artifacts_dir="/data/artifacts",
)
```

Raises `FederationConfigError`, `FederationDeclarationError`, `FederationInvariantError`, or `MigrationPendingError` on misconfiguration.

**Properties**

| Property | Type | Meaning |
| --- | --- | --- |
| `dialect` | `str` | Federation engine label (`federation`) |
| `limits` | [`FederationLimits`](#federationlimits) | Federation coordination limits |
| `default_space_uid` | `str` | Default aetherspace uid for sessions and exports |

Owner-only methods require `role="owner"`. Methods marked **(owner)** refuse with `ConfigError` when `role="consumer"` before mutating shared artifacts. `list_aetherspaces` is visibility-scoped for every role. `export_knowledge` for a named visible space is available to consumers; `apply_knowledge` and all structure writes are owner-only.

`space=None` on session, export, and apply methods resolves to [`default_space_uid`](#aetherfederation).

Mutating facade methods take the instance writer lock and run serially. Guarded names are listed in the exported `MUTATING_ENGINE_METHODS` and `MUTATING_FEDERATION_METHODS` tuples.

### Methods

| Method | Returns | Description |
| --- | --- | --- |
| `inspect_persisted(federation_id, *, artifacts_dir, role="owner")` classmethod | [`PersistedFederationInspection`](#persistedfederationinspection) | Load declaration and roster from a persisted `fed_<id>` tree without member engines. |
| `add_engine(engine)` **(owner)** | `None` | Register a member engine (its `source_id` is its connection name), recompose, and persist. |
| `remove_engine(connection_name)` **(owner)** | `None` | Remove a member engine, prune plan templates, recompose, and persist. |
| `export_federation()` **(owner)** | `dict` | Return the federation declaration document (topology and mappings). |
| `apply_federation(document)` **(owner)** | `None` | Apply a federation declaration document and recompose. |
| `preview_migration_map()` | [`MigrationPreview`](#migrationpreview) | Read-only preview of schema migration impact against stored artifacts. |
| `aetherspace(name=None, space_context=None, *, uid=None, notes_file=None, notes=None)` **(owner)** for writes | [`AetherSpace`](#aetherspace) | **Create:** `name` + `space_context` mints a new `S####` uid (duplicate names raise). **Update:** `uid` + `space_context` overwrites that space (optional `name=` renames the label). **Read:** `uid=` or unique visible `name=` (ambiguous names raise). Define and update require scope ⊆ effective visibility. Cannot redefine the default master space. |
| `delete_aetherspace(name=None, *, uid=None, persist_learning=True)` **(owner)** | `AetherspaceDeleteResult` | Delete one persisted space and its learning partition. Prefer `uid=`; `name=` only when unique among visible spaces. The default master space cannot be deleted. |
| `list_aetherspaces(*, include_system=False)` | `tuple[AetherSpace, ...]` | Return visible [`AetherSpace`](#aetherspace) descriptors. Owners include the implicit master space. Consumers omit master. Credential-default system spaces are omitted unless `include_system=True`. |
| `default_space_uid` (property) | `str` | Default aetherspace uid for this instance: master for an owner, visibility-keyed default for a consumer. |
| `export_context(name)` **(owner)** | `dict` | Read-only export document for one saved scope preset (or implicit `master`). |
| `list_contexts()` **(owner)** | `tuple[str, ...]` | Saved federation-context names plus implicit `master`. |
| `list_templates(*, space=None)` | `tuple[StoredTemplateSummary, ...]` | Caller-visible template summaries from the federation artifact template store. |
| `fetch_template(template_ref, *, space=None)` | [`StoredTemplateDetail`](#storedtemplatedetail) | Detail by `id`, including parameterized SQL and bind slots. |
| `execute_template(template_id, params=None, *, space=None, as_dataframe=False)` | rows or `DataFrame` | Re-run a federation-stored template by `id` with a p-param dict. |
| `export_structure(space=None)` | `dict` | Structural inventory merged with editable structure for the default or one named space, plus federation `members` / `member_count` when members are present. See [Structure document](#structure-document). |
| `apply_structure(document)` **(owner)** | `None` | Apply a structural document to the composite federation graph declaratively; the document becomes the truth. |
| `export_knowledge(space=None)` | `dict` | Knowledge payload for the default or one named space on the composite graph. See [Knowledge document](#knowledge-document). |
| `apply_knowledge(space, document)` **(owner)** | `None` | Replace space domain knowledge and description overlays from one exported document. |
| `apply_migration_map(document)` **(owner)** | `None` | Validate and persist a federation migration map document, then recompose. |
| `session(*, mode="writer", space=None, ephemeral_scope=None, data_row_cap=None)` | [`PipelineSession`](#pipelinesession) | Programmatic session sharing the composite schema graph and template store. `writer` persists space-partition learning; consumers may open `writer` for learning only. |
| `asession(*, mode="writer", space=None)` | [`AsyncPipelineSession`](#asyncpipelinesession) | Async wrapper around [`session`](#pipelinesession) using worker threads. |
| `run_interactive(*, space=None)` | `None` | Stdout interactive loop; prefer `session` for services. |
| `run_seed_warmup(...)` **(owner)** | — | Raises `ConfigError` (`warmup is not supported on AetherFederation`). Run seed warmup on each member [`AetherEngine`](#aetherengine). |
| `run_seed_warmup_from_history(...)` **(owner)** | — | Raises `ConfigError` (`warmup is not supported on AetherFederation`). Run SQL-history warmup on each member engine. |
| `run_seed_warmup_from_query_log(...)` **(owner)** | — | Raises `ConfigError` (`warmup is not supported on AetherFederation`). Run query-log warmup on each member engine. |
| `get_seed_warmup_summary()` | [`SeedWarmupSummarySnapshot`](#seedwarmupsummarysnapshot) | Newest seed-warmup summary text if present. |
| `get_qsim_summary(start, end)` | [`QSimSummarySnapshot`](#qsimsummarysnapshot) | QSim summary lines for inclusive version range. |
| `get_questions_only(version)` | `None` | Print numbered questions and write `qsim_v{version}_questions.txt`. |
| `run_qsim(num_intents=20, num_questions=100, seed=None)` **(owner)** | `None` | QSim routed through federation decomposition checks. |
| `clear_template_store(*, space=None)` **(owner)** | `bool` | Clear composite and member templates (`space=None` clears every partition plus plan templates). |
| `clear_simulation_caches()` **(owner)** | `int` | Clear federation and member QSim and seed-warmup artifacts. |
| `clear_all_learning(*, keep_structure=True, space=None)` **(owner)** | `None` | Clear federation and member learning (space-scoped like the engine when `space` is set). Optionally retain applied structure when `keep_structure=True`. |

---

## Shared lifecycle

[`refresh`](#shared-lifecycle) and [`close`](#shared-lifecycle) are shared by [`AetherEngine`](#aetherengine) and [`AetherFederation`](#aetherfederation).

| Method | Returns | Description |
| --- | --- | --- |
| `refresh(*, reflect=True, credentials=None)` | [`RefreshReport`](#refreshreport) | Re-resolve credentials through `token_provider` or `credentials`, reopen the connection, and reconcile artifacts against the live schema. With `reflect=False`, skip the live probe and perform artifact-side work only. On federation, refresh each member then recompose. |
| `close()` | `None` | Idempotent teardown. Disposes the coordinator or dialect-owned pool, removes the artifact lock file, and clears cached model clients. Member engines passed in by the caller remain open; the federation does not close them. |

### RefreshReport

| Field | Type | Meaning |
| --- | --- | --- |
| `migration_tier` | `MigrationTier` | Classified migration severity |
| `schema_changed` | `bool` | Whether the schema diff was non-empty |
| `tables_added` | `tuple[str, ...]` | Tables added during refresh |
| `tables_removed` | `tuple[str, ...]` | Tables removed during refresh |
| `columns_added` | `tuple[tuple[str, str], ...]` | `(table, column)` pairs added |
| `columns_removed` | `tuple[tuple[str, str], ...]` | `(table, column)` pairs removed |
| `templates_invalidated` | `int` | Templates dropped by store reconciliation |
| `orphans_removed` | `int` | Expired orphan directories removed |
| `bytes_reclaimed` | `int` | Bytes reclaimed from orphan removal |
| `diagnostics` | `tuple[Diagnostic, ...]` | Growth and checkpoint diagnostics |

---

## SpaceContext

Frozen scope for [`AetherSpace`](#aetherspace) definitions. **Concept:** [User guide — AetherSpace](USER_GUIDE.md#aetherspace).

| Field | Type | Meaning |
| --- | --- | --- |
| `tables` | `frozenset[str]` | Allowed table/view names |
| `columns` | `frozenset[str]` | Qualified `table.column` allow list |
| `deny_objects` | `frozenset[str]` | Tables/views excluded from the space |
| `deny_columns` | `frozenset[str]` | Qualified `table.column` deny list |
| `notes_file` | `str` \| `None` | Optional notes path (mutually exclusive with `notes`) |
| `notes` | `str` \| `None` | Inline notes baked into the space snapshot |

Every table and column must exist on the master graph at write time.

---

## AetherSpace

Read-only descriptor returned by [`aetherspace`](#aetherengine). Durable identity is **`uid`**; **`name`** is a display label.

| Field | Type | Meaning |
| --- | --- | --- |
| `uid` | `str` | Stable opaque identity (`master` or `S####`) |
| `name` | `str` | Display label (unique among visible spaces on create) |
| `tables` | `tuple[str, ...]` | Table names in this space subset |
| `columns` | `tuple[str, ...]` | Qualified `table.column` names in this subset |
| `notes` | `str` \| `None` | Merged notes text baked at define time |

Open a session with `session(..., space=uid)` or a unique visible name.

---

## Sessions

### PipelineSession

Sync session from [`AetherEngine.session`](#aetherengine) or [`AetherFederation.session`](#aetherfederation). One **turn** runs from `ask` through terminal `done`. While `done` is `False`, call `step(...)`.

| Method | Returns | Contract |
| --- | --- | --- |
| `ask(question: str)` | [`SessionStep`](#sessionstep) | Start a turn; raises `SessionActiveError` if busy |
| `step(response=None)` | [`SessionStep`](#sessionstep) | Supply user text for a suspend (`"y"` / `"n"` or free text) |
| `ask_until_done(question, *, on_confirm="y")` | [`SessionStep`](#sessionstep) | Auto-answer yes/no suspends only |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | [`SessionStep`](#sessionstep) | Auto-answer until the turn ends |
| `awaiting_prompt()` | `bool` | `True` when the next input must go to `step` |
| `reset()` | `None` | Clear suspend state |
| `cancel()` | `bool` | Cooperative cancel for the in-flight turn |
| `__enter__` / `__exit__` | context manager | Exit calls `cancel()` then `reset()` |

`mode="reader"` skips durable template writes. `mode="writer"` persists space-partition learning and serialises writer turns on the instance lock; consumers may use `writer` for learning only. Suspend `kind` values: [Troubleshooting — SessionStep kinds](TROUBLESHOOTING.md#sessionstep-kind-values).

### AsyncPipelineSession

| Method | Returns | Contract |
| --- | --- | --- |
| `ask(question)` | [`SessionStep`](#sessionstep) | Delegates to `PipelineSession.ask` on a worker thread |
| `step(response=None)` | [`SessionStep`](#sessionstep) | Delegates to `PipelineSession.step` |
| `ask_until_done(question, *, on_confirm="y")` | [`SessionStep`](#sessionstep) | Delegates to `PipelineSession.ask_until_done` |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | [`SessionStep`](#sessionstep) | Delegates to `PipelineSession.accept_until_done` |
| `reset()` | `None` | Delegates via `asyncio.to_thread` |
| `awaiting_prompt()` | `bool` | Delegates via `asyncio.to_thread` |
| `cancel()` | `bool` | Cooperative cancel for the in-flight turn |
| `__aenter__` / `__aexit__` | async context manager | Forwards to the inner session |

### SessionStep

| Field | Type | Meaning |
| --- | --- | --- |
| `done` | `bool` | `True` when the turn finished or ended in error |
| `prompt` | `str` \| `None` | Short instruction before collecting input on suspends |
| `kind` | `str` | Stable stage identifier; see [Troubleshooting](TROUBLESHOOTING.md#sessionstep-kind-values) |
| `reply_shape` | `"yes_no"` \| `"free_text"` \| `None` | Input shape when `done` is `False` |
| `sql` | `str` \| `dict[str, str]` \| `None` | SQL under discussion (dict form carries per-source statements on federation) |
| `data` | `DataFrame` \| `None` | Rows available at this step |
| `data_truncated` | `bool` | Whether more rows exist beyond `data` |
| `parameters` | `tuple[ParameterBinding, ...]` | Bind slots for template replay |
| `intent_summary` | [`IntentSummary`](#intentsummary) \| `None` | Structured intent headline on intent-related steps |
| `semantic_warnings` | `tuple[str, ...]` | Model-authored caveats on intent confirmation |
| `answer` | `str` \| `None` | Rendered metadata answer on terminal non-SQL steps |
| `diagnostics` | `tuple[Diagnostic, ...]` | Structured diagnostics for the step |
| `error` | [`SessionError`](#sessionerror) \| `None` | Structured terminal failure |
| `template_id` | `str` \| `None` | Stored template identifier when relevant |
| `turn_id` | `str` \| `None` | Turn correlation id |
| `elapsed_ms` | `int` \| `None` | Wall time for the step |
| `llm_usage` | [`LlmTurnUsageSummary`](#llmturnusagesummary) \| `None` | Aggregated LLM usage for the turn |

Terminal steps have exactly one populated outcome shape: `answer` for metadata, `sql` with `data` for analytical success, or `error` for failure.

### SessionError

| Field | Type | Meaning |
| --- | --- | --- |
| `code` | [`SessionOutcome`](#sessionoutcome) | Closed outcome that ended the turn |
| `detail_code` | `str` \| `None` | One [`DiagnosticCode`](TROUBLESHOOTING.md#diagnosticcode-catalogue) or [`SqlDiagnosticCode`](TROUBLESHOOTING.md#sqldiagnosticcode-catalogue) responsible for the failure |
| `source_id` | `str` \| `None` | Federation member identity when attributable to one source |
| `phase` | `str` \| `None` | Federation stage (`member`, `coordinator`, …) when relevant |
| `limit_key` | `str` \| `None` | Configured limit name on cost or cap breaches |

### SessionOutcome

Closed enum on [`SessionError.code`](#sessionerror). Mapping guidance: [Troubleshooting — SessionOutcome](TROUBLESHOOTING.md#sessionoutcome).

| Member | Value |
| --- | --- |
| `FORBIDDEN` | `forbidden` |
| `UNSUPPORTED_OPERATION` | `unsupported_operation` |
| `UNANSWERABLE` | `unanswerable` |
| `INSUFFICIENT_KNOWLEDGE` | `insufficient_knowledge` |
| `NOT_A_QUESTION` | `not_a_question` |
| `PARSE_FAILED` | `parse_failed` |
| `VALIDATION_FAILED` | `validation_failed` |
| `EXECUTION_FAILED` | `execution_failed` |
| `EXECUTION_TIMEOUT` | `execution_timeout` |
| `COST_EXCEEDED` | `cost_exceeded` |
| `LIMIT_EXCEEDED` | `limit_exceeded` |
| `DECLINED` | `declined` |
| `CANCELLED` | `cancelled` |
| `MIGRATION_PENDING` | `migration_pending` |
| `INTERNAL_ERROR` | `internal_error` |

### ParameterBinding

| Field | Type | Meaning |
| --- | --- | --- |
| `handle` | `str` | Bind token identifying this slot |
| `current_value` | `ParamValue` \| `None` | Bound value in effect |
| `display_name` | `str` | Human-readable label |
| `column_expr` | `str` | Column expression the slot binds against, when known |

### IntentSummary

| Field | Type | Meaning |
| --- | --- | --- |
| `tables` | `tuple[str, ...]` | Tables referenced |
| `select_cols` | `tuple[str, ...]` | Projected columns |
| `filters` | `tuple[str, ...]` | Filter expressions |
| `group_by` | `tuple[str, ...]` | Grouping columns |
| `order_by` | `tuple[str, ...]` | Sort keys |
| `limit` | `int` \| `None` | Row limit when present |
| `natural_language` | `str` | Short natural-language headline |

### LlmTurnUsageSummary

| Field | Type | Meaning |
| --- | --- | --- |
| `request_count` | `int` | Model requests on the turn |
| `input_tokens` | `int` | Input tokens |
| `cached_input_tokens` | `int` | Cached prefix tokens |
| `output_tokens` | `int` | Output tokens |
| `cost_usd` | `float` \| `None` | Estimated USD cost when priced |

### StoredTemplateSummary

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Template identifier |
| `approval_state` | `str` | `approved` or `pending` |

### StoredTemplateDetail

| Field | Type | Meaning |
| --- | --- | --- |
| `summary` | [`StoredTemplateSummary`](#storedtemplatesummary) | Summary row |
| `parameters` | `tuple[ParameterBinding, ...]` | Bind slots |
| `approval_state` | `str` | `approved` or `pending` |

### ConfigSnapshot

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Frozen redacted configuration text for integrators |

### SchemaStatsSnapshot

| Field | Type | Meaning |
| --- | --- | --- |
| `stats` | `dict[str, Any]` | Schema statistics mapping |

### SeedWarmupSummarySnapshot

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Newest seed-warmup summary text |

### QSimSummarySnapshot

| Field | Type | Meaning |
| --- | --- | --- |
| `lines` | `tuple[str, ...]` | QSim summary lines for a version range |

---

## Structure and knowledge documents

Export methods return dicts; apply methods accept dicts. The library performs no disk I/O for export or apply. Callers own persistence for round-trip edits. Suggested caller-owned file names: `schema_structure.json`, `knowledge.json`, `federation_declaration.json`. After `apply_structure`, the library persists the applied document as `applied_structure.json` under the engine artifact tree.

### Structure document

Returned by [`export_structure`](#aetherengine). Accepted by [`apply_structure`](#aetherengine).

Top-level keys: `table_count`, `tables`, `relationships`, `foreign_keys_add`, `foreign_keys_remove`, `primary_keys_add`, `primary_keys_remove`. Federation exports may also include `members` and `member_count`.

Each `tables[]` entry: `name`, `columns` (each with `name`, `data_type`, and optional `role`, `sensitivity`, `usable`, `boolean_truth_value`), optional `primary_key`, `foreign_keys`, optional table `role`.

Removing a structure edit: `export_structure()`, delete the entry, `apply_structure(document)`.

### Knowledge document

Returned by [`export_knowledge`](#aetherengine). Accepted by [`apply_knowledge`](#aetherengine).

Top-level keys: `uid` (optional), `domain_knowledge`, `table_descriptions`, `column_descriptions`.

Each `domain_knowledge[]` entry: `key`, `kind`, `text`, `referenced_entities` (schema entity names; empty list allowed for glossary entries).

### DomainKnowledgeEntry

Dataclass mirroring one `domain_knowledge[]` row. Construct entries in application code before building a knowledge document.

| Field | Type | Meaning |
| --- | --- | --- |
| `key` | `str` | Stable entry key |
| `kind` | `str` | Entry kind label |
| `text` | `str` | Prose body |
| `referenced_entities` | `tuple[str, ...]` | Anchored schema entities |

---

## Tabular upload

### inspect_tabular_upload

```python
inspect_tabular_upload(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview | IO[bytes] | IO[str],
    *,
    filename: str | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> DataQualityReport
```

Module-level function (not an engine method). Inspects one CSV or Excel upload before any engine exists. Accepts a path, bytes, or a readable stream. Raises `ConfigError` on fatal file failures.

### DataQualityReport

| Field / property | Type | Meaning |
| --- | --- | --- |
| `ok` | `bool` | Whether construction may proceed without blocking issues |
| `issues` | `tuple[Diagnostic, ...]` | Structured findings |
| `narrative` | `str` | Human-readable summary for operators |
| `suggested_selections` | `dict[str, dict[str, Any]]` | Per-filename interpretation hints (same shape as `source_selections`) |
| `confirmed_selections` | `dict[str, dict[str, Any]]` | Selections confirmed after construction or ingest |
| `requires_review` | `bool` (property) | `True` when any issue needs caller confirmation |

`report.suggested_selections` feeds directly into `AetherEngine(source_selections=report.suggested_selections)`.

### source_selections

Outer mapping keys are **file names** (not paths). Each value accepts exactly these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `sheet` | `str` | Excel sheet name |
| `header_row` | `int` | 1-based header row index |
| `skip_rows` | `int` | Leading rows to skip before header detection |
| `table_range` | `str` | A1 range (for example `A1:F120`) |
| `merge_regions` | `tuple[str, ...]` | Merged cell regions to honour |
| `append_regions` | `tuple[str, ...]` | Additional rectangular regions to append |
| `column_transforms` | `tuple[dict, ...]` | Per-column normalisation transforms |

Valid only on the `csv` engine at construction. Raises `ConfigError` when supplied for other engines.

### ingest_upload_sources

[`AetherEngine.ingest_upload_sources`](#aetherengine) validates uploads and materialises relations into an existing `csv` or `duckdb` engine (owner-only).

### Two report surfaces

- [`inspect_tabular_upload`](#inspect-tabular-upload) reports on files before an engine exists.
- [`engine.data_quality_report`](#aetherengine) reports the confirmed post-construction or post-ingest state, including `confirmed_selections`.

### UploadIngestResult

| Field | Type | Meaning |
| --- | --- | --- |
| `relation_names` | `tuple[str, ...]` | Relation names created or updated |
| `report` | [`DataQualityReport`](#dataqualityreport) | Validation report for the ingest |
| `schema_diff` | `SchemaDiff` \| `None` | Schema change summary when applicable |

---

## Federation documents

### Declaration document

Accepted by [`apply_federation`](#aetherfederation). Returned by [`export_federation`](#aetherfederation). Worked example: [Sandbox — Federation declaration format](SANDBOX.md#federation-declaration-format).

Top-level keys accepted on apply: `federation_id`, `aliases`, `cross_source_joins`, `coordinator`, `logical_columns`, `logical_tables`. Export may also include composed connection metadata and `table_namespace` — do not author those fields.

| Section | Keys |
| --- | --- |
| `aliases[]` | `alias`, `source`, `table` |
| `cross_source_joins[]` | `left`, `right`, `kind`, `logical_key` |
| `coordinator` | `row_cap`, `default_source_row_cap`, `default_source_timeout_ms`, `coordinator_timeout_ms`, `plan_timeout_ms`, `semijoin_key_cap`, `spill_row_threshold`, `max_parallel_members`, `total_input_byte_cap` |
| `logical_columns[]` | `logical`, `members`, `role`, `unify_in_graph` |
| `logical_tables[]` | `logical`, `semantics`, `members`, `authoritative_source` |
| `logical_tables[].members[]` | `source`, `table`, `columns` |

### Migration maps

[`preview_migration_map`](#aetherengine) returns a [`MigrationPreview`](#migrationpreview) with `skeleton_document` when migration is pending. `MigrationPendingError` carries the skeleton dict. Apply via [`apply_migration_map`](#aetherengine) with the edited document.

### MigrationPreview

| Field | Type | Meaning |
| --- | --- | --- |
| `tier` | `"compatible"` \| `"remap"` \| `"destructive"` | Severity class |
| `affected_tables` | `tuple[str, ...]` | Tables impacted |
| `affected_columns` | `tuple[tuple[str, str], ...]` | `(table, column)` pairs impacted |
| `skeleton_document` | `dict` | Migration map skeleton to edit |

### PersistedFederationInspection

| Field | Type | Meaning |
| --- | --- | --- |
| `federation_id` | `str` | Federation identifier |
| `federation_dir` | `str` | Persisted tree path |
| `manifest` | `FederationManifest` | Parsed manifest |
| `mappings` | `FederationMappings` | Parsed mappings |
| `roster` | `tuple[tuple[str, str, str, str], ...]` | Registered member tuples |

---

## Observability

### audit_sink

Constructor parameter on [`AetherEngine`](#aetherengine) and [`AetherFederation`](#aetherfederation). Receives [`AuditEvent`](#auditevent) records at lifecycle boundaries. Event inventory: [Troubleshooting — Audit events](TROUBLESHOOTING.md#audit-events).

### AuditEvent

| Field | Type | Meaning |
| --- | --- | --- |
| `event_type` | `str` | Event name |
| `timestamp_iso` | `str` | UTC timestamp |
| `question` | `str` \| `None` | Active question when relevant |
| `schema_hash` | `str` \| `None` | Schema fingerprint |
| `provider` | `"openai"` \| `"azure"` \| `"sandbox"` | LLM provider |
| `details` | `tuple[tuple[str, str], ...]` | Structured detail pairs |
| `turn_id` | `str` \| `None` | Turn correlation id |

### phase_callback

Constructor parameter. Receives [`PhaseProgressEvent`](#phaseprogressevent) during construction and ask turns. The event names its own phase; branch in the callback when you need construction versus ask progress.

### PhaseProgressEvent

| Field | Type | Meaning |
| --- | --- | --- |
| `phase` | `str` | Phase label |
| `timestamp_iso` | `str` | UTC timestamp when emitted |
| `source` | `str` \| `None` | Federation member `source_id` when scoped to one member |
| `stage` | `int` \| `str` \| `None` | Optional sub-stage |
| `turn_id` | `str` \| `None` | Turn correlation id |
| `elapsed_ms` | `int` \| `None` | Milliseconds since previous phase emit on ask turns |

### diagnostic_sink

Constructor parameter. Receives [`Diagnostic`](#diagnostic) records on the diagnostic channel (symmetric with `audit_sink`).

### Diagnostic

| Field | Type | Meaning |
| --- | --- | --- |
| `code` | `str` | [`DiagnosticCode`](TROUBLESHOOTING.md#diagnosticcode-catalogue) or [`SqlDiagnosticCode`](TROUBLESHOOTING.md#sqldiagnosticcode-catalogue) value |
| `level` | `str` | Severity (`info`, `warning`, `error`, …) |
| `message` | `str` | Short message |
| `details` | `tuple[tuple[str, str], ...]` | Structured detail pairs |

Full code catalogues: [Troubleshooting — Diagnostic codes](TROUBLESHOOTING.md#diagnostic-codes).

### LLM usage

[`SessionStep.llm_usage`](#sessionstep) carries per-turn totals. [`Diagnostic`](#diagnostic) rows with code `LLM_TURN_COST` include token and cost detail keys documented under [Troubleshooting](TROUBLESHOOTING.md#common-diagnosticdetails-keys).

---

## Sandbox

Offline sandbox entry point. Walkthrough: [Sandbox guide](SANDBOX.md). Corpus reference: [Sandbox data reference](SANDBOX_DATA_REFERENCE.md).

### Sandbox

| Member | Returns | Purpose |
| --- | --- | --- |
| `Sandbox(*, llm_config=None, artifacts_dir=None, bundle_dir=None, cleanup=True, auto_seed=True, maintainer_access=False)` | context manager | Enter the authoring environment |
| `sandbox.engine(engine_context=None, *, role="owner", include="tables", limits=None)` | [`AetherEngine`](#aetherengine) | Build an engine on the default dataset |
| `sandbox.federation(federation_id, *, declaration_file=None, members=None, context=None)` | [`AetherFederation`](#aetherfederation) | Build a federation over named datasets |
| `sandbox.load_dataset(name, *, seed_sql=None, sql_file=None)` | `str` | Maintainer: seed an additional in-memory database |
| `sandbox.datasets` | `tuple[str, ...]` | Loaded dataset names |
| `sandbox.connection(name="main")` | connection | DuckDB connection for manual construction |
| `sandbox.artifacts_dir` | `str` | Shared artifacts root |
| `sandbox.config_file` | `str` \| `None` | TOML path for sandbox LLM configuration |
| `sandbox.adopt(engine)` | `None` | Idempotent: apply sandbox mock configuration (auto-runs on construction for sandbox-hosted connections) |
| `sandbox.close()` | `None` | Release connections, temp extract dir, and owned artifacts |

`with Sandbox() as sandbox:` plus production-shaped `AetherEngine(..., native_connection=sandbox.connection(), ...)` is the supported offline entry point. `sandbox.engine()` is a convenience wrapper over that constructor. Warmup and QSim raise `ConfigError` on sandbox instances.

### SandboxHandle

Returned when using sandbox helpers that own ephemeral resources. Not exported from `__all__`; documented here for sandbox guide cross-links.

| Member | Role |
| --- | --- |
| `engine` | The [`AetherEngine`](#aetherengine) or [`AetherFederation`](#aetherfederation) instance |
| `connection` | Primary DuckDB connection |
| `member_connections` | Per-member connections for federation handles, else `None` |
| `artifacts_dir` | Artifacts directory string |
| `adopt(engine)` | Idempotent sandbox configuration for a caller-built engine (auto-runs on construction for sandbox-hosted connections) |
| `close()` | Release temp resources and owned artifacts |

---

## Exceptions

Catch `AetherError` for a single handler over every library failure. Catch `RetryableError` (or `isinstance(exc, RetryableError)`) to branch on transient failures.

| Exception | Bases | When raised | What to do |
| --- | --- | --- | --- |
| `AetherError` | `Exception` | Base type for every library failure. | Catch once at service boundaries; branch on subclasses when needed. |
| `AccessError` | `SchemaAccessError`, `RuntimeError` | Database refused `EXPLAIN` or `execute`. | Treat as permission denial; inspect `operation` and `SessionStep.status`. |
| `AggregateJoinFanOutError` | `AetherError` | Join path would duplicate rows at parent grain. | Rephrase or narrow scope; fix schema join metadata. |
| `AmbiguousDateLiteralError` | `AetherError` | Absolute date bound is not valid ISO 8601. | Ask the user for an unambiguous date. |
| `ArtifactLockTimeoutError` | `RuntimeError`, `RetryableError` | Artifact directory lock not acquired in time. | Retry after the holder releases the lock. |
| `ClauseWidenedRowsetError` | `AetherError` | `LIMIT` / `DISTINCT ON` would run on a join-widened row set. | Rephrase the question or adjust scope. |
| `ComparisonJoinScopeExceededError` | `AetherError` | Cross-table comparison needs a join beyond allowed scope. | Narrow tables or declare relationships. |
| `ConfigError` | `AetherError`, `ValueError` | Missing/invalid configuration or TOML. | Fix environment or config file. |
| `DatabaseConnectionError` | `AetherError`, `OSError` | Driver rejected a connection attempt. | Check credentials, network, and engine reachability. |
| `DatabaseExecutionError` | `AetherError` | Driver failed during statement execution. | Inspect `classification` / `retryable`; retry when transient. |
| `DatabasePingFailed` | `DatabaseConnectionError`, `RetryableError` | `SELECT 1` ping failed after retries. | Retry; treat as connectivity blip. |
| `FederationCapExceededError` | `FederationRuntimeError` | Federated row/byte/timeout cap exceeded. | Narrow or re-scope; do not retry unchanged. |
| `FederationConfigError` | `ConfigError` | Invalid federation manifest, mappings, or sidecar. | Fix federation declaration artifacts. |
| `FederationDeclarationError` | `FederationConfigError` | Declaration rejected at build/registration. | Correct declaration JSON. |
| `FederationIneligibleError` | `ConfigError` | Intent cannot run as a federated plan. | Rephrase or use single-engine scope. |
| `FederationInvariantError` | `FederationConfigError` | Federation composition/replay invariant violated. | Fix manifest/roster/mappings. |
| `FederationJoinFanOutError` | `FederationRuntimeError` | Coordinator join exceeded declared grain. | Narrow join keys or question scope. |
| `FederationMalformedMemberAnswerError` | `FederationMemberExecutionError` | Member projection mismatched prepared sub-intent. | Inspect member SQL and mappings. |
| `FederationMappingsAppliedSidecarError` | `FederationConfigError` | Applied mappings sidecar disagrees with mappings file. | Re-export and re-apply mappings. |
| `FederationMemberExecutionError` | `FederationRuntimeError` | One member query failed during execution. | Attribute failure to `source_id`; retry if `RetryableError`. |
| `FederationMemberProbeError` | `FederationRuntimeError` | Member probe failed during federation init. | Fix member connectivity or credentials. |
| `FederationMemberUnprofilableError` | `FederationDeclarationError` | Member schema graph was not profiled. | Profile member engine artifacts first. |
| `FederationPartialFailureError` | `FederationRuntimeError` | One member failed after others succeeded. | Inspect `succeeded` and failing `source_id`. |
| `FederationRuntimeError` | `ConfigError` | Federated execution failed after planning. | Read message and federation diagnostics. |
| `FederationTurnCancelledError` | `FederationRuntimeError` | Federated turn cancelled cooperatively. | Start a new turn if needed. |
| `JoinCandidateCapExceededError` | `AetherError` | Join path enumeration exceeded refusal cap. | Reduce tables or declare explicit joins. |
| `JoinColumnCountMismatchError` | `AetherError` | Join signature column counts mismatch. | Fix schema FK/join metadata. |
| `JoinInjectionAlignmentError` | `AetherError` | Join signatures do not align with SQL carriers. | Repair template or schema graph. |
| `JoinInjectionFailedError` | `AetherError` | Deterministic SQL rewrite with joins failed. | Inspect `det_sql` and join metadata. |
| `JoinPathKeyTypeError` | `AetherError` | Join path pairs incompatible column types. | Fix schema typing or mappings. |
| `JoinPathTieCapExceededError` | `AetherError` | Too many equal-length join paths. | Disambiguate relationships in schema. |
| `JoinProbeEdgeKindMismatchError` | `AetherError` | Join signature / edge-kind lists misaligned. | Internal join metadata inconsistency. |
| `LlmJsonExhausted` | `AetherError` | `llm_json` exhausted retries without valid JSON. | Retry turn or switch model/deployment. |
| `LlmTransientFailure` | `RuntimeError`, `RetryableError` | Transient LLM HTTP failure. | Retry with backoff. |
| `MigrationPendingError` | `AetherError`, `ValueError` | Migration map missing, invalid, or `abort`. | Edit the migration map document and call `apply_migration_map`. |
| `MockFixtureMissingError` | `RuntimeError` | Mock LLM lacks a recorded answer. | Add fixture or ask a recorded question. |
| `NoJoinPathError` | `AetherError` | Requested tables have no join path. | Declare FK/semantic edges or fewer tables. |
| `NullInNegatedListError` | `AetherError` | `NOT IN` list contains null. | Rephrase filter predicate. |
| `OwnerOnlyOperationError` | `ConfigError` | Consumer attempted an owner-only mutation (`apply_structure`, `apply_knowledge`, space writes, and related owner surfaces). | Use an owner-role engine instance. |
| `PipelineSuspended` | `AetherError` | Programmatic turn awaits `session.step`. | Not an error; resume with `step`. |
| `ProbeCtePlacementError` | `AetherError` | Probe CTE used as illegal join anchor. | Rephrase semi/anti-join intent. |
| `RefinementRetry` | `AetherError` | User declined intent/SQL and the turn must restart refinement. | Catch and continue the interactive loop. |
| `RegistryRenderError` | `AetherError` | Window or CASE registry token cannot be resolved during SQL render. | Ensure registry ids match declared steps. |
| `ResultCapExceededError` | `RuntimeError` | Single-engine row/byte cap exceeded. | Narrow question or raise caps. |
| `RetryableDatabaseExecutionError` | `DatabaseExecutionError`, `RetryableError` | Transient database execution failure. | Retry the statement/turn. |
| `RetryableError` | `AetherError` | Marker for transient failures. | `isinstance(exc, RetryableError)` before retry. |
| `RetryableFederationPartialFailureError` | `FederationPartialFailureError`, `RetryableError` | Retryable partial federation failure. | Retry turn when safe. |
| `SchemaAccessError` | `AetherError`, `ValueError` | Scope/graph unreadable at init. | Fix allow/deny lists and credentials. |
| `SchemaInvariantError` | `RuntimeError` | Canonical schema containers out of sync. | Programmer error; fix build pipeline. |
| `SessionActiveError` | `RuntimeError` | `ask` while a turn is active. | Wait for turn completion or use another session. |
| `SessionTurnCancelledError` | `AetherError` | Programmatic turn cancelled. | Start a new `ask` if needed. |
| `StatementTimeoutError` | `RuntimeError`, `RetryableError` | Database statement timeout. | Retry or narrow query cost. |
| `SubdayDateWindowOnDateColumnError` | `AetherError` | Sub-day window on date-only column. | Rephrase date filter. |
| `SuspendedSessionExpiredError` | `SessionActiveError` | Suspended turn exceeded TTL. | Start a new `ask`. |

Federation configuration is rejected at registration when every member is a file (`csv`) engine - load uploads into one CSV engine instead. Relative date-window filters on federated questions are anchored once at turn start so each member statement uses the same instant.

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
