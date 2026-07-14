# API reference

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → this guide → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

Requires Python 3.10 or newer. Install the `aetherdialect` distribution from PyPI; import symbols from the `aetherdialect` package. The authoritative export list is `aetherdialect.__all__`.

## Sections

| Section | Contents |
| --- | --- |
| [EngineContext](#enginecontext) | Scope fields for construction |
| [SpaceContext](#spacecontext) | AetherSpace table/column subset |
| [AetherSpace](#aetherspace) | Read-only named-space descriptor |
| [Configuration](#configuration) | Merge order and TOML flattening table |
| [Observability](#observability) | `Diagnostic`, `AuditEvent`, and code catalogs |
| [AetherEngine methods](#aetherengine-methods) | facade method table |
| [PipelineSession methods](#pipelinesession-methods) | Sync session API |
| [AsyncPipelineSession methods](#asyncpipelinesession-methods) | Async session API |
| [Exceptions](#exceptions) | Raised error types |


Embedding flow, suspend `kind` values, and worked examples are in the [Integrator guide](INTEGRATOR_GUIDE.md). End-to-end setup with inlined TOML and notes: [Getting started](GETTING_STARTED.md).

## EngineContext

Frozen scope input to `AetherEngine`. **Setup walkthrough:** [Getting started — EngineContext](GETTING_STARTED.md#step-4--wire-enginecontext-and-construct-the-engine). **Operator semantics, notes, and sensitivity:** [User guide — EngineContext](USER_GUIDE.md#enginecontext). **Build pipeline:** [How it works — Schema build](HOW_IT_WORKS.md#2-schema-build-and-classification).


| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Context name (`"master"` default). |
| `include` | `"tables" \| "views" \| "both"` | Reflect base tables, views, or both. |
| `allow_objects` / `deny_objects` | `tuple[str, ...]` | Table allow/deny lists. |
| `allow_columns` / `deny_columns` | `tuple[str, ...]` | Qualified `table.column` allow/deny lists. |
| `notes_file` | `str \| None` | Path to domain notes. |
| `sql_file` | `str \| None` | Path to guidance DDL. |


`notes_file` and `sql_file` are set only on `EngineContext` at construction. They are **not** environment variables and are **not** flattened from `config_file` TOML.

Named contexts are subset-only of the persisted **master** graph (one shared graph; non-master never rebuilds or re-hashes). Owner constructs with a full `EngineContext` object to create/overwrite; consumer constructs with `engine_context="context_name"` (string name only). `export_aetherengine(name)` / `list_aetherengines()` dump or list stored contexts (**master-context-only**; see [AetherEngine methods](#aetherengine-methods)).

`deny_columns` entries are absent from the built `SchemaGraph` while the deny remains effective — unlike **restricted** or **hidden** sensitivity, which keep the column in the graph ([User guide — Sensitivity classification](USER_GUIDE.md#sensitivity-classification)). `allow_columns` and `deny_columns` must use qualified names only.

## SpaceContext

Frozen knowledge scope for [AetherSpace](#aetherspace) definitions. **Conceptual guide:** [User guide — AetherSpace](USER_GUIDE.md#aetherspace).


| Field | Type | Meaning |
| --- | --- | --- |
| `tables` | `frozenset[str]` | Allowed table/view names (empty means no extra table filter beyond master). |
| `columns` | `frozenset[str]` | Qualified `table.column` allow list for the space. |
| `deny_objects` | `frozenset[str]` | Tables/views excluded from space knowledge. |
| `deny_columns` | `frozenset[str]` | Qualified `table.column` deny list for the space. |


Every table/column must exist on the master graph at write time. Named spaces are subsets of the master engine context only. There is no TOML block for spaces — snapshots persist under the engine storage directory.

## AetherSpace

Read-only descriptor returned by `AetherEngine.aetherspace(name)` (existence check) or after a successful define/overwrite. No `session()` method — select the space via `AetherEngine.session(..., space=name)`.


| Member / method | Type | Meaning |
| --------------- | ---- | ------- |
| `name`          | `str` | Normalised space name |
| `list_scope()`  | `dict[str, tuple[str, ...]]` | Keys `"tables"` and `"columns"` with tuple values |
| `notes`         | `str \| None` | Optional merged notes text when `notes_file` was supplied at define time |

## Engine storage layout

Resolved storage root:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

where `<artifacts_parent>` is the expanded `artifacts_dir` when provided, otherwise the platform user-data directory for the package, and `<connection_slug>` is derived from database connection parameters. Template shards, fingerprints, and lifecycle are in [How it works — Engine storage](HOW_IT_WORKS.md).

## Configuration

### Merge order

1. Start from a string copy of `os.environ`.
2. When **config_file** is omitted, that copy is the effective environment for `AetherEngine`.
3. When **config_file** is provided, the parsed TOML is the single source of truth for every field listed in the flattening table that appears in the file: non-empty values replace `os.environ`, and fields present with empty values remove the corresponding key from the effective mapping so shell-inherited secrets cannot override an explicit empty assignment. Keys for sections or fields absent from the file keep their `os.environ` values.

The library never mutates `os.environ` during reads. When a TOML key shadows an environment value, one diagnostic per key is emitted with code `CONFIG_FILE_VALUE_APPLIED`.

### `config_file` TOML flattening

Every value is coerced with `str(...)`. Absent sections are skipped. Fields present with empty strings suppress inherited environment values for the mapped key. Parse failures raise `ConfigError`.


| TOML section.key                     | Flattened environment key            | Accepted aliases                                                                                        | Default (if unset)                      | Required                                             |
| ------------------------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `[openai].api_key`                   | `OPENAI_API_KEY`                     | —                                                                                                       | —                                       | yes when OpenAI is the active LLM stack              |
| `[openai].base_url`                  | `OPENAI_BASE_URL`                    | —                                                                                                       | —                                       | no                                                   |
| `[azure_openai].endpoint`            | `AZURE_OPENAI_ENDPOINT`              | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[azure_openai].api_key`             | `AZURE_OPENAI_API_KEY`               | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[azure_openai].api_version`         | `AZURE_OPENAI_API_VERSION`           | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[azure_openai].base_url`            | `AZURE_OPENAI_BASE_URL`              | —                                                                                                       | —                                       | no                                                   |
| `[azure_openai.deployments].light`   | `AZURE_OPENAI_DEPLOYMENT_LIGHT`      | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[azure_openai.deployments].medium`  | `AZURE_OPENAI_DEPLOYMENT_MEDIUM`     | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[azure_openai.deployments].heavy`   | `AZURE_OPENAI_DEPLOYMENT_HEAVY`      | —                                                                                                       | —                                       | yes when Azure OpenAI is the active LLM stack        |
| `[sqlite].path`                      | `SQLITE_PATH`                        | `SQLITE_DATABASE`, `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE` | `:memory:`                              | yes when SQLite is selected (path key for selection) |
| `[sqlite].database`                  | `SQLITE_DATABASE`                    | `SQLITE_PATH`, `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE`     | `:memory:`                              | alias of `path`                                      |
| `[duckdb].path`                      | `DUCKDB_PATH`                        | `DUCKDB_DATABASE`, `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN`                     | `:memory:`                              | yes when DuckDB is selected (path key for selection) |
| `[duckdb].database`                  | `DUCKDB_DATABASE`                    | `DUCKDB_PATH`, `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN`                         | `:memory:`                              | alias of `path`                                      |
| `[duckdb].schema`                    | `DUCKDB_SCHEMA`                      | `DUCKDB_DEFAULT_SCHEMA`                                                                                 | `main`                                  | no                                                   |
| `[csv].directory`                    | `CSV_DIRECTORY`                      | —                                                                                                       | —                                       | yes when CSV is selected (mutually exclusive with `files`) |
| `[csv].files`                        | `CSV_FILES`                          | —                                                                                                       | —                                       | yes when CSV is selected (comma-separated paths or TOML array; mutually exclusive with `directory`) |
| `[mysql].host`                       | `MYSQL_HOST`                         | `MYSQL_SERVER`, `MYSQL_HOSTNAME`                                                                        | `localhost`                             | no                                                   |
| `[mysql].port`                       | `MYSQL_PORT`                         | `MYSQL_TCP_PORT`                                                                                        | `3306`                                  | no                                                   |
| `[mysql].user`                       | `MYSQL_USER`                         | `MYSQL_USERNAME`                                                                                        | `root`                                  | yes when MySQL is selected                           |
| `[mysql].password`                   | `MYSQL_PASSWORD`                     | `MYSQL_PWD`                                                                                             | —                                       | yes when MySQL is selected                           |
| `[mysql].database`                   | `MYSQL_DATABASE`                     | `MYSQL_DB`                                                                                              | —                                       | yes when MySQL is selected                           |
| `[mariadb].host`                     | `MARIADB_HOST`                       | `MARIADB_SERVER`, `MARIADB_HOSTNAME`                                                                    | `localhost`                             | no                                                   |
| `[mariadb].port`                     | `MARIADB_PORT`                       | `MARIADB_TCP_PORT`                                                                                      | `3306`                                  | no                                                   |
| `[mariadb].user`                     | `MARIADB_USER`                       | `MARIADB_USERNAME`                                                                                      | `root`                                  | yes when MariaDB is selected                         |
| `[mariadb].password`                 | `MARIADB_PASSWORD`                   | `MARIADB_PWD`                                                                                           | —                                       | yes when MariaDB is selected                         |
| `[mariadb].database`                 | `MARIADB_DATABASE`                   | `MARIADB_DB`                                                                                            | —                                       | yes when MariaDB is selected                         |
| `[sqlserver].host`                   | `SQLSERVER_HOST`                     | `SQLSERVER_SERVER`, `MSSQL_HOST`, `MSSQL_SERVER`                                                        | `localhost`                             | no                                                   |
| `[sqlserver].port`                   | `SQLSERVER_PORT`                     | `MSSQL_PORT`                                                                                            | `1433`                                  | no                                                   |
| `[sqlserver].user`                   | `SQLSERVER_USER`                     | `SQLSERVER_USERNAME`, `MSSQL_USER`                                                                      | —                                       | yes when SQL auth is used                            |
| `[sqlserver].password`               | `SQLSERVER_PASSWORD`                 | `SQLSERVER_PWD`, `MSSQL_SA_PASSWORD`, `MSSQL_PASSWORD`                                                  | —                                       | yes when SQL auth is used                            |
| `[sqlserver].database`               | `SQLSERVER_DATABASE`                 | `SQLSERVER_DB`, `MSSQL_DATABASE`, `MSSQL_DB`                                                            | —                                       | yes when SQL Server is selected                      |
| `[sqlserver].schema`                 | `SQLSERVER_SCHEMA`                   | `MSSQL_SCHEMA`, `SQLSERVER_DEFAULT_SCHEMA`                                                              | `dbo`                                   | no                                                   |
| `[sqlserver].driver`                 | `SQLSERVER_DRIVER`                   | `MSSQL_DRIVER`, `ODBC_DRIVER`                                                                           | `ODBC Driver 18 for SQL Server`         | no                                                   |
| `[sqlserver].auth_mode`              | `SQLSERVER_AUTH_MODE`                | `MSSQL_AUTH_MODE`                                                                                       | `sql`                                   | no                                                   |
| `[sqlserver].tenant_id`              | `SQLSERVER_TENANT_ID`                | `MSSQL_TENANT_ID`, `AZURE_TENANT_ID`                                                                    | —                                       | yes when Azure AD password auth is used              |
| `[sqlserver].client_id`              | `SQLSERVER_CLIENT_ID`                | `MSSQL_CLIENT_ID`, `AZURE_CLIENT_ID`                                                                    | —                                       | yes when Azure AD service principal auth is used     |
| `[sqlserver].client_secret`          | `SQLSERVER_CLIENT_SECRET`            | `MSSQL_CLIENT_SECRET`, `AZURE_CLIENT_SECRET`                                                            | —                                       | yes when Azure AD service principal auth is used     |
| `[postgresql].host`                  | `POSTGRES_HOST`                      | `POSTGRES_SERVER`, `POSTGRES_HOSTNAME`, `PGHOST`, `PGHOSTADDR`                                          | `localhost`                             | no                                                   |
| `[postgresql].port`                  | `POSTGRES_PORT`                      | `PGPORT`                                                                                                | `5432`                                  | no                                                   |
| `[postgresql].database`              | `POSTGRES_DATABASE`                  | `POSTGRES_DB`, `PGDATABASE`                                                                             | —                                       | yes when PostgreSQL is selected                      |
| `[postgresql].schema`                | `POSTGRES_SCHEMA`                    | `PGSCHEMA`                                                                                              | `public`                                | no                                                   |
| `[postgresql].user`                  | `POSTGRES_USER`                      | `POSTGRES_USERNAME`, `PGUSER`                                                                           | `postgres`                              | yes when PostgreSQL is selected                      |
| `[postgresql].password`              | `POSTGRES_PASSWORD`                  | `POSTGRES_PWD`, `PGPASSWORD`                                                                            | —                                       | yes when PostgreSQL is selected                      |
| `[redshift].host`                    | `REDSHIFT_HOST`                      | `REDSHIFT_SERVER`                                                                                       | `localhost`                             | no                                                   |
| `[redshift].port`                    | `REDSHIFT_PORT`                      | `REDSHIFT_TCP_PORT`                                                                                     | `5439`                                  | no                                                   |
| `[redshift].user`                    | `REDSHIFT_USER`                      | `REDSHIFT_USERNAME`                                                                                     | `awsuser`                               | yes when Redshift is selected                        |
| `[redshift].password`                | `REDSHIFT_PASSWORD`                  | `REDSHIFT_PWD`                                                                                          | —                                       | yes when password auth is used                       |
| `[redshift].database`                | `REDSHIFT_DATABASE`                  | `REDSHIFT_DB`                                                                                           | `dev`                                   | yes when password auth is used                       |
| `[redshift].schema`                  | `REDSHIFT_SCHEMA`                    | —                                                                                                       | `public`                                | no                                                   |
| `[redshift].use_iam`                 | `REDSHIFT_USE_IAM`                   | `REDSHIFT_IAM`                                                                                          | `false`                                 | no                                                   |
| `[redshift].cluster_identifier`      | `REDSHIFT_CLUSTER_IDENTIFIER`        | `REDSHIFT_CLUSTER_ID`                                                                                   | —                                       | yes when IAM auth is used (cluster)                  |
| `[redshift].workgroup`               | `REDSHIFT_WORKGROUP`                 | `REDSHIFT_SERVERLESS_WORKGROUP`                                                                         | —                                       | yes when IAM auth is used (serverless)               |
| `[redshift].region`                  | `REDSHIFT_REGION`                    | `REDSHIFT_AWS_REGION`                                                                                   | —                                       | no                                                   |
| `[databricks].host`                  | `DATABRICKS_HOST`                    | `DATABRICKS_SERVER`, `DATABRICKS_SERVER_HOSTNAME`                                                       | —                                       | yes when Databricks is selected                      |
| `[databricks].http_path`             | `DATABRICKS_HTTP_PATH`               | `DATABRICKS_SQL_HTTP_PATH`, `DATABRICKS_WAREHOUSE_HTTP_PATH`                                            | —                                       | yes when Databricks is selected                      |
| `[databricks].access_token`          | `DATABRICKS_ACCESS_TOKEN`            | `DATABRICKS_TOKEN`, `DATABRICKS_PAT`, `ACCESS_TOKEN`                                                    | —                                       | yes when Databricks is selected                      |
| `[databricks].catalog`               | `DATABRICKS_CATALOG`                 | `SPARK_DEFAULT_CATALOG`                                                                                 | —                                       | no                                                   |
| `[databricks].schema`                | `DATABRICKS_SCHEMA`                  | `DATABRICKS_DEFAULT_SCHEMA`, `SPARK_DEFAULT_SCHEMA`                                                     | —                                       | no                                                   |
| `[snowflake].account`                | `SNOWFLAKE_ACCOUNT`                  | `SNOWSQL_ACCOUNT`, `SF_ACCOUNT`                                                                         | —                                       | yes when Snowflake is selected                       |
| `[snowflake].user`                   | `SNOWFLAKE_USER`                     | `SNOWFLAKE_USERNAME`, `SNOWSQL_USER`                                                                    | —                                       | yes when Snowflake is selected                       |
| `[snowflake].password`               | `SNOWFLAKE_PASSWORD`                 | `SNOWFLAKE_PWD`, `SNOWSQL_PWD`                                                                          | —                                       | yes when password auth is used                       |
| `[snowflake].database`               | `SNOWFLAKE_DATABASE`                 | `SNOWFLAKE_DB`, `SNOWSQL_DATABASE`                                                                      | — (optional; slug uses `db` when unset) | no                                                   |
| `[snowflake].schema`                 | `SNOWFLAKE_SCHEMA`                   | `SNOWSQL_SCHEMA`, `SNOWFLAKE_DEFAULT_SCHEMA`                                                            | `PUBLIC`                                | no                                                   |
| `[snowflake].warehouse`              | `SNOWFLAKE_WAREHOUSE`                | `SNOWSQL_WAREHOUSE`                                                                                     | —                                       | no                                                   |
| `[snowflake].role`                   | `SNOWFLAKE_ROLE`                     | `SNOWSQL_ROLE`                                                                                          | —                                       | no                                                   |
| `[snowflake].private_key_path`       | `SNOWFLAKE_PRIVATE_KEY_PATH`         | `SNOWFLAKE_PRIVATE_KEY`, `SNOWSQL_PRIVATE_KEY_PATH`                                                     | —                                       | yes when key-pair auth is used                       |
| `[snowflake].private_key_passphrase` | `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE`   | `SNOWSQL_PRIVATE_KEY_PASSPHRASE`                                                                        | —                                       | no                                                   |
| `[snowflake].authenticator`          | `SNOWFLAKE_AUTHENTICATOR`            | `SNOWSQL_AUTHENTICATOR`                                                                                 | —                                       | no                                                   |
| `[snowflake].oauth_token`            | `SNOWFLAKE_OAUTH_TOKEN`              | `SNOWFLAKE_OAUTH`, `SNOWSQL_OAUTH_TOKEN`                                                                | —                                       | yes when OAuth auth is used                          |
| `[bigquery].project`                 | `BIGQUERY_PROJECT`                   | `GOOGLE_CLOUD_PROJECT`, `GCP_PROJECT`                                                                   | —                                       | yes when BigQuery is selected                        |
| `[bigquery].dataset`                 | `BIGQUERY_DATASET`                   | `BIGQUERY_DB`, `GCP_DATASET`, `BIGQUERY_SCHEMA`, `BQ_DATASET`                                           | —                                       | yes when BigQuery is selected                        |
| `[bigquery].credentials_path`        | `BIGQUERY_CREDENTIALS_PATH`          | `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_CREDENTIALS_PATH`, `BQ_CREDENTIALS_PATH`                         | —                                       | no (ADC used when unset)                             |
| `[bigquery].location`                | `BIGQUERY_LOCATION`                  | `GCP_LOCATION`, `BQ_LOCATION`, `GOOGLE_CLOUD_LOCATION`                                                  | `US`                                    | no                                                   |
| `[engine].selected`                  | `AETHERDIALECT_ENGINE`               | —                                                                                                       | —                                       | yes when multiple database engines are configured    |
| `[llm].provider`                     | `AETHERDIALECT_LLM_PROVIDER`         | —                                                                                                       | —                                       | yes when both LLM stacks are configured              |
| `[mock].fixtures_file`               | `AETHERDIALECT_MOCK_FIXTURES_FILE`   | —                                                                                                       | —                                       | yes when `[llm] provider = "mock"`                     |
| `[execution].max_query_cost_rows`    | `AETHERDIALECT_MAX_QUERY_COST_ROWS`  | —                                                                                                       | `50000000`                              | no                                                   |
| `[execution].max_query_cost_bytes`   | `AETHERDIALECT_MAX_QUERY_COST_BYTES` | —                                                                                                       | `50000000000`                           | no                                                   |
| `[execution].statement_timeout_ms`   | `AETHERDIALECT_STATEMENT_TIMEOUT_MS` | —                                                                                                       | `30000`                                 | no                                                   |
| `[execution].llm_timeout_ms`         | `AETHERDIALECT_LLM_TIMEOUT_MS`       | —                                                                                                       | `60000`                                 | no                                                   |
| `[execution].profile_timeout_ms`     | `AETHERDIALECT_PROFILE_TIMEOUT_MS`   | —                                                                                                       | `120000`                                | no                                                   |
| `[execution].explain_timeout_ms`     | `AETHERDIALECT_EXPLAIN_TIMEOUT_MS`   | —                                                                                                       | falls back to `statement_timeout_ms`    | no                                                   |


On **BigQuery**, when `AETHERDIALECT_MAX_QUERY_COST_BYTES` is active, execution sets `maximum_bytes_billed` on query jobs from that cap (in addition to optional `job_timeout_ms` from `statement_timeout_ms`).

`AETHERDIALECT_ENGINE` / `[engine] selected` accepts: `sqlite`, `duckdb`, `csv`, `mysql`, `mariadb`, `sqlserver`, `postgresql`, `redshift`, `databricks`, `snowflake`, `bigquery`. When more than one engine block is complete and selectable, this key is required.

When Snowflake `database` is unset, `apply_environment` leaves it empty and the connection slug uses the placeholder segment `db`.

### Database connection reference

Optional DDL for join hints is supplied only via `EngineContext.sql_file` at construction (not env/TOML). Connection slug shapes are in [How it works — Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-artifact-lifecycle).

| Engine     | SQLAlchemy URL form (conceptual)                                             | Primary env keys                                                                                 | Notes                                                                                                                                                                                                   |
| ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SQLite     | `sqlite:///{path}` (`sqlite:///:memory:` in-memory)                          | `SQLITE_PATH` or `SQLITE_DATABASE`                                                               | Path aliases: `SQLITE_DATABASE_PATH`, `SQLITE_FILE`, `SQLITE_DB`, `SQLITE_DSN`, `SQLITE3_DATABASE`. Schema is always `main`. Stdlib `pysqlite`.                                                         |
| DuckDB     | `duckdb:///{path}` (`duckdb:///:memory:` in-memory)                          | `DUCKDB_PATH` or `DUCKDB_DATABASE`                                                               | Path aliases: `DUCKDB_DATABASE_PATH`, `DUCKDB_FILE`, `DUCKDB_DB`, `DUCKDB_DSN`. Optional `DUCKDB_SCHEMA` / `DUCKDB_DEFAULT_SCHEMA` (default `main`).                                                    |
| CSV/Excel  | In-memory DuckDB backend (no persistent URL)                                 | `CSV_DIRECTORY` or `CSV_FILES`                                                                   | Mutually exclusive. `directory` loads every `*.csv` / `*.xlsx` in a folder; `files` is an explicit list. Requires `aetherdialect[csv]` (includes DuckDB). Reflection reads headers only; graph is rebuilt per session. |
| MySQL      | `mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4` | `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`                     | Host aliases: `MYSQL_SERVER`, `MYSQL_HOSTNAME`. Port: `MYSQL_TCP_PORT`. User: `MYSQL_USERNAME`. Password: `MYSQL_PWD`. Database: `MYSQL_DB`.                                                            |
| MariaDB    | `mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4` | `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE`           | **MARIADB_* only** — no `MYSQL_*` fallback. Same sqlglot read=`mysql` backend as MySQL.                                                                                                               |
| SQL Server | `mssql+pyodbc://…` (auth-mode dependent)                                     | `SQLSERVER_HOST`, `SQLSERVER_PORT`, `SQLSERVER_DATABASE`, `SQLSERVER_USER`, `SQLSERVER_PASSWORD` | `MSSQL_*` and `AZURE_*` aliases in the flattening table above. `SQLSERVER_AUTH_MODE`: `sql` (default), `windows`, `aad_password`, `aad_sp`.                                       |
| PostgreSQL | `postgresql+psycopg://{user}:{password}@{host}:{port}/{database}`            | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD`      | `PG*` and `POSTGRES_*` aliases in the flattening table above. Optional `POSTGRES_SCHEMA` (default `public`).                                                                      |
| Redshift   | `redshift+redshift_connector://…`                                            | `REDSHIFT_HOST`, `REDSHIFT_PORT`, `REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `REDSHIFT_DATABASE`      | **Redshift-native aliases only** (no `PGHOST` / `PGUSER`). IAM: `REDSHIFT_USE_IAM`, `REDSHIFT_CLUSTER_IDENTIFIER` or `REDSHIFT_WORKGROUP`, optional `REDSHIFT_REGION`.                                  |
| Databricks | SQL warehouse via `databricks-sql-connector`; optional SQLAlchemy URL        | `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_ACCESS_TOKEN`                             | Token aliases: `DATABRICKS_TOKEN`, `DATABRICKS_PAT`, `ACCESS_TOKEN`. Optional `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`. **Fallback:** Databricks connector → SQLAlchemy → PySpark / `DatabricksSession` when the warehouse trio is absent or connect fails. |
| Snowflake  | `snowflake://{auth}@{account}/?…`                                            | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`                                                            | `database` optional (slug uses `db` when unset). Password, key-pair, OAuth, or external-browser auth per env. **Fallback:** Snowflake connector (Arrow) → SQLAlchemy → Snowpark active session.          |
| BigQuery   | `bigquery://{project}/{dataset}?location=…`                                  | `BIGQUERY_PROJECT`, `BIGQUERY_DATASET`                                                           | Dataset aliases: `BIGQUERY_DB`, `GCP_DATASET`, `BQ_DATASET`. ADC when `BIGQUERY_CREDENTIALS_PATH` unset. **Result fetch:** `bq_storage` → `bq_client` → SQLAlchemy.                                     |

## Observability

Three channels are documented in the [Integrator guide — Observability](INTEGRATOR_GUIDE.md#observability). This section lists types and catalogs only.

### `audit_sink`

Optional `Callable[[AuditEvent], None] \| None` on `AetherEngine` construction. It is **not** a boolean flag and does **not** print anything by itself. When provided, the engine invokes your function with an `AuditEvent` at coarse lifecycle boundaries.

### `Diagnostic`


| Field         | Type                          | Meaning                                |
| ------------- | ----------------------------- | -------------------------------------- |
| `stage`       | `str`                         | Pipeline stage that emitted the row    |
| `level`       | `str`                         | `info`, `warn`, or `error`             |
| `code`        | `str`                         | Stable code string (see catalog below) |
| `message`     | `str`                         | Human-readable text                    |
| `details`     | `tuple[tuple[str, str], ...]` | Optional key-value metadata            |
| `duration_ms` | `int \| None`                   | Wall-clock duration when applicable    |


Each `SessionStep` from `ask` / `step` includes a `diagnostics` tuple produced during that step.

### `AuditEvent`


| Field           | Type                          | Meaning                           |
| --------------- | ----------------------------- | --------------------------------- |
| `event_type`    | `str`                         | Discriminator (see catalog below) |
| `timestamp_iso` | `str`                         | UTC timestamp when emitted        |
| `question`      | `str \| None`                   | Active question when relevant      |
| `schema_hash`   | `str \| None`                   | Schema fingerprint when relevant   |
| `provider`      | `"openai" \| "azure" \| "mock"` | LLM provider for the event         |
| `details`       | `tuple[tuple[str, str], ...]` | Lightweight metadata              |


SQL and per-step detail belong on the terminal `SessionStep`, not on audit rows.

#### Audit `event_type` catalog

**Session lifecycle**


| `event_type`  | When emitted                                                             |
| ------------- | ------------------------------------------------------------------------ |
| `init`        | After successful `AetherEngine` construction                                 |
| `ask_begin`   | Start of `session.ask(...)`                                              |
| `ask_done`    | Turn completed (`details` includes `outcome`, `kind`)                    |
| `ask_error`   | Terminal failure or fatal guard error                                    |
| `ask_blocked` | `ask` rejected before raise (non-`str` question or `SessionActiveError`) |


**Admin operations**


| `event_type`                | When emitted                                |
| --------------------------- | ------------------------------------------- |
| `apply_schema_overrides`    | After `apply_schema_overrides` persists     |
| `clear_persisted_overrides` | After overrides sidecar removal and rebuild |
| `clear_template_store`      | After template tree removal and reload      |
| `clear_simulation_caches`   | After QSim or seed-warmup cache deletion    |
| `clear_all_learning`        | After combined learning clears              |


**Write queue drain** (writer session only)


| `event_type`                    | When emitted                                     |
| ------------------------------- | ------------------------------------------------ |
| `write_queue_feedback_record`   | Writer applied a queued `feedback_record`        |
| `write_queue_template_reject`   | Writer applied a queued `template_reject`        |
| `write_queue_template_accept`   | Writer applied a queued `template_accept`        |
| `write_queue_override_proposal` | Writer materialised a queued `override_proposal` |


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


**Configuration**


| Code                        | Typical use                                |
| --------------------------- | ------------------------------------------ |
| `CONFIG_FILE_VALUE_APPLIED` | TOML value overrode `os.environ` for a key |


**Catch-all**


| Code          | Typical use                          |
| ------------- | ------------------------------------ |
| `ENGINE_INFO` | Progress, CLI echo, schema summaries |


**Validation (lowercase)** — `SessionStep.diagnostics` may also carry EXPLAIN- and validator-derived codes (for example `explain_seq_scan_indexed`). Treat unknown codes as opaque. Soft versus hard EXPLAIN behaviour is in the Security guide (see the README documentation map).

## `AetherEngine` constructor

```python
AetherEngine(
    engine_context: EngineContext | str | None = None,
    *,
    artifacts_dir: str | None = None,
    config_file: str | os.PathLike[str] | None = None,
    execution_engine: Any = None,
    native_connection: Any = None,
    audit_sink: Callable[[AuditEvent], None] | None = None,
    role: SchemaRole = "owner",
) -> None
```


| Parameter          | Type                                              | Meaning                                                                 |
| ------------------ | ------------------------------------------------- | ----------------------------------------------------------------------- |
| `engine_context`   | `EngineContext \| str \| None`                    | Master or named engine context. Pass an `EngineContext` to define scope (owner persists; `name="master"` builds the graph, other names define subset specs over the existing master graph). Pass a `str` name to consume a stored context (consumers must use a name only). When omitted, a persisted master context is loaded from artifacts; if none exists, `ConfigError`. |
| `artifacts_dir`    | `str \| None`                                     | Optional root; engine files under `<root>/aetherdialect/<slug>/`        |
| `config_file`      | `str \| os.PathLike[str] \| None`                 | TOML config path; when omitted, settings come from `os.environ` only    |
| `execution_engine` | `Any`                                             | Optional SQLAlchemy engine (caller-owned pool or read replica)          |
| `native_connection` | `Any`                                          | Optional native DuckDB or SQLite connection for embedded in-memory DBs |
| `audit_sink`       | `Callable[[AuditEvent], None] \| None`            | Optional lifecycle audit callback                                       |
| `role`             | `"owner" \| "consumer"`                           | `owner` may mutate shared artifacts; `consumer` pins owner snapshot id  |


Raises `ConfigError`, `ConnectionError`, `MigrationPendingError`, or other failures documented under [Exceptions](#exceptions).

## `schema_migration_map.json`

Written in the working directory when migration requires operator action; consumed on the next successful `AetherEngine(...)` construction and renamed to `schema_migration_map.applied.json` (timestamped variant when a prior applied file exists). Workflow is in the [User guide — Migration](USER_GUIDE.md).


| Field                                       | Type           | Meaning                                                         |
| ------------------------------------------- | -------------- | --------------------------------------------------------------- |
| `version`                                   | `int`          | Map format version (`1` today)                                  |
| `action`                                    | `str`          | `remap`, `destructive`, or `abort`                              |
| `table_renames`                             | array          | Objects with `from` / `to` table names                          |
| `column_renames`                            | array          | Objects with `table`, `from`, `to` column names                 |
| `dropped_tables`                            | array of `str` | Tables removed from the warehouse                               |
| `dropped_columns`                           | array          | Objects with `table` and `column`                               |
| `added_tables`                              | array of `str` | Tables added to the warehouse                                   |
| `added_columns`                             | array          | Objects with `table` and `column`                               |
| `refresh_existing_descriptions_on_addition` | `bool`         | Default `false`; may refresh descriptions when tables are added |


## Schema overrides JSON (`schema_overrides.json`)

Version `1`. Export writes `./schema_overrides.json` in the process working directory. Apply reads the same path. Workflow and upgrade behaviour when version mismatches: [User guide — Schema overrides](USER_GUIDE.md#schema-overrides) and [Upgrading](USER_GUIDE.md#upgrading-and-version-compatibility).

### Editable analyst and integrator surface

Hand-edited files for `apply_schema_overrides` should use **plain strings** for descriptions and roles and `null` or a string for `sensitivity`. Do **not** add `owner` keys; the engine treats missing owner as analyst content and records it under the internal `user_override` provenance during apply.


| Location             | Editable fields                                                           |
| -------------------- | ------------------------------------------------------------------------- |
| Each table           | `description` (string), `role` (string)                                   |
| Each column          | `description` (string), `role` (string), `sensitivity` (`null` or string), `usable` (`false` only — marks column unusable; cannot re-enable profiler-omitted columns) |
| Graph                | `foreign_keys_add[]`, `foreign_keys_remove[]`                             |
| Primary keys         | `primary_keys_add[]`, `primary_keys_remove[]`                             |
| Internal persistence | `_internal` (block lists; engine-maintained, not hand-authored)           |


### Full shape (illustrative hand-edited file)

```jsonc
{
  "version": 1,
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

## `AetherEngine` methods

Methods marked **(master context only)** require `EngineContext.name == "master"` on the active instance (typically `role="owner"` with the default master context). Named-space and named-context snapshots persist under the engine storage directory; template learning is partitioned by aetherspace name.


| Method                                                                                                                  | Returns                     | Description                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apply_migration_map(path="schema_migration_map.json", *, config_file=None, engine_context, artifacts_dir, execution_engine=None, native_connection=None, role="owner")` classmethod | `AetherEngine`                  | Copies the editor map into the working directory filename `schema_migration_map.json`, then constructs `AetherEngine`. Optional `execution_engine` / `native_connection` share one in-memory database with migration apply. Pair with the [User guide — Migration](USER_GUIDE.md).                                                                                                                 |
| `aetherspace(name, space_context=None, *, notes_file=None)` **(master context only)**                                   | `AetherSpace`               | With `space_context`: owner define/overwrite snapshot. Without: existence check and read-only descriptor. Cannot redefine `master`.                                                                                                                                                       |
| `export_aetherspace(name)` **(master context only)**                                                                    | `Path`                      | Read-only JSON dump of one named space (or implicit `master`).                                                                                                                                                                                                                            |
| `export_aetherengine(name)` **(master context only)**                                                                   | `Path`                      | Read-only JSON dump of one named engine context (or implicit `master`).                                                                                                                                                                                                                   |
| `list_aetherspaces()` **(master context only)**                                                                         | `tuple[str, ...]`           | Saved space names plus implicit `master`.                                                                                                                                                                                                                                                 |
| `list_aetherengines()` **(master context only)**                                                                        | `tuple[str, ...]`           | Saved engine-context names plus implicit `master`.                                                                                                                                                                                                                                        |
| `execute_sql(sql, params=None, *, as_dataframe=False)`                                                                  | `list[tuple]` or `DataFrame` | Standalone validated `SELECT` through the active dialect and **engine context** scope (not aetherspace). Same `FORBIDDEN_SQL`, AST, and EXPLAIN/cost gates as session execution.                                                                                                          |
| `export_schema_overrides()`                                                                                             | `Path`                      | Writes `./schema_overrides.json` atomically from the live graph.                                                                                                                                                                                                                          |
| `apply_schema_overrides()`                                                                                              | `None`                      | Validates `./schema_overrides.json`, mutates graph, persists gzip schema cache and `applied_overrides.json`, prints summary via notify, renames editor files to `schema_overrides.applied.json` and `schema_overrides.applied.schema.json` (timestamped archives when prior files exist). |
| `show_config()`                                                                                                         | `ConfigSnapshot`            | Redacted snapshot of engine, schema scope, database, and LLM settings.                                                                                                                                                |
| `session(*, mode="writer", space="master")`                                                                              | `PipelineSession`           | Context manager; exit calls `reset()`. No stdout. `space` selects a persisted [AetherSpace](USER_GUIDE.md#aetherspace).                                                                                                                                                                     |
| `asession(*, mode="writer", space="master")`                                                                            | `AsyncPipelineSession`      | Same as `session` on worker threads.                                                                                                                                                                                                                                                      |
| `run_interactive(*, space="master")`                                                                                     | `None`                      | Prints to stdout; one question per call. `space` selects a persisted [AetherSpace](USER_GUIDE.md#aetherspace) (same default as `session`). Prefer `session` for services.                                                                                                                                                   |
| `run_seed_warmup(seed_filepath, interactive_gold=True, *, abort_on_gold_failure=False, max_kept_intents=2000)` | `None` | Full seed warmup with LLM phrasing and template writes; `max_kept_intents=None` keeps every intent that passes quality and dedup (no budget cap). |
| `run_seed_warmup_from_history(sql_history_filepath, *, expand=False, max_kept_intents=2000)` | `None` | SQL-history warmup; `expand=True` widens coverage; `max_kept_intents=None` keeps every intent passing quality/dedup. |
| `run_seed_warmup_from_query_log(lookback_days=730, max_queries=5000, *, expand=False, max_kept_intents=2000, min_runs=1, user_filter=None)` | `None` | Warehouse query-log warmup; same `expand` and `max_kept_intents` semantics as SQL-history warmup. |
| `get_schema_stats()`                                                                                                    | `SchemaStatsSnapshot`       | Copy of internal graph counters (for example `table_count`) for dashboards or health checks; not required for normal Q&A.                                                                                                                                                                 |
| `write_queue_path` (property)                                                                                           | `Path`                      | Absolute path to `write_queue.jsonl` under the engine storage directory (`artifacts_dir` / dialect subfolder). Same file readers append to and writers drain.                                                                                                                             |
| `get_seed_warmup_summary()`                                                                                             | `SeedWarmupSummarySnapshot` | Reads newest seed-warmup summary file if present.                                                                                                                                                                                                                                         |
| `get_qsim_summary(start, end)`                                                                                          | `QSimSummarySnapshot`       | Reads QSim summary index for inclusive version range.                                                                                                                                                                                                                                     |
| `get_questions_only(version)`                                                                                           | `None`                      | Prints numbered questions and writes `qsim_v{version}_questions.txt` in the working directory.                                                                                                                                                                                            |
| `run_qsim(num_intents=20, num_questions=100, seed=None)`                                                                | `None`                      | QSim generator; prints summary.                                                                                                                                                                                                                                                           |
| `clear_persisted_overrides()`                                                                                           | `bool`                      | Removes overrides sidecar and schema cache when present; rebuilds init bundle; returns whether a sidecar existed.                                                                                                                                                                         |
| `clear_template_store()`                                                                                                | `bool`                      | Removes `intent_templates/` (header and shards) and legacy `intent_templates.json.gz` when present; rebuilds init bundle.                                                                                                                                                                 |
| `clear_simulation_caches()`                                                                                             | `int`                       | Deletes QSim and seed-warmup artifacts; returns removed file count; rebuilds init bundle.                                                                                                                                                                                                 |
| `clear_all_learning(*, keep_overrides=True)`                                                                            | `None`                      | Clears templates, simulation caches, and optionally overrides; rebuilds init bundle.                                                                                                                                                                                                      |


## `PipelineSession` methods


| Method                                        | Returns                    | Contract                                                       |
| --------------------------------------------- | -------------------------- | -------------------------------------------------------------- |
| `ask(question: str)`                          | `SessionStep`              | Starts a turn; raises `SessionActiveError` if busy.            |
| `step(response=None)`                         | `SessionStep`              | Supplies user text for a suspend.                              |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep`              | Auto-answers yes-or-no suspends; `on_confirm` is `"y"` or `"n"`; raises on free-text suspends. |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | `SessionStep` | Auto-answers yes-or-no and free-text suspends until the turn ends or an unexpected `reply_shape` appears. |
| `reuse_saved_question(question_old, question_new, new_values)` | `SessionStep` | Re-executes a stored template with caller-supplied bind values keyed by template handles; terminal step includes `parameters`. Raises `ConfigError` when no template matches or values are invalid. |
| `awaiting_prompt()`                           | `bool`                     | `True` when the next input must go to `step`.                  |
| `reset()`                                     | `None`                     | Clears suspend state, choice queues, and partial turn state.   |
| `__enter__` / `__exit__`                      | `PipelineSession` / `None` | Exit calls `reset()`.                                          |


`mode="reader"` skips durable template and feedback writes. `mode="writer"` is the default and serialises writer turns with a per-instance lock. Suspend `kind` values and the state machine are in the [Integrator guide](INTEGRATOR_GUIDE.md). Public terminal/suspend kind strings include `execute` (`SESSION_KIND_EXECUTE`) for the separated execute confirmation step after SQL generation.

On terminal success, `step.parameters` is a tuple of frozen `ParameterBinding` rows — one per template handle referenced in the executed SQL — populated by `ask`, `ask_until_done`, `accept_until_done`, and `reuse_saved_question`:

| Field | Type | Meaning |
| --- | --- | --- |
| `handle` | `str` | Bind key (`p1`, `s1`, …); the key callers pass in `reuse_saved_question`'s `new_values`. |
| `current_value` | `str \| int \| float \| bool \| list \| None` | Value bound for this execution. |
| `display_name` | `str` | Short human-readable label for UI display (not a bind key). |
| `upper_handle` | `str` | Optional companion handle for range upper bounds. |
| `unit_handle` | `str` | Optional companion handle for date-window units. |

Display names are resolved once per template and cached in the template store.

## `AsyncPipelineSession` methods


| Method                                        | Returns                         | Contract                                                      |
| --------------------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| `ask(question)`                               | `SessionStep`                   | Delegates to `PipelineSession.ask` on a worker thread.        |
| `step(response=None)`                         | `SessionStep`                   | Delegates to `PipelineSession.step`.                          |
| `ask_until_done(question, *, on_confirm="y")` | `SessionStep`                   | Delegates to `PipelineSession.ask_until_done`.                |
| `accept_until_done(question, *, on_yes_no="y", on_free_text="looks good")` | `SessionStep` | Delegates to `PipelineSession.accept_until_done`.             |
| `reuse_saved_question(question_old, question_new, new_values)` | `SessionStep` | Delegates to `PipelineSession.reuse_saved_question`.          |
| `reset()`                                     | `None`                          | Delegates to `PipelineSession.reset` via `asyncio.to_thread`. |
| `awaiting_prompt()`                           | `bool`                          | Delegates to `PipelineSession.awaiting_prompt`.               |
| `__aenter__` / `__aexit__`                    | `AsyncPipelineSession` / `bool` | Async context manager forwarding to the inner session.        |


## Offline sandbox

Walkthrough: [Sandbox guide](SANDBOX.md).

| Symbol | Role |
| ------ | ---- |
| `AetherEngine.offline_sandbox(...)` | In-memory rental shop + mock LLM; returns a `SandboxHandle` |
| `SandboxHandle.engine` | The `AetherEngine` instance for the open handle |
| `SandboxHandle.session(...)` | Same as `SandboxHandle.engine.session(...)` while the handle is open |
| `SandboxHandle.apply_bundled_schema_overrides()` | Copies bundled override JSON to the working directory and applies or enqueues overrides |
| `SandboxHandle.close()` | Releases temp extract dir and owned artifacts (also called by the context manager) |
| `AetherEngine.sandbox_questions()` | Curated offline practice question list |
| `AetherEngine.sandbox_paraphrase_pairs()` | Canonical→paraphrase pairs from the bundled catalog (corpus build) |
| `AetherEngine.sandbox_validation_failure_demo()` | Questions that should end in terminal validation errors |
| `AetherEngine.sandbox_feedback_demo()` | Anchor question + allowed rejection text for the feedback recipe |
| `MockFixtureMissingError` | Mock LLM has no recorded answer for the requested turn (import from `aetherdialect`) |

Warmup and QSim raise `ConfigError` on sandbox instances. Leaving the handle open leaks temp directories — always use `with AetherEngine.offline_sandbox() as sb:` or call `sb.close()`.

## Exceptions

| Exception | Bases | When raised |
| --- | --- | --- |
| `ConfigError` | `ValueError` | Missing or invalid configuration, ambiguous engine/LLM, or unreadable TOML. |
| `ConnectionError` | `OSError` | Driver-level connection failures after construction. |
| `DatabasePingFailed` | `ConnectionError`, `RetryableError` | Retriable connectivity failures. |
| `LlmTransientFailure` | `RuntimeError`, `RetryableError` | Transient LLM HTTP failures. |
| `MigrationPendingError` | `ValueError` | Migration map missing, invalid, or `action="abort"`. |
| `MockFixtureMissingError` | `RuntimeError` | Mock LLM has no recorded answer for the requested turn (offline sandbox). |
| `SchemaAccessError` | `ValueError` | Unreadable scope, empty visible graph, or ambiguous allow-list entries. |
| `SessionActiveError` | `RuntimeError` | `ask` while a turn is already active. |
| `StatementTimeoutError` | `RuntimeError`, `RetryableError` | Statement timeout from the database engine. |


Catch `RetryableError` to branch on transient failures.

---

**See also:** [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [How it works](HOW_IT_WORKS.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
