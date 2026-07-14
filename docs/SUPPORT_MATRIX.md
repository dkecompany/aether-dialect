# Support matrix

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → this file.

Per-engine capabilities, dialect-specific notes, and constructs the intent IR cannot emit directly. The offline sandbox runs on **in-memory DuckDB only** — other engines apply in production connections.

## Sections

| Section | Contents |
| --- | --- |
| [Quick unsupported SQL](#quick-unsupported-sql) | First-class IR gaps |
| [Legend](#legend) | Capability table terms |
| [IR-unsupported constructs](#ir-unsupported-constructs-and-reformulations) | Reformulation rules |
| [Dialect-specific notes](#dialect-specific-notes) | Per-engine behavior |
| [Engine capabilities](#engine-capabilities) | Feature matrix |

## Quick unsupported SQL

The engine does **not** generate these as first-class IR (the intent parser is instructed to reformulate where possible):

- DML/DDL (`INSERT`, `UPDATE`, `DELETE`, `MERGE`, `CREATE`, …)
- Set operations (`UNION`, `EXCEPT`, `INTERSECT`)
- `EXISTS` / `NOT EXISTS`, lateral joins, recursive CTEs, correlated subqueries
- `DISTINCT ON`, row-skipping `OFFSET` / `FETCH FIRST`
- Many window functions outside the whitelist, JSON path operators, custom aggregates

See [IR-unsupported constructs and reformulations](#ir-unsupported-constructs-and-reformulations) for reformulation rules. Enforcement detail is in the Security guide (README documentation map).

## Legend

Cells in **Engine capabilities** use these terms:

- **native** — the IR represents the feature directly; the dialect renders it with the corresponding native SQL construct.
- **translated** — the IR represents the feature; the dialect rewrites it into one or more native constructs (for example Databricks `DATETRUNC` argument reordering or `ARRAY_CONTAINS` for array membership).
- **partial** — the feature is supported with documented restrictions inside the IR.
- **not supported** — the IR has no first-class representation. The intent-parser system prompt instructs the LLM to reformulate using available primitives where possible.
- **blocked** — the structural validator rejects the construct; it cannot be emitted regardless of how the question is phrased.

Additional notes:

- **"Native" on warehouse SQL engines** (Databricks, Snowflake, BigQuery) means the IR maps to that engine's SQL through its dialect adapter; runtime correctness depends on the warehouse version and grants.
- **sqlglot-backed engines** (all except PostgreSQL DDL parse) use sqlglot AST validation and dialect-specific render hooks; see per-engine notes below.
- **BigQuery** does not reflect foreign keys from the catalog. Supply `EngineContext.sql_file` with join hints when multi-table questions need edges the graph cannot infer.
- **"Blocked"** features cannot be reached even by reformulating the question. The IR is intentionally narrower than full SQL; see the Security guide (README documentation map).

## IR-unsupported constructs and reformulations

This table is the **single source of truth** for constructs the IR cannot emit directly and the reformulations the intent parser is instructed to prefer.


| Construct                                                 | Reason                                                                                                       | Reformulation the engine uses                                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Set difference (`A EXCEPT B`)                             | IR cannot emit                                                                                               | Rewrite as left join from A to B with `IS NULL` on the B side.                                                 |
| `EXISTS` / `NOT EXISTS`                                   | IR cannot emit                                                                                               | Rewrite as left join to the inner subject, `IS NOT NULL` for EXISTS or `IS NULL` for NOT EXISTS.               |
| Anti-join                                                 | IR cannot emit                                                                                               | Same pattern as NOT EXISTS.                                                                                    |
| Correlated subquery                                       | IR cannot emit                                                                                               | Rewrite as an aggregation inside a CTE keyed on the correlation column, joined back.                           |
| Lateral join                                              | IR cannot emit                                                                                               | Rewrite as a CTE-projected aggregate.                                                                          |
| Recursive CTE                                             | not representable; refused.                                                                                  | —                                                                                                              |
| AND-of-OR predicates `(a OR b) AND (c OR d)`              | the `filter_group` IR represents OR-of-AND (DNF); flat `bool_op` chains do not preserve grouping precedence. | The intent parser is instructed to rewrite into an equivalent DNF when feasible; otherwise the engine refuses. |
| `DISTINCT ON`                                             | not representable as a bare clause                                                                           | rewrite as a window-function `ROW_NUMBER() = 1` filter.                                                        |
| Self-join on the same physical table without a CTE bridge | refused.                                                                                                     | —                                                                                                              |


## Dialect-specific notes

### PostgreSQL

- AST validation uses `pglast`. The dialect adapter applies structural rejection rules (CTE checks, `WHERE` subtree, lateral, EXISTS, subquery, set ops, self-join).
- Profiling uses `TABLESAMPLE BERNOULLI` for large tables and `LIMIT` on views; categorical stats land in `frequent_values`, FK overlap probes in `value_overlap_sample`.
- `EngineContext.sql_file` DDL is parsed with **pglast** (CREATE/ALTER grammar) before regex reflect or LLM fallback.
- `EXPLAIN` is run as a smoke check; failures are classified by code.

### Databricks

- AST validation uses `sqlglot` with the `databricks` dialect token (not legacy `spark`).
- `CROSS JOIN` is allowed for scalar subquery emission, with the rhs name in the scalar-allowed set (same rule as other sqlglot engines).
- `DATETRUNC` post-pass reorders arguments to match the warehouse's expected form when the Spark dialect renders date truncations.
- Sub-day temporal arithmetic uses `INTERVAL` directly; day and above use `date_sub` / `add_months`.
- The Databricks SQL connector is preferred when the SQL warehouse trio (host, HTTP path, token) is configured. PySpark is the fallback.
- Profiling large tables uses `TABLESAMPLE (pct)`; `EngineContext.sql_file` DDL is parsed with **sqlglot** (Spark dialect).

### MySQL

- AST validation uses `sqlglot` with the MySQL dialect.
- Profiling and execution use SQLAlchemy with `pymysql`.
- Array columns use `JSON_CONTAINS` / `JSON_TABLE` render paths.
- Foreign keys are reflected from `information_schema` for InnoDB tables.
- Partition pruning injects predicates from `information_schema.partitions` when partition metadata is reflected.
- ENUM/SET domains, generated columns, AUTO_INCREMENT PK detection, and index metadata from `information_schema.statistics` feed EXPLAIN index-awareness diagnostics.
- Profiling row-count sampling uses a `WHERE RAND()` predicate inside a subquery (not `TABLESAMPLE`). `EngineContext.sql_file` DDL is parsed with **sqlglot** (MySQL dialect).

### MariaDB

- **MySQL alias** — dialect behavior is identical to MySQL; configure `[engine] selected = "mariadb"` with `MARIADB_*` env keys when targeting MariaDB.
- Registered as engine `mariadb` with sqlglot read=`mysql` and pymysql execution (`mysql+pymysql://…` URLs).
- Connection env keys are **MARIADB_*** only (`MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE`); no `MYSQL_*` fallback.
- AST validation uses `sqlglot` with the MySQL dialect token; render and array paths match MySQL (`JSON_CONTAINS`, `JSON_TABLE`, `WHERE RAND()` profiling subquery, `EXPLAIN FORMAT=JSON`, `MAX_EXECUTION_TIME` timeouts, InnoDB FK reflection, partition inject).

### DuckDB

- AST validation uses `sqlglot` with the DuckDB dialect.
- Embedded file or `:memory:` database via `duckdb:///` SQLAlchemy URLs.
- Native `VARCHAR[]` / `LIST` arrays use `list_contains` and `UNNEST` render paths; `ILIKE` is native.
- EXPLAIN text is scanned for cross-product / nested-loop shapes; row estimates parse `EC=` markers when present.
- Profiling large tables uses `USING SAMPLE … PERCENT (bernoulli)` inside a sampled subquery.
- Schema reflection uses SQLAlchemy over `information_schema`; `EngineContext.sql_file` DDL is parsed with **sqlglot** (DuckDB dialect).

### CSV / Excel (`csv`)

- Reads `*.csv` / `*.xlsx` files from `CSV_DIRECTORY` or `CSV_FILES`, loads them into an in-memory DuckDB database each session, and reuses the DuckDB SQL/execution path.
- Schema is derived from file headers plus sample-based type inference; persisted schema graphs lock column types across reloads.
- DDL probe uses source-file content hashes and mtimes (no live catalog or `sql_file`).

### SQLite

- AST validation uses `sqlglot` with the SQLite dialect.
- Embedded file or `:memory:` database via `sqlite:///` SQLAlchemy URLs (stdlib `pysqlite`).
- **Limitations:** weak catalog typing (value types come from profiling heuristics); no native arrays (JSON1 `json_each` emulation); date windows and diffs use `date('now', …)` / `julianday` rather than `DATE_TRUNC`; EXPLAIN QUERY PLAN provides diagnostics only (no row-count estimate, so cost gates are inactive); no `TABLESAMPLE` (profiling uses `LIMIT` sampling).
- DDL probe uses `sqlite_master` + `PRAGMA table_info` instead of `information_schema`.
- `EngineContext.sql_file` DDL is parsed with **sqlglot** (SQLite dialect).

### SQL Server

- AST validation uses `sqlglot` with the T-SQL dialect.
- EXPLAIN uses `SET SHOWPLAN_ALL ON` for row estimates and `SET SHOWPLAN_XML ON` for missing-index and scan-shape diagnostics (separate batches).
- Query-log warmup prefers Query Store (`sys.query_store_*`) when enabled, falling back to `sys.dm_exec_query_stats`.
- Azure AD auth modes (`aad_password`, `aad_sp`) are wired into SQLAlchemy ODBC URLs.
- Reflection captures identity/computed columns and index metadata (including columnstore/filtered indexes).
- `LIMIT` is transpiled to `OFFSET … FETCH NEXT …` where supported.
- Array columns use `OPENJSON` render paths.
- Profiling large tables uses `TABLESAMPLE SYSTEM`; statement timeouts at **execute** use the ODBC driver command timeout, not a session `SET`. `EngineContext.sql_file` DDL is parsed with **sqlglot** (T-SQL).

### Snowflake

- AST validation uses `sqlglot` with the Snowflake dialect.
- EXPLAIN uses `EXPLAIN USING JSON`; parses partition assignment and spill warnings.
- Array columns use `ARRAY_CONTAINS` and `LATERAL FLATTEN` render paths.
- When a native `snowflake.connector` connection is available, execution can use the Arrow result backend.
- Profiling large tables uses `SAMPLE (pct)`; execute timeouts use `ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS`. `EngineContext.sql_file` DDL is parsed with **sqlglot** (Snowflake dialect).

### BigQuery

- AST validation uses `sqlglot` with the BigQuery dialect.
- EXPLAIN is implemented as a dry-run query job (validation without billed bytes).
- Execution uses `google-cloud-bigquery` (`bq_client` reader kind) via the transitive dependency of `sqlalchemy-bigquery`.
- **No foreign-key reflection.** Multi-table join planning requires `EngineContext.sql_file` (DDL with join hints) or `foreign_keys_add` overrides.
- Bound parameters are rewritten from `:name` to `@name` at execution time.
- Tables with `require_partition_filter` may receive synthetic partition predicates when the intent carries date signals. Shared guard logic lives in the dialect pruning helper used at SQL finalization.
- Profiling large tables uses `TABLESAMPLE SYSTEM`. At execute, `AETHERDIALECT_MAX_QUERY_COST_BYTES` maps to `maximum_bytes_billed` on query jobs. `EngineContext.sql_file` DDL is parsed with **sqlglot** (BigQuery dialect).

### Amazon Redshift

- AST validation uses `sqlglot` with the Redshift dialect.
- Largely Postgres-compatible rendering; `ILIKE` is native.
- SUPER-typed array columns use json-extract render paths.
- EXPLAIN text is scanned for broadcast (`DS_BCAST_INNER`) and redistribution (`DS_DIST_*`) plan shapes.
- Sort-key / distkey / diststyle metadata drives partition-style predicate injection when the intent carries matching filters.
- Foreign-key reflection is partial; `EngineContext.sql_file` is recommended when join edges are missing from the catalog.
- Profiling row-count sampling uses a `WHERE RANDOM()` predicate inside a subquery (not `TABLESAMPLE`). `EngineContext.sql_file` DDL is parsed with **sqlglot** (Redshift dialect).

### Engine capabilities


| Capability | SQLite | DuckDB | CSV/Excel | MySQL | MariaDB | SQL Server | PostgreSQL | Redshift | Databricks | Snowflake | BigQuery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXPLAIN form | `EXPLAIN QUERY PLAN` | plain `EXPLAIN` (text) | n/a | `EXPLAIN FORMAT=JSON` | `EXPLAIN FORMAT=JSON` | `SHOWPLAN_ALL` + `SHOWPLAN_XML` | `EXPLAIN (FORMAT JSON)` | plain `EXPLAIN` | `EXPLAIN COST` | `EXPLAIN USING JSON` | dry-run job |
| Query-log source | n/a | n/a | n/a | `performance_schema` | `performance_schema` | Query Store | `pg_stat_statements` | `svl_qlog` | `system.query.history` | `QUERY_HISTORY` | `JOBS` |
| SQL-file parse grammar | sqlglot (sqlite) | sqlglot (duckdb) | n/a | sqlglot (mysql) | sqlglot (mysql) | sqlglot (tsql) | pglast | sqlglot (redshift) | sqlglot (databricks) | sqlglot (snowflake) | sqlglot (bigquery) |
| Partition / cluster pruning | no-op | inject | n/a | partition inject | partition inject | partition inject | partition inject | sortkey/distkey inject | Delta (inject) | cluster-key inject | partition inject + guard |
| FK catalog reflection | PRAGMA | SQLAlchemy | n/a | full (InnoDB) | full (InnoDB) | full (`sys.*`) | full | partial | Unity Catalog | full | **none** |
| Result reader backends | SQLAlchemy | SQLAlchemy | SQLAlchemy | SQLAlchemy | SQLAlchemy | SQLAlchemy | SQLAlchemy | SQLAlchemy | connector → Spark | connector → Snowpark | `bq_storage` → SQLAlchemy |


---

**See also:** [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [API reference](API_REFERENCE.md) · [How it works](HOW_IT_WORKS.md) · [Security](SECURITY.md) · [README](../README.md#documentation)
