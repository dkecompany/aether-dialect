# Support matrix

What the engine refuses, how to reformulate refused shapes, per-dialect behaviour notes, and the per-engine capability tables. Closed-enough openness guarantees (determinism, replay, numerics, cost caps): [Integrator guide - Guarantees](INTEGRATOR_GUIDE.md#guarantees). Trust boundaries and disclosure rules: [Security](SECURITY.md). The offline sandbox runs on **in-memory DuckDB only** - other engines apply in production connections.

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [User guide](USER_GUIDE.md) -> [Integrator guide](INTEGRATOR_GUIDE.md) -> [Sandbox guide](SANDBOX.md) -> [API reference](API_REFERENCE.md) -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> this document.

## Sections

| Section | Contents |
| --- | --- |
| [Supported intent constructs](#supported-intent-constructs) | Question shapes the engine answers natively |
| [Refused constructs](#refused-constructs) | Honest refusals and rewrite guidance |
| [Dialect-specific notes](#dialect-specific-notes) | Per-engine behaviour |
| [Legend](#legend) | Vocabulary used in the capability tables |
| [Engine capabilities](#engine-capabilities) | Feature matrix |
| [Integrator guarantees](INTEGRATOR_GUIDE.md#guarantees) | Determinism, replay limits, federation numerics, cost caps, configuration boundary |

---

## Supported intent constructs

These question shapes are **supported natively**. The engine compiles them deterministically without emitting literal `EXISTS`, `NOT EXISTS`, or bare `DISTINCT ON` SQL.

| Construct | How the engine renders it |
| --- | --- |
| Semi-join (rows in A with a match in B) | Inner join to a distinct key projection from the matched side |
| Anti-join (rows in A with no match in B; set difference over keys) | Left join plus a null test on a synthetic presence marker |
| Per-partition top one (`DISTINCT ON` semantics) | Row-number = 1 over the partition (never bare `DISTINCT ON` SQL) |

Cross-member union is **not** a set operator inside one warehouse statement - it uses declared logical tables and coordinator glue.

### Negated filter rendering

Negated comparisons (`!=`, `NOT IN`, `NOT LIKE`, `NOT BETWEEN`) on a **nullable** column render with an extra `OR column IS NULL` branch so rows with an unknown value are included. On a **non-nullable** column the comparison is emitted without that branch. Use `IS NULL` or `IS NOT NULL` when the question is only about missing values.

Text equality comparisons ignore letter case. When the column collation is already case-insensitive, the engine compares values directly; otherwise it folds both sides to lower case.

Integer division always yields a fractional result: when the warehouse would otherwise truncate integer operands, the numerator is cast before division.

Aggregates over an empty row set follow SQL semantics: `COUNT` is `0`; `SUM`, `AVG`, `MIN`, and `MAX` are unknown (`NULL`). When a preserved anchor row survives a join with no matching detail rows, counts are coalesced to zero but other aggregates remain unknown.

## Refused constructs

Each refusal records an honest reason. Reformulation is suggested only where one exists. Enforcement of the analytical subset is in [Security - Why SQL injection is not possible](SECURITY.md#4-why-sql-injection-is-not-possible).

| Construct | Reason |
| --- | --- |
| DML/DDL (`INSERT`, `UPDATE`, `DELETE`, `MERGE`, `CREATE`, ...) | Outside the analytical read-only subset |
| Set operations inside one member statement (`UNION`, `EXCEPT`, `INTERSECT`) | Cross-member union uses declared logical tables and coordinator glue |
| `LATERAL` / lateral join | Correlated row generation has no fixed shape across engines; reformulate as a CTE-projected aggregate |
| Recursive CTE | Requires fixpoint semantics and a termination bound the engine does not model |
| Correlated subquery beyond intermediate CTE shapes | Reformulate as an aggregation inside a CTE keyed on the correlation column, then join back |
| Predicate nesting beyond depth 3 | Simplify the boolean logic or split the question |
| Self-join on the same physical table without a CTE bridge | Lift one branch into a CTE step, then join |
| Literal `EXISTS` / `NOT EXISTS` subqueries in output SQL | Use existence (semi-join) or absence (anti-join) question shapes instead |
| Bare `DISTINCT ON` SQL | Use the per-partition top-one question shape instead |
| Row-skipping `OFFSET` / `FETCH FIRST` | Row identity under offset depends on a total order the engine does not guarantee |
| Window functions outside the whitelist | Not modeled in this release |
| JSON path operators | Escape the typed column model the schema graph is built on |
| Custom aggregates | Not modeled in this release |
| `PIVOT` | Output column set depends on data values, so the shape is not knowable at validation |
| `GROUPING SETS`, `ROLLUP`, `CUBE` | Subtotals change result grain mid-set; not modeled |
| Unrelated table pairing (cross product) | No cardinality bound below the product of the inputs; a filter after pairing does not restore one |
| Comparison-only table beyond join reach | A table brought in only for a cross-table comparison must be reachable through declared keys or semantic edges |
| Ambiguous calendar date literal | Absolute date-window bounds must use ISO 8601 form such as `2020-01-15` |

You can compare an entity to itself, and compare value ranges between two tables when the engine can already relate them through declared keys or edges.

## Dialect-specific notes

Registered engines (canonical order): `sqlite`, `duckdb`, `csv`, `mysql`, `mariadb`, `sqlserver`, `postgresql`, `redshift`, `databricks`, `snowflake`, `bigquery`. There is **no** separate `excel` engine - Excel workbooks (`.xlsx`, not `.xls`) are loaded by the **`csv`** file engine into in-memory DuckDB.

### PostgreSQL

- AST validation uses `pglast`. The dialect adapter rejects lateral joins, existential subqueries, disallowed nested subqueries, top-level set operations, recursive CTEs, and direct self-joins on the same table.
- Profiling uses `TABLESAMPLE BERNOULLI` for large tables and `LIMIT` on views; categorical statistics and foreign-key overlap sampling feed the schema graph.
- Supplied DDL files are parsed with **pglast** (CREATE/ALTER grammar) before regex reflect or provider fallback.
- `EXPLAIN (FORMAT JSON, COSTS true)` is run as a smoke check; failures are classified by code.
- Query-log warmup uses `pg_stat_statements` when the extension is available.

### Databricks

- AST validation uses `sqlglot` with the `databricks` dialect token.
- `CROSS JOIN` is allowed for scalar subquery emission, with the rhs name in the scalar-allowed set (same rule as other sqlglot engines).
- `DATETRUNC` post-pass reorders arguments to match the warehouse's expected form when the `databricks` dialect renders date truncations.
- Sub-day temporal arithmetic uses `INTERVAL` directly; day and above use `date_sub` / `add_months`.
- The Databricks SQL connector is preferred when the SQL warehouse trio (host, HTTP path, token) is configured. PySpark is the fallback.
- Profiling large tables uses `TABLESAMPLE (pct)`; supplied DDL files are parsed with **sqlglot** (databricks dialect).
- Schema profiling does not apply a portable session statement-timeout SQL hook on this engine; `profile_timeout_ms` is ignored during profiling even when configured.
- EXPLAIN uses `EXPLAIN COST`. Partition pruning injects Delta partition predicates when reflected metadata and your question's filters align.
- Ordered string aggregation is not supported; unordered `string_agg` uses `collect_list` / `array_join`.
- FK reflection uses Unity Catalog `information_schema`.

### MySQL

- AST validation uses `sqlglot` with the MySQL dialect.
- Profiling and execution use SQLAlchemy with `pymysql`.
- Array columns use `JSON_CONTAINS` / `JSON_TABLE` render paths.
- Foreign keys are reflected from `information_schema` for InnoDB tables.
- Partition pruning injects predicates from `information_schema.partitions` when partition metadata is reflected.
- ENUM/SET domains, generated columns, AUTO_INCREMENT PK detection, and index metadata from `information_schema.statistics` feed EXPLAIN index-awareness diagnostics.
- Profiling row-count sampling uses a `WHERE RAND()` predicate inside a subquery (not `TABLESAMPLE`). Supplied DDL files are parsed with **sqlglot** (MySQL dialect).
- EXPLAIN uses `EXPLAIN FORMAT=JSON`. Query-log warmup uses `performance_schema`.
- Median aggregate is not supported on this engine.

### MariaDB

- **MySQL alias** - dialect behaviour is identical to MySQL; configure `[engine] selected = "mariadb"` with `MARIADB_*` env keys when targeting MariaDB.
- Registered as engine `mariadb` with sqlglot read=`mysql` and pymysql execution (`mysql+pymysql://...` URLs).
- Connection env keys are **MARIADB_*** only (`MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE`); no `MYSQL_*` fallback.
- AST validation uses `sqlglot` with the MySQL dialect token; render and array paths match MySQL (`JSON_CONTAINS`, `JSON_TABLE`, `WHERE RAND()` profiling subquery, `EXPLAIN FORMAT=JSON`, `MAX_EXECUTION_TIME` timeouts, InnoDB FK reflection, partition inject).
- Median aggregate is not supported on this engine.

### DuckDB

- AST validation uses `sqlglot` with the DuckDB dialect.
- Embedded file or `:memory:` database via `duckdb:///` SQLAlchemy URLs.
- Native `VARCHAR[]` / `LIST` arrays use `list_contains` and `UNNEST` render paths; `ILIKE` is native.
- EXPLAIN text is scanned for cross-product / nested-loop shapes; row estimates parse `EC=` markers when present.
- Profiling large tables uses `USING SAMPLE ... PERCENT (bernoulli)` inside a sampled subquery.
- Schema reflection uses SQLAlchemy over `information_schema`; supplied DDL files are parsed with **sqlglot** (DuckDB dialect).
- Partition pruning injects predicates from reflected partition metadata when present.
- Query-log warmup is unavailable on embedded DuckDB.
- Federated execution uses **in-process DuckDB** as the coordinator that glues per-source member results.

### CSV (`csv`)

- Single registered file engine. Reads `*.csv` and `*.xlsx` files (not `.xls`) from `CSV_DIRECTORY` or `CSV_FILES`, loads them into an in-memory DuckDB database each session, and reuses the DuckDB SQL / EXPLAIN / execution path (`CsvDialect` subclasses `DuckDBDialect`).
- Schema is derived from file headers plus sample-based type inference; persisted schema graphs lock column types across reloads. File sources declare no catalog foreign keys.
- DDL probe uses source-file content hashes and mtimes (no live catalog or supplied DDL file).
- **Upload inspection** — **Inspect first, then construct.** Four severities apply consistently across documentation and code: **Advisory** (auto-applied single interpretation), **Review** (caller must choose via `source_selections`), **Blocking** (readable file, no coherent table - `inspect_tabular_upload` returns `ok=False`), **Fatal** (unreadable or unsupported - `inspect_tabular_upload` raises `ConfigError`). Call `inspect_tabular_upload` before `AetherEngine(...)`; pass accepted `source_selections` on construction. Construction without selections when **Review** issues remain raises `ConfigError` with the report attached. Operator walkthrough: [User guide - CSV and Excel uploads](USER_GUIDE.md#csv-and-excel-uploads).
- **Federation:** requires at least one non-file member — a federation whose members are *all* file (`csv`) engines is refused at registration. A database engine **can** be federated with uploaded files; the CSV member participates in per-source decomposition while the coordinator remains in-process DuckDB.
- Statistical aggregates (`stddev`, `variance`), `array_contains` predicates, and explicit window frame bounds are not supported.

### SQLite

- AST validation uses `sqlglot` with the SQLite dialect.
- Embedded file or `:memory:` database via `sqlite:///` SQLAlchemy URLs (stdlib `pysqlite`).
- **Limitations:** weak catalog typing (value types come from profiling heuristics); no native arrays (JSON1 `json_each` emulation); date windows and diffs use `date('now', ...)` / `julianday` rather than `DATE_TRUNC`; `EXPLAIN QUERY PLAN` provides diagnostics only (no row-count estimate, so cost gates are inactive); no `TABLESAMPLE` (profiling uses `LIMIT` sampling); partition pruning is a no-op.
- Statistical aggregates (`stddev`, `variance`) and `array_contains` predicates are not supported.
- DDL probe uses `sqlite_master` + `PRAGMA table_info` instead of `information_schema`. FK reflection uses `PRAGMA foreign_key_list` when foreign keys are enabled.
- Supplied DDL files are parsed with **sqlglot** (SQLite dialect).
- Can participate as a federation member (cross-source decomposition); the coordinator remains DuckDB.

### SQL Server

- AST validation uses `sqlglot` with the T-SQL dialect.
- EXPLAIN uses `SET SHOWPLAN_ALL ON` for row estimates and `SET SHOWPLAN_XML ON` for missing-index and scan-shape diagnostics (separate batches).
- Query-log warmup prefers Query Store (`sys.query_store_*`) when enabled, falling back to `sys.dm_exec_query_stats`.
- Azure AD auth modes (`aad_password`, `aad_sp`) are wired into SQLAlchemy ODBC URLs.
- Reflection captures identity/computed columns and index metadata (including columnstore/filtered indexes).
- `LIMIT` is transpiled to `OFFSET ... FETCH NEXT ...` where supported.
- `array_contains` predicates are not supported; `OPENJSON` render paths apply to array unnest only.
- Profiling large tables uses `TABLESAMPLE SYSTEM`; statement timeouts at **execute** use the ODBC driver command timeout, not a session `SET`. Schema profiling does not apply a portable `profile_statement_timeout_sql` hook on SQL Server, so `profile_timeout_ms` does not bound profiling queries. Supplied DDL files are parsed with **sqlglot** (T-SQL).
- Partition pruning injects predicates when your question's filters match reflected partition metadata.

### Snowflake

- AST validation uses `sqlglot` with the Snowflake dialect.
- Snowpark is a client API (DataFrame layer) over the same Snowflake SQL engine, not a separate SQL grammar. Every parse, validation, and render path uses the Snowflake dialect token; an active Snowpark session is only a result-fetch backend.
- EXPLAIN uses `EXPLAIN USING JSON`; parses partition assignment and spill warnings.
- Array columns use `ARRAY_CONTAINS` and `LATERAL FLATTEN` render paths.
- Result backends prefer native `snowflake.connector` Arrow, then SQLAlchemy, then an active Snowpark session when reachable.
- Profiling large tables uses `SAMPLE (pct)`; execute timeouts use `ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS`. Supplied DDL files are parsed with **sqlglot** (Snowflake dialect).
- Cluster-key predicate injection when your question filters on columns that match declared cluster keys. Query-log warmup uses `INFORMATION_SCHEMA.QUERY_HISTORY`.

### BigQuery

- AST validation uses `sqlglot` with the BigQuery dialect.
- EXPLAIN is implemented as a dry-run query job (validation without billed bytes).
- Execution uses `google-cloud-bigquery` via the transitive dependency of `sqlalchemy-bigquery`; row fetch prefers the BigQuery Storage API (`bq_storage`) when available, else SQLAlchemy.
- **No foreign-key reflection.** Multi-table join planning requires a supplied DDL file with join hints or schema overrides when profiling cannot infer edges.
- Bound parameters are rewritten from `:name` to `@name` at execution time.
- Tables with `require_partition_filter` may receive synthetic partition predicates when your question includes date filters on partition columns.
- Profiling large tables uses `TABLESAMPLE SYSTEM`. At execute, the configured maximum query cost maps to `maximum_bytes_billed` on query jobs. Supplied DDL files are parsed with **sqlglot** (BigQuery dialect).
- Query-log warmup uses `INFORMATION_SCHEMA.JOBS`.

### Amazon Redshift

- AST validation uses `sqlglot` with the Redshift dialect.
- Largely Postgres-compatible rendering; `ILIKE` is native.
- SUPER-typed array columns use json-extract render paths.
- EXPLAIN text is scanned for broadcast (`DS_BCAST_INNER`) and redistribution (`DS_DIST_*`) plan shapes.
- Sort-key / distkey / diststyle metadata drives partition-style predicate injection when your question's filters match reflected key columns.
- Foreign-key reflection loads from `information_schema` and `svv_foreign_keys` but is treated as **partial** in practice; a supplied DDL file is recommended when join edges are missing from the catalog.
- Profiling row-count sampling uses a `WHERE RANDOM()` predicate inside a subquery (not `TABLESAMPLE`). Supplied DDL files are parsed with **sqlglot** (Redshift dialect).
- Query-log warmup uses `svl_qlog`.

## Legend

Cells in **[Engine capabilities](#engine-capabilities)** use the vocabulary below.

| Cell vocabulary | Meaning |
| --- | --- |
| Concrete SQL / API form (for example `EXPLAIN QUERY PLAN`, `EXPLAIN FORMAT=JSON`, `SHOWPLAN_ALL` + `SHOWPLAN_XML`, dry-run job) | The exact EXPLAIN-class or catalog mechanism that dialect uses. |
| Query-log source (for example `performance_schema`, `pg_stat_statements`, `svl_qlog`, `QUERY_HISTORY`, `JOBS`) | Catalog view or table used for query-log warmup when available. |
| `n/a` | The capability does not apply to that engine (for example query-log warmup on embedded engines, or SQL-file DDL parse on the CSV file engine which fingerprints source files instead). For **Federation coordinator**, `n/a` means the engine does not *host* the coordinator (only in-process DuckDB does); the engine may still be a federation *member*. |
| `inject` / `partition inject` / `sortkey/distkey inject` / `Delta (inject)` / `cluster-key inject` / `partition inject + guard` | Partition or cluster pruning appends predicates from reflected partition, sortkey, distkey, cluster, or Delta metadata when your question's filters match. |
| `no-op` | Partition pruning leaves the SQL unchanged (SQLite). |
| `sqlglot (dialect)` / `pglast` | Grammar used to parse supplied DDL files before reflection. |
| `full (InnoDB)` / `full` / `full (\`sys.*\`)` / `Unity Catalog` / `PRAGMA` / `SQLAlchemy` | Foreign-key (and related constraint) reflection source and completeness. |
| `partial` | Catalog FK reflection exists but is incomplete enough that operators should supply a DDL file or overrides when edges are missing (Redshift). |
| `none` | Dialect returns no live FK metadata (BigQuery). |
| `per-source + glue` | Federated plans run a member statement per source, then the DuckDB coordinator combines results. |
| `member only` | Engine may be a federation member; it does not host the coordinator. |
| `DuckDB in-process` | The in-process DuckDB coordinator hosts federation glue for this engine. |
| `->` chains in **Result reader backends** | Preference order of row-fetch backends (first available wins), for example `connector -> SQLAlchemy -> Spark` or `bq_storage -> SQLAlchemy`. |

Additional notes:

- Warehouse SQL engines map questions through their dialect adapter; runtime correctness depends on warehouse version and grants.
- All engines except PostgreSQL use sqlglot AST validation and dialect-specific render hooks; PostgreSQL uses `pglast`.
- Session results: single-engine `SessionStep.sql` is a dialect string; federated analytical turns return `step.sql` as a member `source_id` → SQL mapping. Schema/business-knowledge questions can finish as `kind="meta"` without SQL. After accept, re-run with `execute_template(step.template_id, params)`; use `export_knowledge()`, `export_space_knowledge(space=...)`, and `export_metadata(space=...)` for structured knowledge and inventory read-back (details: [User guide - Asking a question](USER_GUIDE.md#asking-a-question)).

## Engine capabilities

| Capability | SQLite | DuckDB | CSV | MySQL | MariaDB | SQL Server | PostgreSQL | Redshift | Databricks | Snowflake | BigQuery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXPLAIN form | `EXPLAIN QUERY PLAN` | plain `EXPLAIN` (text) | plain `EXPLAIN` (text) | `EXPLAIN FORMAT=JSON` | `EXPLAIN FORMAT=JSON` | `SHOWPLAN_ALL` + `SHOWPLAN_XML` | `EXPLAIN (FORMAT JSON)` | plain `EXPLAIN` | `EXPLAIN COST` | `EXPLAIN USING JSON` | dry-run job |
| Query-log source | n/a | n/a | n/a | `performance_schema` | `performance_schema` | Query Store -> `dm_exec_query_stats` | `pg_stat_statements` | `svl_qlog` | `system.query.history` | `QUERY_HISTORY` | `JOBS` |
| SQL-file parse grammar | sqlglot (sqlite) | sqlglot (duckdb) | n/a | sqlglot (mysql) | sqlglot (mysql) | sqlglot (tsql) | pglast | sqlglot (redshift) | sqlglot (databricks) | sqlglot (snowflake) | sqlglot (bigquery) |
| Partition / cluster pruning | no-op | inject | inject | partition inject | partition inject | partition inject | partition inject | sortkey/distkey inject | Delta (inject) | cluster-key inject | partition inject + guard |
| FK catalog reflection | PRAGMA | SQLAlchemy | n/a | full (InnoDB) | full (InnoDB) | full (`sys.*`) | full | partial | Unity Catalog | full | **none** |
| Result reader backends | SQLAlchemy | SQLAlchemy | SQLAlchemy | connector -> SQLAlchemy | connector -> SQLAlchemy | SQLAlchemy | SQLAlchemy | connector -> SQLAlchemy | connector -> SQLAlchemy -> Spark | connector (Arrow) -> SQLAlchemy -> Snowpark | `bq_storage` -> SQLAlchemy |
| Federation coordinator | n/a | DuckDB in-process | member only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Cross-source decomposition | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue | per-source + glue |

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [README](../README.md)
