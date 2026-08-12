"""Artifact persistence, locks, manifests, write-queue, and migration helpers."""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import MagicMock

from packaging.version import InvalidVersion, Version

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
from ._config import (
    EngineLimits,
)
from ._constants import (
    AETHERSPACES_SEGMENT,
    ARTIFACT_DIR_MODE,
    ARTIFACT_FILE_MODE,
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_LOCK_FILENAME,
    ARTIFACT_LOCK_POLL_INTERVAL_SECONDS,
    ARTIFACT_LOCK_TIMEOUT_SECONDS,
    ARTIFACT_MANIFEST_FILENAME,
    AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY,
    AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT,
    DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL,
    DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK,
    DIAGNOSTIC_CODE_WRITE_QUEUE_FULL,
    FAILURE_TRACE_ROTATE_BYTES,
    FEDERATION_ARTIFACT_FORMAT_VERSION,
    JSON_COMPACT_SEPARATORS,
    KNOWLEDGE_EXPORT_FORMAT_VERSION,
    LEGACY_ARTIFACT_FILENAMES,
    LEGACY_ARTIFACT_GLOBS,
    MIN_COMPATIBLE_PACKAGE_VERSION,
    SIMULATION_CACHE_EXACT_FILENAMES,
    SIMULATION_CACHE_GLOB_PATTERNS,
    TEMPLATE_STORE_SEGMENT,
    WRITE_QUEUE_FILENAME,
)
from ._contracts_base import (
    ArtifactLockTimeoutError,
    ArtifactManifest,
    ConfigError,
    DomainKnowledgeEntry,
    MigrationTier,
    OpenResourceInventory,
    StructuralKnowledgeFact,
)
from ._contracts_core import (
    LlmExecutionConfig,
    StepResult,
    WriteQueueEvent,
)
from ._contracts_schema import (
    SchemaGraph,
)
from ._utils import (
    coerce_format_version,
    debug,
    domain_knowledge_artifact_path,
    domain_knowledge_digest,
    format_failure_trace,
    format_versions_match,
    notify,
    resolved_engine_limits,
    structural_knowledge_artifact_path,
)

_RESOURCE_REGISTRY_LOCK = threading.Lock()


_LOCK_REENTRY = threading.local()
_ARTIFACT_LOCK_TIMEOUT_DEFAULT = object()

_POISONED_CONNECTION_LOCK = threading.Lock()

_POISONED_CONNECTION_IDS: set[int] = set()
_FORK_PARENT_OWNED_CONNECTION_IDS: set[int] = set()
_FORK_PARENT_OWNED_DIALECT_IDS: set[int] = set()
_REGISTERED_DIALECT_IDS: set[int] = set()

_AFTER_FORK_CALLBACKS: list[Callable[[], None]] = []

_OPEN_ARTIFACT_LOCKS: set[str] = set()
_OWNED_TEMP_DIRECTORIES: set[str] = set()
_LIVE_CONNECTIONS: set[int] = set()
_OWNER_CLOSE_RESOURCES: dict[int, Any] = {}


_structural_migration_handler: Callable[..., None] | None = None
_knowledge_migration_handler: Callable[..., None] | None = None


def read_gzip_json(path: str) -> Any:
    """Load a JSON value from a UTF-8 document stored as gzip."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _chmod_artifact_file(path: str | os.PathLike[str]) -> None:
    try:
        os.chmod(path, ARTIFACT_FILE_MODE)
    except OSError:
        pass


def write_gzip_json_atomic(path: str, obj: Any, *, sort_keys: bool) -> None:
    """Serialize ``obj`` to compact UTF-8 JSON, gzip it, and replace. ``path`` atomically."""
    raw = json.dumps(obj, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=sort_keys).encode("utf-8")
    compressed = gzip.compress(raw)
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    lock_path = abs_path + ".__write.lock"
    with _file_lock(lock_path, timeout=ARTIFACT_LOCK_TIMEOUT_SECONDS):
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json.gz", dir=directory)
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(compressed)
            for attempt in range(5):
                try:
                    os.replace(tmp_path, abs_path)
                    _chmod_artifact_file(abs_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def write_text_atomic(path: str | os.PathLike[str], text: str, *, suffix: str = ".json.tmp") -> None:
    """Replace *path* with *text* atomically via a temporary file in the same directory."""
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    os.makedirs(directory, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
        for attempt in range(5):
            try:
                os.replace(tmp_path, abs_path)
                _chmod_artifact_file(abs_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def write_json_atomic(
    path: str | os.PathLike[str],
    obj: Any,
    *,
    sort_keys: bool = True,
    indent: int | None = None,
) -> None:
    """Serialize ``obj`` to UTF-8 JSON and replace ``path`` atomically."""
    if indent is not None:
        text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=JSON_COMPACT_SEPARATORS, sort_keys=sort_keys)
    write_text_atomic(path, text)


def save_domain_knowledge_artifact(
    artifacts_dir: str | os.PathLike[str],
    entries: Sequence[DomainKnowledgeEntry],
    *,
    notes_sha256: str | None = None,
    scope_fingerprint: str | None = None,
) -> None:
    """Persist master domain knowledge next to the schema cache (atomic replace)."""
    payload: dict[str, Any] = {
        "format_version": KNOWLEDGE_EXPORT_FORMAT_VERSION,
        "domain_knowledge": [
            {
                "key": e.key,
                "kind": e.kind,
                "text": e.text,
                "referenced_entities": sorted(e.referenced_entities),
            }
            for e in entries
        ],
        "domain_knowledge_digest": domain_knowledge_digest(entries),
    }
    notes_hash = str(notes_sha256 or "").strip()
    if notes_hash:
        payload["notes_sha256"] = notes_hash
    scope_fp = str(scope_fingerprint or "").strip()
    if scope_fp:
        payload["scope_fingerprint"] = scope_fp
    write_json_atomic(domain_knowledge_artifact_path(artifacts_dir), payload, sort_keys=True, indent=2)


def save_structural_knowledge_artifact(
    artifacts_dir: str | os.PathLike[str],
    facts: Sequence[StructuralKnowledgeFact],
    *,
    notes_sha256: str | None = None,
    scope_fingerprint: str | None = None,
) -> None:
    """Persist structural knowledge extracted from notes (single write- back point)."""
    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "structural_knowledge": [f.to_dict() for f in facts],
    }
    notes_hash = str(notes_sha256 or "").strip()
    if notes_hash:
        payload["notes_sha256"] = notes_hash
    scope_fp = str(scope_fingerprint or "").strip()
    if scope_fp:
        payload["scope_fingerprint"] = scope_fp
    write_json_atomic(structural_knowledge_artifact_path(artifacts_dir), payload, sort_keys=True, indent=2)


def _new_owner_close_resources() -> Any:
    class _OwnerCloseResources:
        __slots__ = ("temp_directories", "bundle_archives", "write_queue_handles", "live_connections")

        def __init__(self) -> None:
            self.temp_directories: list[str] = []
            self.bundle_archives: list[Any] = []
            self.write_queue_handles: list[Any] = []
            self.live_connections: list[Any] = []

    return _OwnerCloseResources()


def _owner_resource_bundle_locked(owner: Any) -> Any:
    bundle = _OWNER_CLOSE_RESOURCES.get(id(owner))
    if bundle is None:
        bundle = _new_owner_close_resources()
        _OWNER_CLOSE_RESOURCES[id(owner)] = bundle
    return bundle


def _owner_resource_bundle(owner: Any) -> Any:
    with _RESOURCE_REGISTRY_LOCK:
        return _owner_resource_bundle_locked(owner)


def open_resource_inventory() -> OpenResourceInventory:
    """Return counts of open locks, temp directories, and live connections owned by the library."""
    with _RESOURCE_REGISTRY_LOCK:
        return OpenResourceInventory(
            locks=len(_OPEN_ARTIFACT_LOCKS),
            temp_directories=len(_OWNED_TEMP_DIRECTORIES),
            live_connections=len(_LIVE_CONNECTIONS),
        )


def track_close_temp_directory(owner: Any, path: str) -> None:
    """Record a library-owned temporary directory to remove during :meth:`release_close_resources`."""
    abs_path = os.path.abspath(path)
    with _RESOURCE_REGISTRY_LOCK:
        bundle = _owner_resource_bundle_locked(owner)
        if abs_path not in bundle.temp_directories:
            bundle.temp_directories.append(abs_path)
        _OWNED_TEMP_DIRECTORIES.add(abs_path)


def register_live_connection(connection: Any, *, owner: Any | None = None) -> None:
    """Record a live database handle attributable to the library."""
    with _RESOURCE_REGISTRY_LOCK:
        _LIVE_CONNECTIONS.add(id(connection))
        if owner is not None:
            bundle = _owner_resource_bundle_locked(owner)
            if connection not in bundle.live_connections:
                bundle.live_connections.append(connection)


def unregister_live_connection(connection: Any) -> None:
    """Drop a live database handle from the library inventory."""
    with _RESOURCE_REGISTRY_LOCK:
        _LIVE_CONNECTIONS.discard(id(connection))


def track_close_bundle_archive(owner: Any, archive: Any) -> None:
    """Record an open bundle archive to close during :meth:`release_close_resources`."""
    with _RESOURCE_REGISTRY_LOCK:
        bundle = _owner_resource_bundle_locked(owner)
        if archive not in bundle.bundle_archives:
            bundle.bundle_archives.append(archive)


def track_close_write_queue_handle(owner: Any, handle: Any) -> None:
    """Record an open write-queue file handle to close during :meth:`release_close_resources`."""
    with _RESOURCE_REGISTRY_LOCK:
        bundle = _owner_resource_bundle_locked(owner)
        if handle not in bundle.write_queue_handles:
            bundle.write_queue_handles.append(handle)


def register_dialect_live_handles(dialect: Any, *, owner: Any | None = None) -> None:
    """Register dialect connection handles for inventory and close release."""
    with _RESOURCE_REGISTRY_LOCK:
        _REGISTERED_DIALECT_IDS.add(id(dialect))
    for attr in ("connection", "engine", "_native_connection", "_snowflake_connection"):
        if not hasattr(dialect, attr):
            continue
        handle = getattr(dialect, attr, None)
        if handle is None or isinstance(handle, MagicMock):
            continue
        register_live_connection(handle, owner=owner)


def unregister_dialect_live_handles(
    dialect: Any,
    *,
    borrowed_execution_engine: Any | None = None,
    borrowed_native_connection: Any | None = None,
) -> None:
    """Drop dialect connection handles from the library inventory before disposal."""
    with _RESOURCE_REGISTRY_LOCK:
        _REGISTERED_DIALECT_IDS.discard(id(dialect))
    for attr in ("connection", "engine", "_native_connection", "_snowflake_connection"):
        handle = getattr(dialect, attr, None)
        if handle is None:
            continue
        if attr == "engine" and handle is borrowed_execution_engine:
            continue
        if (
            attr in ("connection", "_native_connection", "_snowflake_connection")
            and handle is borrowed_native_connection
        ):
            continue
        unregister_live_connection(handle)


def _remove_owned_temp_directory(path: str) -> None:
    abs_path = os.path.abspath(path)
    with _RESOURCE_REGISTRY_LOCK:
        _OWNED_TEMP_DIRECTORIES.discard(abs_path)
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path, ignore_errors=True)


def _close_tracked_handle(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except (OSError, AttributeError, TypeError):
            pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        else:
            return True


def _lock_metadata_path(lock_path: str) -> str:
    return f"{lock_path}.holder"


def _parse_lock_metadata_bytes(raw: bytes) -> tuple[int | None, float | None]:
    if not raw:
        return None, None
    if sys.platform == "win32" and len(raw) > 1:
        payload_raw = raw[1:]
        try:
            payload = json.loads(payload_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            raw = payload_raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    pid_raw = payload.get("pid")
    mono_raw = payload.get("monotonic")
    pid = int(pid_raw) if isinstance(pid_raw, int) or (isinstance(pid_raw, str) and pid_raw.isdigit()) else None
    mono: float | None
    try:
        mono = float(mono_raw) if mono_raw is not None else None
    except (TypeError, ValueError):
        mono = None
    return pid, mono


def _read_lock_metadata(lock_path: str) -> tuple[int | None, float | None]:
    sidecar = _lock_metadata_path(lock_path)
    try:
        with open(sidecar, "rb") as fh:
            raw = fh.read()
    except OSError:
        raw = b""
    if raw:
        pid, mono = _parse_lock_metadata_bytes(raw)
        if pid is not None:
            return pid, mono
    try:
        with open(lock_path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None, None
    return _parse_lock_metadata_bytes(raw)


def _remove_lock_metadata(lock_path: str) -> None:
    try:
        os.unlink(_lock_metadata_path(lock_path))
    except OSError:
        pass


def _release_artifact_lock_files(artifacts_dir: str) -> None:
    root = os.path.abspath(artifacts_dir)
    lock_path = os.path.join(root, ARTIFACT_LOCK_FILENAME)
    try:
        os.unlink(lock_path)
    except OSError:
        pass
    _remove_lock_metadata(lock_path)
    with _RESOURCE_REGISTRY_LOCK:
        _OPEN_ARTIFACT_LOCKS.discard(os.path.normcase(lock_path))


def release_close_resources(owner: Any) -> None:
    """Release locks, temp dirs, bundle archives, and write-queue handles owned by *owner*."""
    owner_id = id(owner)
    with _RESOURCE_REGISTRY_LOCK:
        bundle = _OWNER_CLOSE_RESOURCES.pop(owner_id, None)
    if bundle is None:
        bundle = _new_owner_close_resources()

    write_queue_handle = getattr(owner, "_write_queue_handle", None)
    if write_queue_handle is not None:
        bundle.write_queue_handles.append(write_queue_handle)
    bundle_archive = getattr(owner, "_bundle_archive", None)
    if bundle_archive is not None:
        bundle.bundle_archives.append(bundle_archive)
    for handle in list(bundle.write_queue_handles):
        _close_tracked_handle(handle)

    for archive in list(bundle.bundle_archives):
        _close_tracked_handle(archive)

    temp_dirs = list(bundle.temp_directories)
    for path in list(getattr(owner, "_owned_temp_dirs", ()) or ()):
        if path not in temp_dirs:
            temp_dirs.append(str(path))
    for path in temp_dirs:
        _remove_owned_temp_directory(path)

    artifacts_dir = getattr(owner, "_artifacts_dir", None)
    if artifacts_dir is not None:
        _release_artifact_lock_files(str(artifacts_dir))
    federation_storage_dir = getattr(owner, "_federation_storage_dir", None)
    if federation_storage_dir is not None:
        _release_artifact_lock_files(str(federation_storage_dir))

    for connection in list(bundle.live_connections):
        unregister_live_connection(connection)


def _write_lock_metadata(lock_path: str, fh: Any) -> None:
    payload = json.dumps(
        {"pid": os.getpid(), "monotonic": time.monotonic()}, separators=JSON_COMPACT_SEPARATORS
    ).encode(
        "utf-8",
    )
    sidecar = _lock_metadata_path(lock_path)
    directory = os.path.dirname(os.path.abspath(sidecar)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".holder_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(payload)
        os.replace(tmp_path, sidecar)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    if sys.platform == "win32":
        return
    else:
        fh.seek(0)
        fh.truncate(0)
        fh.write(payload)
        try:
            fh.flush()
        except OSError:
            pass


def _artifact_lock_timeout_seconds(explicit: float | object) -> float:
    if explicit is not _ARTIFACT_LOCK_TIMEOUT_DEFAULT:
        return float(cast(float, explicit))
    return float(resolved_engine_limits().artifact_lock_timeout_seconds)


def _artifacts_dir_is_local_filesystem(path: str) -> bool:
    """Return True when *path* resides on a local filesystem mount."""
    abs_path = os.path.abspath(path)
    if sys.platform == "win32":
        if abs_path.startswith("\\\\"):
            return False
        drive, _ = os.path.splitdrive(abs_path)
        if not drive:
            return True
        import ctypes

        remote = 4
        return cast(bool, ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") != remote)
    else:
        return True


def warn_if_artifacts_dir_not_local(artifacts_dir: str) -> None:
    """Emit ``ARTIFACTS_DIR_NOT_LOCAL`` when *artifacts_dir* is not on a local filesystem."""
    if _artifacts_dir_is_local_filesystem(artifacts_dir):
        return
    notify(
        f"artifacts directory {artifacts_dir!r} is not on a local filesystem; advisory artifact locks may be unreliable",
        stage="init",
        code=DIAGNOSTIC_CODE_ARTIFACTS_DIR_NOT_LOCAL,
        level="warning",
    )


def _get_reentry_map() -> dict[str, int]:
    m = getattr(_LOCK_REENTRY, "map", None)
    if m is None:
        m = {}
        _LOCK_REENTRY.map = m
    return m


@contextmanager
def _file_lock(lock_path: str, *, timeout: float, artifacts_dir: str | None = None) -> Iterator[None]:
    """Acquire an exclusive OS-level lock on ``lock_path`` for the duration of the context."""
    abs_lock_path = os.path.abspath(lock_path)
    lock_directory = artifacts_dir or os.path.dirname(abs_lock_path) or "."
    os.makedirs(os.path.dirname(abs_lock_path) or ".", mode=ARTIFACT_DIR_MODE, exist_ok=True)
    deadline = time.monotonic() + max(timeout, 0.0)
    stale_retried = False
    fh: Any = None
    acquired = False
    try:
        while not acquired:
            if fh is not None:
                fh.close()
                fh = None
            fh = open(abs_lock_path, "a+b")
            while True:
                if sys.platform == "win32":
                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError:
                        locked = False
                    else:
                        locked = True
                else:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        locked = False
                    else:
                        locked = True
                if locked:
                    acquired = True
                    _write_lock_metadata(abs_lock_path, fh)
                    with _RESOURCE_REGISTRY_LOCK:
                        _OPEN_ARTIFACT_LOCKS.add(os.path.normcase(abs_lock_path))
                    break
                if time.monotonic() >= deadline:
                    holder_pid, _ = _read_lock_metadata(abs_lock_path)
                    raise ArtifactLockTimeoutError(
                        lock_directory,
                        holder_pid,
                        timeout=timeout,
                        lock_path=abs_lock_path,
                    )
                holder_pid, _ = _read_lock_metadata(abs_lock_path)
                if not stale_retried and holder_pid is not None and not _process_exists(holder_pid):
                    notify(
                        f"removing stale artifact lock at {abs_lock_path!r} left by pid {holder_pid}",
                        stage="pipeline",
                        code=DIAGNOSTIC_CODE_STALE_ARTIFACT_LOCK,
                        level="warning",
                    )
                    fh.close()
                    fh = None
                    try:
                        os.unlink(abs_lock_path)
                    except OSError:
                        pass
                    _remove_lock_metadata(abs_lock_path)
                    stale_retried = True
                    break
                time.sleep(ARTIFACT_LOCK_POLL_INTERVAL_SECONDS)
        yield
    finally:
        if acquired and fh is not None:
            if sys.platform == "win32":
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()
            with _RESOURCE_REGISTRY_LOCK:
                _OPEN_ARTIFACT_LOCKS.discard(os.path.normcase(abs_lock_path))
            try:
                os.unlink(abs_lock_path)
            except OSError:
                pass
            _remove_lock_metadata(abs_lock_path)
        elif fh is not None:
            fh.close()


@contextmanager
def artifact_lock(
    artifacts_dir: str,
    *,
    timeout: float | object = _ARTIFACT_LOCK_TIMEOUT_DEFAULT,
) -> Iterator[None]:
    """Reentrant per-``artifacts_dir`` lock covering load, mutate, and save sequences for template learning. The lock is advisory: it serialises cooperating processes only and requires a local filesystem for reliable mutual exclusion. The lock file path joins *artifacts_dir* with :data:`ARTIFACT_LOCK_FILENAME` from ``aetherdialect._config``. Nested ``with artifact_lock`` blocks on the same directory bump a per-thread refcount without deadlocking. Cross-thread and cross-process callers serialize at the OS level when they cooperate on the same lock file."""
    resolved_timeout = _artifact_lock_timeout_seconds(timeout)
    abs_dir = os.path.abspath(artifacts_dir)
    os.makedirs(abs_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    lock_path = os.path.join(abs_dir, ARTIFACT_LOCK_FILENAME)
    key = os.path.normcase(lock_path)
    reentry = _get_reentry_map()
    depth = reentry.get(key, 0)
    if depth > 0:
        reentry[key] = depth + 1
        try:
            yield
        finally:
            reentry[key] -= 1
            if reentry[key] <= 0:
                reentry.pop(key, None)
        return
    reentry[key] = 1
    try:
        with _file_lock(lock_path, timeout=resolved_timeout, artifacts_dir=abs_dir):
            yield
    finally:
        reentry[key] -= 1
        if reentry[key] <= 0:
            reentry.pop(key, None)


def artifact_package_version_string() -> str:
    try:
        return version("aetherdialect")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _manifest_path(artifacts_dir: str) -> str:
    """Return the absolute path to ``artifact_manifest.json`` under ``artifacts_dir``."""
    return os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)


def read_artifact_manifest(artifacts_dir: str) -> ArtifactManifest | None:
    """Load artifact manifest JSON if present."""
    path = _manifest_path(artifacts_dir)
    with artifact_lock(artifacts_dir):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            ver = coerce_format_version(data.get("artifact_format_version", "0") or "0")
        except (TypeError, ValueError):
            ver = "0"
        return ArtifactManifest(
            artifact_format_version=ver,
            created_with_package_version=str(data.get("created_with_package_version", "") or ""),
            min_compatible_package_version=str(data.get("min_compatible_package_version", "") or ""),
            last_action=str(data.get("last_action", "") or ""),
            last_action_at=str(data.get("last_action_at", "") or ""),
            structural_hash=str(data.get("structural_hash", "") or ""),
            profiling_hash=str(data.get("profiling_hash", "") or ""),
            scope_hash=str(data.get("scope_hash", "") or ""),
            effective_structural_hash=str(data.get("effective_structural_hash", "") or ""),
            schema_graph_id=str(data.get("schema_graph_id", "") or ""),
            notes_hash=str(data.get("notes_hash", "") or ""),
            semantic_edges_hash=str(data.get("semantic_edges_hash", "") or ""),
            last_migration_tier=str(data.get("last_migration_tier", "") or ""),
            last_migration_at=str(data.get("last_migration_at", "") or ""),
            source_probe=str(data.get("source_probe", "") or ""),
            store_fingerprint=str(data.get("store_fingerprint", "") or ""),
            ingest_source_probe=str(data.get("ingest_source_probe", "") or ""),
        )


def write_artifact_manifest(
    artifacts_dir: str,
    *,
    structural_hash: str = "",
    profiling_hash: str = "",
    scope_hash: str = "",
    effective_structural_hash: str = "",
    schema_graph_id: str = "",
    notes_hash: str = "",
    semantic_edges_hash: str = "",
    last_migration_tier: str = "",
    last_migration_at: str | None = None,
    last_action: str = "compat_wipe",
    last_corruption_at: str = "",
    source_probe: str | None = None,
    store_fingerprint: str | None = None,
    ingest_source_probe: str | None = None,
) -> None:
    """Write manifest with format version, package version, optional. hashes, and last action. Persists atomically via a temporary file in *artifacts_dir* followed by ``os.replace``."""
    os.makedirs(artifacts_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
    path = _manifest_path(artifacts_dir)
    prior_probe = ""
    prior_store_fp = ""
    prior_ingest_probe = ""
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                prior_data = json.load(fh)
            if isinstance(prior_data, dict):
                prior_probe = str(prior_data.get("source_probe", "") or "")
                prior_store_fp = str(prior_data.get("store_fingerprint", "") or "")
                prior_ingest_probe = str(prior_data.get("ingest_source_probe", "") or "")
        except (json.JSONDecodeError, OSError):
            pass
    mig_at = last_migration_at if last_migration_at is not None else ""
    if last_migration_tier and not mig_at:
        mig_at = datetime.now(UTC).isoformat()
    probe = source_probe if source_probe is not None else prior_probe
    store_fp = store_fingerprint if store_fingerprint is not None else prior_store_fp
    ingest_probe = ingest_source_probe if ingest_source_probe is not None else prior_ingest_probe
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_with_package_version": artifact_package_version_string(),
        "min_compatible_package_version": MIN_COMPATIBLE_PACKAGE_VERSION,
        "last_action": last_action,
        "last_action_at": datetime.now(UTC).isoformat(),
        "structural_hash": structural_hash,
        "profiling_hash": profiling_hash,
        "scope_hash": scope_hash,
        "effective_structural_hash": effective_structural_hash,
        "schema_graph_id": schema_graph_id,
        "notes_hash": notes_hash,
        "semantic_edges_hash": semantic_edges_hash,
        "last_migration_tier": last_migration_tier,
        "last_migration_at": mig_at,
        "last_corruption_at": last_corruption_at or "",
        "source_probe": probe,
        "store_fingerprint": store_fp,
        "ingest_source_probe": ingest_probe,
    }
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json.tmp",
            prefix=".artifact_manifest_",
            dir=artifacts_dir,
            delete=False,
        ) as tf:
            tmp_path = tf.name
            json.dump(payload, tf, ensure_ascii=False, indent=2)
        assert tmp_path is not None
        os.replace(tmp_path, path)
        _chmod_artifact_file(path)
        tmp_path = None
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    debug(f"[core_utils.write_artifact_manifest] path={path}")


def _write_queue_engine_limits() -> EngineLimits:
    return resolved_engine_limits()


def _normalize_write_queue_space_name(space_name: str) -> str:
    """Return a normalized lowercase template-store space segment."""
    raw = str(space_name).strip()
    if not raw or raw in (".", ".."):
        raise ValueError(f"invalid template space name: {space_name!r}")
    lower = raw.lower()
    if "/" in lower or "\\" in lower:
        raise ValueError(f"invalid template space name: {space_name!r}")
    try:
        sanitized = sanitize_name_segment(lower)
    except ValueError:
        raise ValueError(f"invalid template space name: {space_name!r}") from None
    if sanitized != lower:
        raise ValueError(f"invalid template space name: {space_name!r}")
    return lower


def emit_write_queue_event(artifacts_dir: str, event: WriteQueueEvent, *, space_name: str) -> None:
    """Append one JSON line representing a deferred writer event to the artifact write queue."""
    norm_space = _normalize_write_queue_space_name(space_name)
    obj: dict[str, Any] = {
        "kind": event.kind,
        "schema_graph_id": event.schema_graph_id,
        "schema_hash": event.schema_hash,
        "produced_at": event.produced_at,
        "payload": [list(pair) for pair in event.payload],
        "space_name": norm_space,
    }
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    line_bytes = len(line.encode("utf-8"))
    limits = _write_queue_engine_limits()
    record_cap = limits.write_queue_max_record_bytes
    if record_cap is not None and line_bytes > record_cap:
        raise ConfigError(f"write queue record size {line_bytes} exceeds write_queue_max_record_bytes ({record_cap})")
    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    with artifact_lock(artifacts_dir):
        os.makedirs(artifacts_dir, mode=ARTIFACT_DIR_MODE, exist_ok=True)
        file_cap = limits.write_queue_max_file_bytes
        if file_cap is not None:
            current_bytes = os.path.getsize(path) if os.path.isfile(path) else 0
            if current_bytes + line_bytes > file_cap:
                notify(
                    f"write queue file size would exceed write_queue_max_file_bytes ({file_cap}); drain required",
                    stage="pipeline",
                    code=DIAGNOSTIC_CODE_WRITE_QUEUE_FULL,
                    details=(
                        ("path", path),
                        ("file_bytes", str(current_bytes)),
                        ("record_bytes", str(line_bytes)),
                    ),
                )
                return
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def decode_write_queue_event(obj: dict[str, Any]) -> WriteQueueEvent | None:
    """Parse one write-queue JSON object into a :class:`WriteQueueEvent`, or return ``None`` when invalid."""
    kinds = {
        "template_accept",
        "template_reject",
        "paraphrase_emit",
        "override_proposal",
        "feedback_record",
    }
    kind = str(obj.get("kind") or "")
    if kind not in kinds:
        return None
    schema_graph_id = str(obj.get("schema_graph_id") or "")
    schema_hash = str(obj.get("schema_hash") or "")
    produced_at = str(obj.get("produced_at") or "")
    if not schema_graph_id:
        return None
    raw_pl = obj.get("payload")
    if not isinstance(raw_pl, list):
        return None
    pairs: list[tuple[str, str]] = []
    for row in raw_pl:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        pairs.append((str(row[0]), str(row[1])))
    write_kind = cast(
        Literal[
            "template_accept",
            "template_reject",
            "paraphrase_emit",
            "override_proposal",
            "feedback_record",
        ],
        kind,
    )
    return WriteQueueEvent(
        kind=write_kind,
        schema_graph_id=schema_graph_id,
        schema_hash=schema_hash,
        produced_at=produced_at,
        payload=tuple(pairs),
    )


def write_queue_event_space_name(obj: dict[str, Any]) -> str | None:
    """Return the aetherspace partition name stamped on a write-queue JSON object."""
    raw = obj.get("space_name")
    if raw is None:
        return None
    cleaned = str(raw).strip().lower()
    return cleaned or None


def wipe_filenames(artifacts_dir: str, names: tuple[str, ...]) -> int:
    """Remove named files directly under *artifacts_dir*; return count removed."""
    removed = 0
    for name in names:
        fp = os.path.join(artifacts_dir, name)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    return removed


def wipe_globs(artifacts_dir: str, patterns: tuple[str, ...]) -> int:
    """Remove files matching glob patterns relative to *artifacts_dir*; return count removed."""
    removed = 0
    for pattern in patterns:
        for fp in glob.glob(os.path.join(artifacts_dir, pattern)):
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1
    return removed


def wipe_versioned_artifacts(artifacts_dir: str) -> None:
    """Remove on-disk template and simulation cache files under *artifacts_dir*."""
    wipe_filenames(artifacts_dir, LEGACY_ARTIFACT_FILENAMES)
    wipe_globs(artifacts_dir, LEGACY_ARTIFACT_GLOBS)
    refresh_migration_simulation_caches(artifacts_dir)
    _clear_write_queue_file(artifacts_dir)
    _remove_aetherspace_snapshots(artifacts_dir)
    partitioned = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    if os.path.isdir(partitioned):
        shutil.rmtree(partitioned, ignore_errors=True)


def refresh_migration_simulation_caches(artifacts_dir: str) -> int:
    """Remove QSim and seed-warmup simulation artifacts; return count of files removed."""
    count = wipe_filenames(artifacts_dir, SIMULATION_CACHE_EXACT_FILENAMES)
    count += wipe_globs(artifacts_dir, SIMULATION_CACHE_GLOB_PATTERNS)
    return count


def _clear_write_queue_file(artifacts_dir: str) -> bool:
    path = os.path.join(artifacts_dir, WRITE_QUEUE_FILENAME)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def _remove_aetherspace_snapshots(artifacts_dir: str) -> bool:
    root = os.path.join(artifacts_dir, AETHERSPACES_SEGMENT)
    if not os.path.isdir(root):
        return False
    shutil.rmtree(root, ignore_errors=True)
    return True


def refresh_migration_auxiliary_artifacts(artifacts_dir: str, *, tier: MigrationTier) -> None:
    """Refresh or wipe auxiliary learning artifacts for the given migration tier."""
    if tier == MigrationTier.DESTRUCTIVE:
        refresh_migration_simulation_caches(artifacts_dir)
        _clear_write_queue_file(artifacts_dir)
        _remove_aetherspace_snapshots(artifacts_dir)
        return
    refresh_migration_simulation_caches(artifacts_dir)
    _clear_write_queue_file(artifacts_dir)


def detect_legacy_artifacts(artifacts_dir: str) -> list[str]:
    """Return artifact filenames suggesting a pre-manifest install populated this directory. A directory is considered pre-manifest when at least one versioned artifact (schema graph snapshot, template store, qsim run, seed warmup cache) is present *but* no ``artifact_manifest.json`` exists alongside it. Such artifacts use an on-disk format that predates the migration manifest and cannot be safely loaded by the current code path."""
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return []
    if os.path.isfile(os.path.join(artifacts_dir, ARTIFACT_MANIFEST_FILENAME)):
        return []
    found: set[str] = set()
    for name in LEGACY_ARTIFACT_FILENAMES:
        if os.path.isfile(os.path.join(artifacts_dir, name)):
            found.add(name)
    for pattern in LEGACY_ARTIFACT_GLOBS:
        for fp in glob.glob(os.path.join(artifacts_dir, pattern)):
            if os.path.isfile(fp):
                found.add(os.path.basename(fp))
    return sorted(found)


def manifest_matches_schema(manifest: ArtifactManifest, schema: SchemaGraph) -> bool:
    if manifest.schema_graph_id and schema.schema_graph_id:
        if manifest.schema_graph_id != schema.schema_graph_id:
            return False
    return (
        manifest.structural_hash == schema.structural_hash
        and manifest.profiling_hash == schema.profiling_hash
        and manifest.scope_hash == schema.scope_hash
        and manifest.effective_structural_hash == schema.effective_structural_hash
        and (manifest.notes_hash or "") == (schema.notes_hash or "")
        and (manifest.semantic_edges_hash or "") == (schema.semantic_edges_hash or "")
    )


def artifact_manifest_incompatible_with_package(manifest: ArtifactManifest | None) -> bool:
    """Return True when the manifest requires a newer package or unknown artifact format."""
    if manifest is None:
        return False
    fmt = coerce_format_version(manifest.artifact_format_version)
    if not (
        format_versions_match(fmt, ARTIFACT_FORMAT_VERSION)
        or format_versions_match(fmt, FEDERATION_ARTIFACT_FORMAT_VERSION)
        or fmt in {"0", ""}
    ):
        return True
    min_cv = (manifest.min_compatible_package_version or "").strip()
    if not min_cv:
        return False
    try:
        return Version(artifact_package_version_string()) < Version(min_cv)
    except (InvalidVersion, TypeError, ValueError):
        return True


def register_structural_migration_handler(handler: Callable[..., None]) -> None:
    """Register the owner-side structural migration callback."""
    global _structural_migration_handler
    _structural_migration_handler = handler


def register_knowledge_migration_handler(handler: Callable[..., None]) -> None:
    """Register the owner-side knowledge migration callback."""
    global _knowledge_migration_handler
    _knowledge_migration_handler = handler


def apply_structural_migration_to_persisted_scopes(
    engine_dir: str,
    *,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
    column_retypes: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Apply table/column migration to persisted aetherspace and named context specs."""
    if _structural_migration_handler is None:
        raise RuntimeError("structural migration handler is not registered")
    _structural_migration_handler(
        engine_dir,
        dropped_tables=dropped_tables,
        dropped_columns=dropped_columns,
        table_renames=table_renames,
        column_renames=column_renames,
        column_retypes=column_retypes,
    )


def migrate_engine_knowledge_artifacts(
    engine_dir: str,
    schema: Any,
    *,
    schema_json_path: str | None = None,
    dropped_tables: tuple[str, ...] = (),
    dropped_columns: tuple[str, ...] = (),
    table_renames: tuple[tuple[str, str], ...] = (),
    column_renames: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Migrate engine DK + structural knowledge for delete/rename via the registered handler."""
    if _knowledge_migration_handler is None:
        return
    _knowledge_migration_handler(
        engine_dir,
        schema,
        schema_json_path=schema_json_path,
        dropped_tables=dropped_tables,
        dropped_columns=dropped_columns,
        table_renames=table_renames,
        column_renames=column_renames,
    )


def load_runtime_config(
    *,
    merged_env: Mapping[str, str],
) -> LlmExecutionConfig:
    """
    Merge built-in defaults with a caller-supplied environment.

    snapshot into one frozen LLM execution config. Resolution order is defaults first, then the environment layer keyed by the canonical Azure OpenAI and execution-limit variable names. Args: merged_env: Mapping of effective environment strings used for the environment merge layer. Returns: The frozen :class:`LlmExecutionConfig`.

    Raises: ValueError: When numeric fields are negative after merge.
    """

    def _env_text(name: str) -> str:
        return str(merged_env.get(name, "") or "").strip()

    defaults: dict[str, Any] = {
        "azure_endpoint": "",
        "azure_api_key": "",
        "azure_api_version": "",
        "deployment_light": "",
        "deployment_heavy": "",
        "max_query_cost_rows": 50_000_000,
        "max_query_cost_bytes": 50_000_000_000,
        "statement_timeout_ms": 30_000,
        "llm_timeout_ms": 60_000,
        "profile_timeout_ms": 120_000,
        "explain_timeout_ms": None,
    }
    env_map: dict[str, str] = {
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_api_key": "AZURE_OPENAI_API_KEY",
        "azure_api_version": "AZURE_OPENAI_API_VERSION",
        "deployment_light": AZURE_OPENAI_ENV_DEPLOYMENT_LIGHT,
        "deployment_heavy": AZURE_OPENAI_ENV_DEPLOYMENT_HEAVY,
        "max_query_cost_rows": "AETHERDIALECT_MAX_QUERY_COST_ROWS",
        "max_query_cost_bytes": "AETHERDIALECT_MAX_QUERY_COST_BYTES",
        "statement_timeout_ms": "AETHERDIALECT_STATEMENT_TIMEOUT_MS",
        "llm_timeout_ms": "AETHERDIALECT_LLM_TIMEOUT_MS",
        "profile_timeout_ms": "AETHERDIALECT_PROFILE_TIMEOUT_MS",
        "explain_timeout_ms": "AETHERDIALECT_EXPLAIN_TIMEOUT_MS",
    }
    merged: dict[str, Any] = dict(defaults)
    for canon, env_name in env_map.items():
        raw = _env_text(env_name)
        if not raw:
            continue
        if canon in {
            "max_query_cost_rows",
            "max_query_cost_bytes",
            "statement_timeout_ms",
            "llm_timeout_ms",
            "profile_timeout_ms",
        }:
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            if iv < 0:
                raise ValueError(f"Invalid non-negative integer for {env_name}")
            merged[canon] = iv
        elif canon == "explain_timeout_ms":
            try:
                iv = int(raw, 10)
            except ValueError:
                continue
            merged[canon] = None if iv <= 0 else iv
        else:
            merged[canon] = raw
    for name in (
        "max_query_cost_rows",
        "max_query_cost_bytes",
        "statement_timeout_ms",
        "llm_timeout_ms",
        "profile_timeout_ms",
    ):
        v = merged.get(name)
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"Invalid runtime config for {name}")
    exm = merged.get("explain_timeout_ms")
    if exm is not None and (not isinstance(exm, int) or exm < 0):
        raise ValueError("Invalid runtime config for explain_timeout_ms")
    cfg = LlmExecutionConfig(
        azure_endpoint=str(merged.get("azure_endpoint") or ""),
        azure_api_key=str(merged.get("azure_api_key") or ""),
        azure_api_version=str(merged.get("azure_api_version") or ""),
        deployment_light=str(merged.get("deployment_light") or ""),
        deployment_heavy=str(merged.get("deployment_heavy") or ""),
        max_query_cost_rows=int(merged["max_query_cost_rows"]),
        max_query_cost_bytes=int(merged["max_query_cost_bytes"]),
        statement_timeout_ms=int(merged["statement_timeout_ms"]),
        llm_timeout_ms=int(merged["llm_timeout_ms"]),
        profile_timeout_ms=int(merged["profile_timeout_ms"]),
        explain_timeout_ms=merged.get("explain_timeout_ms"),
    )
    return cfg


def mark_connection_poisoned(conn: Any) -> None:
    """Mark *conn* as unsafe to reuse after a timed-out or aborted worker."""
    with _POISONED_CONNECTION_LOCK:
        _POISONED_CONNECTION_IDS.add(id(conn))


def is_connection_poisoned(conn: Any) -> bool:
    """Return whether *conn* was marked unsafe for pool reuse."""
    with _POISONED_CONNECTION_LOCK:
        return id(conn) in _POISONED_CONNECTION_IDS


def clear_connection_poison(conn: Any) -> None:
    """Clear any poison mark on *conn* before returning it to active use."""
    with _POISONED_CONNECTION_LOCK:
        _POISONED_CONNECTION_IDS.discard(id(conn))


def is_fork_parent_owned_connection(conn: Any) -> bool:
    """Return whether *conn* was inherited from the parent process after fork."""
    with _RESOURCE_REGISTRY_LOCK:
        return id(conn) in _FORK_PARENT_OWNED_CONNECTION_IDS


def assert_connection_usable_after_fork(conn: Any) -> None:
    """Raise when *conn* is a parent-owned handle in a forked child process."""
    if is_fork_parent_owned_connection(conn):
        raise RuntimeError(
            "Database connection inherited from the parent process after fork. "
            "Construct a new AetherEngine in the child process instead.",
        )


def assert_dialect_usable_after_fork(dialect: Any) -> None:
    """Raise when *dialect* is a parent-owned handle in a forked child process."""
    with _RESOURCE_REGISTRY_LOCK:
        if id(dialect) in _FORK_PARENT_OWNED_DIALECT_IDS:
            raise RuntimeError(
                "Database connection inherited from the parent process after fork. "
                "Construct a new AetherEngine in the child process instead.",
            )


def mark_fork_parent_owned_live_handles() -> None:
    """Mark every tracked live connection and dialect as parent-owned in a fork child."""
    with _RESOURCE_REGISTRY_LOCK:
        _FORK_PARENT_OWNED_CONNECTION_IDS.update(_LIVE_CONNECTIONS)
        _FORK_PARENT_OWNED_DIALECT_IDS.update(_REGISTERED_DIALECT_IDS)


def register_after_fork_callback(callback: Callable[[], None]) -> None:
    """Register a child-process hook invoked after ``fork()``."""
    _AFTER_FORK_CALLBACKS.append(callback)


def _after_fork_in_child() -> None:
    for callback in _AFTER_FORK_CALLBACKS:
        callback()
    mark_fork_parent_owned_live_handles()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def sanitize_name_segment(name: str) -> str:
    """Return a filesystem-safe name segment for templates, spaces, and artifact paths."""
    safe = re.sub(r"[^a-z0-9_-]+", "-", str(name).strip().lower()).strip("-")
    if not safe:
        raise ValueError("name must contain at least one alphanumeric character after sanitization")
    return safe


def _rotate_failure_trace_if_needed(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < FAILURE_TRACE_ROTATE_BYTES:
        return
    rotated = Path(f"{path}.1")
    if rotated.is_file():
        rotated.unlink()
    path.rename(rotated)


def append_failure_trace(step: StepResult | list[StepResult] | object | None, path: str | os.PathLike[str]) -> None:
    """Append a formatted failure trace to the specified results file."""
    if step is None:
        return
    text = format_failure_trace(step)
    if not text:
        return
    p = Path(path)
    _rotate_failure_trace_if_needed(p)
    needs_sep = p.is_file() and p.stat().st_size > 0
    with open(path, "a", encoding="utf-8") as fh:
        if needs_sep:
            fh.write("\n\n" + "=" * 80 + "\n\n")
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
