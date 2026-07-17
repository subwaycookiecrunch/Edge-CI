"""Streaming, session-cached file hashing."""

from __future__ import annotations

import hashlib
import os
import time
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Callable


CHUNK_SIZE = 64 * 1024
_HASH_DEADLINE: ContextVar[
    tuple[float | None, Callable[[], float]] | None
] = ContextVar("edgeci_hash_deadline", default=None)


def hash_file(
    path: Path | str,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> str:
    """Return the SHA-256 hex digest of a local file.

    Files are streamed in 64 KiB chunks. The cache key includes stable file
    metadata, so an overwritten binary or model is hashed again while repeat
    reads of the same file during a session are free.

    Args:
        path: File to hash.
        deadline: Optional absolute deadline for streaming very large files.
        continuous_now: Clock associated with ``deadline``.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is not a regular file.
        OSError: If file metadata or contents cannot be read.
        TimeoutError: If ``deadline`` expires while hashing.
    """

    clock = continuous_now or time.monotonic
    _check_deadline(deadline, clock)
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise IsADirectoryError(f"cannot hash non-file path: {resolved}")
    stat = resolved.stat()
    identity = _stat_identity(stat)
    token = _HASH_DEADLINE.set((deadline, clock))
    try:
        result = _hash_cached(resolved, *identity)
    finally:
        _HASH_DEADLINE.reset(token)
    _check_deadline(deadline, clock)
    if _stat_identity(resolved.stat()) != identity:
        raise OSError(f"file changed while resolving its cached hash: {resolved}")
    return result


def clear_hash_cache() -> None:
    """Clear cached digests, primarily for long-lived embedding processes."""

    _hash_cached.cache_clear()


@lru_cache(maxsize=None)
def _hash_cached(
    path: Path,
    device: int,
    inode: int,
    size: int,
    modified_ns: int,
    changed_ns: int,
) -> str:
    deadline_context = _HASH_DEADLINE.get()
    deadline, clock = deadline_context or (None, time.monotonic)
    _check_deadline(deadline, clock)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        expected = (device, inode, size, modified_ns, changed_ns)
        observed = _stat_identity(before)
        if observed != expected:
            raise OSError(f"file changed before hashing began: {path}")
        while chunk := stream.read(CHUNK_SIZE):
            _check_deadline(deadline, clock)
            digest.update(chunk)
        _check_deadline(deadline, clock)
        after = os.fstat(stream.fileno())
    if _stat_identity(after) != expected:
        raise OSError(f"file changed while it was being hashed: {path}")
    if _stat_identity(path.stat()) != expected:
        raise OSError(f"file path changed while it was being hashed: {path}")
    return digest.hexdigest()


def _check_deadline(
    deadline: float | None, continuous_now: Callable[[], float]
) -> None:
    """Raise before a large-file hash crosses its enclosing deadline."""

    if deadline is not None and continuous_now() >= deadline:
        raise TimeoutError("comparison deadline expired while hashing provenance")


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return metadata fields that identify immutable hash input."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = ("clear_hash_cache", "hash_file")
