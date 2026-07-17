"""Machine-wide, non-blocking EdgeCI process lock."""

from __future__ import annotations

import fcntl
import errno
import os
import stat
from pathlib import Path
from types import TracebackType
from typing import BinaryIO


DEFAULT_LOCK_PATH = Path("/tmp/edgeci.lock")


class EdgeCILockError(RuntimeError):
    """Raised when the machine-wide EdgeCI lock cannot be acquired."""


class EdgeCILock:
    """Exclusive machine-wide file lock for benchmark serialization.

    The lock file remains on disk after release. Removing it would permit an
    inode race in which two processes both believe they hold the lock.
    """

    def __init__(self, path: Path | str = DEFAULT_LOCK_PATH) -> None:
        """Initialize an unacquired lock.

        Args:
            path: Lock-file location. Defaults to ``/tmp/edgeci.lock``.
        """

        self.path = Path(path)
        self._stream: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        """Return whether this instance currently owns the lock."""

        return self._stream is not None

    def acquire(self) -> None:
        """Acquire the lock immediately or raise ``EdgeCILockError``."""

        if self._stream is not None:
            raise EdgeCILockError(f"lock is already acquired: {self.path}")
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        writable = True
        try:
            descriptor = os.open(self.path, flags, 0o644)
        except PermissionError:
            # A lock created by another local account may be read-only to this
            # process. BSD flock permits locking a read-only descriptor; the
            # holder PID simply cannot be refreshed in that case.
            writable = False
            read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            read_flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, read_flags)
            except OSError as exc:
                raise EdgeCILockError(
                    f"cannot open EdgeCI lock {self.path}: {exc}"
                ) from exc
        except OSError as exc:
            raise EdgeCILockError(f"cannot open EdgeCI lock {self.path}: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EdgeCILockError(
                    f"EdgeCI lock must be a single-link regular file: {self.path}"
                )
            if writable and metadata.st_uid == os.getuid():
                # Other local accounts need only read access: BSD flock permits
                # an exclusive advisory lock on a read-only descriptor.
                os.fchmod(descriptor, 0o644)
            stream = os.fdopen(
                descriptor,
                "r+b" if writable else "rb",
            )
        except Exception:
            os.close(descriptor)
            raise
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            holder = _read_holder_pid(stream)
            stream.close()
            suffix = f"; holder PID {holder}" if holder else ""
            raise EdgeCILockError(
                f"another EdgeCI instance is running (lock: {self.path}{suffix})"
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                holder = _read_holder_pid(stream)
                stream.close()
                suffix = f"; holder PID {holder}" if holder else ""
                raise EdgeCILockError(
                    f"another EdgeCI instance is running (lock: {self.path}{suffix})"
                ) from exc
            stream.close()
            raise EdgeCILockError(f"cannot acquire EdgeCI lock {self.path}: {exc}") from exc

        if writable:
            try:
                stream.seek(0)
                stream.truncate()
                stream.write(str(os.getpid()).encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            except OSError as exc:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                finally:
                    stream.close()
                raise EdgeCILockError(
                    f"cannot record ownership in EdgeCI lock {self.path}: {exc}"
                ) from exc
        self._stream = stream

    def release(self) -> None:
        """Release the lock. Calling this on an unlocked instance is safe."""

        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> EdgeCILock:
        """Acquire and return this lock."""

        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release this lock regardless of body outcome."""

        del exc_type, exc_value, traceback
        self.release()


def _read_holder_pid(stream: BinaryIO) -> str:
    """Read a bounded, digits-only advisory holder PID from a lock file."""

    try:
        stream.seek(0)
        candidate = stream.read(32).strip()
    except OSError:
        return ""
    if not candidate or len(candidate) > 20 or not candidate.isdigit():
        return ""
    return candidate.decode("ascii")


__all__ = ("DEFAULT_LOCK_PATH", "EdgeCILock", "EdgeCILockError")
