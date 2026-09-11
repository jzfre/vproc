from pathlib import Path

from filelock import FileLock, Timeout

from vproc.errors import IndexBusyError


class IndexWriteLock:
    """Local process lock; nested use of the same object is reentrant per thread.

    Keep the lock file in place: unlinking it lets a new writer lock a different
    inode while the previous owner still holds the original lock.
    """

    def __init__(self, directory, *, name=".vproc-write.lock"):
        self.directory = Path(directory).expanduser().resolve()
        self._lock = FileLock(self.directory / name, timeout=0)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire()
        except Timeout:
            raise IndexBusyError(
                "Another writer is using this index or its frames. Retry when it finishes."
            ) from None
        return self

    def __exit__(self, *exc):
        self._lock.release()
