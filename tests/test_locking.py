from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest


def _hold_lock(path, connection):
    from vproc.locking import IndexWriteLock

    with IndexWriteLock(path):
        connection.send("locked")
        connection.recv()


def test_lock_is_reentrant_but_excludes_other_instances(tmp_path):
    from vproc.errors import IndexBusyError
    from vproc.locking import IndexWriteLock

    owner = IndexWriteLock(tmp_path / "index")
    contender = IndexWriteLock(tmp_path / "index")
    with owner:
        with owner:
            with pytest.raises(IndexBusyError):
                with contender:
                    pytest.fail("a second writer acquired the lock")
        with pytest.raises(IndexBusyError):
            with contender:
                pytest.fail("inner release must not unlock the outer writer")
    with contender:
        pass


def test_same_lock_object_excludes_another_thread(tmp_path):
    from vproc.errors import IndexBusyError
    from vproc.locking import IndexWriteLock

    lock = IndexWriteLock(tmp_path / "index")

    def compete():
        with lock:
            return "unexpected writer"

    with ThreadPoolExecutor(max_workers=1) as executor:
        with lock:
            with pytest.raises(IndexBusyError):
                executor.submit(compete).result(timeout=5)
        assert executor.submit(compete).result(timeout=5) == "unexpected writer"


@pytest.mark.parametrize("terminate", [False, True])
def test_process_exit_releases_lock(tmp_path, terminate):
    from vproc.errors import IndexBusyError
    from vproc.locking import IndexWriteLock

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_hold_lock, args=(str(tmp_path / "index"), child))
    proc.start()
    child.close()
    try:
        assert parent.poll(10), "child failed to acquire lock"
        assert parent.recv() == "locked"
        with pytest.raises(IndexBusyError):
            with IndexWriteLock(tmp_path / "index"):
                pytest.fail("a second process acquired the writer lock")
        if terminate:
            proc.terminate()
        else:
            parent.send("release")
        proc.join(timeout=10)
        assert not proc.is_alive()
        with IndexWriteLock(tmp_path / "index"):
            pass
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
        parent.close()


def test_symlink_alias_uses_same_lock(tmp_path):
    from vproc.errors import IndexBusyError
    from vproc.locking import IndexWriteLock

    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with IndexWriteLock(actual):
        with pytest.raises(IndexBusyError):
            with IndexWriteLock(alias):
                pytest.fail("directory aliases must share a lock")
