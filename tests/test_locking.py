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


def _storage_row(sid, text):
    return {"id": sid, "project_id": "default", "memory_id": "meeting", "speaker": "Ada",
            "start_ts": 0., "end_ts": 1., "said_text": text, "on_screen_text": "",
            "embed_text": text, "source_video": "synthetic.mp4", "frame_path": "", "vector": [1., 0.]}


def _pause_replacement(path, phase, connection):
    from vproc.store.lancedb_store import Store

    store = Store(path)
    with store.write_lock():
        if phase == "before_commit":
            connection.send("paused")
            connection.recv()
        else:
            # The row transaction has committed when FTS rebuild starts. Simulate
            # process death there, before the writer reports a successful return.
            def pause_fts(table):
                connection.send("paused")
                connection.recv()
            store._rebuild_fts = pause_fts
        store.replace_memory("meeting", "default", [_storage_row("new", "new evidence")], "m")


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_interrupted_real_replacement_is_atomic_and_releases_writer(tmp_path, phase):
    from vproc.errors import IndexBusyError
    from vproc.store.lancedb_store import Store

    path = str(tmp_path / "db")
    reader = Store(path)
    reader.add([_storage_row("old", "old evidence")], embedding_model="m")
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_pause_replacement, args=(path, phase, child))
    process.start()
    child.close()
    try:
        assert parent.poll(20), "child did not reach the transaction boundary"
        assert parent.recv() == "paused"
        expected = "old" if phase == "before_commit" else "new"
        # The same reader connection sees a complete version, even while a
        # different process owns the writer lock; it does not cache stale rows.
        assert [r["id"] for r in reader.memory_rows("meeting", "default")] == [expected]
        with pytest.raises(IndexBusyError):
            reader.replace_memory("meeting", "default", [_storage_row("contender", "x")], "m")
        process.terminate()
        process.join(timeout=10)
        assert not process.is_alive()
        reopened = Store(path)
        assert [r["id"] for r in reopened.vector_search([1., 0.], 5, embedding_model="m")] == [expected]
        # A killed writer releases its OS lock and leaves the next transaction usable.
        reopened.replace_memory("meeting", "default", [_storage_row("recovered", "recovered evidence")], "m")
        assert [r["id"] for r in reader.memory_rows("meeting", "default")] == ["recovered"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent.close()


def test_release_preserves_the_lock_file(tmp_path):
    from vproc.locking import IndexWriteLock

    lock = IndexWriteLock(tmp_path / "index")
    path = tmp_path / "index" / ".vproc-write.lock"
    with lock:
        before = path.stat().st_ino
    assert path.exists() and path.stat().st_ino == before
    with lock:
        assert path.stat().st_ino == before
