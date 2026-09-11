import pytest
from contextlib import nullcontext

import vproc.cli as cli
from vproc.ingest import naming as N


class _FakeStore:
    def __init__(self, path):
        self.path = path
        self.renames = []
        self.rows_updated = 1

    def write_lock(self):
        return nullcontext()

    def update_speaker(self, memory_id, project_id, old, new):
        self.renames.append((memory_id, project_id, old, new))
        return self.rows_updated


def _setup(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    mem_dir = str(frames / "default" / "standup")
    N.save_speaker_map(mem_dir, {
        "mapping": {"SPEAKER_00": "Repan, Jozef"},
        "suggestions": {"SPEAKER_02": {"votes": {"PATINO, DANIEL": 1}, "evidence": [{"t": 209, "name": "PATINO, DANIEL"}]}},
        "votes": []})
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(frames))
    monkeypatch.setenv("VPROC_INDEX_PATH", str(tmp_path / "db"))
    fake = _FakeStore(str(tmp_path / "db"))
    monkeypatch.setattr(cli, "_store_for", lambda cfg: fake)
    return mem_dir, fake


def test_speakers_list(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup"])
    cli.main()
    out = capsys.readouterr().out
    assert "SPEAKER_00" in out and "Repan, Jozef" in out
    assert "SPEAKER_02" in out and "PATINO, DANIEL" in out  # suggestion shown with evidence


def test_speakers_set_updates_store_and_map(tmp_path, monkeypatch, capsys):
    mem_dir, fake = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup", "--set", "SPEAKER_02=PATINO, DANIEL"])
    cli.main()
    assert fake.renames == [("standup", "default", "SPEAKER_02", "PATINO, DANIEL")]
    m = N.load_speaker_map(mem_dir)
    assert m["mapping"]["SPEAKER_02"] == "PATINO, DANIEL"
    assert "SPEAKER_02" not in m["suggestions"]
    assert "renamed SPEAKER_02 -> PATINO, DANIEL in 'standup' (1 rows)" in capsys.readouterr().out


def test_speakers_set_zero_matches_warns(tmp_path, monkeypatch, capsys):
    mem_dir, fake = _setup(tmp_path, monkeypatch)
    before = N.load_speaker_map(mem_dir)
    fake.rows_updated = 0
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup", "--set", "SPEAKER_02=PATINO, DANIEL"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "warning: no rows matched SPEAKER_02" in out.err
    assert N.load_speaker_map(mem_dir) == before


def test_speakers_set_malformed_exits_without_touching_store(tmp_path, monkeypatch, capsys):
    mem_dir, fake = _setup(tmp_path, monkeypatch)
    before = N.load_speaker_map(mem_dir)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup", "--set", "SPEAKER_02="])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert fake.renames == []
    assert N.load_speaker_map(mem_dir) == before


def test_speakers_set_bootstraps_missing_map(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    mem_dir = str(frames / "default" / "nomap")
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(frames))
    monkeypatch.setenv("VPROC_INDEX_PATH", str(tmp_path / "db"))
    fake = _FakeStore(str(tmp_path / "db"))
    monkeypatch.setattr(cli, "_store_for", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "nomap", "--set", "SPEAKER_00=Repan, Jozef"])
    cli.main()
    assert fake.renames == [("nomap", "default", "SPEAKER_00", "Repan, Jozef")]
    m = N.load_speaker_map(mem_dir)
    assert m["mapping"] == {"SPEAKER_00": "Repan, Jozef"}


def test_speakers_missing_map_is_friendly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(tmp_path / "frames"))
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "nope"])
    cli.main()
    assert "no speaker map" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("args", [["standup", "--set"], ["standup", "--unknown"],
                                  ["standup", "--set", "A=B", "extra"]])
def test_speakers_rejects_malformed_arguments(tmp_path, monkeypatch, args):
    _, store = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert store.renames == []


def test_speaker_edit_cannot_traverse_frame_directory(tmp_path, monkeypatch):
    _, store = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "../outside", "--set", "A=B"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert store.renames == []
    assert not (tmp_path / "frames" / "outside").exists()


def test_busy_frame_storage_prevents_speaker_edit(tmp_path, monkeypatch, capsys):
    from vproc.locking import IndexWriteLock

    mem_dir, store = _setup(tmp_path, monkeypatch)
    before = N.load_speaker_map(mem_dir)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup", "--set", "SPEAKER_00=New"])
    with IndexWriteLock(tmp_path / "frames", name=".vproc-frames.lock"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 1
    assert "writer" in capsys.readouterr().err
    assert store.renames == []
    assert N.load_speaker_map(mem_dir) == before


def test_ingest_compatibility_failure_has_actionable_cli_error(monkeypatch, capsys):
    from vproc.errors import IndexCompatibilityError
    import vproc.ingest.pipeline as pipeline

    def mismatch(*args):
        raise IndexCompatibilityError("Use a new VPROC_INDEX_PATH and re-ingest")

    monkeypatch.setattr(pipeline, "ingest_video", mismatch)
    monkeypatch.setattr("sys.argv", ["vproc", "ingest", "test.mp4"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "VPROC_INDEX_PATH" in capsys.readouterr().err


@pytest.mark.parametrize("argument", ["--help", "-h", "help"])
def test_help_succeeds_without_loading_runtime_config(monkeypatch, capsys, argument):
    def no_dotenv():
        pytest.fail("help must not depend on a local .env file")

    monkeypatch.setattr(cli, "load_dotenv", no_dotenv)
    monkeypatch.setenv("VPROC_PORT", "invalid")
    monkeypatch.setattr("sys.argv", ["vproc", argument])
    cli.main()
    out = capsys.readouterr().out
    assert "doctor" in out
    assert "ingest" in out and "serve" in out and "speakers" in out


def test_version_uses_installed_metadata(monkeypatch, capsys):
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "1.2.3")
    monkeypatch.setenv("VPROC_PORT", "invalid")
    monkeypatch.setattr("sys.argv", ["vproc", "--version"])
    cli.main()
    assert capsys.readouterr().out.strip() == "vproc 1.2.3"


def test_version_falls_back_to_source_metadata(monkeypatch, capsys):
    from importlib import metadata
    from pathlib import Path
    import tomllib

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setattr("sys.argv", ["vproc", "--version"])
    cli.main()
    source_version = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]["version"]
    assert capsys.readouterr().out.strip() == f"vproc {source_version}"


@pytest.mark.parametrize("content", [None, "project = ['invalid']", "not valid TOML"])
def test_version_without_usable_metadata_still_succeeds(tmp_path, monkeypatch, capsys, content):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "vproc" / "cli.py"))
    if content is not None:
        (tmp_path / "pyproject.toml").write_text(content)
    monkeypatch.setattr("sys.argv", ["vproc", "--version"])
    cli.main()
    assert capsys.readouterr().out.strip() == "vproc 0+unknown"


def test_doctor_invalid_configuration_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("VPROC_PORT", "invalid")
    monkeypatch.setattr("sys.argv", ["vproc", "doctor"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "VPROC_PORT" in capsys.readouterr().err
