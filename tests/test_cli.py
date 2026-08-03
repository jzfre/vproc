import json
import os

import vproc.cli as cli
from vproc.ingest import naming as N


class _FakeStore:
    def __init__(self, path):
        self.path = path
        self.renames = []

    def update_speaker(self, memory_id, project_id, old, new):
        self.renames.append((memory_id, project_id, old, new))


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


def test_speakers_missing_map_is_friendly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(tmp_path / "frames"))
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "nope"])
    cli.main()
    assert "no speaker map" in capsys.readouterr().out.lower()
