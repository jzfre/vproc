from vproc.ingest.naming import NameVote, apply_names, resolve_names
from vproc.ingest.transcribe import TranscriptSegment


def _v(spk, name, t=0.0, visible=()):
    return NameVote(speaker=spk, t=t, name=name, visible_names=list(visible))


def test_canonicalization_groups_ocr_jitter():
    # The spike's real jitter: 4 spellings of one person + 2 distinct other names.
    votes = [
        _v("SPEAKER_00", "Vanco, Pavol", visible=["Repan, Jozef", "PATINO, DANIEL"]),
        _v("SPEAKER_00", "Yanco, Pavol", visible=["Vanco, Pavol"]),
        _v("SPEAKER_00", "Wanco, Pavol", visible=["Vanco, Pavlo"]),
    ]
    mapping, suggestions = resolve_names(votes)
    # 3 jitter spellings converge on the most frequent form and clear the 2-vote majority.
    assert mapping == {"SPEAKER_00": "Vanco, Pavol"}
    assert suggestions == {}


def test_cooccurring_similar_names_stay_distinct():
    # One frame's visible_names lists both spellings together: they are confirmed to be
    # two different people despite passing the 0.75 similarity threshold (0.966 ratio).
    votes = [
        _v("SPEAKER_00", "Patino, Daniel", visible=["Patino, Daniel", "Patino, Daniela"]),
        _v("SPEAKER_00", "Patino, Daniel"),
        _v("SPEAKER_01", "Patino, Daniela"),
        _v("SPEAKER_01", "Patino, Daniela"),
    ]
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_00": "Patino, Daniel", "SPEAKER_01": "Patino, Daniela"}


def test_jitter_without_cooccurrence_still_groups():
    # Control: same kind of near-miss similarity, but the spellings never co-occur in a
    # single frame, so jitter-merging still collapses them into one canonical name.
    votes = [_v("SPEAKER_00", "Vanco, Pavol"), _v("SPEAKER_00", "Yanco, Pavol")]
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_00": "Vanco, Pavol"}


def test_distinct_names_do_not_merge():
    votes = [_v("SPEAKER_00", "Repan, Jozef"), _v("SPEAKER_00", "PATINO, DANIEL"),
             _v("SPEAKER_00", "Repan, Jozef")]
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_00": "Repan, Jozef"}  # 2/3 strict majority


def test_split_votes_become_suggestion_not_mapping():
    votes = [_v("SPEAKER_02", "Selrico Lamont Martin", t=992),
             _v("SPEAKER_02", "PATINO, DANIEL", t=209),
             _v("SPEAKER_02", None, t=173)]
    mapping, suggestions = resolve_names(votes)
    assert mapping == {}
    s = suggestions["SPEAKER_02"]
    assert s["votes"] == {"Selrico Lamont Martin": 1, "PATINO, DANIEL": 1}
    assert {"t": 992, "name": "Selrico Lamont Martin"} in s["evidence"]


def test_single_vote_is_not_enough():
    mapping, suggestions = resolve_names([_v("SPEAKER_03", "Selrico Lamont Martin")])
    assert mapping == {} and "SPEAKER_03" in suggestions  # >=2 votes required


def test_two_clusters_sharing_winner_both_map():
    votes = ([_v("SPEAKER_01", "Selrico Lamont Martin")] * 2
             + [_v("SPEAKER_03", "Selrico Lamont Martin")] * 2)
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_01": "Selrico Lamont Martin",
                       "SPEAKER_03": "Selrico Lamont Martin"}


def test_null_votes_excluded_from_majority_denominator():
    # 2 name votes + 3 nulls: majority is over the 2 non-null votes.
    votes = [_v("SPEAKER_04", "Repan, Jozef")] * 2 + [_v("SPEAKER_04", None)] * 3
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_04": "Repan, Jozef"}


def test_apply_names_relabels_only_mapped():
    t = [TranscriptSegment(0.0, 1.0, "a", speaker="SPEAKER_00"),
         TranscriptSegment(1.0, 2.0, "b", speaker="SPEAKER_01")]
    apply_names(t, {"SPEAKER_00": "Repan, Jozef"})
    assert [s.speaker for s in t] == ["Repan, Jozef", "SPEAKER_01"]


import os

import vproc.ingest.naming as N
from vproc.ingest.diarize import SpeakerTurn


def test_parse_vote_lenient():
    assert N._parse_vote('{"speaking": "A B", "visible_names": ["A B", "C D"]}') == ("A B", ["A B", "C D"])
    assert N._parse_vote('noise before {"speaking": null, "visible_names": []} after') == (None, [])
    assert N._parse_vote('{"speaking": "  ", "visible_names": "not-a-list"}') == (None, [])
    assert N._parse_vote("total garbage") == (None, [])


def test_sample_name_votes_probes_longest_turns(monkeypatch):
    monkeypatch.setattr(N, "_frame_at", lambda video, t, out: None)  # no real ffmpeg
    probed = []
    def probe(path):
        probed.append(path)
        return '{"speaking": "Repan, Jozef", "visible_names": ["Repan, Jozef"]}'
    # 8 turns for one speaker: only the 6 longest get probed; 1 turn for another.
    turns = [SpeakerTurn(i * 10.0, i * 10.0 + 1.0 + i, "SPEAKER_00") for i in range(8)]
    turns.append(SpeakerTurn(500.0, 520.0, "SPEAKER_01"))
    votes = N.sample_name_votes("/v.mkv", turns, cfg=None, probe=probe)
    assert len([v for v in votes if v.speaker == "SPEAKER_00"]) == 6
    assert len([v for v in votes if v.speaker == "SPEAKER_01"]) == 1
    assert all(v.name == "Repan, Jozef" for v in votes)
    # longest turns won: the two shortest SPEAKER_00 turns (i=0,1) were skipped
    probed_ts = {v.t for v in votes if v.speaker == "SPEAKER_00"}
    assert (0.0 + 1.0) / 2 not in probed_ts and (10.0 + 12.0) / 2 not in probed_ts


def test_sample_name_votes_skips_failed_probes(monkeypatch, capsys):
    monkeypatch.setattr(N, "_frame_at", lambda video, t, out: None)
    calls = {"n": 0}
    def probe(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return '{"speaking": "X Y", "visible_names": []}'
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(20.0, 30.0, "SPEAKER_00")]
    votes = N.sample_name_votes("/v.mkv", turns, cfg=None, probe=probe)
    assert len(votes) == 1  # first probe lost, ingest-level behavior unaffected
    assert "warning: 1/2 name probes failed" in capsys.readouterr().err


def test_sample_name_votes_all_failed_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(N, "_frame_at", lambda video, t, out: None)
    def probe(path):
        raise TimeoutError("slow")
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(20.0, 30.0, "SPEAKER_00")]
    votes = N.sample_name_votes("/v.mkv", turns, cfg=None, probe=probe)
    assert votes == []
    assert "warning: 2/2 name probes failed" in capsys.readouterr().err


def test_speaker_map_round_trip(tmp_path):
    data = {"mapping": {"SPEAKER_04": "Repan, Jozef"},
            "suggestions": {"SPEAKER_02": {"votes": {"A": 1}, "evidence": []}},
            "votes": [{"speaker": "SPEAKER_04", "t": 889, "name": "Repan, Jozef"}]}
    mem_dir = str(tmp_path / "frames" / "default" / "test")
    N.save_speaker_map(mem_dir, data)
    assert N.load_speaker_map(mem_dir) == data
    assert os.path.exists(os.path.join(mem_dir, "speakers.json"))


def test_failed_speaker_map_save_preserves_previous_file(tmp_path):
    from vproc.ingest.naming import load_speaker_map, save_speaker_map
    import pytest

    previous = {"mapping": {"SPEAKER_00": "Ada"}, "suggestions": {}, "votes": []}
    save_speaker_map(str(tmp_path), previous)
    with pytest.raises(TypeError):
        save_speaker_map(str(tmp_path), {"mapping": {"not JSON serializable"}})
    assert load_speaker_map(str(tmp_path)) == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["speakers.json"]
