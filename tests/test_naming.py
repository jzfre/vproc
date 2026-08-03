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
