from vproc.answer.evidence import build_evidence, evidence_block

def _hit(id, said, screen=""):
    return {"id": id, "memory_id": "standup", "start_ts": 12.0, "end_ts": 14.0,
            "speaker": "Tim V", "said_text": said, "on_screen_text": screen}

def test_build_evidence_assigns_sequential_keys():
    ev = build_evidence([_hit("s1", "cars are great"), _hit("s2", "budget is tight", "Q3 Budget")])
    assert [e.key for e in ev] == ["E1", "E2"]
    assert ev[0].segment_id == "s1"
    assert ev[0].memory_title == "standup" and ev[0].speaker == "Tim V"
    assert "budget is tight" in ev[1].text and "[screen] Q3 Budget" in ev[1].text

def test_evidence_block_format():
    ev = build_evidence([_hit("s1", "hello world")])
    block = evidence_block(ev)
    assert block == ('[E1] "Meeting: standup\nSpeaker: Tim V\n'
                     'Time: 00:12-00:14\nhello world"')
