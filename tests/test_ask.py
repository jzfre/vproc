from vproc.config import Config, Endpoint
from vproc.answer.ask import ask_memory, search_memory, ABSTAIN
from vproc.answer.generate import generate

def _cfg(sim_floor=0.25):
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, sim_floor, 0.5, None)

def _hit(id, said):
    return {"id": id, "memory_id": "standup", "start_ts": 12.0, "end_ts": 14.0,
            "speaker": "Tim V", "said_text": said, "on_screen_text": ""}

def test_answer_is_grounded_and_cited():
    def fake_retrieve(store, cfg, q, k=8, where=None):
        return [_hit("s1", "Tim said cars are electric now")], 0.9
    def fake_generate(cfg, q, block):
        return {"answered": True, "claims": [{"text": "cars are electric now", "evidence_ids": ["E1"]}]}
    ans = ask_memory(None, _cfg(), "did Tim talk about cars?", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=fake_generate)
    assert ans.answered is True
    assert "cars are electric now" in ans.text
    assert "[standup · 00:12 · Tim V]" in ans.text     # code-owned citation
    assert ans.claims[0].citations[0].speaker == "Tim V"

def test_hallucination_guard_abstains_on_empty_retrieval():
    def fake_retrieve(*a, **k):
        return [], 0.0
    ans = ask_memory(None, _cfg(), "how does a petrol engine work?", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=lambda *a: 1 / 0)  # must not be called
    assert ans.answered is False and ans.abstained is True
    assert ans.text == ABSTAIN

def test_abstains_below_similarity_floor():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "unrelated")], 0.10  # below floor 0.25
    ans = ask_memory(None, _cfg(), "q", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=lambda *a: 1 / 0)
    assert ans.text == ABSTAIN

def test_abstains_when_hhem_rejects_all():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "Tim said cars are electric")], 0.9
    def fake_generate(*a):
        return {"answered": True, "claims": [{"text": "cars are petrol", "evidence_ids": ["E1"]}]}
    ans = ask_memory(None, _cfg(), "q", scorer=lambda p, h: 0.1,  # HHEM rejects
                     _retrieve=fake_retrieve, _generate=fake_generate)
    assert ans.text == ABSTAIN

def test_claim_evidence_ids_filtered_to_returned_evidence():
    def fake_retrieve(store, cfg, q, k=8, where=None):
        return [_hit("s1", "cars are electric now")], 0.9
    def fake_generate(cfg, q, block):
        return {"answered": True,
                "claims": [{"text": "cars are electric now", "evidence_ids": ["E1", "E99"]}]}
    ans = ask_memory(None, _cfg(), "q", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=fake_generate)
    keys = {e.key for e in ans.evidence}
    assert ans.claims[0].evidence_ids == ["E1"]          # dangling E99 dropped
    assert all(eid in keys for eid in ans.claims[0].evidence_ids)

def test_search_memory_returns_evidence():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "cars")], 0.9
    ev = search_memory(None, _cfg(), "cars", _retrieve=fake_retrieve)
    assert ev[0].key == "E1" and ev[0].speaker == "Tim V"


def test_answer_can_ground_speaker_identity_from_segment_metadata():
    def fake_chat(base_url, model, system, user, **kwargs):
        if "Speaker: Tim V" not in user:
            return '{"answered": false, "claims": []}'
        return ('{"answered": true, "claims": '
                '[{"text": "Tim V said cars are electric now", "evidence_ids": ["E1"]}]}')

    def scorer(premise, hypothesis):
        return 0.9 if "Speaker: Tim V" in premise and "cars are electric now" in premise else 0.0

    ans = ask_memory(
        None, _cfg(), "Who said cars are electric now?", scorer=scorer,
        _retrieve=lambda *a, **k: ([_hit("s1", "cars are electric now")], 0.9),
        _generate=lambda cfg, question, block: generate(cfg, question, block, chat=fake_chat),
    )
    assert ans.answered is True
    assert ans.claims[0].text == "Tim V said cars are electric now"
    assert ans.claims[0].citations[0].speaker == "Tim V"
