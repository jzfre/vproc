from vproc.models import Evidence
from vproc.answer.faithfulness import filter_claims

def _ev(key, text):
    return Evidence(key=key, segment_id=key, memory_title="m", start_ts=0, end_ts=1,
                    speaker="SPEAKER_0", text=text)

def test_filter_keeps_supported_drops_unsupported():
    by_key = {"E1": _ev("E1", "the product ships in July")}
    claims = [
        {"text": "ships in July", "evidence_ids": ["E1"]},   # supported
        {"text": "ships in March", "evidence_ids": ["E1"]},  # contradicted
    ]
    # fake scorer: high if the claim's key words appear in premise
    def scorer(premise, hypothesis):
        return 0.9 if "July" in premise and "July" in hypothesis else 0.1
    kept = filter_claims(claims, by_key, scorer, threshold=0.5)
    assert kept == [{"text": "ships in July", "evidence_ids": ["E1"]}]

def test_filter_drops_when_no_premise():
    claims = [{"text": "x", "evidence_ids": ["E9"]}]  # E9 not in evidence
    kept = filter_claims(claims, {}, lambda p, h: 1.0, threshold=0.5)
    assert kept == []
