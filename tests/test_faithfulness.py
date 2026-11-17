import pytest

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


@pytest.mark.parametrize("score", [float("inf"), 1.1, True, None, "0.9"])
def test_filter_drops_invalid_faithfulness_scores(score):
    claims = [{"text": "ships in July", "evidence_ids": ["E1"]}]
    by_key = {"E1": _ev("E1", "the product ships in July")}
    assert filter_claims(claims, by_key, lambda p, h: score, threshold=0.5) == []


def _fake_transformers(monkeypatch, from_pretrained):
    import sys
    import types

    auto = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoModelForSequenceClassification=auto))


def test_hhem_public_model_loads_despite_a_stale_hf_token(monkeypatch):
    # Hugging Face answers 401 to an invalid HF_TOKEN even for public repos, so a stale
    # diarization token must not take the (public) answer verifier down with it.
    from vproc.answer.faithfulness import HHEM

    calls = []

    def from_pretrained(model_id, **kw):
        calls.append(kw)
        if kw.get("token") is not False:
            raise OSError("401 Client Error: invalid user token")
        return "model"

    _fake_transformers(monkeypatch, from_pretrained)
    assert HHEM("vectara/hallucination_evaluation_model").model == "model"
    assert [kw.get("token") for kw in calls] == [None, False]


def test_hhem_load_failure_without_token_cause_still_raises(monkeypatch):
    from vproc.answer.faithfulness import HHEM

    def from_pretrained(model_id, **kw):
        raise OSError("model files are missing")

    _fake_transformers(monkeypatch, from_pretrained)
    with pytest.raises(OSError, match="missing"):
        HHEM("some/private-model")


def test_repeated_citations_do_not_repeat_evidence_or_change_gate_scores():
    by_key = {"E1": _ev("E1", "ships in July"), "E2": _ev("E2", "launch owner is Ada")}
    claims = [{"text": "ships in July", "evidence_ids": ["E1", "E99", "E1", "E2", "E2"]}]
    seen = []

    def scorer(premise, hypothesis):
        seen.append(premise)
        return 0.9

    kept = filter_claims(claims, by_key, scorer, threshold=0.5)
    assert seen[0].count("ships in July") == 1
    assert seen[0].count("launch owner is Ada") == 1
    assert kept[0]["evidence_ids"] == ["E1", "E2"]
    assert claims[0]["evidence_ids"] == ["E1", "E99", "E1", "E2", "E2"]
