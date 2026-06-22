from vproc.config import Config, Endpoint
from vproc.answer.generate import generate, SYSTEM

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_generate_parses_and_keeps_cited_claims():
    def fake_chat(base_url, model, system, user, **kwargs):
        assert "EVIDENCE" in user
        return '{"answered": true, "claims": [{"text": "ships in July", "evidence_ids": ["E1"]}]}'
    out = generate(_cfg(), "when does it ship?", "[E1] \"ships in July\"", chat=fake_chat)
    assert out["answered"] is True
    assert out["claims"] == [{"text": "ships in July", "evidence_ids": ["E1"]}]

def test_generate_drops_claims_without_evidence():
    def fake_chat(*a, **k):
        return '{"answered": true, "claims": [{"text": "made up", "evidence_ids": []}]}'
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is False and out["claims"] == []

def test_generate_recovers_from_noisy_json():
    def fake_chat(*a, **k):
        return 'Sure!\n{"answered": false, "claims": []}\nThanks'
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is False

def test_system_prompt_forbids_outside_knowledge():
    assert "only" in SYSTEM.lower() and "outside" in SYSTEM.lower()
