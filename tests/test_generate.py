import json

import pytest

from vproc.config import Config, Endpoint
from vproc.answer.generate import generate, SYSTEM

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_generate_parses_and_keeps_cited_claims():
    def fake_chat(base_url, model, system, user, **kwargs):
        assert "EVIDENCE" in user
        # the runaway-thinking bound and the thinking switch must reach the client call
        assert kwargs["max_tokens"] == _cfg().grounding_max_tokens
        assert kwargs["think"] == _cfg().grounding_thinking
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

def test_generate_survives_malformed_but_valid_json():
    # valid JSON, wrong shape: claims null / strings / top-level array must not crash
    for raw in ('{"answered": true, "claims": null}',
                '{"answered": true, "claims": ["ships in July [E1]"]}',
                '[{"text": "x", "evidence_ids": ["E1"]}]'):
        out = generate(_cfg(), "q", "[E1] \"x\"", chat=lambda *a, r=raw, **k: r)
        assert out["answered"] is False and out["claims"] == []

def test_generate_drops_claims_with_non_string_evidence_ids():
    def fake_chat(*a, **k):
        return '{"answered": true, "claims": [{"text": "x", "evidence_ids": [1, 2]}]}'
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is False and out["claims"] == []

def test_generate_extracts_json_after_leading_brace_reasoning():
    def fake_chat(*a, **k):
        return ('Reasoning: the schema is {"answered": bool, "claims": [...]}. '
                'Final answer: {"answered": true, '
                '"claims": [{"text": "ships in July", "evidence_ids": ["E1"]}]}')
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is True
    assert out["claims"] == [{"text": "ships in July", "evidence_ids": ["E1"]}]

def test_generate_strips_markdown_code_fences():
    def fake_chat(*a, **k):
        return ('```json\n{"answered": true, '
                '"claims": [{"text": "ships in July", "evidence_ids": ["E1"]}]}\n```')
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is True and out["claims"][0]["text"] == "ships in July"


@pytest.mark.parametrize("answered", ["false", "true", 1, [True], {"value": True}])
def test_generate_requires_explicit_boolean_answered(answered):
    raw = json.dumps({"answered": answered, "claims": [
        {"text": "ships in July", "evidence_ids": ["E1"]},
    ]})
    out = generate(_cfg(), "q", '[E1] "ships in July"', chat=lambda *a, **k: raw)
    assert out == {"answered": False, "claims": []}


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_generate_drops_blank_claims(text):
    raw = json.dumps({"answered": True, "claims": [{"text": text, "evidence_ids": ["E1"]}]})
    out = generate(_cfg(), "q", '[E1] "ships in July"', chat=lambda *a, **k: raw)
    assert out == {"answered": False, "claims": []}
