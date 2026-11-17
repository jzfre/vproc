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


def test_claims_are_bounded_in_the_schema_and_in_code():
    # Observed on a 27B model with thinking off: after answering, it kept appending claims about
    # everything in the evidence until max_tokens (8192 tokens, 259 s) and the truncated JSON
    # became an abstention. Constrained decoding enforces the schema bound; the slice covers
    # backends that ignore it.
    import json

    from vproc.answer import generate as G

    assert G.ANSWER_SCHEMA["properties"]["claims"]["maxItems"] == G.MAX_CLAIMS
    many = {"answered": True, "claims": [{"text": f"claim {i}", "evidence_ids": ["E1"]} for i in range(20)]}
    cfg = type("C", (), {"grounding": type("E", (), {"base_url": "u", "model": "m"})(),
                         "grounding_max_tokens": 64, "grounding_thinking": False})()
    out = G.generate(cfg, "q", "[E1] evidence", chat=lambda *a, **k: json.dumps(many))
    assert [c["text"] for c in out["claims"]] == [f"claim {i}" for i in range(G.MAX_CLAIMS)]


def test_generate_skips_json_schema_in_reasoning_before_final_answer():
    raw = ('First inspect {"type": "object", "properties": {"answered": {"type": "boolean"}}}. '
           'Final: {"answered": true, "claims": [{"text": "ships in July", "evidence_ids": ["E1"]}]}')
    result = generate(_cfg(), "q", '[E1] "ships in July"', chat=lambda *a, **k: raw)
    assert result["answered"] is True
    assert result["claims"][0]["text"] == "ships in July"


def test_generate_uses_final_answer_after_a_reasoning_draft():
    raw = ('<think>{"answered": true, "claims": [{"text": "draft claim", "evidence_ids": ["E1"]}]}</think>\n'
           '{"answered": false, "claims": []}')
    result = generate(_cfg(), "q", '[E1] "ships in July"', chat=lambda *a, **k: raw)
    assert result == {"answered": False, "claims": []}


def test_generate_does_not_answer_from_unclosed_reasoning():
    raw = '<think>{"answered": true, "claims": [{"text": "draft claim", "evidence_ids": ["E1"]}]}'
    result = generate(_cfg(), "q", '[E1] "ships in July"', chat=lambda *a, **k: raw)
    assert result == {"answered": False, "claims": []}


def test_generate_uses_final_top_level_object_not_nested_claim_json():
    # The claim itself can quote JSON. Its braces cannot be reinterpreted as another answer.
    final = {"answered": True, "claims": [{
        "text": 'The literal payload is {"answered": false, "claims": []}.', "evidence_ids": ["E1"]}]}
    raw = 'Example: {"answered": false, "claims": []}\nFinal: ' + json.dumps(final)
    result = generate(_cfg(), "q", "[E1] quoted payload", chat=lambda *a, **k: raw)
    assert result == final
