import json

from vproc.llm import client

SYSTEM = (
    "You answer ONLY from the numbered EVIDENCE provided below. For every claim you make, "
    "cite the evidence id(s) it comes from. If the evidence does not contain the answer, set "
    'answered to false and return an empty claims list. Never use outside knowledge or your '
    "own assumptions. "
    'Respond as strict JSON: {"answered": bool, "claims": [{"text": str, "evidence_ids": [str]}]}'
)

# Schema for constrained decoding (json_schema response_format). Structurally enforces the
# {answered, claims:[{text, evidence_ids}]} shape the rest of the pipeline depends on.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
            },
        },
    },
    "required": ["answered", "claims"],
}


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Thinking models emit reasoning (often restating the schema, braces and all) around the
    # JSON, so scan every '{' for the first object json can actually decode.
    text = raw.replace("```json", "").replace("```", "")
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            return dec.raw_decode(text[i:])[0]
        except Exception:
            i = text.find("{", i + 1)
    return {"answered": False, "claims": []}


def generate(cfg, question: str, evidence_block: str, chat=client.chat_json) -> dict:
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_block}\n\nJSON:"
    data = _parse(chat(cfg.grounding.base_url, cfg.grounding.model, SYSTEM, user,
                       schema=ANSWER_SCHEMA, max_tokens=cfg.grounding_max_tokens,
                       think=cfg.grounding_thinking))
    data = data if isinstance(data, dict) else {}
    raw_claims = data.get("claims")
    claims = [
        c for c in (raw_claims if isinstance(raw_claims, list) else [])
        if isinstance(c, dict) and isinstance(c.get("text"), str)
        and isinstance(c.get("evidence_ids"), list)
        and c["evidence_ids"] and all(isinstance(e, str) for e in c["evidence_ids"])
    ]
    return {"answered": bool(data.get("answered")) and len(claims) > 0, "claims": claims}
