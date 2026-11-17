import json
import re

from vproc.llm import client

SYSTEM = (
    "You answer ONLY from the numbered EVIDENCE provided below. For every claim you make, "
    "cite the evidence id(s) it comes from. If the evidence does not contain the answer, set "
    'answered to false and return an empty claims list. Never use outside knowledge or your '
    "own assumptions. "
    'Respond as strict JSON: {"answered": bool, "claims": [{"text": str, "evidence_ids": [str]}]}'
)

# An answer is a handful of claims. Unbounded, a non-thinking model was seen answering correctly
# and then appending claims about everything in the evidence until max_tokens: minutes of
# generation ending in truncated JSON, i.e. an abstention. Constrained decoding enforces the
# schema bound; generate() also slices, for backends that ignore it.
MAX_CLAIMS = 6

# Schema for constrained decoding (json_schema response_format). Structurally enforces the
# {answered, claims:[{text, evidence_ids}]} shape the rest of the pipeline depends on.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "claims": {
            "type": "array",
            "maxItems": MAX_CLAIMS,
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
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        pass
    # Some backends put tagged reasoning in content. Drafts inside those tags
    # are never answers, including when generation stopped before the closing tag.
    text = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.S)
    # Recover the last complete answer object from prose/fences. Skip decoded
    # objects as a unit: braces in claim text or schema properties are not new
    # top-level answers. A schema object before the final response must not mask it.
    dec = json.JSONDecoder()
    answer = {}
    i = text.find("{")
    while i != -1:
        try:
            data, end = dec.raw_decode(text, i)
        except (ValueError, RecursionError):
            i = text.find("{", i + 1)
            continue
        if (isinstance(data, dict) and type(data.get("answered")) is bool
                and isinstance(data.get("claims"), list)):
            answer = data
        i = text.find("{", end)
    return answer


def generate(cfg, question: str, evidence_block: str, chat=client.chat_json) -> dict:
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_block}\n\nJSON:"
    data = _parse(chat(cfg.grounding.base_url, cfg.grounding.model, SYSTEM, user,
                       schema=ANSWER_SCHEMA, max_tokens=cfg.grounding_max_tokens,
                       think=cfg.grounding_thinking))
    data = data if isinstance(data, dict) else {}
    if data.get("answered") is not True:
        return {"answered": False, "claims": []}
    raw_claims = data.get("claims")
    claims = [
        c for c in (raw_claims if isinstance(raw_claims, list) else [])
        if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
        and isinstance(c.get("evidence_ids"), list)
        and c["evidence_ids"] and all(isinstance(e, str) for e in c["evidence_ids"])
    ]
    claims = claims[:MAX_CLAIMS]
    return {"answered": bool(claims), "claims": claims}
