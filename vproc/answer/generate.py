import json

from vproc.llm import client

SYSTEM = (
    "You answer ONLY from the numbered EVIDENCE provided below. For every claim you make, "
    "cite the evidence id(s) it comes from. If the evidence does not contain the answer, set "
    'answered to false and return an empty claims list. Never use outside knowledge or your '
    "own assumptions. "
    'Respond as strict JSON: {"answered": bool, "claims": [{"text": str, "evidence_ids": [str]}]}'
)


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except Exception:
                pass
    return {"answered": False, "claims": []}


def generate(cfg, question: str, evidence_block: str, chat=client.chat_json) -> dict:
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_block}\n\nJSON:"
    data = _parse(chat(cfg.grounding.base_url, cfg.grounding.model, SYSTEM, user))
    claims = [c for c in data.get("claims", []) if c.get("evidence_ids")]
    return {"answered": bool(data.get("answered")) and len(claims) > 0, "claims": claims}
