from vproc.answer.evidence import evidence_text
from vproc.models import Evidence

HHEM_MODEL = "vectara/hallucination_evaluation_model"


class HHEM:
    """Lazy wrapper around Vectara HHEM-2.1-Open (loaded on first use)."""

    def __init__(self, model_id: str = HHEM_MODEL):
        from transformers import AutoModelForSequenceClassification  # lazy: heavy import

        load = AutoModelForSequenceClassification.from_pretrained
        try:
            self.model = load(model_id, trust_remote_code=True)
        except OSError:
            # A Hub that rejects the ambient HF_TOKEN (kept for gated diarization) must not take
            # the public default verifier down with it: retry this repo anonymously once. The
            # model's remote code loads its foundation config/tokenizer on its own, still with
            # the ambient token, so a cold cache can fail regardless; the service's 503 then
            # points at HF_TOKEN. A genuinely private or missing model fails the same way again.
            self.model = load(model_id, trust_remote_code=True, token=False)

    def score(self, premise: str, hypothesis: str) -> float:
        return float(self.model.predict([(premise, hypothesis)])[0])


def filter_claims(claims: list[dict], evidence_by_key: dict[str, Evidence],
                  scorer, threshold: float, scores: list[float] | None = None) -> list[dict]:
    """Keep the claims their cited evidence supports. `scores`, when given, collects every
    valid gate score (kept or not) so callers can report numbers without claim text."""
    kept: list[dict] = []
    for claim in claims:
        # Repeating an ID cannot supply additional evidence. It used to repeat
        # the premise (changing HHEM's score/context) and duplicate citations.
        ids = list(dict.fromkeys(k for k in claim["evidence_ids"] if k in evidence_by_key))
        premise = "\n".join(evidence_text(evidence_by_key[k]) for k in ids)
        if not premise:
            continue
        score = scorer(premise, claim["text"])
        # A score must be a probability; invalid model output cannot verify a claim.
        if type(score) in (int, float) and 0.0 <= score <= 1.0:
            if scores is not None:
                scores.append(float(score))
            if score >= threshold:
                kept.append({**claim, "evidence_ids": ids})
    return kept
