from vproc.models import Evidence

HHEM_MODEL = "vectara/hallucination_evaluation_model"


class HHEM:
    """Lazy wrapper around Vectara HHEM-2.1-Open (loaded on first use)."""

    def __init__(self, model_id: str = HHEM_MODEL):
        from transformers import AutoModelForSequenceClassification  # lazy: heavy import

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return float(self.model.predict([(premise, hypothesis)])[0])


def filter_claims(claims: list[dict], evidence_by_key: dict[str, Evidence],
                  scorer, threshold: float) -> list[dict]:
    kept: list[dict] = []
    for claim in claims:
        premise = "\n".join(
            evidence_by_key[k].text for k in claim["evidence_ids"] if k in evidence_by_key
        )
        if premise and scorer(premise, claim["text"]) >= threshold:
            kept.append(claim)
    return kept
