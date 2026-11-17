"""Synthetic grounded-answer contract evaluations; no model downloads or private data.

Real LanceDB retrieval, response parsing, verification gate, and citations run together.
Embedding/chat/scorer outputs are controlled: this evaluates application behavior,
not the semantic accuracy of a deployed model.
"""
import json
from functools import partial

import pytest

from vproc.answer.ask import ask_memory
from vproc.answer.generate import generate
from vproc.config import Config, Endpoint
from vproc.retrieve.retriever import retrieve
from vproc.store.lancedb_store import Store


def claim(text, evidence_ids=None):
    return {"text": text, "evidence_ids": evidence_ids if evidence_ids is not None else ["E1"]}


SUPPORTED = "The release ships in July."
SCREEN = "The budget is 42 credits."
SPEAKER = "Ada owns the release."

# Each tuple declares input model behavior and independently expected application behavior.
CASES = [
    ("spoken", "release", [1., 0.], [claim(SUPPORTED)], None, [SUPPORTED]),
    ("screen", "budget", [1., 0.], [claim(SCREEN)], None, [SCREEN]),
    ("speaker", "owner", [1., 0.], [claim(SPEAKER)], None, [SPEAKER]),
    ("contradiction", "release", [1., 0.], [claim("The release ships in December.")], "unverified", []),
    ("mixed", "release", [1., 0.], [claim(SUPPORTED), claim("The budget is unlimited.")], None, [SUPPORTED]),
    ("invented-citation", "release", [1., 0.], [claim(SUPPORTED, ["E99"])], "unverified", []),
    ("missing-citation", "release", [1., 0.], [claim(SUPPORTED, [])], "no_answer", []),
    ("off-topic", "orbital mechanics", [0., 1.], [claim(SUPPORTED)], "low_similarity", []),
    ("declined", "release", [1., 0.], [], "no_answer", []),
    ("wrong-scope", "release", [1., 0.], [claim(SUPPORTED)], "no_hits", []),
]


@pytest.mark.parametrize("name,query,vector,claims,reason,expected", CASES, ids=[c[0] for c in CASES])
def test_grounded_answer_contract(tmp_path, name, query, vector, claims, reason, expected):
    ep = Endpoint("unused", "synthetic")
    cfg = Config(ep, ep, ep, str(tmp_path / "db"), "127.0.0.1", 8765, 0.25, 0.5, None)
    store = Store(cfg.index_path)
    store.add([{
        "id": "release-1", "project_id": "default", "memory_id": "Synthetic planning",
        "speaker": "Ada", "start_ts": 12., "end_ts": 20.,
        "said_text": "The release ships in July. I own the release.",
        "on_screen_text": SCREEN, "embed_text": "Ada release ships July budget 42 credits",
        "source_video": "synthetic.mp4", "frame_path": "", "vector": [1., 0.],
    }], embedding_model=ep.model)
    generated = []

    def chat(*args, **kwargs):
        generated.append(True)
        return json.dumps({"answered": bool(claims), "claims": claims})

    def scorer(premise, hypothesis):
        # Explicit fixture oracle, not a substitute for HHEM quality evaluation.
        facts = {SUPPORTED: SUPPORTED, SCREEN: SCREEN, SPEAKER: "Speaker: Ada"}
        return 0.95 if hypothesis in facts and facts[hypothesis] in premise else 0.05

    result = ask_memory(
        store, cfg, query, scorer,
        where="project_id = 'other'" if name == "wrong-scope" else None,
        _retrieve=partial(retrieve, embed=lambda *a: [vector]),
        _generate=partial(generate, chat=chat),
    )
    assert result.abstain_reason == reason
    assert result.answered == bool(expected)
    assert result.abstained == (not expected)
    assert [c.text for c in result.claims] == expected
    if reason in {"no_hits", "low_similarity"}:
        assert not generated
    for c in result.claims:
        assert c.evidence_ids == ["E1"]
        assert [(cite.memory_title, cite.start_ts, cite.speaker) for cite in c.citations] == [
            ("Synthetic planning", 12., "Ada")]
    if not expected:
        assert result.claims == [] and result.evidence == []
