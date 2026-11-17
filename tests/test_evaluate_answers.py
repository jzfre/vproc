"""Harness contracts use controlled models; these tests do not measure model accuracy."""
import importlib.util
import json
from pathlib import Path

import pytest

from vproc.config import Config, Endpoint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_answers", ROOT / "scripts" / "evaluate_answers.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def raw_dataset():
    return {
        "segments": [{"id": "fact", "memory_id": "meeting", "project_id": "default", "speaker": "Ada",
                      "start_ts": 12, "end_ts": 15, "said_text": "Secret Aurora launch is July 17."}],
        "cases": [{"id": "date", "question": "Secret launch date?", "expected_segment_ids": ["fact"],
                   "should_abstain": False, "expected_phrases": ["July 17"]},
                  {"id": "missing", "question": "Secret unrelated question?", "project": "missing",
                   "expected_segment_ids": [], "should_abstain": True}],
    }


def cfg(tmp_path):
    ep = Endpoint("unused", "synthetic")
    return Config(ep, ep, ep, str(tmp_path / "must-not-create"), "localhost", 8765, 0.25, 0.5, None)


def models():
    return dict(embed=lambda *a: [[1., 0.] for _ in a[2]],
                chat=lambda *a, **k: json.dumps({"answered": True, "claims": [
                    {"text": "The launch is July 17.", "evidence_ids": ["E1"]}]}),
                scorer=lambda p, h: 0.95)


def test_verified_evaluation_uses_temporary_index_and_private_report(tmp_path):
    config = cfg(tmp_path)
    report = E.evaluate(E.Dataset.model_validate(raw_dataset()), config, **models())
    assert report["passed"] is True
    assert report["summary"]["completed"] == 2
    assert report["summary"]["mean_recall_at_k"] == 1.0
    assert report["summary"]["mean_abstention_correct"] == 1.0
    assert report["summary"]["mean_citation_source_accuracy"] == 1.0
    assert report["summary"]["mean_lexical_phrase_recall"] == 1.0
    assert not Path(config.index_path).exists()
    assert "Secret" not in json.dumps(report) and "July" not in json.dumps(report)


def test_retrieval_only_does_not_load_or_call_answer_models(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Retrieval-only mode must not load a verifier or call grounding")

    monkeypatch.setattr(E, "HHEM", forbidden)
    report = E.evaluate(E.Dataset.model_validate(raw_dataset()), cfg(tmp_path), retrieval_only=True,
                        embed=models()["embed"], chat=forbidden, scorer=forbidden)
    assert report["passed"] is True and report["mode"] == "retrieval_only"
    assert report["summary"]["mean_abstention_correct"] is None
    assert report["summary"]["mean_citation_source_accuracy"] is None
    assert report["summary"]["mean_lexical_phrase_recall"] is None


def test_wrong_phrase_fails_even_with_supported_claims(tmp_path):
    data = raw_dataset()
    data["cases"][0]["expected_phrases"] = ["December"]
    report = E.evaluate(E.Dataset.model_validate(data), cfg(tmp_path), **models())
    assert report["passed"] is False
    assert report["cases"][0]["lexical_phrase_recall"] == 0


def test_citation_to_wrong_source_fails(tmp_path):
    data = raw_dataset()
    data["segments"].append({**data["segments"][0], "id": "wrong", "said_text": "Secret lunch menu."})
    fake = models()
    # A supported-looking answer with the distractor's evidence is not a correct source.
    fake["chat"] = lambda *a, **k: json.dumps({"answered": True, "claims": [
        {"text": "The launch is July 17.", "evidence_ids": ["E2"]}]})
    report = E.evaluate(E.Dataset.model_validate(data), cfg(tmp_path), **fake)
    assert report["passed"] is False
    assert report["cases"][0]["citation_source_accuracy"] == 0


def test_failed_retrieval_recall_is_not_hidden_by_correct_answer(tmp_path):
    data = raw_dataset()
    data["segments"].append({**data["segments"][0], "id": "other", "said_text": "Unrelated engineering lunch."})
    fake = models()
    def embed(url, model, texts):
        return [[0., 1.], [1., 0.]] if len(texts) == 2 else [[1., 0.]]
    fake["embed"] = embed
    data["cases"][0]["question"] = "Describe the shipping schedule."
    report = E.evaluate(E.Dataset.model_validate(data), cfg(tmp_path), k=1, **fake)
    assert report["cases"][0]["recall_at_k"] == 0
    assert report["passed"] is False


@pytest.mark.parametrize("stage", ["corpus_embedding", "retrieval", "answer", "verifier_load"])
def test_backend_errors_fail_closed_without_content_or_credentials(tmp_path, monkeypatch, stage):
    fake = models()
    def fail(*a, **k):
        raise RuntimeError("Secret prompt and bearer s3cret")
    if stage == "corpus_embedding":
        fake["embed"] = fail
    elif stage == "retrieval":
        original = fake["embed"]
        calls = 0
        def embed(*a):
            nonlocal calls
            calls += 1
            return original(*a) if calls == 1 else fail()
        fake["embed"] = embed
    elif stage == "answer":
        fake["chat"] = fail
    else:
        fake["scorer"] = None
        monkeypatch.setattr(E, "HHEM", fail)
    report = E.evaluate(E.Dataset.model_validate(raw_dataset()), cfg(tmp_path), **fake)
    assert report["passed"] is False
    assert report["error"] == {"stage": stage, "type": "RuntimeError"}
    assert report["summary"]["completed"] == 0
    assert "Secret" not in json.dumps(report) and "s3cret" not in json.dumps(report)


@pytest.mark.parametrize("problem", ["unknown", "duplicate_source", "duplicate_case", "duplicate_expected", "wrong_scope", "no_source"])
def test_bad_gold_data_rejected_before_model_calls(problem):
    data = raw_dataset()
    if problem == "unknown":
        data["cases"][0]["expected_segment_ids"] = ["unknown"]
    elif problem == "duplicate_source":
        data["segments"].append(data["segments"][0])
    elif problem == "duplicate_case":
        data["cases"].append(data["cases"][0])
    elif problem == "duplicate_expected":
        data["cases"][0]["expected_segment_ids"] = ["fact", "fact"]
    elif problem == "wrong_scope":
        data["cases"][0]["project"] = "missing"
    else:
        data["cases"][0]["expected_segment_ids"] = []
    with pytest.raises(ValueError):
        E.Dataset.model_validate(data)


def test_shipped_dataset_is_valid_and_has_real_distractors():
    dataset = E.Dataset.model_validate_json((ROOT / "evals" / "answers.json").read_text())
    assert len(dataset.segments) > 8  # default top-k cannot return every source
    assert len(dataset.cases) >= 8


def test_cli_invalid_dataset_does_not_print_private_validation_input(tmp_path, capsys):
    path = tmp_path / "private.json"
    path.write_text('{"segments": "Secret transcript"}')
    assert E.main(["--dataset", str(path)]) == 2
    assert "Secret" not in capsys.readouterr().err


@pytest.mark.parametrize("passed", [True, False])
def test_cli_exit_status_matches_report(tmp_path, monkeypatch, passed):
    path, output = tmp_path / "data.json", tmp_path / "report.json"
    path.write_text(json.dumps(raw_dataset()))
    monkeypatch.setattr(E, "load_dotenv", lambda: None)
    monkeypatch.setattr(E, "load_config", lambda: cfg(tmp_path))
    monkeypatch.setattr(E, "evaluate", lambda *a, **k: {"passed": passed})
    assert E.main(["--dataset", str(path), "--output", str(output)]) == (0 if passed else 1)
    assert json.loads(output.read_text()) == {"passed": passed}


def test_citation_with_wrong_timestamp_fails_even_for_correct_source(tmp_path, monkeypatch):
    real_ask = E.ask_memory
    def broken_citation(*args, **kwargs):
        answer = real_ask(*args, **kwargs)
        if answer.claims:
            answer.claims[0].citations[0].start_ts += 1
        return answer
    monkeypatch.setattr(E, "ask_memory", broken_citation)
    report = E.evaluate(E.Dataset.model_validate(raw_dataset()), cfg(tmp_path), **models())
    assert report["passed"] is False
    assert report["cases"][0]["citation_source_accuracy"] == 0


def test_missing_embeddings_fail_before_store_write(tmp_path):
    fake = models()
    fake["embed"] = lambda *a: []
    report = E.evaluate(E.Dataset.model_validate(raw_dataset()), cfg(tmp_path), **fake)
    assert report["error"] == {"stage": "corpus_embedding", "type": "ValueError"}
    assert report["passed"] is False
