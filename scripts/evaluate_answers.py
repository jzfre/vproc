#!/usr/bin/env python3
"""Evaluate configured models on a synthetic (or user-supplied) gold corpus.

All index writes go to temporary storage. Reports omit questions, meeting content,
model responses, URLs and exception messages. See evals/README.md for metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vproc.answer.ask import ask_memory
from vproc.answer.faithfulness import HHEM
from vproc.answer.generate import generate
from vproc.config import load_config, load_dotenv
from vproc.llm import client
from vproc.retrieve.retriever import retrieve
from vproc.service import Query, build_where
from vproc.store.lancedb_store import Store


class GoldSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=128)
    memory_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(default="default", min_length=1, max_length=128)
    speaker: str = Field(min_length=1, max_length=128)
    start_ts: float = Field(ge=0)
    end_ts: float = Field(ge=0)
    said_text: str = Field(default="", max_length=16000)
    on_screen_text: str = Field(default="", max_length=16000)

    @model_validator(mode="after")
    def valid(self):
        if self.end_ts < self.start_ts or not (self.said_text.strip() or self.on_screen_text.strip()):
            raise ValueError("Segment must contain content and ordered times")
        if not all(s.strip() for s in (self.id, self.memory_id, self.project_id, self.speaker)):
            raise ValueError("Identifiers must not be blank")
        return self


class GoldCase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=8192)
    expected_segment_ids: list[str] = Field(default_factory=list, max_length=200)
    should_abstain: bool = Field(strict=True)
    expected_phrases: list[str] = Field(default_factory=list, max_length=20)
    min_recall: float = Field(default=1.0, ge=0, le=1)
    project: str | None = None
    memory: str | None = None
    speaker: str | None = None

    @field_validator("id", "question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @model_validator(mode="after")
    def valid(self):
        if len(self.expected_segment_ids) != len(set(self.expected_segment_ids)):
            raise ValueError("Expected segment IDs must be unique")
        if any(not phrase.strip() for phrase in self.expected_phrases):
            raise ValueError("Expected phrases must not be blank")
        if not self.should_abstain and not self.expected_segment_ids:
            raise ValueError("Answerable cases require expected source segments")
        if self.should_abstain and self.expected_phrases:
            raise ValueError("Abstaining cases cannot require answer phrases")
        build_where(Query(project=self.project, memory=self.memory, speaker=self.speaker))
        return self


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[GoldSegment] = Field(min_length=1, max_length=200)
    cases: list[GoldCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def references(self):
        by_id = {s.id: s for s in self.segments}
        if len(by_id) != len(self.segments) or len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Segment and case IDs must each be unique")
        for case in self.cases:
            for sid in case.expected_segment_ids:
                if sid not in by_id:
                    raise ValueError("Unknown expected source segment")
                segment = by_id[sid]
                for field, value in (("project_id", case.project), ("memory_id", case.memory),
                                     ("speaker", case.speaker)):
                    if value and getattr(segment, field) != value:
                        raise ValueError("Expected source is outside the case filters")
        return self


def _rows(dataset):
    from vproc.ingest.align import make_embed_text
    from vproc.ingest.transcribe import TranscriptSegment

    rows = []
    for segment in dataset.segments:
        spoken = ([TranscriptSegment(segment.start_ts, segment.end_ts, segment.said_text,
                                     speaker=segment.speaker)] if segment.said_text else [])
        rows.append({**segment.model_dump(), "source_video": "synthetic-eval.mp4", "frame_path": "",
                     "embed_text": make_embed_text(segment.on_screen_text, spoken)})
    return rows


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def evaluate(dataset: Dataset, cfg, *, retrieval_only=False, k=8,
             embed=client.embed_texts, chat=client.chat_json, scorer=None):
    """Run real application stages; injectable model functions are for harness tests only."""
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    gold_by_id = {segment.id: segment for segment in dataset.segments}
    report = {"mode": "retrieval_only" if retrieval_only else "verified_answers",
              "k": k, "cases": [], "error": None,
              "configuration": {"embedding_model": cfg.embed.model, "grounding_model": cfg.grounding.model,
                                "hhem_model": cfg.hhem_model, "sim_floor": cfg.sim_floor,
                                "hhem_threshold": cfg.hhem_threshold,
                                "query_prefix_sha256": hashlib.sha256(cfg.embed_query_prefix.encode()).hexdigest()},
              "dataset_sha256": hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()}
    start, stage = time.monotonic(), "setup"
    try:
        client.configure(cfg)
        if not retrieval_only and scorer is None:
            stage = "verifier_load"
            scorer = HHEM(cfg.hhem_model).score
        rows = _rows(dataset)
        stage = "corpus_embedding"
        vectors = embed(cfg.embed.base_url, cfg.embed.model, [row["embed_text"] for row in rows])
        if len(vectors) != len(rows):
            raise ValueError("Embedding response cardinality mismatch")
        for row, vector in zip(rows, vectors):
            row["vector"] = vector
        with tempfile.TemporaryDirectory(prefix="vproc-eval-") as work:
            stage = "temporary_index"
            store = Store(str(Path(work) / "index"))
            store.add(rows, embedding_model=cfg.embed.model)
            for case in dataset.cases:
                case_start = time.monotonic()
                stage = "retrieval"
                where = build_where(Query(project=case.project, memory=case.memory, speaker=case.speaker))
                hits, top_sim = retrieve(store, cfg, case.question, k=k, where=where, embed=embed)
                expected = set(case.expected_segment_ids)
                recall = len(expected & {hit["id"] for hit in hits}) / len(expected) if expected else None
                scoped = all((not case.project or hit["project_id"] == case.project)
                             and (not case.memory or hit["memory_id"] == case.memory)
                             and (not case.speaker or hit["speaker"] == case.speaker) for hit in hits)
                result = {"id": case.id, "recall_at_k": recall, "scope_correct": scoped,
                          "abstention_correct": None, "citation_source_accuracy": None,
                          "lexical_phrase_recall": None, "passed": scoped and (recall is None or recall >= case.min_recall)}
                if not retrieval_only:
                    stage = "answer"
                    answer = ask_memory(
                        store, cfg, case.question, scorer, where=where, k=k,
                        _retrieve=lambda *a, **kw: (hits, top_sim),
                        _generate=lambda config, question, evidence: generate(config, question, evidence, chat=chat))
                    result["abstention_correct"] = answer.abstained == case.should_abstain
                    sources = {e.key: e.segment_id for e in answer.evidence}
                    citations = []
                    for claim in answer.claims:
                        # Verify the code-owned timestamps/speaker too; a correct
                        # evidence ID paired with the wrong playback target is a failure.
                        for i in range(max(len(claim.evidence_ids), len(claim.citations))):
                            sid = sources.get(claim.evidence_ids[i]) if i < len(claim.evidence_ids) else None
                            citation = claim.citations[i] if i < len(claim.citations) else None
                            gold = gold_by_id.get(sid)
                            citations.append(bool(sid in expected and gold and citation
                                and citation.memory_title == gold.memory_id and citation.speaker == gold.speaker
                                and citation.start_ts == gold.start_ts and citation.end_ts == gold.end_ts))
                    if citations:
                        result["citation_source_accuracy"] = sum(citations) / len(citations)
                    if case.expected_phrases:
                        claims_text = "\n".join(c.text for c in answer.claims).casefold()
                        result["lexical_phrase_recall"] = sum(
                            phrase.casefold() in claims_text for phrase in case.expected_phrases) / len(case.expected_phrases)
                    result["passed"] = (result["passed"] and result["abstention_correct"]
                                        and (case.should_abstain or result["citation_source_accuracy"] == 1.0)
                                        and (result["lexical_phrase_recall"] in (None, 1.0)))
                result["latency_s"] = round(time.monotonic() - case_start, 4)
                report["cases"].append(result)
    except Exception as exc:
        # Model errors may contain audio/text, credentials, or authenticated URLs.
        # Fail fast so a backend outage does not trigger one long timeout per case.
        report["error"] = {"stage": stage, "type": type(exc).__name__}
    report["summary"] = {
        "case_count": len(dataset.cases), "completed": len(report["cases"]),
        "passed": sum(case["passed"] for case in report["cases"]),
        "mean_recall_at_k": _mean(case["recall_at_k"] for case in report["cases"]),
        "mean_abstention_correct": _mean(case["abstention_correct"] for case in report["cases"]),
        "mean_citation_source_accuracy": _mean(case["citation_source_accuracy"] for case in report["cases"]),
        "mean_lexical_phrase_recall": _mean(case["lexical_phrase_recall"] for case in report["cases"]),
        "total_latency_s": round(time.monotonic() - start, 4),
    }
    report["passed"] = report["error"] is None and report["summary"]["passed"] == len(dataset.cases)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "evals" / "answers.json")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.k <= 100:
        parser.error("--k must be between 1 and 100")
    try:
        dataset = Dataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        load_dotenv()
        cfg = load_config()
    except Exception as exc:
        # Pydantic's normal error text includes input values; don't print it for private datasets.
        print(f"Evaluation setup failed ({type(exc).__name__}); check dataset and configuration.", file=sys.stderr)
        return 2
    report = evaluate(dataset, cfg, retrieval_only=args.retrieval_only, k=args.k)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            args.output.write_text(payload, encoding="utf-8")
        except OSError:
            print("Could not write evaluation report; check the output location.", file=sys.stderr)
            return 2
    else:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
