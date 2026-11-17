#!/usr/bin/env python3
"""Reproduce storage/catalogue scaling with synthetic, disposable local data.

Run from an installed development environment:
    uv run --no-sync python scripts/benchmark_store.py --rows 10000

No model endpoints or existing indexes are used. JSON goes to stdout. Timings
are observations, not CI pass/fail performance thresholds. tracemalloc measures
Python allocations only; it does not include native Arrow/Lance memory.
"""
import argparse
import gc
from importlib.metadata import version
import json
import platform
import tempfile
import time
import tracemalloc

from vproc.store.lancedb_store import Store


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def previous_catalogue(store):
    """The old HTTP catalogue algorithm, for a reproducible before/after comparison."""
    by_memory = {}
    for row in store.all_rows():
        if row.get("project_id", "default") == "default":
            by_memory.setdefault(row["memory_id"], []).append(row)
    return [{"memory_id": mid, "segment_count": len(rows),
             "duration_s": max(row["end_ts"] for row in rows),
             "speakers": sorted({row["speaker"] for row in rows})}
            for mid, rows in sorted(by_memory.items())]


def measure(call):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = call()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {"seconds": round(elapsed, 6), "python_peak_MiB": round(peak / 1024**2, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive, default=10000)
    parser.add_argument("--dimensions", type=positive, default=128)
    parser.add_argument("--text-chars", type=positive, default=2000)
    args = parser.parse_args()
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "lancedb": version("lancedb"), "pyarrow": version("pyarrow"),
              "rows": args.rows, "memories": (args.rows + 99) // 100,
              "dimensions": args.dimensions, "text_chars_per_field": args.text_chars}
    text = ("Synthetic discussion. " * (args.text_chars // 22 + 1))[:args.text_chars]
    with tempfile.TemporaryDirectory(prefix="vproc-benchmark-") as directory:
        store = Store(directory)

        def row(i, memory=None):
            return {"id": str(i), "project_id": "default", "memory_id": memory or f"m{i // 100}",
                    "speaker": "Ada", "start_ts": float(i % 100), "end_ts": float(i % 100 + 1),
                    "said_text": text, "on_screen_text": "", "embed_text": text,
                    "source_video": "synthetic.mp4", "frame_path": "",
                    "vector": [1.] + [0.] * (args.dimensions - 1)}

        rows = [row(i) for i in range(args.rows)]
        start = time.perf_counter()
        store.add(rows, embedding_model="synthetic-benchmark")
        report["initial_add_with_fts_seconds"] = round(time.perf_counter() - start, 6)
        del rows
        before, report["previous_catalogue"] = measure(lambda: previous_catalogue(store))
        after, report["streaming_catalogue"] = measure(store.memory_summaries)
        if before != after:
            raise RuntimeError("Catalogue output changed")
        query = [1.] + [0.] * (args.dimensions - 1)
        hits, report["filtered_cosine"] = measure(lambda: store.vector_search(
            query, 8, "memory_id = 'm0'", embedding_model="synthetic-benchmark"))
        if len(hits) != min(8, args.rows) or any(hit["memory_id"] != "m0" for hit in hits):
            raise RuntimeError("Filtered vector query returned wrong rows")
        start = time.perf_counter()
        store.replace_memory("m0", "default", [row(args.rows + i, "m0") for i in range(min(100, args.rows))],
                             embedding_model="synthetic-benchmark")
        report["replace_100_or_fewer_with_fts_seconds"] = round(time.perf_counter() - start, 6)
        if sum(item["segment_count"] for item in store.memory_summaries()) != args.rows:
            raise RuntimeError("Replacement changed library size")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
