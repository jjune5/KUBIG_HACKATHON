"""run_sample_eval.py — Evaluate baseline_rag pipeline on sample_corpus + test_set_34.

Imports build_index / retrieve / generate_answer from baseline_rag (no modification).
Saves submission_sample.csv compatible with sample_test_set/score.py.

Usage:
    $env:CORPUS_DIR = "sample_corpus/sample_corpus/enron"
    python run_sample_eval.py
"""

import json
import os
import sys
import time

from baseline_rag import build_index, retrieve, generate_answer, SOLAR_MODEL, _flags
from upstage_tracker import UpstageTracker


CORPUS_DIR    = os.environ.get("CORPUS_DIR", "sample_corpus/sample_corpus/enron")
TEST_SET_PATH = "sample_test_set/test_set_34.jsonl"
OUTPUT_PATH   = "submission_sample.csv"


def main() -> None:
    print(f"[Config] CORPUS_DIR    = {CORPUS_DIR}")
    print(f"[Config] SOLAR_MODEL   = {SOLAR_MODEL}")
    print(f"[Config] ABLATION_FLAGS = {sorted(_flags())}")
    print(f"[Config] TEST_SET_PATH = {TEST_SET_PATH}")
    print()

    print(f"[1/3] Building index from {CORPUS_DIR}...")
    t0 = time.time()
    index = build_index(CORPUS_DIR)
    print(f"  → index built in {time.time() - t0:.1f}s\n")

    print(f"[2/3] Loading test set...")
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"  → {len(questions)} questions\n")

    print(f"[3/3] Running pipeline (model={SOLAR_MODEL})...")
    tracker = UpstageTracker(model=SOLAR_MODEL)
    for q in questions:
        qid = q["question_id"]
        try:
            context = retrieve(q["question"], index)
            answer  = generate_answer(
                question     = q["question"],
                context      = context,
                tracker      = tracker,
                question_id  = qid,
                token        = "sample",
            )
            print(f"  [{qid}] {answer[:80]}")
        except Exception as e:
            print(f"  [{qid}] !! ERROR: {type(e).__name__}: {e}")

    print()
    tracker.save_csv(OUTPUT_PATH)
    print(f"\nDone. Submission saved → {OUTPUT_PATH}")
    print(f"Run: python sample_test_set/score.py {OUTPUT_PATH} {TEST_SET_PATH}")


if __name__ == "__main__":
    sys.exit(main())
