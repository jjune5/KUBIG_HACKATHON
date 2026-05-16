"""run_sample_eval_v2.py — Evaluate v2 baseline_rag on sample_corpus + test_set_34.

v2 changes: Phase 1 patches A+B+C+D+G (sanitize / PII / SYSTEM_PROMPT / Spotlighting).

Imports build_index / retrieve / generate_answer from local v2 baseline_rag.
Reuses sample_corpus + sample_test_set from ../tech-starterkit/ (same data).
Saves submission_sample_v2.csv compatible with sample_test_set/score.py.

Usage (from tech-starterkit_v2/):
    python run_sample_eval_v2.py
"""

import json
import os
import sys
import time

from baseline_rag import build_index, retrieve, generate_answer, SOLAR_MODEL, _flags
from upstage_tracker import UpstageTracker


CORPUS_DIR    = os.environ.get(
    "CORPUS_DIR",
    "../tech-starterkit/sample_corpus/sample_corpus/enron",
)
TEST_SET_PATH = "../tech-starterkit/sample_test_set/test_set_34.jsonl"
OUTPUT_PATH   = "submission_sample_v2.csv"


def main() -> None:
    print(f"[Config] CORPUS_DIR     = {CORPUS_DIR}")
    print(f"[Config] SOLAR_MODEL    = {SOLAR_MODEL}")
    print(f"[Config] ABLATION_FLAGS = {sorted(_flags())}")
    print(f"[Config] TEST_SET_PATH  = {TEST_SET_PATH}")
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
    print(f"Run: python ../tech-starterkit/sample_test_set/score.py {OUTPUT_PATH} {TEST_SET_PATH}")


if __name__ == "__main__":
    sys.exit(main())
