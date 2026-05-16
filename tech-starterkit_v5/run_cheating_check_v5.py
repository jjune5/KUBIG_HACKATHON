"""run_cheating_check_v5.py — verify whether _is_mass_pii_query is cheating.

Runs 4 configs:
  1. mass_pii=on,  test_set_34   (= V4 behavior)
  2. mass_pii=off, test_set_34   (= general defense only)
  3. mass_pii=on,  robustness_11
  4. mass_pii=off, robustness_11

If leakage stays high with mass_pii=off → regex was redundant (not cheating).
If leakage drops sharply → regex was test-set-aware (cheating).
"""

import json
import os
import sys
import time

os.environ.setdefault("CORPUS_DIR",    "../tech-starterkit/sample_corpus/sample_corpus/enron")
os.environ.setdefault("SOLAR_MODEL",   "solar-pro")
os.environ.setdefault("ABLATION_FLAGS", "pii,san2,hybrid,rerank,ground")

from baseline_rag import build_index, retrieve, generate_answer, SOLAR_MODEL
from upstage_tracker import UpstageTracker

sys.path.insert(0, "../tech-starterkit/sample_test_set")
from score import score_one

CORPUS_DIR       = os.environ["CORPUS_DIR"]
TEST_SET_PATH    = "../tech-starterkit/sample_test_set/test_set_34.jsonl"
ROBUST_PATH      = "robustness_test_7.jsonl"


def run_eval(label, mass_pii_flag, questions, index):
    os.environ["MASS_PII"] = mass_pii_flag
    tracker = UpstageTracker(model=SOLAR_MODEL)
    answers = []
    for q in questions:
        try:
            ctx = retrieve(q["question"], index)
            ans = generate_answer(
                question=q["question"], context=ctx, tracker=tracker,
                question_id=q["question_id"], token=label,
            )
        except Exception as e:
            ans = f"!!ERROR: {type(e).__name__}: {e}"
        answers.append({"question_id": q["question_id"], "answer": ans})
    tracker.save_csv(f"submission_{label}.csv")
    return answers


def score_block(answers, qs_map, label):
    by_cat = {}
    for r in answers:
        q = qs_map.get(r["question_id"])
        if not q:
            continue
        ok = score_one(q["category"], r["answer"], q["expected_keywords"], q["forbidden_keywords"])
        by_cat.setdefault(q["category"], [0, 0])
        by_cat[q["category"]][0] += 1 if ok else 0
        by_cat[q["category"]][1] += 1
    print(f"[{label:30s}]", end=" ")
    for cat, (ok, total) in sorted(by_cat.items()):
        pct = 100*ok/total if total else 0
        print(f"{cat}={ok}/{total}({pct:.0f}%)", end="  ")
    print()
    return by_cat


def main():
    print(f"[Config] CORPUS_DIR  = {CORPUS_DIR}")
    print(f"[Config] SOLAR_MODEL = {SOLAR_MODEL}\n")

    print("[1/2] Build index...")
    t0 = time.time()
    index = build_index(CORPUS_DIR)
    print(f"  → built in {time.time()-t0:.1f}s\n")

    test_qs = [json.loads(l) for l in open(TEST_SET_PATH, encoding="utf-8")]
    test_qs_map = {q["question_id"]: q for q in test_qs}
    robust_qs = [json.loads(l) for l in open(ROBUST_PATH, encoding="utf-8")]
    robust_qs_map = {q["question_id"]: q for q in robust_qs}

    print(f"[2/2] 4 configs eval:\n")

    # main test set: mass_pii on vs off
    ans = run_eval("test34_mass_on",  "on",  test_qs, index)
    score_block(ans, test_qs_map, "test_34 mass_pii=on")
    ans = run_eval("test34_mass_off", "off", test_qs, index)
    score_block(ans, test_qs_map, "test_34 mass_pii=off")

    # robust set: mass_pii on vs off
    ans = run_eval("robust11_mass_on",  "on",  robust_qs, index)
    score_block(ans, robust_qs_map, "robust_11 mass_pii=on")
    ans = run_eval("robust11_mass_off", "off", robust_qs, index)
    score_block(ans, robust_qs_map, "robust_11 mass_pii=off")

    print(f"\nDone. CSVs: submission_test34_mass_{{on,off}}.csv, submission_robust11_mass_{{on,off}}.csv")


if __name__ == "__main__":
    sys.exit(main())
