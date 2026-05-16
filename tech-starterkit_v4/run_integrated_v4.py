"""run_integrated_v4.py — V4 main eval + robustness + 12-flag ablation in ONE build.

Strategy:
  - build_index ONCE (~4 min)
  - eval test_set_34 with all-flags (= V4 score)
  - eval robustness_test_11 with all-flags (= cheating verification)
  - eval test_set_34 with 12 flag combinations (= ablation matrix)

Outputs:
  - submission_v4.csv          (34 questions, all flags)
  - submission_robust_v4.csv   (11 robustness questions)
  - ablation_matrix.csv        (12 configs × 34 questions = 408 rows)
"""

import json
import os
import sys
import time
import csv

# preflight: set defaults BEFORE importing baseline_rag (env captured at import)
os.environ.setdefault("CORPUS_DIR",    "../tech-starterkit/sample_corpus/sample_corpus/enron")
os.environ.setdefault("SOLAR_MODEL",   "solar-pro")
os.environ.setdefault("ABLATION_FLAGS", "pii,san2,hybrid,rerank,ground")

from baseline_rag import build_index, retrieve, generate_answer, SOLAR_MODEL, _flags
from upstage_tracker import UpstageTracker

sys.path.insert(0, "../tech-starterkit/sample_test_set")
from score import score_one, hackathon_score   # noqa

CORPUS_DIR        = os.environ["CORPUS_DIR"]
TEST_SET_PATH     = "../tech-starterkit/sample_test_set/test_set_34.jsonl"
ROBUST_TEST_PATH  = "robustness_test_7.jsonl"

ABLATION_CONFIGS = [
    ("baseline",   ""),
    ("pii",        "pii"),
    ("san2",       "san2"),
    ("hybrid",     "hybrid"),
    ("rerank",     "rerank"),
    ("ground",     "ground"),
    ("all",        "pii,san2,hybrid,rerank,ground"),
    ("all-pii",    "san2,hybrid,rerank,ground"),
    ("all-san2",   "pii,hybrid,rerank,ground"),
    ("all-hybrid", "pii,san2,rerank,ground"),
    ("all-rerank", "pii,san2,hybrid,ground"),
    ("all-ground", "pii,san2,hybrid,rerank"),
]


def _run_questions(questions: list[dict], index, tracker: UpstageTracker, label: str) -> list[dict]:
    out = []
    for q in questions:
        qid = q["question_id"]
        try:
            ctx = retrieve(q["question"], index)
            ans = generate_answer(
                question=q["question"], context=ctx, tracker=tracker,
                question_id=qid, token=label,
            )
        except Exception as e:
            ans = f"!!ERROR: {type(e).__name__}: {e}"
        out.append({"question_id": qid, "answer": ans})
    return out


def _score_block(answers: list[dict], qs: dict, label: str) -> None:
    by_cat = {}
    for r in answers:
        q = qs.get(r["question_id"])
        if not q:
            continue
        ok = score_one(q["category"], r["answer"], q["expected_keywords"], q["forbidden_keywords"])
        by_cat.setdefault(q["category"], [0, 0])
        by_cat[q["category"]][0] += 1 if ok else 0
        by_cat[q["category"]][1] += 1
    print(f"  [{label}]", end=" ")
    for cat, (ok, total) in sorted(by_cat.items()):
        print(f"{cat}={ok}/{total}", end="  ")
    print()


def main() -> None:
    print(f"[Config] CORPUS_DIR    = {CORPUS_DIR}")
    print(f"[Config] SOLAR_MODEL   = {SOLAR_MODEL}")
    print()

    print(f"[1/5] Build index ({CORPUS_DIR})...")
    t0 = time.time()
    index = build_index(CORPUS_DIR)
    print(f"  → built in {time.time()-t0:.1f}s\n")

    print(f"[2/5] Load test_set_34 + robustness_test_11...")
    test_qs = [json.loads(l) for l in open(TEST_SET_PATH, encoding="utf-8")]
    test_qs_map = {q["question_id"]: q for q in test_qs}
    robust_qs = [json.loads(l) for l in open(ROBUST_TEST_PATH, encoding="utf-8")]
    robust_qs_map = {q["question_id"]: q for q in robust_qs}
    print(f"  → main {len(test_qs)} + robust {len(robust_qs)}\n")

    # ── Phase A: V4 main eval (all flags) ──────────────────────────────────
    print(f"[3/5] V4 main eval (all flags)...")
    os.environ["ABLATION_FLAGS"] = "pii,san2,hybrid,rerank,ground"
    print(f"  ABLATION_FLAGS = {sorted(_flags())}")
    tracker = UpstageTracker(model=SOLAR_MODEL)
    ans_v4 = _run_questions(test_qs, index, tracker, "v4_main")
    tracker.save_csv("submission_v4.csv")
    _score_block(ans_v4, test_qs_map, "V4 main")

    # ── Phase B: V4 robustness ─────────────────────────────────────────────
    print(f"\n[4/5] V4 robustness (all flags, novel patterns)...")
    tracker_r = UpstageTracker(model=SOLAR_MODEL)
    ans_robust = _run_questions(robust_qs, index, tracker_r, "v4_robust")
    tracker_r.save_csv("submission_robust_v4.csv")
    _score_block(ans_robust, robust_qs_map, "V4 robust")

    # ── Phase C: 12-flag ablation matrix ───────────────────────────────────
    print(f"\n[5/5] 12-flag ablation (12 configs × 34 questions = 408 evals)...")
    matrix_rows = []
    for cfg_name, flags in ABLATION_CONFIGS:
        os.environ["ABLATION_FLAGS"] = flags
        t = UpstageTracker(model=SOLAR_MODEL)
        ans = _run_questions(test_qs, index, t, f"abl_{cfg_name}")
        by_cat = {}
        for r in ans:
            q = test_qs_map.get(r["question_id"])
            if not q:
                continue
            ok = score_one(q["category"], r["answer"], q["expected_keywords"], q["forbidden_keywords"])
            by_cat.setdefault(q["category"], [0, 0])
            by_cat[q["category"]][0] += 1 if ok else 0
            by_cat[q["category"]][1] += 1
        rows_for_hackathon = [
            {"category": qq["category"],
             "correct": score_one(qq["category"], rr["answer"], qq["expected_keywords"], qq["forbidden_keywords"])}
            for rr, qq in zip(ans, [test_qs_map[r["question_id"]] for r in ans])
        ]
        hs = hackathon_score(rows_for_hackathon)
        row = {
            "config": cfg_name,
            "flags":  flags,
            "L1":     f"{by_cat.get('L1',[0,0])[0]}/{by_cat.get('L1',[0,0])[1]}",
            "L2":     f"{by_cat.get('L2',[0,0])[0]}/{by_cat.get('L2',[0,0])[1]}",
            "L3":     f"{by_cat.get('L3',[0,0])[0]}/{by_cat.get('L3',[0,0])[1]}",
            "poi":    f"{by_cat.get('poisoning',[0,0])[0]}/{by_cat.get('poisoning',[0,0])[1]}",
            "lek":    f"{by_cat.get('leakage',[0,0])[0]}/{by_cat.get('leakage',[0,0])[1]}",
            "weighted_score": hs["weighted_score"],
        }
        matrix_rows.append(row)
        print(f"  [{cfg_name:12s}] L1={row['L1']} L2={row['L2']} L3={row['L3']} poi={row['poi']} lek={row['lek']} score={row['weighted_score']}")

    with open("ablation_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=matrix_rows[0].keys())
        w.writeheader()
        for r in matrix_rows:
            w.writerow(r)
    print(f"\n  → ablation_matrix.csv saved ({len(matrix_rows)} rows)")

    print(f"\nDone. Files: submission_v4.csv, submission_robust_v4.csv, ablation_matrix.csv")


if __name__ == "__main__":
    sys.exit(main())
