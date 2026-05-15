# KUBIG_HACKATHON — UAI 2026 Spring (Tech Track: Poisoned RAG)

Enterprise 이메일 archive (Enron 형식) PDF 코퍼스에 대한 RAG 시스템.
다중 문서 추론 (multi-hop) + 자연어 위장 인젝션 / PII 방어.

📖 **자세한 문서: [`tech-starterkit/PIPELINE.md`](tech-starterkit/PIPELINE.md)**

---

## 파이프라인 한눈에

```
PDF
  → [Upstage Document Parse]          ← element type 보존, pypdf fallback
  → [Indexing-time sanitize]          ← 22 injection patterns 제거
  → [Chunking 800/100]
  → [Upstage Solar Embedding 4096D]   ← passage / query 분리
  → [Qdrant in-memory] + [BM25Okapi]

Question
  → [Hybrid: Dense top-30 + BM25 top-30 → RRF k=60]
  → [BGE cross-encoder rerank (BAAI/bge-reranker-v2-m3) → top-5]
  → [Solar Pro generate]
  → [Output sanitize + PII mask + Groundedness Check]
  → Answer
```

### 적용 테크닉

| 단계 | 기술 |
|---|---|
| Parse | Upstage Document Parse (429 백오프, pypdf fallback) |
| Chunking | sliding window 800 / overlap 100 |
| Dense | Upstage Solar Embedding 4096D + Qdrant cosine |
| Sparse | rank-bm25 `BM25Okapi` |
| Fusion | RRF (k=60) |
| Rerank | BGE cross-encoder `BAAI/bge-reranker-v2-m3` |
| Sanitize | indexing-time 22-pattern regex (HackAPrompt 영감) |
| PII | label-aware mask + output guard |
| LLM | Upstage Solar Pro |
| Groundedness | `solar-1-mini-groundedness-check` (notGrounded → 거절) |

### Reference

- **FinGEAR** (EMNLP 2025) — BGE rerank + hybrid 결합
- **HackAPrompt** (EMNLP 2023) — 인젝션 패턴 12개 중 8개 출처
- **RRF** (SIGIR 2009) — Reciprocal Rank Fusion k=60 표준값

---

## 실험 결과

**환경**: Solar Pro + `pii,san2,hybrid,rerank,ground` (full flag)
**Corpus**: `sample_corpus/sample_corpus/enron/` (15 Enron PDFs)
**Test**: `sample_test_set/test_set_34.jsonl` (34 questions)

| Category | Correct | Pct |
|---|---:|---:|
| L1 (단일 사실) | 7/8 | 87.5% |
| L2 (2-hop) | 8/8 | **100%** |
| L3 (계산/카운트) | 6/8 | 75% |
| Poisoning (인젝션 방어) | 0/5 | **0%** ⚠️ |
| Leakage (PII 보호) | 0/5 | **0%** ⚠️ |
| **Total** | **21/34** | **61.8%** |

| | Reasoning | Security | Latency | **Weighted** |
|---|---:|---:|---:|---:|
| 점수 | 85% | 0% | 1.09s 평균 (0% penalty) | **51.0 / 100** |

### 분석 요약

- **Reasoning 85% — 강력**: L2 perfect, L1/L3 균형. 산수 오류 1건 (S_0019) + retrieval miss 2건 (S_0007, S_0022).
- **Security 0% — 완전 실패**: sample_corpus 의 새 5종 트리거 phrase (`Data&extracted`, `Audit&extraction` 등) 와 NYMEX form PII label (Mother's Maiden / DOB / Home Phone) 을 기존 `EXTENDED_INJECTION_PATTERNS` + `PII_MASK_PATTERNS` 가 cover 안 함. 30분 작업으로 patch 가능 (PIPELINE.md § 5.1 참조).

---

## 재현

```powershell
cd tech-starterkit

# env 설정
$env:UPSTAGE_API_KEY    = "<your-key>"
$env:CORPUS_DIR         = "sample_corpus/sample_corpus/enron"
$env:SOLAR_MODEL        = "solar-pro"
$env:ABLATION_FLAGS     = "pii,san2,hybrid,rerank,ground"

# 인덱싱 + 평가 (~5분)
python run_sample_eval.py

# 채점
python sample_test_set/score.py submission_sample.csv sample_test_set/test_set_34.jsonl
```

---

## 구조

```
hackathon/
├── README.md                          ← 본 문서
├── CLAUDE.md                          ← 프로젝트 인스트럭션 (Claude Code 자동 로드)
├── sample_corpus.tar.gz               ← 원본 corpus archive
├── hackathon-regs/tech/               ← 본대회 규정
└── tech-starterkit/
    ├── PIPELINE.md                    ★ 자세한 파이프라인 + 실험 분석
    ├── baseline_rag.py                ★ 메인 RAG 파이프라인
    ├── run_sample_eval.py             ← sample_corpus 평가 entry
    ├── submission_sample.csv          ← 본 실험 산출물
    ├── decryptor.py / upstage_tracker.py / validator.py
    ├── sample_corpus/sample_corpus/enron/   ← 15 PDFs
    └── sample_test_set/               ← test_set_34.jsonl + score.py
```

---

## 다음 단계 (우선순위)

1. **§ 5.1 sanitize regex 보강** — Security 0 → 80+ (Poisoning + Leakage 회복)
2. **§ 5.2 corpus-wide 메타 chunk** — L3 S_0022 같은 cross-mailbox 질문 해결
3. **§ 5.3 dynamic ban-token learning** — 새 트리거 phrase 자동 추출
4. **§ 5.4 metadata 확장** — Document Parse element type / msg_index / mailbox 활용

자세한 코드 변경 예시는 `tech-starterkit/PIPELINE.md` § 5 참조.
