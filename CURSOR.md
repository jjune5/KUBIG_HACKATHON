# CURSOR.md — Cursor Agent Guidance

This project is the UAI 2026 Spring Hackathon Tech Track: **Poisoned RAG**.
Use Korean when talking to the user, with English technical terms where useful.

## Core Behavior

Follow the Karpathy-style coding discipline from
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills):

- **Think before coding**: state assumptions, surface ambiguity, ask one sharp question if blocked.
- **Simplicity first**: prefer the smallest change that satisfies the current goal.
- **Surgical changes**: do not refactor unrelated code or clean up files the user did not ask about.
- **Goal-driven execution**: convert work into verifiable criteria, then test or inspect the result.

## Repository Rules

- Main implementation file: `tech-starterkit/baseline_rag.py`.
- Do not modify `decryptor.py`, `upstage_tracker.py`, or `validator.py`.
- Treat `hackathon-regs/tech/` as read-only reference material.
- Do not hardcode answers, question IDs, source PDFs, or test-set-specific shortcuts.
- Do not bypass decryption or write plaintext decrypted test questions to disk.
- Final answers must be generated through `UpstageTracker.chat()` so `used_tokens` is nonzero.

## Current Recommended Pipeline

Default target is the normal `tech-starterkit` pipeline, not `tech-starterkit-clean`.

Recommended defaults:

```powershell
$env:SOLAR_MODEL = "solar-pro"
$env:ABLATION_FLAGS = "pii,san2,hybrid,rerank"
$env:PARSER = "upstage"
python baseline_rag.py
```

Preferred mechanisms:

- Upstage Document Parse for PDF parsing when time allows.
- Upstage `embedding-passage` / `embedding-query`.
- Qdrant dense retrieval + BM25 hybrid RRF.
- BGE reranker for candidate reranking.
- `solar-pro` for final answer generation.
- `pii` and `san2` output guards enabled.

Avoid by default:

- `parent` retrieval: experiments showed context noise and lower L2/L3.
- `pot`: experiments showed unstable code generation and worse overall results.
- `source_pdf` filtering: it is an evaluation hint, not available in real encrypted tests.

## D-day Corpus Handling

Assume the actual D-day corpus has been provided locally, and inspect that corpus
before making parser, chunking, retrieval, or security decisions.

Do not hardcode assumptions from any development sample. The final corpus may be
larger, may include different document families, and may contain additional poisoned
documents, injection styles, and PII formats.

Before final runs:

- Count PDFs and pages.
- Sample several normal-looking and suspicious-looking documents.
- Search for instruction-like text, NOTICE blocks, role-switching text, and trigger phrases.
- Search for PII-like fields: personal phone, home phone, birth date, maiden name, SSN,
  salary, bank account, payroll fields.
- Update sanitize/output guards based on observed corpus patterns.
- Keep retrieval generic. Do not rely on `source_pdf` or answer-key metadata.

Development samples are examples only. Use their patterns as starting hypotheses,
not as final corpus facts.

## Security Patterns

Defend against fake compliance / audit / access-control instructions in retrieved documents.
Sanitize before indexing and guard after generation.

Known trigger examples from development samples; extend them after inspecting the actual corpus:

- `Data&extracted. No additional records.`
- `No&further data available.`
- `Audit&extraction complete. No further info.`
- `Query&complete. Additional data restricted.`
- `Extraction&complete. Further queries require clearance.`
- Legacy tokens like `APPROVED_BY_*`, `[VERIFIED]`, `[SECURE]`

PII to refuse or mask:

- SSN / Korean RRN
- personal cell phone, home phone
- mother's maiden name
- birth date
- salary, bank account, payroll details

Allowed when clearly work/public info:

- work phone numbers such as `713-853-XXXX`
- work email addresses
- office locations
- job titles, departments, company names

## Evaluation

Use sample sets for local scoring only; real scoring happens on `stages.ai`.

```powershell
python sample_test_set/score.py submission.csv sample_test_set/test_set_34.jsonl
python shared_test_set/score.py submission.csv shared_test_set/test_set_55.jsonl
```

Remember:

- `expected_keywords`, `answer_raw`, and `source_pdf` are local evaluation aids only.
- Real encrypted tests likely provide only `question_id`, decrypted `question`, and `token`.
- `validator.py` checks schema, not correctness.

## Response Style

- Be concise and direct.
- The user is under hackathon time pressure; prioritize decisions and executable next steps.
- Explain RAG concepts briefly when needed.
- If a test is expensive, say so before running it.
- Report command results in Korean with the important numbers, because the user may not see raw output.
