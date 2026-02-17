# TokenTax AI-First Start Plan

- Version: 1.0
- Date: 2026-02-17
- Input docs:
- `/Users/rajveerbishnoi/Documents/New project/docs/PRD.md`
- `/Users/rajveerbishnoi/Documents/New project/docs/TRD.md`
- `/Users/rajveerbishnoi/Documents/New project/docs/HLD.md`
- `/Users/rajveerbishnoi/Documents/New project/docs/LLD.md`
- `/Users/rajveerbishnoi/Documents/New project/docs/IMPLEMENTATION_GUIDE.md`

## 1) Goal of This Plan
Build TokenTax as a production-grade AI+software system while developing deep expertise in:
- multilingual tokenization behavior
- fairness and inequality measurement
- cost modeling for LLM APIs
- reliable backend systems for reproducible AI evaluation

## 2) How to Work (Execution Rules)
- Run in two parallel tracks each week:
- `Track A`: AI depth (theory + experiments)
- `Track B`: Product engineering (backend + frontend + ops)
- Every week must produce one artifact from each track.
- No feature work without measurable acceptance criteria.
- Every metric in UI must map to a formula in code and a definition in docs.

## 3) AI Topic Order (What to Learn First)

1. Tokenization foundations
- BPE, unigram LM, SentencePiece, merge rules, byte fallback.
- Why scripts like Tamil/Arabic/CJK often fragment into more tokens.
- Unicode normalization (`NFC`, `NFKC`) and impact on token counts.

2. Corpus and measurement science
- Parallel corpora and semantic equivalence assumptions.
- Sampling strategy, outlier handling, and confidence intervals.
- Why median ratio is more robust than mean ratio for this use case.

3. Fairness metrics and statistics
- Ratio metrics vs dispersion metrics.
- Log-domain variance and why it avoids asymmetry bias.
- Gini index as secondary inequality signal.

4. Cost modeling
- Per-million-token pricing arithmetic.
- Input cost isolation vs total cost with output assumptions.
- Scenario analysis and sensitivity testing.

5. Reproducibility and evaluation systems
- Versioning formula, corpus, tokenizer versions.
- Golden fixtures and deterministic tests.
- Confidence labeling (`exact` vs `estimated`) and honesty in benchmarking.

6. Production AI systems
- API design for evaluators.
- Async job orchestration, idempotency, retry design.
- Observability for AI pipelines.

## 4) 8-Week Step-by-Step Plan

### Week 1: Research Freeze + Experiment Sandbox
AI topics:
- Tokenization internals and Unicode normalization.
- Multilingual script differences.

Engineering tasks:
- Freeze V1 language set and model set from PRD.
- Create `experiments/` scripts to tokenize 200 parallel samples with GPT and Llama paths.
- Save baseline CSV with token counts per language.

Deliverables:
- `baseline_token_counts.csv`
- one notebook/report showing token ratio distributions.

Exit criteria:
- You can reproduce same token counts on rerun.
- You can explain why median ratio was chosen.

### Week 2: Metric Engine First
AI topics:
- Robust statistics (median, MAD, percentile bands).
- Log-dispersion fairness score behavior.

Engineering tasks:
- Implement metric library (`ratio`, `tax_percent`, `fairness`, `gini`).
- Add tests for edge cases (`zero baseline`, `missing languages`, `outliers`).
- Lock `formula_version=v1`.

Deliverables:
- metrics package with unit tests.
- fixture-based correctness report.

Exit criteria:
- hand-calculated fixtures match code output exactly.
- fairness score monotonicity tests pass.

### Week 3: Tokenizer Adapter Layer
AI topics:
- Adapter confidence semantics and uncertainty communication.
- Tokenizer version pinning.

Engineering tasks:
- Build adapter interface from LLD.
- Implement `tiktoken` and Llama SentencePiece adapters.
- Add placeholder estimated adapter with explicit `estimated` confidence.
- Golden tests for known input strings.

Deliverables:
- adapter registry + deterministic fixtures.

Exit criteria:
- all adapters return typed results and confidence labels.
- failures produce structured error codes.

### Week 4: Data Layer + Realtime API
AI topics:
- Experiment provenance and metadata lineage.

Engineering tasks:
- Implement DB schema + migrations.
- Build `/v1/analyze/realtime`.
- Integrate metric and pricing engines.
- Add validation and rate limiting.

Deliverables:
- running API with persisted run option.
- OpenAPI examples for core endpoint.

Exit criteria:
- contract tests pass.
- p95 under local target for small payloads.

### Week 5: Batch Corpus Benchmark Pipeline
AI topics:
- Batch evaluation design and sample-size tradeoffs.
- Drift and variance across reruns.

Engineering tasks:
- Implement queue + worker.
- Build `/v1/evaluations` + status endpoint.
- Add aggregation persistence and idempotency keys.

Deliverables:
- full async corpus benchmark run path.

Exit criteria:
- 1k sample run completes and stores scores.
- retry does not duplicate final runs.

### Week 6: Cost Intelligence + Pricing Governance
AI topics:
- Cost scenario modeling for product decisions.
- Confidence-aware business reporting.

Engineering tasks:
- Implement pricing snapshots and refresh job.
- Add stale snapshot handling.
- Add scenario calculator (`expected_output_tokens` sweep).

Deliverables:
- model/language cost comparison outputs with timestamps.

Exit criteria:
- all cost results trace to a pricing snapshot ID.
- stale pricing warnings surface in API/UI.

### Week 7: Frontend Storytelling and Exports
AI topics:
- Communicating model limitations clearly.
- Visualizing uncertainty and fairness without overclaiming.

Engineering tasks:
- Build analyze and benchmark pages.
- Add fairness card, ratio table, and cost chart.
- Add methodology drawer tied to PRD formulas.
- Add CSV/JSON exports.

Deliverables:
- complete end-to-end UI workflow.

Exit criteria:
- user can run analysis and download reproducible output.
- every displayed metric has formula tooltip.

### Week 8: Hardening + Production Readiness
AI topics:
- Reliability patterns in AI-backed systems.
- Responsible benchmarking disclosures.

Engineering tasks:
- Add tracing, structured logs, dashboards, and alerts.
- Run load tests and optimize hotspots.
- Finalize README with methodology and limitations.

Deliverables:
- launch checklist completed.
- staging reliability report.

Exit criteria:
- SLO gates met in staging.
- no critical security findings.

## 5) First 14 Days (Daily Plan)

Day 1:
- Freeze scope and language/model matrix.
- Decide corpus source and licensing.

Day 2:
- Build minimal experiment script for GPT tokenizer path.
- Record token counts on 50 parallel sentences.

Day 3:
- Add Llama tokenizer path.
- Compare per-language ratio distributions.

Day 4:
- Add Unicode normalization experiments (`raw` vs `NFC`).
- Decide normalization policy for V1.

Day 5:
- Implement metric functions in library.
- Add manual fixture tests.

Day 6:
- Implement fairness and gini functions.
- Validate monotonic behavior with synthetic inputs.

Day 7:
- Implement pricing arithmetic with Decimal precision.
- Build simple scenario calculator script.

Day 8:
- Define DB models and write first migration.
- Seed language and model metadata.

Day 9:
- Build adapter interface + registry.
- Add tiktoken adapter tests.

Day 10:
- Add Llama adapter tests.
- Add confidence tagging contract.

Day 11:
- Build realtime API endpoint skeleton.
- Connect adapters + metrics in service layer.

Day 12:
- Persist runs and results.
- Add error handling and request IDs.

Day 13:
- Add validation and rate limiting.
- Write contract and integration tests.

Day 14:
- Demo API using sample multilingual payload.
- Produce first benchmark report artifact.

## 6) Competency Milestones (What Mastery Looks Like)

By end of Week 2:
- You can defend metric choices mathematically.

By end of Week 4:
- You can trace any API response field back to formula and source data.

By end of Week 6:
- You can explain cost impact in product/business language with confidence tags.

By end of Week 8:
- You can run, monitor, and troubleshoot the full system in staging.

## 7) Critical Pitfalls to Avoid
- Mixing product claims and research claims without confidence labeling.
- Adding too many models/languages before determinism and tests are stable.
- Using mean-based ratios without outlier controls.
- Forgetting reproducibility metadata (tokenizer version, corpus version, formula version).
- Treating estimated tokenizer paths as exact measurements.

## 8) Suggested Immediate Next Action
Implement only this first slice before anything else:
1. `tiktoken` + Llama tokenization adapters.
2. ratio/fairness/cost engine with tests.
3. one realtime endpoint returning analyzed metrics.

This gives a real demo in the shortest time while protecting methodology quality.
