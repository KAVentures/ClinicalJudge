# One-shot Codex / Claude Code implementation prompt

The repository contains a runnable v0.1.0 prototype. Give the following complete task to Codex or Claude Code **from the cloned repository root**. The agent should implement, test and leave a reviewable PR or commit as permitted. It must not claim that algorithmic tests establish medical validation, must not fabricate physician judgments and must not incur API/GPU charges without explicit permission.

---

**OBJECTIVE**

Turn ClinicalJudge into a fully documented, reproducible, MIT-licensed open-source clinical AI evaluation toolkit usable as a Codex or Claude Code skill, Python API, CLI and optional MCP server. Its scientific purpose is to enable *independent automated scoring* of open-ended medical AI responses while measuring, not assuming, agreement with blinded physicians. Use upstream PrecepTron's released task-specific adapters where appropriate; do not copy their models, copyrighted vignettes, or GRAND-ROUNDS data into this repository. Preserve the current code, improve rather than replace it, and maintain reproducibility of prior artifacts.

**HARD RULES**

- The coding agent must orchestrate; it must never provide its own clinical ratings. All actual scores originate from an explicitly configured, separate, preferably open-weight judge model/checkpoint, deterministic verifier, or named physician, with provenance and correct labels. No silent fallbacks.
- Same-family Qwen comparisons must be labeled non-independent or use a separately validated alternative judge. No judge is declared unbiased merely because it is open-weight. Benchmark generator-family bias rather than assuming neutrality.
- No invented doctors, annotations, validation scores, medical evidence or clinical-equivalence claims. Fake/demo data are visibly marked and excluded from clinical metrics.
- Keep all training, calibration, validation and final test cases strictly separated by clinical case and perturbation families. Reject leakage and log reproducible hashes.
- No patient-identifiable data or proprietary vignettes in public outputs. Remote transfers default to disabled; explicit researcher approval, HTTPS, licensing and institutional rules are required.
- MIT only for original code. Respect distinct model, adapters and dataset licensing. Never expose API keys.

**ENGINEERING TASKS**

1. Audit existing source, fix correctness and security issues, add test coverage. Keep a clean Python package and CLI with pinned version tags, model/adapter revisions, exact rubric hashes and structured inference provenance.
2. Complete real end-to-end separate-model scoring against an independently launched Qwen3-32B-compatible endpoint and upstream PrecepTron adapters. Add tests using a fake local server and conditional opt-in live tests. Reproduce PrecepTron's published scores on allowed public fixtures if datasets are accessible; document unavailable copyrighted cases. No access? Record BLOCKED rather than inventing results.
3. Implement modular evaluation instruments for diagnostic accuracy, differential/clinical reasoning, treatment and management, medical note quality, clinical safety, evidence sufficiency, missing information, response to contradictory evidence, abstention, helpfulness and evidence quality, plus blinded *pairwise comparison* as a separate instrument. Each has a fixed schema, exact criterion anchors, registered version and validation status. Pairwise comparison must randomize presentation order, retain answer rendering parity, and test order bias; never conflate pairwise preference with an absolute clinical score.
4. Add adapters mapping existing KAVentures research schemas from `clinical-evidence-sufficiency-llm`, `health-ai-readiness-robustness` and `clinical-ai-reconciliation` into normalized JSONL, plus GRAND-ROUNDS/PrecepTron adapters where legal. Do not execute paid model inference unless permitted. Use saved model outputs when present. Validate every converted record and preserve case grouping.
5. Maintain authoritative reference evidence for each case: answerability, reference diagnosis if available, critical missing fields, known contradictions, guideline-based dangerous action rules, valid alternatives, and source evidence. Require these where necessary; do not derive hidden ground truth from candidate model text. Verify objective attributes deterministically where possible and use an independent judge for interpretation. Prevent prompt injection via untrusted response content, strict parsing and no tool execution of judge text.
6. Add `calibrate`/`train` optional LoRA workflow with explicit physician labels and reproducible case splits; *never require training for basic use*. Compare untuned zero-shot/Qwen, existing PrecepTron adapters and any newly trained adapter on exactly the same untouched holdout. Reject improvements that only reflect test-set reuse.
7. Add human-review packet generation for one or more independent physician raters (randomized, blinded, full rubric anchors, clinician expertise, conflict-of-interest fields, provenance), with CSV/JSON imports, disagreement analysis, adjudication and overlap checks. Preserve rater-level data. Do not generate artificial human labels.
8. Complete statistical evaluation: exact/win-within-tolerance, ordinal quadratic-weighted Cohen kappa, inter-physician comparisons on the same cases, case-cluster bootstrap intervals, safety-critical error sensitivity/specificity/false reassurance, stratification by response generator, specialty and perturbation type. Add predeclared equivalence/noninferiority methods with powered sample-design inputs; don't claim equivalence by comparing point estimates alone.
9. Add stress tests: equivalent wording, length, Markdown/citations, relative certainty, contradictory findings, missing facts, language variants, adversarial instructions embedded in candidate answers, style-conditioned self-preference and cross-family generalization. Evaluate clinical *misjudgments*, not only JSON compliance.
10. Make all existing and future externally contributed validation studies usable through a machine-readable per-study manifest, frozen instruments, protocol links, model revision, clinician population, rater independence, held-out dataset checksum, analysis output and signed-off review status. Let other teams contribute new rubrics and human comparisons by PR. Never auto-approve an equivalence claim from a submitted manifest.
11. Deliver working Codex and Claude Code `SKILL.md` files, quickstart, Python API, CLI, self-hosted deployment instructions and opt-in MCP tools, and reproducible export to publication-ready JSON/CSV/Markdown with per-task limitations. Keep all endpoint/model calls explicit and auditable. Support offline no-network operation for syntax/validation tests.
12. Add GitHub Actions CI and full unit/integration/property-style tests, dependency review, documentation and examples. Make setup extremely simple for researchers without GPU knowledge, but never imply a model can run without compute or credentials. If hosting is optional, distinguish inference charges and privacy risk. Do not change the clinical label on a scored result without recomputing or issuing a new model/rubric release.

**ACCEPTANCE GATES**

- A fresh clone installs and all offline tests pass on Python 3.10 and 3.12.
- A synthetic fixture goes from ingest→validate→blind→separate mock HTTP endpoint→structured scores→physician-comparison smoke test→report with no cloud tokens or claimed physician validation.
- A real configured independent backend works on representative de-identified fixture input when accessible, and inference provenance is pinned. Same-family judging is blocked by default.
- Altering reference/response/rubric/model revision cannot silently reuse cached ratings. Any unsupported task is explicitly blocked or experimental.
- A contributed external study is schema-validated and registered as *submitted/unreviewed* until independent acceptance.
- A full clinical validation suite cannot be declared "passing" unless actual independent physician ratings have been supplied and a preregistered analysis was executed. Surface missing evidence as a documented gap.
- Run the suite, iterate until code defects are fixed, provide exact passing test counts, unresolved scientific limitations, changes and the next validation experiment. Do not promise future autonomous work.

**FINISH BY** providing a truthful release report: implemented features, blocked validation prerequisites, actual test outputs, current judge tasks/versions, exact model/dataset dependencies, and a minimal usage example.

## Mandatory provenance acceptance criteria (ClinicalJudge v0.2.0)

- Preserve `judge.transport_provenance` from the real API response: returned
  model ID, API-created timestamp, request ID when available, completion ID,
  system fingerprint when available, finish reason, usage, request/response UTC
  timestamps, system prompt and full request hashes, exact decoding settings.
- Never replace a missing response field with an invented or request-side value.
  Track missing fields in `clinicaljudge provenance` audit reports.
- For Qwen/PrecepTron local models support distinct **base model, tokenizer,
  and LoRA adapter** commit IDs in a validated manifest; `--require-pinned`
  prevents missing declarations. Show that a declared SHA does not prove the
  server actually loaded the corresponding weights; require a real serving
  command/attestation for stronger claims.
- Distinguish model *release date* (verified from release/model-card provenance)
  from evaluation date (`request_sent_at_utc`) and API response `created`.
- Produce a publication-ready reproducibility manifest/report with all relevant
  model IDs/commits, runtime version, rubric and instruction hashes, random
  seeds, temperature and generation settings, package lock/environment info.
- Capture `PrecepTron` inference transport metadata without changing its
  canonical task prompts and reference rubrics. Test real endpoint outputs
  locally using mocked OpenAI-compatible HTTP servers (no API spending).
- Explicitly reject attempts to claim that a manually entered model alias or
  SHA implies independently verified checkpoint identity.
