# ClinicalJudge (prototype v0.2.0)

**An open, reproducible framework for clinical AI evaluation with independent inference backends, immutable-at-run-time rubrics, and physician validation tooling.**

> **Status: experimental research prototype. No included generic rubric has been physician-validated. No claim of physician-equivalence, clinical safety certification, or medical-device suitability is made.**

ClinicalJudge lets researchers use Codex and Claude Code to *orchestrate* clinical evaluations while ensuring neither agent becomes the judge. Instead, the software sends blinded inputs to a deliberately specified independent judge endpoint (e.g. locally hosted Qwen-based model or a compatible PrecepTron adapter). There is no fallback to the coding agent's own model, and all scores retain rubric hashes and declared judge/generator families.

## Try it (no model required)

Requires Python 3.10+. From the repository root:

```bash
python -m clinicaljudge rubrics
python -m clinicaljudge lint --input examples/cases.jsonl
python -m unittest discover -s tests -v
python -m clinicaljudge run --backend demo --input examples/cases.jsonl --output runs/demo.jsonl
python -m clinicaljudge validate --predictions runs/demo.jsonl --physicians examples/physician_ratings.csv --output runs/demo_validation.json
```

**The `demo` backend produces intentionally fabricated, nonmedical placeholder scores.** It exists only for command, schema and report smoke tests. Do not present demo output as research results or physician validation.

## Run a real separate judge

1. Serve a genuinely separate open-weight model with an OpenAI-compatible `/v1/chat/completions` API (for example vLLM with a pinned Qwen model checkpoint and verified model weights).
2. Review that input contains no PHI and set an explicit model identity and revision.
3. Run:

```bash
export JUDGE_API_KEY=EMPTY
python -m clinicaljudge run \
  --input examples/cases.jsonl \
  --output runs/qwen_experimental.jsonl \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen-judge \
  --model-revision PINNED_MODEL_SHA_HERE \
  --judge-family qwen
```

This runs the **experimental rubrics only**. Changing the model and endpoint does not magically confer physician calibration or validated critical-error sensitivity. A hosted URL requires `--permit-remote` and HTTPS. The tool does *not* download or serve a model on its own.

## Use PrecepTron for its actual released tasks

Read [docs/BACKENDS.md](docs/BACKENDS.md). Install the upstream MIT-licensed [PrecepTron](https://github.com/2v/PrecepTron) package and serve its LoRA adapters on Qwen3-32B. Provide task names `cpc_bond`, `diagnostic_reasoning`, `management_reasoning`, or `r_idea`, and the reference fields documented by PrecepTron.

```bash
python -m clinicaljudge run \
 --input /path/to/preceptron_cases.jsonl \
 --output runs/preceptron.jsonl \
 --backend preceptron \
 --base-url http://127.0.0.1:8000/v1 \
 --model PrecepTron-32B-CPC \
 --model-revision PINNED_LORA_COMMIT \
 --judge-family qwen
```

The released PrecepTron training/validation figures apply to **its published tasks**, not automatically to custom safety or missing-information rubrics. `ClinicalJudge` does not redistribute PrecepTron weights or GRAND-ROUNDS data.

## Available experimental rubric tasks

`diagnosis`, `clinical_reasoning`, `clinical_management`, `documentation`, `safety`, `missing_information`, `contradiction`, `evidence_sufficiency`, `helpfulness`, `evidence_quality`. Every rubric is JSON with ID/version and criterion weights. You can add entirely new tasks without touching engine code by contributing a new JSON file with tests. Changes to a published rubric must increment the version and ideally preserve the old version as a separate artifact.

Pairwise preference and complex reference adjudication are **planned**, not falsely marked as implemented. There is no universal score that substitutes for all clinical constructs.

## Input records (JSONL)

```json
{"case_id":"case1","response_id":"response1","task":"missing_information","case":"Clinically de-identified vignette","response":"Model's actual response","generator_family":"openai","reference":{"answerable":false,"critical_missing_facts":["one required test"]}}
```

Responses in one clinical case, across perturbations and models, must share `case_id` for grouping. `reference` should be established *before* inspecting model outputs. The grading backend does **not** receive generator family or model identity. No record should contain identifiable patient data.

## Research provenance: precise model identity and evaluation timestamps

Every real API call now captures the **requested and API-returned model IDs**,
UTC send/receive times, provider-created timestamp, request/completion IDs,
optional backend fingerprint, token usage, finish reason, instruction/full-input
hashes and the framework's source-code fingerprint. The engine records which
fields are missing rather than claiming a false exact snapshot.

For locally hosted Qwen or PrecepTron, provide `--model-manifest` with the *real*
40-character base checkpoint SHA, optional tokenizer SHA, **separate LoRA SHA**,
inference runtime/version and verified model release date (if known). Add
`--require-pinned` to refuse missing/invalid manifests. The metadata explicitly
says the researcher-declared commits are **not automatically verified against
server-loaded weights**: you must actually pin the serving configuration.

```bash
python -m clinicaljudge run --input cases.jsonl --output scores.jsonl \
 --backend preceptron --base-url http://127.0.0.1:8000/v1 \
 --model PrecepTron-32B-CPC --judge-family qwen \
 --model-manifest runs/model_manifest.json --require-pinned
python -m clinicaljudge provenance \
 --predictions scores.jsonl --output scores_provenance.json
```

Read [docs/PROVENANCE.md](docs/PROVENANCE.md) for model snapshot/LoRA pinning,
publication reporting, runtime reproducibility and limitations. `--model-revision`
remains supported for compatibility but is **only a manual string** without a
verified manifest.

## Human validation (CSV)

```text
response_id,case_id,task,rater_id,score,critical_error
response1,case1,missing_information,doctor_01,0.7,1
```

`score` is normalized 0–1 **using the same task rubric and anchors as the AI**. `critical_error` is a human binary rating (0/1) or blank if not assessed. Blinded physicians should score independently. For a real study use an externally held-out dataset and specify case-level calibration exclusion:

```bash
python -m clinicaljudge validate \
 --predictions runs/qwen_experimental.jsonl \
 --physicians path/to/real_physician_labels.csv \
 --train-cases path/to/calibration_case_ids.txt \
 --output runs/validation_report.json
```

The report provides per-task within-tolerance agreement (Wilson and clustered bootstrap intervals), median-physician weighted kappa, physician-physician weighted kappa on double-rated responses, and safety-critical sensitivity/specificity with confidence intervals where binary labels exist. These **descriptive measures do not establish noninferiority**; formal equivalence demands a preregistered margin, appropriate clinical panels and adequate power. See [VALIDATION.md](VALIDATION.md).

## Install as agent skill

Clone this repository into your research workspace, then place a link or a copy of the relevant `SKILL.md` under your project `.agents/skills/clinicaljudge/` for Codex or `.claude/skills/clinicaljudge/` for Claude Code. The agent runs the CLI; the agent is *not* authorized to grade clinical answers itself. See [docs/AGENT_SETUP.md](docs/AGENT_SETUP.md).

## Scientific design constraints

- No same-family grading by default; unknown family identities must be reviewed. This is only a proxy for independence, not proof of absence of model bias.
- Human labels are never generated or impersonated by AI.
- Case/perturbation groups, not response rows, are the unit of held-out splitting and uncertainty analysis.
- Rubric changes invalidate cached grades (hash checked at resume).
- Outputs are clearly labeled `NOT_VALIDATED_ON_THIS_TASK` until independently supported by a separate validation report.
- No default network access to patient data and no automatic inference-cost escalation.
- Structured result explanations/quotes are **claims made by the model**, not independently verified clinical facts.
- Do not publish data without rights and consent. MIT applies to *this code*, not external datasets, model weights or clinical cases.

## Contribute validation

Follow [CONTRIBUTING.md](CONTRIBUTING.md), [VALIDATION.md](VALIDATION.md), and [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md). Submit task-specific protocols, reviewer agreements and independently reproducible performance outputs. Separate model weights/checkpoints and appropriately licensed data from the core MIT codebase.

## Publication / licensing

Code: MIT. Upstream PrecepTron code/adapters: see their MIT license and model cards; Qwen3-32B base weights: separate Apache-2.0 terms; source datasets and physician annotations: their own terms. This is not a clinical decision-support product.
