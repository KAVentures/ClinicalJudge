# Physician-scored missing-information pilot (50 model responses)

**State as of 2026-10-23: source matching and data preparation supported; actual independent Qwen/PrecepTron inference NOT executed. Do not cite as a validation of the new judge.**

The full source is publicly available at frozen commit `48c6b67102e4494bf224f63fd9a61816aa9da920` of [KAVentures/health-ai-readiness-robustness](https://github.com/KAVentures/health-ai-readiness-robustness). It contains 50 blinded response items, 33 distinct source cases and 150 physician ratings from three clinicians (two external, one study author). Previous automated votes (OpenAI, Anthropic, xAI and Google) were independently compared with this cohort in `validations/pilots/2026-10-23-existing-judge-physician-baseline.md`.

## Why this differs from native PrecepTron

PrecepTron offers four specialized physician-trained adapters for **different** rubrics. These do not establish physician equivalence for this study's appropriate-uncertainty criterion. For a first independent judge experiment, select an explicitly pinned *separate* open-weight model (e.g. the Qwen3 backbone) and the **experimental `appropriate_uncertainty_binary` rubric** that matches the original physicians' exact 0/1 label definition. Such a model is uncalibrated for the task; its performance must be measured, not presumed.

## Reproduce, without exposure of held-out physician labels to the judge

```bash
python -m pip install -e '.[dev]'
python scripts/prepare_healthbench_physician50.py --output-dir runs/healthbench_physician50
python -m clinicaljudge lint --input runs/healthbench_physician50/cases_blinded.jsonl

# Configure an independently served open-weight Qwen model; do not set --backend demo.
# Pin model checkpoint and inference runtime; see docs/PROVENANCE.md.
# localhost example: no uploaded physician labels or source metadata go to the judge.
clinicaljudge run \
  --input runs/healthbench_physician50/cases_blinded.jsonl \
  --output runs/healthbench_physician50/qwen_predictions.jsonl \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-family qwen \
  --model YOUR_EXACT_SERVED_QWEN_MODEL_ID

clinicaljudge validate \
  --predictions runs/healthbench_physician50/qwen_predictions.jsonl \
  --physicians runs/healthbench_physician50/physicians_HOLDOUT.csv \
  --tolerance 0 \
  --output runs/healthbench_physician50/qwen_human_comparison.json
clinicaljudge provenance \
  --predictions runs/healthbench_physician50/qwen_predictions.jsonl \
  --output runs/healthbench_physician50/qwen_provenance.json
```

`cases_blinded.jsonl` contains no reference labels or legacy judge votes. `physicians_HOLDOUT.csv` is separate and must not be included in model inference, prompt optimization or training for a genuine holdout comparison. It is *not enough* simply to tell the coding agent not to read the labels: use isolated directories, permissions or a separate process for stricter blinded designs.

If using a remote API endpoint, first review benchmark licensing and data-sharing terms, pass `--permit-remote` consciously, and avoid transmitting any non-public cases or PHI. We intentionally do not hardcode endpoint credentials.

## What to measure

- Exact agreement and Cohen's κ against the **three-rater majority**, reporting author involvement.
- Independent-only physician references `O` and `G`, reported separately and compared with their own 74% agreement (κ≈0.472) on these 50 responses.
- 95% intervals accounting for clustering by the 33 clinical cases (not treating all responses as distinct cases).
- Missed incorrect-uncertainty responses (`judge=1` when physician reference=0), and excess conservative responses (`judge=0` when physicians=1).
- By-model-family behavior for GPT, Claude, Gemini and Grok generated responses.
- Original scores from the old judges as a reference, *never* as substitute new predictions.

**Limitations**: This is a binary behavior score, not a ground truth for clinical harm or diagnostic correctness. Case removal does not automatically create an unanswerable question. ClinicalJudge's generic rubric engine and its Qwen backend remain unvalidated for this task until new results exist. Reproducible weight hashes, temperatures and prompt hashes are required for a research result; see provenance instructions.

## Current remote-compute blocker

On 2026-10-23, a request to Hugging Face Jobs to run even a CPU-basic preflight returned HTTP **402 Payment Required**. The user's Hugging Face account has no confirmed usable positive Jobs credit balance. No real model inference was run; do not record any `ClinicalJudge`/`PrecepTron` agreement, sensitivity or κ values without actual inference outputs. The released PrecepTron adapters also have access-gating requiring model-license consent on Hugging Face. Their specialized validation does not transfer automatically to this task.