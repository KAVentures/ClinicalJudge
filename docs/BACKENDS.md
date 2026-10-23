# Backends: never substitute a coding agent's own model

## `openai-compatible`

Calls the explicit local or user-authorized HTTPS judge chat endpoint with temperature 0. Requires model ID, base URL and declared judge family. JSON criterion-by-criterion grading is attempted and strictly parsed; wrong/missing IDs abort scoring without writing a fabricated prediction. For v0.1.0 there is no automatic repair/retry of invalid model output, to prevent hidden cost and biased re-asking.

Example local vLLM command (ensure the checkpoint is pinned and hardware is sufficient):

```bash
vllm serve Qwen/Qwen3-32B --served-model-name qwen-judge --max-model-len 8192
```

Then run the CLI as documented in README. A local server does not imply the model is a physician-calibrated judge.

## `preceptron`

This backend imports `from preceptron import score` from **the upstream package** and uses the task-specific trained adapter served by an OpenAI-compatible endpoint. Install upstream source following https://github.com/2v/PrecepTron and install `openai`. You must select the correct adapter matching your case task; the software does not check that an arbitrary `--model` actually loads the expected adapter (pin it and verify deployment). Native normalized scores are stored with raw 0–max scores. Their own validations are task-specific.

Official upstream task→requirements:

- `cpc_bond`: `reference.final_diagnosis`
- `diagnostic_reasoning`: `reference.question_text`, `reference.final_diagnosis` + `case`
- `management_reasoning`: `reference.question_text`, `reference.rubric` (including `max_score`) + `case`
- `r_idea`: `reference.question_text`

The released PrecepTron `CPC` model was partly trained on a nonpublic emergency-department task. Full original training set is not reproducible from public GRAND-ROUNDS alone. Avoid claiming otherwise.

## `demo`

Produces placeholder scores solely to test the workflow. These are ALWAYS tagged `NOT_A_REAL_JUDGE`. Never cite them in study results.

## Endpoint safety

No patient data should be put in this software, unless an institution has separately authorized the storage/processing arrangement. Remote endpoints are denied by default. The user must explicitly add `--permit-remote` and provide HTTPS. Credentials are read from `JUDGE_API_KEY`. No automatic provider fallback is performed. Web-hosted endpoints can retain prompts; review their policy first.

## Judge independence and limitations

The open-weight model should not be among the candidate models, and should not be fine-tuned on the held-out test cases. Model family separation prevents only simple self-family comparisons. It does not rule out shared training data, model-to-model preference, stylistic biases, or cross-provider contamination. Require blinded human validation and generator-stratified audits to make stronger independence claims.

## Model identity and publication provenance

See [PROVENANCE.md](PROVENANCE.md). Both backends capture OpenAI-compatible HTTP response provenance. PrecepTron preserves its own canonical prompts using upstream `build_messages` and `parse_judge_response`, rather than discarding the HTTP response metadata via `score()`. LoRA and base-model commits must be declared separately, and the inference server must actually be launched with the pinned weights; the software cannot prove this from the HTTP model alias alone.
