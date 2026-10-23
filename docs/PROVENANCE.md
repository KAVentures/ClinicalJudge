# Research model provenance (v0.2.0)

**Identity and execution provenance are mandatory research metadata, not equivalent to physician validation.**

## What ClinicalJudge automatically records

Each graded row has a `judge` object with requested model ID, declared family,
framework version and source-code SHA-256, method, temperature, rubric SHA-256,
input SHA-256, and `transport_provenance` containing the exact API-reported model
ID (if returned), call/response UTC timestamps, API-created time (if returned),
completion ID, HTTP request ID (if returned), system fingerprint (if returned),
finish reason, token usage, service tier, system prompt SHA-256, full messages SHA-256,
and inference-request SHA-256. Missing API fields are saved as `null`, not invented.

`messages_sha256` hashes the complete input **including case and response**. The
`system_prompt_sha256` hashes the stable system instruction separately. Neither
field discloses clinical case text. For public data and publication, ensure source
cases and derived content have appropriate redistribution rights.

## Open-weight models: separately pin the base AND LoRA adapter

Create a model manifest (this is a TEMPLATE, not a verified real checkpoint):

```json
{
  "schema_version": 1,
  "served_model": "PrecepTron-32B-CPC",
  "backbone_model": "Qwen/Qwen3-32B",
  "backbone_commit": "REPLACE_WITH_ACTUAL_40_CHARACTER_GIT_COMMIT_SHA",
  "adapter_model": "tbuckley/PrecepTron-32B-CPC",
  "adapter_commit": "REPLACE_WITH_ACTUAL_40_CHARACTER_GIT_COMMIT_SHA",
  "tokenizer_model": "Qwen/Qwen3-32B",
  "tokenizer_commit": "REPLACE_WITH_ACTUAL_40_CHARACTER_GIT_COMMIT_SHA",
  "inference_runtime": "vllm",
  "runtime_version": "REPLACE_WITH_REAL_VERSION",
  "inference_dtype": "bfloat16",
  "quantization": null,
  "release_date": null
}
```

Replace placeholders with real, independently obtained Git commit IDs and software
versions; replace `null` with a vendor/model-card release date **and** source URL
only if verified. Store it in `runs/model_manifest.json`. Pin the actual serving
runtime, too; for vLLM `--revision <base commit>` and
`--tokenizer-revision <tokenizer commit>` control the base model/tokenizer, while
LoRA adapters must be downloaded from their specific committed revision and
served from that checkout. Merely writing a commit in a JSON file does NOT
make the server use it. The metadata is marked `COMMIT_DECLARED_NOT_SERVER_ATTESTED`.

```bash
clinicaljudge run \
  --input cases.jsonl --output results.jsonl \
  --backend preceptron --judge-family qwen \
  --base-url http://127.0.0.1:8000/v1 \
  --model PrecepTron-32B-CPC \
  --model-manifest runs/model_manifest.json --require-pinned
```

In a study methods section, publish base model repo + commit, LoRA repo + commit,
loaded inference server/flags and version, tokenizer, decoding configuration, code
release, rubrics, clinical task, evaluation UTC range, and validation details.

**Limit:** The manifest is a researcher declaration, checked for shape and SHA
syntax but **not a cryptographic attestation of the weights loaded in the
inference server**. For stronger verification, record serving command, resolved
Hub snapshots, artifact hashes and a reproducible container/runtime manifest.

## API model stamps vs open weights

The API's returned `model` field is captured separately from the requested
model. A server may return a friendly served alias, not the underlying base
checkpoint. Some APIs return a backend fingerprint or request ID; others do not.
Do not infer weights, a release date, or immutability from a timestamp or alias.

Commercial models might have provider-pinned snapshots; document the provider's
snapshot semantics. Hosting a Qwen or PrecepTron model behind an OpenAI-compatible
API does **not** mean the OpenAI model is judging: it is merely a compatible HTTP
interface. The local weights and exact adapter matter.

## Export a methods-ready machine-readable provenance audit

```bash
clinicaljudge provenance --predictions results.jsonl \
  --output results_provenance.json
```

The audit summarizes reported and requested model IDs, execution period, prompt
and rubric hashes, pin declaration status, and counts of missing API metadata.
If the judge configuration varies, runs are separated into groups rather than
misrepresented as one fixed model. Review `limitations` before copying into a
manuscript. The audit is **not** itself proof of identical model weights or
bitwise determinism (including at `temperature: 0`).
