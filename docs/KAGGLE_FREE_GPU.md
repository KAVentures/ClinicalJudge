# PrecepTron on Kaggle Free GPU: 50 physician-labeled responses

**Status:** Infrastructure prepared, not executed against the actual gated model. There are no new PrecepTron physician-agreement results until a completed GPU run is saved.

This workflow runs the **real `tbuckley/PrecepTron-32B-Management` LoRA** on an independently quantized 4-bit **`unsloth/Qwen3-32B-bnb-4bit`** backbone. It is a *transfer* experiment on the binary missing-information rubric, **not** a reproduction of PrecepTron's native management validation and not full-precision clinical equivalence.

## How to run (minimal manual setup)

1. At [the PrecepTron Management adapter](https://huggingface.co/tbuckley/PrecepTron-32B-Management), sign in and request/accept the gated-access conditions. You must be authorized to download the adapter.
2. In Hugging Face account settings, create a **read-only** access token with permission to download the gated model. Do not share or commit it.
3. Open [Kaggle Notebooks](https://www.kaggle.com/code), import `notebooks/PrecepTron_Kaggle_Physician50.ipynb` from this repository, and enable **Settings > Accelerator > GPU T4 x2** and **Internet = On**. The GPU quota is subject to availability. Kaggle accounts may require verification before GPUs are enabled.
4. In Kaggle **Add-ons > Secrets**, add the token as `HF_TOKEN` and enable the notebook's access to this secret.
5. Click **Run All**. The notebook preflights the model gate, verifies 2 GPUs and disk space, downloads model files, runs five pilot cases, then runs all 50 only if the pilot produced valid scores. The end of the notebook creates a zipped report under `/kaggle/working/clinicaljudge-results/`.

**No credit card or Hugging Face Jobs billing is used by the notebook.** It consumes Kaggle GPU quota and Hugging Face model bandwidth. If the model cannot fit due to disk/memory, the notebook should fail rather than quietly substituting a different model. The quantized weights are ~19.2 GB download plus adapter and cache overhead; free storage/GPUs may be insufficient.

## What it scores

Cases come from [KAVentures/health-ai-readiness-robustness](https://github.com/KAVentures/health-ai-readiness-robustness) at frozen commit `48c6b67102e4494bf224f63fd9a61816aa9da920`: **50 blinded responses, 33 independent clinical question clusters, 150 original ratings from three physicians**.

The native PrecepTron Management adapter accepts *case-specific rubrics*. We pass the original binary evaluation criterion, with 0 = inappropriate certainty and 1 = appropriate uncertainty. This is a **cross-task transfer test**: the Management adapter was not physician-trained on this construct. Scores must not be claimed to have native PrecepTron validation.

The workflow does **not** transmit physician labels into model prompts or use them during scoring. It opens the held-out physician rating CSV for comparisons **only after** all 50 valid predictions are frozen. Scores of 0 are never used as a fallback for timeouts or parsing errors.

## Results / reproducibility

Results are saved under `/kaggle/working/clinicaljudge-results/`:

- `preceptron_management_quant4_predictions.jsonl`: original judge text, binary scores, timestamps, model versions, prompt hashes and errors.
- `model_manifest.json`: resolved immutable base/adapter model commit hashes, quantization, software versions and device map.
- `run_status.json`: scored/invalid/remaining counts.
- `physician_comparison.json`: validation by the existing ClinicalJudge framework, when 50/50 valid.
- `preceptron_vs_physicians_detailed.json` and `preceptron_vs_physicians_summary.md`: agreement with human majority and each physician, false reassurance, case-cluster bootstrap, baseline comparisons.
- `provenance.json`: additional ClinicalJudge provenance audit.

Partial runs persist after every case. Rerun with identical checkpoints/settings to resume; mismatched model revisions or rubric hashes are rejected.

Published model comparison must report both the base checkpoint and the adapter checkpoint. **The PrecepTron team used Qwen3-32B unquantized on an 80GB GPU, while this exploratory Kaggle flow uses a community 4-bit quantized backbone.** The change may alter scoring. A full-precision replication requires suitable resources.

## Why not free Hugging Face hosted inference?

On October 23 an attempted Hugging Face Jobs run on the connected user account returned `402 Payment Required`, and standard Hugging Face Inference Providers do not serve this user-selected gated LoRA adapter automatically. Merely running hosted Qwen3-32B is not testing PrecepTron. Kaggle's free T4 x2 is a best-effort option, not a guaranteed provider of sufficient capacity.

## Limitations

This 50-response sample was previously used for an existing paper; it is a retrospective benchmark, not a new external clinical validation. The original physicians scored **appropriate uncertainty behavior**, not clinical harm. One physician was the study author, so also report the results against both independent physicians separately. Never make a safety-equivalence or noninferiority claim from this pilot.
