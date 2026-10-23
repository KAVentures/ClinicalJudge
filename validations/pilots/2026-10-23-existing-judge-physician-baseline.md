# ClinicalJudge validation-framework pilot: existing physician labels (2026-10-23)

**Status: completed retrospective reanalysis of legacy LLM-judge votes; NOT a new ClinicalJudge/PrecepTron independent-model inference run.**

## Primary dataset

Repository: [KAVentures/health-ai-readiness-robustness](https://github.com/KAVentures/health-ai-readiness-robustness), frozen at commit `48c6b67102e4494bf224f63fd9a61816aa9da920`. The `runs/human_eval/key.json` index and `runs/judge_panel/votes.jsonl` were linked by `(subject_model, item, prompt_id)`. Physician ratings came from `runs/human_eval/human_labels.csv`, `labels_O.csv`, and `labels_G.csv`. All 50 matched. The cases represent 33 distinct `prompt_id` values; four different generator families are represented. Responses, not underlying independent questions, are the unit of the crude agreement percentages. Cluster bootstrap uses `prompt_id` groups.

**Construct:** whether a model's response shows appropriate uncertainty when context is missing. A score of 1 means *appropriate*; 0 means *not appropriate*. No adjudicated critical-harm label exists in this 50-item sample. The `critical_error_detection` fields in the tool report are intentionally null, rather than treating a missing-information error as a physician-adjudicated dangerous medical error.

| Existing judge | Agreement with physician majority | 95% case-cluster bootstrap CI | Cohen's κ vs majority | Missed inappropriate-uncertainty labels (false appropriate) |
|---|---:|---|---:|---:|
| OpenAI | 33/50 (66%) | 52.1–77.8% | 0.281 | 16/23 |
| Anthropic | 31/50 (62%) | 47.8–76.0% | 0.202 | 16/23 |
| xAI | 36/50 (72%) | 58.3–84.8% | 0.425 | 10/23 |
| Google | 35/50 (70%) | 56.3–83.6% | 0.378 | 12/23 |

Two independent clinicians O and G agreed on **37/50 = 74%**, Cohen's κ = **0.472**. Across all three physician raters, raw Fleiss' κ = **0.643** as reported by the original study. The majority includes the study author; this is disclosed. In the independent-only reference O, xAI matched **39/50 = 78% (κ=0.554)**; in the independent-only reference G, Anthropic and OpenAI each matched **39/50 = 78%**. These references differ and do not support a single decisive rank ordering. Human ratings and legacy judge votes were not collected as a new blinded comparative experiment for ClinicalJudge.

ClinicalJudge's existing `validate` function was executed with **exact binary-match tolerance 0** against the three rater scores and legacy predicted scores, and `python -m clinicaljudge validate` was separately smoke-tested on the xAI import. The framework's 18 software unit tests passed with the repository as Python working directory.

## Secondary dataset: evidence sufficiency

Repository: [KAVentures/clinical-evidence-sufficiency-llm](https://github.com/KAVentures/clinical-evidence-sufficiency-llm), frozen at commit `b5e5818aa704200f6491779157bde4f1eba2990e`.

The public `outputs/doctor_review/{doctor_review,adjudication,sensitivity}_{A,B,C}.csv` files contain **blank rating columns** (templates); they are not the returned physician responses. Consequently, this dataset cannot be recomputed as a new item-level human comparison solely from the publicly accessible files. Precomputed source-repository aggregate reports exist at `outputs/tables/adjudication_report_final.json` and `sensitivity_report_excludeC.json`, but are **not newly generated ClinicalJudge scores**.

From the source's precomputed 66 judgeable cases in the positive-enriched sensitivity set (raters A/B), the primary GPT-5.4-nano judge had 7 true positives, 0 false negatives, 26 false positives and 33 true negatives for the source's unsafe-overconfidence construct: **sensitivity 7/7 (100%, highly uncertain), specificity 33/59 (55.9%), raw agreement 60.6%**. This is an existing published result, not a fresh independent replication. The adjudication set has only two clinician-unsafe cases among 69 judgeable and is unsuitable for reliable sensitivity estimation.

## Why the new judge was not run

The repository's `demo` backend produces deliberate fake scores and must not be used for clinical validation. The actual `openai-compatible` and `preceptron` backends need an independently running model endpoint. No Qwen/PrecepTron inference endpoint or model weights were available in this execution environment. The PrecepTron 32B README documents an approximately 80-GB GPU serving requirement; launching paid GPU compute without an explicit budget is inappropriate. Therefore **no new independent physician-calibrated judge scores have been produced**.

## Next valid experiment

1. Stand up pinned Qwen3-32B + PrecepTron models on suitable GPU infrastructure, or supply a trusted pinned endpoint; record base-model and LoRA commits.
2. Route matching stored response text and case-reference fields (not just historical score votes) to the judge in **blind mode**. PrecepTron diagnostic/reasoning adapters are not validated for this exact binary missing-information rubric; use an experimental custom module, not a claim of PrecepTron-equivalence.
3. Freeze the rubric, prohibit any overlap between judge calibration cases and the held-out physician sample, and test all 50 cases with identical inputs.
4. Compare to independent clinicians O/G and the three-rater majority separately, report within-case bootstrap intervals, missed inappropriate responses, calibration, and generator-family bias.
5. For evidence sufficiency, recover *actual returned* deidentified physician rating rows before computing new item-level comparison.

**No physician-equivalence, non-inferiority, clinical safety, or independent-judge performance claim is supported by this pilot.**