# ClinicalJudge validation program

**A working tool is not a validated clinical rater.** Physician equivalence is task-, population-, rubric-, and judge-version-specific.

## Required validation questions

1. Agreement on identical held-out clinical cases under exactly the same rubric for AI and physicians (both percentage within tolerance and ordinal weighted kappa, with uncertainty).
2. Inter-physician agreement on the **same** cases, to contextualize judge-vs-physician scores.
3. Whether judge misses clinically consequential errors; report sensitivity, specificity, and false-reassurance examples, with confidence intervals. Enrich critical cases for sensitivity analysis but do not interpret positive predictive value as population prevalence.
4. Whether clinically similar answers from different generator families get systematically different ratings at equal human-judged quality.
5. Robustness to answer length, confident framing, syntactic transformations, citations, languages, and missing/context-contradictory case versions.
6. Whether performance generalizes across new clinical specialties, institutions, models and years. No claim of universal equivalence.

## Label collection design

- Register a written protocol **before** any held-out scoring, with case eligibility, clinical reference standard, sampling scheme, primary metric, clinical equivalence/noninferiority margin, exclusion rules, and analysis plan.
- At least two independent, blinded physicians are preferable for an inter-physician comparison. Resolve material disagreements using a third adjudicator where feasible, and retain individual scores separately.
- Clinicians must use the **exact same rubric and presentation** as the model; response model identity and intervention arm must be scrubbed.
- Train, calibration, development and final validation must be partitioned by clinical **case**, with all candidate model answers, paraphrases, and perturbations grouped together. Make split manifests auditable and freeze the final held-out set.
- Never put personally identifiable clinical data, unlicensed vignettes, or proprietary physician annotations in public pull requests.
- Include cases with dangerous errors and over-abstention, rather than only easy diagnosis examples.
- Establish specialties and country-specific clinical guidelines relevant to the scoring protocol.

## Reporting

For each validation, submit a machine-readable task/version manifest, protocol, scoring and physician guidelines, case count (distinct cases and responses), confidence intervals, failure taxonomy, generator-family breakdown, and reproducibility details. The CLI returns **descriptive agreement** and can prevent direct overlap with a supplied calibration manifest. It does NOT calculate a preregistered noninferiority analysis and cannot authenticate human rater identities.

## Proposed evidence grades (future feature)

- **Experimental:** unvalidated scoring output.
- **Physician compared:** descriptive external human comparison; not proof of equivalence.
- **Validated for scope X:** only after a preregistered, appropriately powered external validation with prespecified margins and acceptance criteria; requires independent review.

Version v0.1.0 intentionally publishes no 'validated for scope X' certificates.

## Data governance

Primary code is MIT, but independently acquired physician scoring data and clinical vignettes need their own provenance, licensing and ethics review. Public release of model weights does not imply redistribution rights to clinical cases. Separately identify synthetic data.
