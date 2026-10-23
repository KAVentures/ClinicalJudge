# Contributing

Contributions from physicians, AI evaluation researchers, statisticians, and engineers are welcome.

## New clinical task

1. Create `clinicaljudge/rubrics/<task_name>.json` with versioned fixed criteria, anchors and critical/error definitions.
2. Provide a rationale for construct validity. For subjective judgments, explain what is *not* directly observable.
3. Add synthetic fixtures, automated schema tests, and a reference policy. Avoid contamination from existing test responses.
4. Mark it experimental until independent validation is submitted and reviewed.
5. Preserve existing versioned evaluations. Do not silently alter prior rubrics under the same version.

## New physician validation

1. Fork the repo and submit a PR with a protocol filled from `docs/RESEARCH_PROTOCOL.md` **and** a non-sensitive validation report.
2. Include distinct case count, total response count, physician-panel method, blinded scoring instructions, judge revision/hash, dataset provenance, and model-family stratification.
3. State whether physician ratings were used for training, prompt selection, threshold fitting or any previous validation.
4. Report errors/misses and confidence intervals, not only one agreement percentage.
5. Provide rerunnable artifacts where legally possible; never upload PHI, copyrighted cases without rights or raw clinician PII.
6. Do not present output as 'physician-equivalent' without a preregistered, independently reviewed equivalence protocol.

## Software PRs

- Add tests for code changes and run `python -m unittest discover -s tests -v`.
- Keep agent/model identity blinded from grading payloads.
- Do not add automatic fallback from independent backend to Claude/Codex or an unvalidated LLM.
- Do not silently permit remote upload or external API charges.
- Keep all clinical software uses marked as research-only until separately validated/approved.
