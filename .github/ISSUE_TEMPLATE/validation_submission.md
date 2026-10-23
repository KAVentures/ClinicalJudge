---
name: Validation study submission
about: Submit independent physician validation evidence for a rubric/model version
labels: validation
---

### Validation scope
- Clinical task, rubric version and rubric SHA256:
- Judge model checkpoint, LoRA revision and runtime:
- Dataset source/licensing, specialty, country, date and inclusion rules:
- Distinct cases / response items / physician ratings:
- Rater clinical expertise, conflicts, and blinding procedure:
- Case-level train/calibration/validation partitioning and leakage check:
- Prespecified endpoints and margins (if making equivalence claims):

### Results
- Judge vs clinician within tolerance and weighted kappa + CIs:
- Clinician vs clinician comparisons on the same cases:
- Critical-error sensitivity/specificity and denominators:
- Generator-family and formatting-bias checks:
- Protocol deviations / material failures:

### Reproducibility
- Code/adapter commit(s):
- Anonymized rating/analysis artifacts or lawful access instructions:
- Ethics/permissions determination:

**Do not include PHI, unreleased patient cases or raw identifiable clinician data.**
