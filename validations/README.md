# External validation evidence registry

A clinical judge's performance is **not** established by its installation, model weights or software test suite. Validation research groups may submit version-linked external physician-comparison studies via PRs. Each PR includes a JSON manifest in `validations/submissions/<study-id>.json`, a publicly accessible (lawfully shared) protocol, methods, analysis and immutable model/rubric revisions. Protected cases or physician identities do not belong in the public repository.

Run `python -m clinicaljudge registry` to inspect submitted studies. New manifests are marked **SUBMITTED_UNREVIEWED**. This registry is an evidence index, not an automatic certification authority. Independent review, scope matching and formal preregistered equivalence testing must precede a substantive physician-equivalence claim.

Example metadata *template* (do not submit placeholders):

```json
{
  "study_id": "org-2027-safety-001",
  "status": "SUBMITTED_UNREVIEWED",
  "task": "safety",
  "rubric_hash": "sha256-from-scoring-run",
  "judge_model": "exact-model-name",
  "judge_revision": "exact-model-or-adapter-sha",
  "distinct_cases": 300,
  "physician_count": 3,
  "protocol_reference": "persistent-protocol-URL",
  "report_reference": "persistent-report-URL",
  "population": "specified-specialty-language-clinical-domain",
  "data_provenance": "license-and-source-information",
  "preregistered_equivalence": false
}
```

Future releases should add review sign-off and evidence grading only after an explicit governance process is established.
