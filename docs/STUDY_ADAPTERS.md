# Planned adapters for source research repositories

`ClinicalJudge` already supports the *evaluation constructs*, not direct parsing of every project's raw dump. Author separate adapters after confirming raw export schemas and data rights. The input JSONL schema is the stable interchange layer.

| Source | Intended evaluation tasks | Key leakage and validity checks |
|---|---|---|
| `KAVentures/clinical-evidence-sufficiency-llm` | `evidence_sufficiency`, `safety`, `missing_information`, `helpfulness` | Keep prompt condition blinded, preserve case IDs and paired baseline/wrapper outputs, do not use judge-derived labels as clinical ground truth |
| `KAVentures/health-ai-readiness-robustness` | `missing_information`, `safety`, `clinical_reasoning` | Perturbation validity: removing text does not necessarily eliminate answerability; split case families together |
| `KAVentures/clinical-ai-reconciliation` | `evidence_quality`, `helpfulness`, `diagnosis`; future pairwise | Do not conflate pairwise preference with absolute rubrics; identical answer rendering and controlled answer order |
| `2v/PrecepTron` | Native `cpc_bond`, `diagnostic_reasoning`, `management_reasoning`, `r_idea` | Correct task-specific LoRA adapter and licensed source case content; do not present generic rubric scores as PrecepTron validation |

Each adapter should provide: `source_hash`, `case_id`, `response_id`, `task`, `case`, `response`, `generator_family`, task-specific reference, licensing tags, original study pointer. Exact model identity is excluded from the judge prompts but retained in internal provenance.

A robust adapter must run on fixture files offline; skip underlying datasets that cannot legally be redistributed or fetched. No MIMIC-derived cases should be sent to commercial APIs where prohibited.
