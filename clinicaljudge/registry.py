"""Version-linked registry of *submitted* external study reports. No auto-certification."""
import json
from pathlib import Path

REQUIRED = ("study_id", "task", "rubric_hash", "judge_model", "judge_revision", "distinct_cases",
            "physician_count", "protocol_reference", "report_reference", "population", "data_provenance")


def read_registry(directory: str | Path) -> dict:
    folder = Path(directory)
    if not folder.is_dir():
        raise ValueError(f"Validation submissions directory does not exist: {folder}")
    studies, seen = [], set()
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(f"{path.name}: missing registry keys: {missing}")
        if data["study_id"] in seen:
            raise ValueError("Duplicate study ID")
        seen.add(data["study_id"])
        if not isinstance(data["distinct_cases"], int) or data["distinct_cases"] <= 0:
            raise ValueError("distinct_cases must be a positive integer")
        if not isinstance(data["physician_count"], int) or data["physician_count"] < 1:
            raise ValueError("physician_count must be a positive integer")
        if data.get("status") != "SUBMITTED_UNREVIEWED":
            raise ValueError("Registry never automatically certifies clinical validity; status must be SUBMITTED_UNREVIEWED")
        studies.append(data)
    return {"status": "EVIDENCE_INDEX_NOT_CLINICAL_CERTIFICATION", "number_of_submissions": len(studies),
            "tasks_with_submissions": sorted({s["task"] for s in studies}), "studies": studies}
