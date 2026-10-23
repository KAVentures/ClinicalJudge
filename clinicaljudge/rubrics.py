"""Versioned, score-anchored rubric registry. Custom rubrics are research instruments."""
import json
from pathlib import Path
from .schema import digest

RUBRICS = Path(__file__).parent / "rubrics"


def get_rubric(task: str, directory: str | None = None) -> dict:
    if not task.replace("_", "").isalnum():
        raise ValueError("Invalid task name")
    source = Path(directory) if directory else RUBRICS
    path = source / (task + ".json")
    if not path.is_file():
        raise ValueError(f"Rubric not found: {task}. Define and validate a rubric before scoring.")
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if rubric.get("task") != task or not rubric.get("version"):
        raise ValueError("Rubric task/version mismatch")
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Rubric must have criteria")
    ids = [c.get("id") for c in criteria]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Rubric criteria IDs must be distinct non-empty strings")
    if any(not isinstance(c.get("description"), str)
           or not isinstance(c.get("weight"), (int, float))
           or c.get("weight", 0) <= 0
           or c.get("severity") not in ("ordinary", "critical") for c in criteria):
        raise ValueError("Invalid rubric criterion")
    return rubric


def rubric_hash(rubric: dict) -> str:
    return digest(rubric)
