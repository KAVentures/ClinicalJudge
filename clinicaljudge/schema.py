"""Input validation and protection against accidental identity leakage."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PRECEPTRON_TASKS = {"cpc_bond", "diagnostic_reasoning", "management_reasoning", "r_idea"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{n}: record must be an object")
            results.append(entry)
    return results


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_case(item: dict) -> None:
    for key in ("case_id", "response_id", "task", "case", "response", "generator_family"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"{item.get('response_id', '?')}: missing non-empty string '{key}'")
    if "reference" in item and not isinstance(item["reference"], dict):
        raise ValueError("reference must be a dict")
    if item["task"] in PRECEPTRON_TASKS and not isinstance(item.get("reference"), dict):
        raise ValueError("PrecepTron task requires reference inputs")
    must_have = {
        "missing_information": ("answerable", "critical_missing_facts"),
        "contradiction": ("contradictions",),
        "evidence_sufficiency": ("answerable",),
        "diagnosis": ("final_diagnosis",),
    }
    for field in must_have.get(item["task"], ()):
        if field not in item.get("reference", {}):
            raise ValueError(f"{item['response_id']}: reference.{field} is required for {item['task']}")


def validate_batch(items: list[dict]) -> None:
    if not items:
        raise ValueError("Input must contain at least one case")
    seen = set()
    for item in items:
        validate_case(item)
        if item["response_id"] in seen:
            raise ValueError(f"Duplicate response_id: {item['response_id']}")
        seen.add(item["response_id"])


def blinded_case(item: dict) -> dict:
    """Only send content necessary for grading. No generator or response id metadata."""
    return {"case": item["case"], "response": item["response"], "reference": item.get("reference", {})}
