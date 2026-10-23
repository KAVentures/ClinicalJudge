#!/usr/bin/env python3
"""Reanalyze historical judge votes against published physician labels.

This is a retrospective baseline, NOT a new ClinicalJudge/PrecepTron inference
or a claim of clinical equivalence. Fetches pinned public source artifacts.
"""
import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinicaljudge.validation import validate

REPO = "KAVentures/health-ai-readiness-robustness"
SHA = "48c6b67102e4494bf224f63fd9a61816aa9da920"
PROVIDERS = ("openai", "anthropic", "xai", "google")
RATERS = ("R1", "O", "G")


def fetch(path):
    url = f"https://raw.githubusercontent.com/{REPO}/{SHA}/{path}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def read_labels(path):
    labels = {}
    for row in csv.DictReader(io.StringIO(fetch(path))):
        annotation_id = row["annotation_id"]
        label = int(row["human_label"])
        if annotation_id in labels or label not in (0, 1):
            raise ValueError("Invalid or duplicate physician label")
        labels[annotation_id] = label
    return labels


def load_source():
    keys = json.loads(fetch("runs/human_eval/key.json"))
    doctors = {
        "R1": read_labels("runs/human_eval/human_labels.csv"),
        "O": read_labels("runs/human_eval/labels_O.csv"),
        "G": read_labels("runs/human_eval/labels_G.csv"),
    }
    votes = {}
    for line in fetch("runs/judge_panel/votes.jsonl").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["subject_model"], str(record["item"]), record["prompt_id"])
        if key in votes:
            raise ValueError("Duplicate judge vote")
        votes[key] = record["votes"]
    rows = []
    for response_id, item in sorted(keys.items()):
        vote_key = (item["subject_model"], str(item["item"]), item["prompt_id"])
        if vote_key not in votes:
            raise ValueError(f"Unmatched response: {response_id}")
        rows.append({
            "response_id": response_id,
            "case_id": item["prompt_id"],
            "generator_family": item["subject_model"],
            "labels": {r: doctors[r][response_id] for r in RATERS},
            "votes": {p: int(bool(votes[vote_key][p])) for p in PROVIDERS},
        })
    return rows


def analyze(rows):
    if not rows or len({r["response_id"] for r in rows}) != len(rows):
        raise ValueError("No rows or duplicate responses")
    physicians = [
        {"response_id": x["response_id"], "case_id": x["case_id"],
         "task": "missing_information", "rater_id": r,
         "score": float(x["labels"][r]), "critical_error": None}
        for x in rows for r in RATERS
    ]
    scores = {}
    for provider in PROVIDERS:
        predictions = [
            {"response_id": x["response_id"], "case_id": x["case_id"],
             "task": "missing_information", "score": float(x["votes"][provider]),
             "generator_family": x["generator_family"],
             "judge": {"backend": "legacy_vote_import", "model": provider}}
            for x in rows
        ]
        scores[provider] = validate(predictions, physicians, tolerance=0)["tasks"]["missing_information"]
    return {
        "provenance": {"repository": REPO, "commit": SHA,
                       "type": "LEGACY_LLM_VOTES_NOT_CLINICALJUDGE_INFERENCE"},
        "responses": len(rows),
        "distinct_cases": len({r["case_id"] for r in rows}),
        "scores": scores,
        "interpretation": "Descriptive retrospective comparison; NOT physician equivalence."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/legacy_physician_comparison.json"))
    args = parser.parse_args()
    report = analyze(load_source())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, stats in report["scores"].items():
        print(f"{name}: agreement={stats['within_tolerance']}, "
              f"kappa={stats['weighted_kappa_vs_physician_median']}")
    print(f"Report: {args.output} (existing judge votes only)")
