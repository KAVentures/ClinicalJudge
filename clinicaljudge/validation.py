"""Auditable descriptive human-judge comparisons; never equates agreement with validity."""
from __future__ import annotations
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from .schema import load_jsonl


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    edge = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return [round(max(0, centre-edge), 4), round(min(1, centre+edge), 4)]


def bootstrap_case_ci(data: list[dict], score_fn, *, seed: int = 42, repeats: int = 400) -> list[float] | None:
    groups = defaultdict(list)
    for v in data:
        groups[v["case_id"]].append(v)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(repeats):
        selected = [groups[rng.choice(keys)] for _ in keys]
        flat = [x for subset in selected for x in subset]
        v = score_fn(flat)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025*(len(vals)-1))], 4), round(vals[int(0.975*(len(vals)-1))], 4)]


def load_physicians(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No physician ratings")
    seen = set()
    for x in rows:
        for col in ("response_id", "case_id", "task", "rater_id", "score"):
            if not x.get(col):
                raise ValueError(f"Missing {col} in physician ratings")
        key = (x["response_id"], x["rater_id"])
        if key in seen:
            raise ValueError(f"Duplicate physician assessment: {key}")
        seen.add(key)
        x["score"] = float(x["score"])
        if not math.isfinite(x["score"]) or not 0 <= x["score"] <= 1:
            raise ValueError("Human scores must be normalized to 0..1 using SAME anchors as judge")
        if x.get("critical_error", "") != "":
            val = x["critical_error"].lower()
            if val not in ("0", "1", "true", "false"):
                raise ValueError("critical_error must be 0/1 or empty")
            x["critical_error"] = val in ("1", "true")
        else:
            x["critical_error"] = None
    return rows


def _cohen_weighted_kappa(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    matrix = [[0.0] * (bins+1) for _ in range(bins+1)]
    for x, y in pairs:
        i = max(0, min(bins, int(round(x * bins))))
        j = max(0, min(bins, int(round(y * bins))))
        matrix[i][j] += 1
    n = len(pairs)
    row = [sum(x) for x in matrix]
    col = [sum(matrix[i][j] for i in range(bins+1)) for j in range(bins+1)]
    observed = sum(((i-j) / bins)**2 * matrix[i][j]/n for i in range(bins+1) for j in range(bins+1))
    expected = sum(((i-j) / bins)**2 * row[i]*col[j]/n**2 for i in range(bins+1) for j in range(bins+1))
    return None if expected == 0 else round(1-observed/expected, 4)


def validate(predictions: list[dict], physicians: list[dict], training_cases: set[str] | None = None,
             tolerance: float = 0.1) -> dict:
    if not 0 <= tolerance <= 1:
        raise ValueError("Tolerance must be 0 to 1")
    results = {}
    pred_lookup = {}
    for pred in predictions:
        key = pred.get("response_id")
        if not key or key in pred_lookup:
            raise ValueError("Predictions have missing/duplicate response IDs")
        if not isinstance(pred.get("score"), (int, float)) or not 0 <= pred["score"] <= 1:
            raise ValueError("Prediction score missing/out of range")
        pred_lookup[key] = pred
    by_response = defaultdict(list)
    for rate in physicians:
        p = pred_lookup.get(rate["response_id"])
        if p:
            if p["case_id"] != rate["case_id"] or p["task"] != rate["task"]:
                raise ValueError("Case/task mismatch between rating and prediction")
            if training_cases and p["case_id"] in training_cases:
                raise ValueError(f"TRAIN/TEST LEAKAGE: {p['case_id']} appears in calibration/training set")
            by_response[rate["response_id"]].append(rate)
    for task in sorted(set(p["task"] for p in predictions)):
        records, judge_pairs, physician_pairs, safety_pairs = [], [], [], []
        for response_id, ratings in by_response.items():
            prediction = pred_lookup[response_id]
            if prediction["task"] != task:
                continue
            values = [r["score"] for r in ratings]
            reference = median(values)
            records.append({"case_id": prediction["case_id"], "auto": prediction["score"],
                            "human": reference, "generator_family": prediction.get("generator_family", "unknown")})
            judge_pairs.append((prediction["score"], reference))
            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    physician_pairs.append((values[i], values[j]))
            ground_safety = [r["critical_error"] for r in ratings if r["critical_error"] is not None]
            if ground_safety and isinstance(prediction.get("critical_error_flag"), bool):
                # Majority tie is not silently classified; only unanimous or strict majority.
                if sum(ground_safety)*2 != len(ground_safety):
                    safety_pairs.append((prediction["critical_error_flag"],
                                         sum(ground_safety) > len(ground_safety)/2))
        if not records:
            continue
        inside = sum(abs(x["auto"]-x["human"]) <= tolerance for x in records)
        agreed = inside / len(records)
        mean_error = mean(abs(x["auto"]-x["human"]) for x in records)
        tp = sum(pred and actual for pred, actual in safety_pairs)
        fn = sum(not pred and actual for pred, actual in safety_pairs)
        tn = sum(not pred and not actual for pred, actual in safety_pairs)
        fp = sum(pred and not actual for pred, actual in safety_pairs)
        strata = {}
        for family in sorted({r["generator_family"] for r in records}):
            subset = [r for r in records if r["generator_family"] == family]
            strata[family] = {"responses": len(subset),
                              "distinct_cases": len({r["case_id"] for r in subset}),
                              "within_tolerance": round(mean(abs(r["auto"]-r["human"]) <= tolerance for r in subset), 4),
                              "mean_absolute_error": round(mean(abs(r["auto"]-r["human"]) for r in subset), 4),
                              "signed_error": round(mean(r["auto"]-r["human"] for r in subset), 4)}
        results[task] = {
            "generator_family_descriptives": strata,
            "matched_responses": len(records), "distinct_cases": len({x["case_id"] for x in records}),
            "within_tolerance": round(agreed, 4), "within_tolerance_wilson_95": wilson(inside, len(records)),
            "within_tolerance_cluster_bootstrap_95": bootstrap_case_ci(
                records, lambda rs: mean(abs(x["auto"]-x["human"]) <= tolerance for x in rs)),
            "mean_absolute_error": round(mean_error, 4),
            "weighted_kappa_vs_physician_median": _cohen_weighted_kappa(judge_pairs),
            "physician_physician_weighted_kappa": _cohen_weighted_kappa(physician_pairs),
            "physician_pairs": len(physician_pairs),
            "critical_error_detection": {
                "n_labeled": len(safety_pairs), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                "sensitivity": round(tp/(tp+fn), 4) if tp+fn else None,
                "sensitivity_wilson_95": wilson(tp, tp+fn),
                "specificity": round(tn/(tn+fp), 4) if tn+fp else None,
                "specificity_wilson_95": wilson(tn, tn+fp),
            },
            "interpretation": "DESCRIPTIVE AGREEMENT AUDIT ONLY. Not clinical equivalence or noninferiority."
        }
    if not results:
        raise ValueError("No matching scored responses and physician ratings")
    return {
        "validation_status": ("DEMO_NOT_VALIDATION" if any(p.get("judge", {}).get("backend") == "demo" for p in predictions)
                              else "HUMAN_COMPARISON_PERFORMED_NOT_EQUIVALENCE"),
        "notes": ["Do not treat multiple ratings of one clinical case as independent cases.",
                  "Hold out entire cases and all paraphrases/perturbations from calibration.",
                  "Physician median need not represent adjudicated clinical truth.",
                  "Only task-matched rating scales and anchors are comparable.",
                  "Formal noninferiority and external generalization require a preregistered design."],
        "tasks": results
    }


def load_training_case_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return {r["case_id"] for r in load_jsonl(p)}
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
