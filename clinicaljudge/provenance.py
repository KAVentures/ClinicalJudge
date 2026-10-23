"""Transparent model identity declarations and per-call reproducibility audits.

A commit hash in a user-created manifest is *not* proof that an inference server
really loaded those weights. Only a measured or attested server can establish that.
"""
from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from .schema import digest

SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")


def read_manifest(path: str | None, *, served_model: str, backend: str,
                  require_pinned: bool = False) -> dict | None:
    if not path:
        if require_pinned:
            raise ValueError("--require-pinned requires --model-manifest with checkpoint commit IDs")
        return None
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Model manifest must be a JSON object with schema_version: 1")
    allowed = {"schema_version", "served_model", "backbone_model", "backbone_commit",
               "adapter_model", "adapter_commit", "tokenizer_model", "tokenizer_commit",
               "inference_runtime", "runtime_version", "release_date", "release_date_source",
               "inference_dtype", "quantization"}
    if set(manifest) - allowed:
        raise ValueError("Unexpected model manifest fields; only research identity metadata is allowed")
    if manifest.get("served_model") != served_model:
        raise ValueError("Model manifest served_model does not match requested --model")
    required_str = ("backbone_model", "backbone_commit", "inference_runtime", "runtime_version")
    for field in required_str:
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Model manifest requires {field}")
    for field in ("backbone_commit", "adapter_commit", "tokenizer_commit"):
        if manifest.get(field) is not None and not SHA40.fullmatch(str(manifest[field])):
            raise ValueError(f"Manifest {field} must be a 40-character git commit SHA")
    if (manifest.get("adapter_model") is None) != (manifest.get("adapter_commit") is None):
        raise ValueError("Model manifest adapter_model and adapter_commit must both be present or absent")
    if backend == "preceptron" and not manifest.get("adapter_commit"):
        raise ValueError("PrecepTron requires both backbone and adapter pins in its manifest")
    if require_pinned and not SHA40.fullmatch(manifest["backbone_commit"]):
        raise ValueError("Model backbone is not commit-pinned")
    if manifest.get("release_date") and not manifest.get("release_date_source"):
        raise ValueError("release_date needs a public release_date_source URL or citation")
    if "release_date" in manifest:
        release_date = manifest["release_date"]
        if release_date is not None and (not isinstance(release_date, str) or
                                          not re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date)):
            raise ValueError("release_date must be an ISO YYYY-MM-DD string or null")
    return manifest


def framework_code_sha256() -> str:
    """Hash the executable framework code, independent of VCS availability."""
    root = Path(__file__).resolve().parent
    filenames = ("cli.py", "engine.py", "schema.py", "rubrics.py", "registry.py",
                 "provenance.py", "validation.py")
    h = hashlib.sha256()
    for name in filenames:
        h.update(name.encode("utf-8"))
        h.update((root / name).read_bytes())
    return h.hexdigest()


def provenance_report(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No prediction rows supplied")
    runs = {}
    for row in rows:
        judge = row.get("judge", {})
        if not isinstance(judge, dict):
            raise ValueError("Missing judge metadata")
        key = (judge.get("model"), judge.get("backend"), judge.get("model_revision"),
               judge.get("model_manifest_sha256"))
        label = json.dumps(key)
        group = runs.setdefault(label, {"judge_model_requested": key[0], "backend": key[1],
                                        "declared_revision": key[2], "manifest_sha256": key[3],
                                        "n_responses": 0, "api_response_models": set(),
                                        "timestamps_utc": [], "missing_api_model": 0,
                                        "missing_api_created_time": 0, "missing_request_id": 0,
                                        "prompt_hashes": set(), "rubric_hashes": set(),
                                        "model_pinning_statuses": set(),
                                        "inference_runtime": None, "framework_versions": set(),
                                        "code_hashes": set(), "system_prompt_hashes": set()})
        group["n_responses"] += 1
        transport = judge.get("transport_provenance") or {}
        if transport.get("response_model"):
            group["api_response_models"].add(transport["response_model"])
        else:
            group["missing_api_model"] += 1
        if not transport.get("api_created_at_utc"):
            group["missing_api_created_time"] += 1
        if not transport.get("api_request_id"):
            group["missing_request_id"] += 1
        if transport.get("request_sent_at_utc"):
            group["timestamps_utc"].append(transport["request_sent_at_utc"])
        if transport.get("system_prompt_sha256"):
            group["system_prompt_hashes"].add(transport["system_prompt_sha256"])
        if transport.get("messages_sha256"):
            group["prompt_hashes"].add(transport["messages_sha256"])
        if judge.get("rubric_sha256"):
            group["rubric_hashes"].add(judge["rubric_sha256"])
        group["model_pinning_statuses"].add(judge.get("model_pin_status", "UNKNOWN"))
        if judge.get("framework_code_sha256"):
            group["code_hashes"].add(judge["framework_code_sha256"])
        if judge.get("framework_version"):
            group["framework_versions"].add(judge["framework_version"])
        if judge.get("model_manifest"):
            manifest = judge["model_manifest"]
            group["inference_runtime"] = {"name": manifest.get("inference_runtime"),
                                          "version": manifest.get("runtime_version")}
    result = []
    for group in runs.values():
        result.append({
            **{k: v for k, v in group.items() if not isinstance(v, set) and k != "timestamps_utc"},
            "api_response_models": sorted(group["api_response_models"]),
            "prompt_hashes": sorted(group["prompt_hashes"]),
            "system_prompt_hashes": sorted(group["system_prompt_hashes"]),
            "code_hashes": sorted(group["code_hashes"]),
            "rubric_hashes": sorted(group["rubric_hashes"]),
            "model_pinning_statuses": sorted(group["model_pinning_statuses"]),
            "framework_versions": sorted(group["framework_versions"]),
            "first_request_utc": min(group["timestamps_utc"]) if group["timestamps_utc"] else None,
            "last_request_utc": max(group["timestamps_utc"]) if group["timestamps_utc"] else None,
        })
    return {"schema": "clinicaljudge.provenance.v1", "n_groups": len(result), "groups": result,
            "limitations": ["Model declarations are not evidence of server-loaded weights.",
                            "API-reported IDs may be aliases, not immutable snapshots.",
                            "Model release dates must come from vendor/model-card source, not request timestamps.",
                            "Temperature zero does not ensure bitwise deterministic inference."]}
