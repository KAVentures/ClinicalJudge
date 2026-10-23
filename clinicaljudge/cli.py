"""Research-grade CLI. No provider accounts, judge models or real clinical validation bundled."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from . import __version__
from .engine import ensure_independent, family_name, grade
from .rubrics import RUBRICS, get_rubric, rubric_hash
from .schema import load_jsonl, validate_batch, write_jsonl
from .validation import load_physicians, load_training_case_ids, validate
from .registry import read_registry
from .provenance import read_manifest, provenance_report, framework_code_sha256
from .schema import digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinicaljudge", description="Independent clinical AI research evaluation framework")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rubrics", help="List experimental rubric IDs")
    registry = sub.add_parser("registry", help="Inspect external validation study submissions")
    registry.add_argument("--submissions", default="validations/submissions")
    lint = sub.add_parser("lint", help="Check cases and rubric coverage without inference")
    lint.add_argument("--input", required=True)
    lint.add_argument("--rubric-dir")
    run = sub.add_parser("run", help="Blind, structured rubric scoring")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=("openai-compatible", "preceptron", "demo"), required=True)
    run.add_argument("--base-url", default="")
    run.add_argument("--model", default="UNSPECIFIED_DEMO")
    run.add_argument("--model-revision", default="UNPINNED", help="Legacy user-declared snapshot/commit; not verified")
    run.add_argument("--model-manifest", help="JSON declaring base-model and adapter git SHAs and runtime version")
    run.add_argument("--require-pinned", action="store_true", help="Fail if the model manifest lacks pinned checkpoint commits")
    run.add_argument("--judge-family", default="qwen", help="Explicit base-model family; preceptron requires qwen")
    run.add_argument("--rubric-dir")
    run.add_argument("--permit-remote", action="store_true", help="Explicitly authorize transmission to a remote host")
    run.add_argument("--allow-overlap", action="store_true", help="Permit same-family/unknown-family grading, clearly labeled")
    run.add_argument("--force", action="store_true", help="Overwrite an existing response score, preserving new file")
    inspect = sub.add_parser("provenance", help="Export model identities, dates, hashes and missing metadata")
    inspect.add_argument("--predictions", required=True)
    inspect.add_argument("--output", required=True)
    review = sub.add_parser("validate", help="Compare scores with independently supplied human ratings")
    review.add_argument("--predictions", required=True)
    review.add_argument("--physicians", required=True)
    review.add_argument("--train-cases", help="Case IDs or calibration JSONL; detect train/test leakage")
    review.add_argument("--output", required=True)
    review.add_argument("--tolerance", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "rubrics":
            for path in sorted(RUBRICS.glob("*.json")):
                d = get_rubric(path.stem)
                print(f"{d['task']}\t{d['version']}\t{d['status']}")
            return 0
        if args.command == "registry":
            print(json.dumps(read_registry(args.submissions), indent=2, ensure_ascii=False))
            return 0
        if args.command == "lint":
            rows = load_jsonl(args.input)
            validate_batch(rows)
            for item in rows:
                if item["task"] not in ("cpc_bond", "diagnostic_reasoning", "management_reasoning", "r_idea"):
                    get_rubric(item["task"], args.rubric_dir)
            print(f"OK: {len(rows)} unique responses / {len({x['case_id'] for x in rows})} clinical cases")
            return 0
        if args.command == "provenance":
            report = provenance_report(load_jsonl(args.predictions))
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
            print(f"Saved model provenance audit to {path}")
            return 0
        if args.command == "run":
            items = load_jsonl(args.input)
            validate_batch(items)
            if args.backend == "preceptron" and family_name(args.judge_family) != "qwen":
                raise ValueError("Released PrecepTron adapters use Qwen3-32B: judge-family must be qwen")
            if args.backend != "demo" and args.model == "UNSPECIFIED_DEMO":
                raise ValueError("Real inference requires explicit --model and --base-url")
            manifest = read_manifest(args.model_manifest, served_model=args.model,
                                     backend=args.backend, require_pinned=args.require_pinned)
            manifest_sha = digest(manifest) if manifest else None
            if manifest and args.model_revision != "UNPINNED":
                expected_revision = (manifest.get("adapter_commit") if args.backend == "preceptron"
                                     else manifest["backbone_commit"])
                if args.model_revision != expected_revision:
                    raise ValueError("--model-revision differs from declared manifest checkpoint; omit it or fix the mismatch")
            ensure_independent(items, args.judge_family, args.backend, args.allow_overlap)
            destination = Path(args.output)
            existing = load_jsonl(destination) if destination.is_file() else []
            prior = {entry["response_id"]: entry for entry in existing}
            if len(prior) != len(existing):
                raise ValueError("Existing output has duplicate response IDs")
            count = 0
            for item in items:
                if item["response_id"] in prior and not args.force:
                    old = prior[item["response_id"]]
                    from .schema import blinded_case, digest
                    from .engine import independence
                    if old.get("judge", {}).get("input_sha256") != digest(blinded_case(item)):
                        raise ValueError(f"Existing response {item['response_id']} changed: use a new output path")
                    if (old.get("judge", {}).get("model") != args.model
                            or old.get("judge", {}).get("backend") != args.backend
                            or old.get("judge", {}).get("family") != args.judge_family
                            or old.get("judge", {}).get("model_revision", "UNPINNED") != args.model_revision
                            or old.get("judge", {}).get("model_manifest_sha256") != manifest_sha
                            or old.get("judge", {}).get("framework_code_sha256") != framework_code_sha256()):
                        raise ValueError("Cannot resume using changed model/backend/family/revision/manifest/code")
                    if old.get("judge", {}).get("independence") != independence(item["generator_family"], args.judge_family, backend=args.backend):
                        raise ValueError("Independence metadata changed")
                    if args.backend != "preceptron":
                        expected = rubric_hash(get_rubric(item["task"], args.rubric_dir))
                        if old["judge"].get("rubric_sha256") != expected:
                            raise ValueError("Cannot silently resume after rubric changes")
                    continue
                result = grade(item, backend=args.backend, model=args.model,
                               judge_family=args.judge_family, base_url=args.base_url,
                               api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"),
                               permit_remote=args.permit_remote, allow_overlap=args.allow_overlap,
                               rubric_dir=args.rubric_dir)
                result["judge"]["model_revision"] = args.model_revision
                result["judge"]["framework_version"] = __version__
                result["judge"]["framework_code_sha256"] = framework_code_sha256()
                result["judge"]["model_manifest"] = manifest
                result["judge"]["model_manifest_sha256"] = manifest_sha
                result["judge"]["model_pin_status"] = (
                    "COMMIT_DECLARED_NOT_SERVER_ATTESTED" if manifest else "UNVERIFIED_NO_MANIFEST")
                result["judge"]["reproducibility_warning"] = (
                    "NO_CHECKPOINT_MANIFEST" if manifest is None else
                    "SERVER_WEIGHTS_NOT_ATTESTED")
                prior[item["response_id"]] = result
                # Save after each record to allow resumption. Failed calls never receive fabricated scores.
                write_jsonl(destination, [prior[k] for k in sorted(prior)])
                count += 1
            print(f"Scored {count} new responses; total {len(prior)} in {destination}")
            if args.backend == "demo":
                print("WARNING: DEMO output is NOT a clinical judge; do not include in research results")
            return 0
        if args.command == "validate":
            predictions = load_jsonl(args.predictions)
            doctors = load_physicians(args.physicians)
            train = load_training_case_ids(args.train_cases)
            report = validate(predictions, doctors, train, args.tolerance)
            p = Path(args.output)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
            print(f"Saved descriptive human comparison to {p}")
            return 0
        raise ValueError("Unknown command")
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
