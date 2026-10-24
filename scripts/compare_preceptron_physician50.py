#!/usr/bin/env python3
"""Analyze frozen experimental PrecepTron scores against the original physician panel.

Only run after 50 completed valid predictions. Never feed physician labels back to judge.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def cohen_binary(a:list[int],b:list[int])->float|None:
    n=len(a)
    if n==0:return None
    acc=sum(x==y for x,y in zip(a,b))/n
    pa=sum(a)/n; pb=sum(b)/n
    expected=pa*pb+(1-pa)*(1-pb)
    return None if expected==1 else (acc-expected)/(1-expected)


def describe(preds:list[int],truth:list[int])->dict:
    n=len(preds)
    tp=sum(p==1 and h==1 for p,h in zip(preds,truth))
    tn=sum(p==0 and h==0 for p,h in zip(preds,truth))
    fp=sum(p==1 and h==0 for p,h in zip(preds,truth))
    fn=sum(p==0 and h==1 for p,h in zip(preds,truth))
    return {"n":n,"accuracy":round((tp+tn)/n,4),
            "cohen_kappa":None if (k:=cohen_binary(preds,truth)) is None else round(k,4),
            "confusion_tp_tn_fp_fn":[tp,tn,fp,fn],
            "false_reassurance_count":fp, "excess_caution_count":fn,
            "positive_rate":round(sum(preds)/n,4)}


def bootstrap_cases(pairs:list[dict], *, repeats=2000, seed=20261024)->list[float]:
    groups=defaultdict(list)
    for p in pairs:groups[p["case_id"]].append(p)
    keys=sorted(groups); rng=random.Random(seed); values=[]
    for _ in range(repeats):
        chosen=[item for key in (rng.choice(keys) for _ in keys) for item in groups[key]]
        values.append(mean(item["score"]==item["truth"] for item in chosen))
    values.sort(); return [round(values[int(.025*(repeats-1))],4), round(values[int(.975*(repeats-1))],4)]


def evaluate(predictions:list[dict], ratings:list[dict])->dict:
    if len(predictions)!=50 or {r["response_id"] for r in predictions}!={f"A{i:03}" for i in range(1,51)}:
        raise ValueError("Require exactly 50 distinct blinded PrecepTron responses")
    if any(r.get("score") not in (0,1) for r in predictions):
        raise ValueError("Invalid or missing binary PrecepTron score; do NOT impute failures")
    if any(r.get("judge",{}).get("model")!="tbuckley/PrecepTron-32B-Management" for r in predictions):
        raise ValueError("Incorrect judge model identity; do NOT mislabel base Qwen as PrecepTron")
    grouped=defaultdict(dict)
    for x in ratings:
        key=x["response_id"]
        if x["rater_id"] in grouped[key]: raise ValueError("duplicate physician rating")
        grouped[key][x["rater_id"]]=int(float(x["score"]))
    if set(grouped)!={r["response_id"] for r in predictions}:
        raise ValueError("physician IDs do not match")
    ordered=sorted(predictions,key=lambda r:r["response_id"])
    if any(set(grouped[r["response_id"]])!={"R1","O","G"} for r in ordered):
        raise ValueError("Expected original three-rater panel")
    rows=[]
    for pred in ordered:
        rid=pred["response_id"]; values=grouped[rid]
        p=int(pred["score"])
        majority=int(sum(values.values())>=2)
        rows.append({"response_id":rid,"case_id":pred["case_id"],"generator_family":pred["generator_family"],
                     "score":p,"R1":values["R1"],"O":values["O"],"G":values["G"],"majority":majority})
    results={
        "status":"INDEPENDENT_PRECEPTRON_ADAPTER_INFERENCE_EXPLO RATORY".replace(" ",""),
        "limitation":"Quantized Qwen3-32B base + PrecepTron Management LoRA transferred to non-native appropriate-uncertainty rubric. Not validated clinical safety or physician equivalence.",
        "n_responses":50,"n_independent_case_clusters":len({r['case_id'] for r in rows}),
        "vs_majority":describe([r['score'] for r in rows],[r['majority'] for r in rows]),
        "accuracy_case_cluster_bootstrap_95":bootstrap_cases([dict(r,truth=r["majority"]) for r in rows]),
        "vs_independent_O":describe([r['score'] for r in rows],[r['O'] for r in rows]),
        "vs_independent_G":describe([r['score'] for r in rows],[r['G'] for r in rows]),
        "vs_author_R1":describe([r['score'] for r in rows],[r['R1'] for r in rows]),
        "O_vs_G":describe([r['O'] for r in rows],[r['G'] for r in rows]),
        "by_generator":{
            fam: describe([r['score'] for r in rows if r['generator_family']==fam],
                          [r['majority'] for r in rows if r['generator_family']==fam])
            for fam in sorted({r['generator_family'] for r in rows})
        },
        "legacy_judges_vs_majority":{
            "GPT/OpenAI": {"accuracy":0.66,"kappa":0.2808798646362099},
            "Claude/Anthropic": {"accuracy":0.62,"kappa":0.2016806722689075},
            "Grok/xAI": {"accuracy":0.72,"kappa":0.425287356321839},
            "Gemini/Google": {"accuracy":0.70,"kappa":0.37810945273631824},
        },
        "row_level":rows,
    }
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--physicians",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    preds=[json.loads(x) for x in args.predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
    with args.physicians.open(newline="",encoding="utf-8") as f: doctors=list(csv.DictReader(f))
    report=evaluate(preds,doctors)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"preceptron_vs_physicians_detailed.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    summary=["# PrecepTron-32B-Management: quantized transfer to appropriate uncertainty", "",
             "**Experimental; not task-validated physician equivalence.**", "",
             f"50 blinded model responses / {report['n_independent_case_clusters']} clinical case clusters / 3 original physician raters.", "",
             "| Comparator | Exact agreement | Cohen's kappa | False appropriate |", "|---|---:|---:|---:|"]
    for k,label in [("vs_majority","Physician majority"),("vs_independent_O","Independent physician O"),
                    ("vs_independent_G","Independent physician G"),("vs_author_R1","Study author")]:
        item=report[k]
        summary.append(f"| {label} | {item['accuracy']:.1%} | {item['cohen_kappa']} | {item['false_reassurance_count']} |")
    summary.extend(["","Two independent physicians: " + str(report["O_vs_G"]["accuracy"]),
                    "95% case-cluster bootstrap interval for agreement with majority: " + str(report["accuracy_case_cluster_bootstrap_95"]),
                    "","Historical legacy judge accuracy on same sample: GPT 66%, Claude 62%, Grok 72%, Gemini 70%.",
                    "","All values above are descriptive. Model quantization, reused training sources and task transfer are material limitations."])
    (args.output_dir/"preceptron_vs_physicians_summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary))

if __name__=="__main__":main()
