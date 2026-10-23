#!/usr/bin/env python3
"""Prepare the 50 physician-rated missing-information cases without outcome leakage.

The physician label CSV is saved SEPARATELY from the JSONL given to the judge.
This script DOES NOT perform model inference or assert clinical equivalence.

Source files are read from a pinned public-repository commit, not a moving branch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = 'KAVentures/health-ai-readiness-robustness'
SOURCE_COMMIT = '48c6b67102e4494bf224f63fd9a61816aa9da920'
DATA_FILES = (
    'runs/human_eval/annotation_sheet.md',
    'runs/human_eval/key.json',
    'runs/human_eval/human_labels.csv',
    'runs/human_eval/labels_O.csv',
    'runs/human_eval/labels_G.csv',
)
RATER_PATHS = {'R1':'runs/human_eval/human_labels.csv',
               'O':'runs/human_eval/labels_O.csv',
               'G':'runs/human_eval/labels_G.csv'}
TASK='appropriate_uncertainty_binary'


def pinned_url(path: str) -> str:
    if path not in DATA_FILES:
        raise ValueError('Unapproved input path: ' + path)
    return f'https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{path}'


def fetch(path: str, *, source_dir: Path | None = None) -> bytes:
    if path not in DATA_FILES:
        raise ValueError('Unexpected source filename')
    if source_dir is not None:
        return (source_dir / Path(path).name).read_bytes()
    with urlopen(pinned_url(path), timeout=45) as res:
        return res.read()


def parse_sheet(text: str) -> dict[str, dict]:
    """Read raw blinded conversations, avoiding intro, ratings, and provider IDs."""
    blocks = re.split(r'(?m)^## (A\d{3})\s*$', text)
    if len(blocks) < 3 or (len(blocks)-1) % 2:
        raise ValueError('Annotation sheet headings not found')
    cases = {}
    for idx in range(1, len(blocks), 2):
        response_id, body = blocks[idx], blocks[idx+1]
        head = re.search(r'(?m)^### Conversation \(information partially removed\)\s*$', body)
        sep = re.search(r'(?m)^### Assistant response\s*$', body)
        if not head or not sep or sep.start() <= head.end():
            raise ValueError(f'Missing conversation/response delimiters in {response_id}')
        case = body[head.end():sep.start()].strip()
        response = body[sep.end():].strip()
        # The original sheet uses horizontal separators between items; remove
        # only a final standalone Markdown divider, preserving source wording.
        response = re.sub(r'(?m)\n\s*---\s*$', '', response).strip()
        if not case or not response or response_id in cases:
            raise ValueError(f'Duplicate or empty input for {response_id}')
        cases[response_id]={'case':case, 'response':response}
    return cases


def parse_rater_csv(text: str) -> dict[str, int]:
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        rid, value = row.get('annotation_id'), row.get('human_label')
        if not rid or value not in ('0', '1') or rid in out:
            raise ValueError('Malformed or duplicated human label')
        out[rid] = int(value)
    return out


def prepare(contents: dict[str, bytes]) -> tuple[list[dict], list[dict], dict]:
    hashes={path: hashlib.sha256(contents[path]).hexdigest() for path in DATA_FILES}
    raw=contents['runs/human_eval/annotation_sheet.md'].decode('utf-8')
    case_blocks=parse_sheet(raw)
    key=json.loads(contents['runs/human_eval/key.json'].decode('utf-8'))
    labels={r: parse_rater_csv(contents[p].decode('utf-8')) for r,p in RATER_PATHS.items()}
    ids=set(case_blocks)
    if len(ids)!=50 or ids != set(key) or any(ids!=set(lab) for lab in labels.values()):
        raise ValueError('Source IDs inconsistent or not exactly 50; source may have changed')
    inputs,physicians=[],[]
    for rid in sorted(ids):
        info=key[rid]
        family=info['subject_model'].lower()
        if family.startswith(('opus', 'claude')): family='anthropic'
        elif family.startswith('gpt'): family='openai'
        elif family.startswith('gemini'): family='google'
        elif family.startswith('grok'): family='xai'
        else: raise ValueError('Unrecognized generator model family, cannot prove independence')
        # NO original physician scores or machine votes in the judge input.
        inputs.append({'response_id':rid, 'case_id':info['prompt_id'], 'task':TASK,
                       'case':case_blocks[rid]['case'], 'response':case_blocks[rid]['response'],
                       'generator_family':family})
        for rater in ('R1','O','G'):
            physicians.append({'response_id':rid, 'case_id':info['prompt_id'],
                               'task':TASK, 'rater_id':rater, 'score':labels[rater][rid],
                               'critical_error':''})
    if len({item['case_id'] for item in inputs}) != 33:
        raise ValueError('Expected 33 distinct case/prompt IDs')
    manifest={'study':'health-ai-readiness-robustness physician-rated missing-information 50',
              'repository':REPOSITORY,'commit':SOURCE_COMMIT,
              'source_sha256':hashes,'n_responses':50,'n_distinct_cases':33,
              'n_ratings':150,'physicians':['R1_author','O_independent','G_independent'],
              'rubric_id':TASK,
              'judge_input_includes_physician_labels':False,
              'annotation_construct':'Appropriate uncertainty when user information was removed',
              'known_limitation':'Not a task-validated physician-equivalent judge; source perturbations may be answerable despite deletion; do not treat as harm ground truth',
              'sources':{p:pinned_url(p) for p in DATA_FILES}}
    return inputs,physicians,manifest


def save_files(output_dir: Path, cases: list[dict], doctors: list[dict], manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'cases_blinded.jsonl').open('w',encoding='utf-8') as f:
        for x in cases:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (output_dir/'physicians_HOLDOUT.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['response_id','case_id','task','rater_id','score','critical_error']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(doctors)
    (output_dir/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,help='Optional offline directory holding the five source basenames')
    parser.add_argument('--output-dir',type=Path,default=Path('runs/healthbench_physician50'))
    args=parser.parse_args()
    payloads={name:fetch(name,source_dir=args.source_dir) for name in DATA_FILES}
    cases,doctors,manifest=prepare(payloads)
    save_files(args.output_dir,cases,doctors,manifest)
    print(f'Prepared {len(cases)} blinded responses, {len(set(x["case_id"] for x in cases))} clinical cases and {len(doctors)} withheld human ratings')
    print('Run clinicaljudge lint --input',args.output_dir/'cases_blinded.jsonl')
    print('Do NOT use the physician_HOLDOUT.csv ratings as judge inputs or calibration on these test cases.')

if __name__=='__main__': main()