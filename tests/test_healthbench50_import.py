import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_healthbench_physician50 import parse_sheet, prepare, save_files, DATA_FILES


def _fake():
    note = '# Blinded annotation\n'
    for n in range(1,51):
        note += (f'\n## A{n:03d}\n\n### Conversation (information partially removed)\n'
                 f'**USER:** Clinical question {n}\n\n### Assistant response\n'
                 f'Please clarify case {n}.\n\n---\n')
    key={f'A{i:03d}':{'subject_model':'gpt-5_5-high' if i%2 else 'grok-4_3',
                        'prompt_id':f'case-{i%33}'} for i in range(1,51)}
    out={'runs/human_eval/annotation_sheet.md':note.encode(),
         'runs/human_eval/key.json':json.dumps(key).encode()}
    for path in DATA_FILES[2:]:
        out[path]=('annotation_id,human_label\n' + ''.join(f'A{i:03d},{i%2}\n' for i in range(1,51))).encode()
    return out


def test_import_does_not_leak_physician_scores(tmp_path):
    cases,doctors,manifest=prepare(_fake())
    assert len(cases)==50 and len(doctors)==150
    assert len({x['case_id'] for x in cases})==33
    assert cases[0]['generator_family']=='openai'
    assert 'labels' not in cases[0] and 'score' not in cases[0]
    assert 'Please clarify' in cases[0]['response']
    save_files(tmp_path,cases,doctors,manifest)
    blind=(tmp_path/'cases_blinded.jsonl').read_text()
    assert 'human_label' not in blind and 'human_score' not in blind
    assert json.loads(blind.splitlines()[0])['task']=='appropriate_uncertainty_binary'


def test_parser_refuses_missing_response():
    try:
        parse_sheet('## A001\n### Conversation (information partially removed)\nfoo')
    except ValueError: pass
    else: assert False, 'Expected error'


def test_import_rejects_missing_rating():
    src=_fake()
    path='runs/human_eval/labels_G.csv'
    src[path]=src[path].replace(b'A003,1\n',b'')
    try: prepare(src)
    except ValueError: pass
    else: assert False, 'Missing human ratings must fail'