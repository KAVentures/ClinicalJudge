"""Offline tests: no gated model access or GPU is required."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_file(filename):
    spec=importlib.util.spec_from_file_location(filename,ROOT/'scripts'/filename)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[filename]=mod
    spec.loader.exec_module(mod)
    return mod

runner=load_file('run_preceptron_kaggle.py')
compare=load_file('compare_preceptron_physician50.py')


def fixture():
    pred=[];docs=[]
    for i in range(1,51):
        rid=f'A{i:03}'; case_id=f'case_{((i-1)%33):02d}'
        score=i%2
        pred.append({'response_id':rid,'case_id':case_id,'generator_family':'anthropic',
                     'score':score,'judge':{'model':'tbuckley/PrecepTron-32B-Management'}})
        for rater in ('R1','O','G'):
            docs.append({'response_id':rid,'case_id':case_id,'rater_id':rater,'score':score})
    return pred,docs


def test_hashes_deterministic():
    assert runner.sha(runner.normalized_json(runner.RUBRIC))==runner.sha(runner.normalized_json(runner.RUBRIC))
    assert runner.inference_settings()['temperature']==0
    assert runner.TASK=='appropriate_uncertainty_binary'


def test_binary_kappa():
    assert compare.cohen_binary([0,1,0,1],[0,1,0,1])==1.0
    assert compare.describe([0,1,0,1],[0,1,1,1])['false_reassurance_count']==0
    assert compare.describe([0,1,0,1],[0,1,1,1])['excess_caution_count']==1


def test_evaluate_expects_actual_adapter():
    predictions,doctors=fixture()
    out=compare.evaluate(predictions,doctors)
    assert out['n_responses']==50
    assert out['n_independent_case_clusters']==33
    assert out['vs_majority']['accuracy']==1
    assert out['vs_independent_O']['accuracy']==1
    assert len(out['accuracy_case_cluster_bootstrap_95'])==2
    predictions[0]['judge']['model']='Qwen/Qwen3-32B'
    try:
        compare.evaluate(predictions,doctors)
    except ValueError as exc:
        assert 'model' in str(exc)
    else:
        raise AssertionError('Should reject wrongly labeled judge')


def test_evaluate_requires_full_50_and_no_imputation():
    predictions,doctors=fixture()
    try:compare.evaluate(predictions[:5],doctors)
    except ValueError as e:assert '50' in str(e)
    else:raise AssertionError('Partial sample cannot count as validation')
    predictions[0]['score']=None
    try:compare.evaluate(predictions,doctors)
    except ValueError as e:assert 'binary' in str(e)
    else:raise AssertionError('Invalid model outputs cannot be imputed')


def test_blinded_case_loader(tmp_path):
    predictions,_=fixture()
    rows=[]
    for i,p in enumerate(predictions):
        rows.append({'response_id':p['response_id'],'case_id':p['case_id'],
                     'task':runner.TASK,'case':'clinical conversation',
                     'response':'model generated response','generator_family':p['generator_family']})
    path=tmp_path/'cases.jsonl';path.write_text('\n'.join(json.dumps(x) for x in rows))
    assert len(runner.read_blinded(path))==50
    rows[0]['human_label']=1
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    try:runner.read_blinded(path)
    except ValueError as e:assert 'Unblinded' in str(e)
    else:raise AssertionError('Human labels must never be sent as model inputs')
