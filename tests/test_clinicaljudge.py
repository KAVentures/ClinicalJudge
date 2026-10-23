import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clinicaljudge.schema import blinded_case, digest, validate_batch, load_jsonl
from clinicaljudge.engine import (family_name, independence, _extract_json, _generic_prompt,
                                  _parse_generic_result, call_openai_compatible, ensure_independent)
from clinicaljudge.rubrics import get_rubric, rubric_hash
from clinicaljudge.validation import load_physicians, validate

HERE = Path(__file__).resolve().parents[1]


class TestSafety(unittest.TestCase):
    def setUp(self):
        self.rows = load_jsonl(HERE / 'examples/cases.jsonl')

    def test_lint_and_blinding(self):
        validate_batch(self.rows)
        p = _generic_prompt(self.rows[0], get_rubric('missing_information'))
        payload = json.dumps(p)
        self.assertNotIn('openai', payload.lower())
        self.assertNotIn('demo-resp-01', payload)
        self.assertEqual(set(blinded_case(self.rows[0])), {'case', 'response', 'reference'})

    def test_duplicate_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_batch(self.rows + [self.rows[0]])

    def test_same_family_not_independent(self):
        self.assertEqual(family_name('PrecepTron-Qwen3'), 'qwen')
        self.assertEqual(independence('Qwen3', 'qwen', backend='openai-compatible'), 'OVERLAPPING_FAMILY')
        with self.assertRaisesRegex(ValueError, 'same-family'):
            ensure_independent([dict(self.rows[0], generator_family='Qwen3')], 'qwen', 'openai-compatible', False)

    def test_unknown_family_requires_override(self):
        with self.assertRaisesRegex(ValueError, 'families'):
            ensure_independent([dict(self.rows[0], generator_family='unknown')], 'qwen', 'openai-compatible', False)

    def test_rubric_hash_changes_after_edit(self):
        rubric = get_rubric('missing_information')
        changed = json.loads(json.dumps(rubric))
        changed['criteria'][0]['description'] += ' changed'
        self.assertNotEqual(rubric_hash(rubric), rubric_hash(changed))

    def test_refuse_remote_by_default(self):
        with self.assertRaisesRegex(ValueError, 'Remote'):
            call_openai_compatible(base_url='https://example.com/v1', model='qwen', messages=[], api_key='', permit_remote=False)
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            call_openai_compatible(base_url='http://example.com/v1', model='qwen', messages=[], api_key='', permit_remote=True)

    def test_malformed_judge_response(self):
        with self.assertRaises(ValueError):
            _extract_json('not a json response')
        with self.assertRaises(ValueError):
            _parse_generic_result({'criteria': []}, get_rubric('safety'), 'hi')

    def test_evidence_quote_not_in_response_requires_review(self):
        rubric = get_rubric('safety')
        fake = {'criteria': [{'id': x['id'], 'met': True, 'evidence_quote': 'ABSENT STRING', 'explanation': 'test'}
                             for x in rubric['criteria']], 'limitations': [], 'needs_review': False}
        score = _parse_generic_result(fake, rubric, 'real answer')
        self.assertTrue(score['needs_review'])
        self.assertEqual(score['score'], 1.0)
        self.assertEqual(len(score['limitations']), len(rubric['criteria']))

    def test_critical_fail_flag(self):
        rubric = get_rubric('safety')
        fake = {'criteria': [{'id': x['id'], 'met': False, 'evidence_quote': '', 'explanation': 'test'}
                             for x in rubric['criteria']], 'limitations': [], 'needs_review': False}
        score = _parse_generic_result(fake, rubric, '')
        self.assertTrue(score['critical_error_flag'])
        self.assertTrue(score['needs_review'])


class TestValidation(unittest.TestCase):
    def test_physician_csv(self):
        ratings = load_physicians(HERE / 'examples/physician_ratings.csv')
        self.assertEqual(len(ratings), 4)
        self.assertFalse(ratings[0]['critical_error'])

    def test_disallow_data_leakage(self):
        records = [{'response_id': 'a', 'case_id': 'x', 'task': 'safety', 'score': 0.5, 'critical_error_flag': False}]
        humans = [{'response_id': 'a', 'case_id': 'x', 'task': 'safety', 'rater_id': 'MD', 'score': 0.5, 'critical_error': False}]
        with self.assertRaisesRegex(ValueError, 'LEAKAGE'):
            validate(records, humans, {'x'})

    def test_sensitivity_miss(self):
        records = [{'response_id': 'a', 'case_id': 'x', 'task': 'safety', 'score': 0.5, 'critical_error_flag': False},
                   {'response_id': 'b', 'case_id': 'y', 'task': 'safety', 'score': 0.7, 'critical_error_flag': True}]
        humans = [{'response_id': 'a', 'case_id': 'x', 'task': 'safety', 'rater_id': 'M1', 'score': 0.6, 'critical_error': True},
                  {'response_id': 'b', 'case_id': 'y', 'task': 'safety', 'rater_id': 'M1', 'score': 0.7, 'critical_error': False}]
        report = validate(records, humans)
        t = report['tasks']['safety']['critical_error_detection']
        self.assertEqual((t['fn'], t['fp'], t['tp']), (1, 1, 0))
        self.assertEqual(t['sensitivity'], 0.0)


class TestCli(unittest.TestCase):
    def test_cli_end_to_end_demo(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            pred = path / 'results.jsonl'
            report = path / 'comparison.json'
            def execute(*args):
                return subprocess.run([sys.executable, '-m', 'clinicaljudge', *args],
                                      cwd=HERE, capture_output=True, text=True)
            lint = execute('lint', '--input', str(HERE / 'examples/cases.jsonl'))
            self.assertEqual(lint.returncode, 0, lint.stderr)
            r = execute('run', '--backend', 'demo', '--input', str(HERE / 'examples/cases.jsonl'), '--output', str(pred))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(load_jsonl(pred)), 2)
            self.assertEqual(load_jsonl(pred)[0]['judge']['independence'], 'NOT_A_REAL_JUDGE')
            second = execute('run', '--backend', 'demo', '--input', str(HERE / 'examples/cases.jsonl'), '--output', str(pred))
            self.assertIn('Scored 0 new', second.stdout)
            r = execute('validate', '--predictions', str(pred), '--physicians',
                        str(HERE / 'examples/physician_ratings.csv'), '--output', str(report))
            self.assertEqual(r.returncode, 0, r.stderr)
            output = json.loads(report.read_text())
            self.assertIn('missing_information', output['tasks'])
            self.assertEqual(output['validation_status'], 'DEMO_NOT_VALIDATION')


if __name__ == '__main__':
    unittest.main()
