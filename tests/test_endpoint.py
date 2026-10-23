"""Integrates with a fake local HTTP endpoint; never uses paid models or medical data."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinicaljudge.engine import grade
from clinicaljudge.schema import load_jsonl
from clinicaljudge.registry import read_registry

ROOT = Path(__file__).resolve().parents[1]

class _FakeChat(BaseHTTPRequestHandler):
    recorded = []
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        received = json.loads(self.rfile.read(n))
        self.recorded.append(received)
        assert self.path == '/v1/chat/completions'
        input_payload = json.loads(received['messages'][1]['content'])
        criteria = input_payload['rubric']['criteria']
        output = {'criteria': [{'id': c['id'], 'met': True, 'evidence_quote': '', 'explanation': 'fixture'}
                               for c in criteria], 'needs_review': True, 'limitations': ['fixture endpoint']}
        body = json.dumps({'choices': [{'message': {'content': json.dumps(output)}}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, format, *args):
        pass


class TestLocalApi(unittest.TestCase):
    def test_real_call_goes_to_separate_host(self):
        _FakeChat.recorded.clear()
        server = HTTPServer(('127.0.0.1', 0), _FakeChat)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            input_item = load_jsonl(ROOT / 'examples/cases.jsonl')[0]
            result = grade(input_item, backend='openai-compatible', model='independent-judge', judge_family='qwen',
                           base_url=f'http://127.0.0.1:{server.server_port}/v1')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.assertEqual(len(_FakeChat.recorded), 1)
        self.assertEqual(result['score'], 1.0)
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['judge']['independence'], 'CROSS_FAMILY_NOT_PROOF_OF_UNBIASEDNESS')
        transmitted = json.dumps(_FakeChat.recorded)
        self.assertNotIn('generator_family', transmitted)
        self.assertNotIn('openai', transmitted.lower())

    def test_empty_registry_is_explicitly_unvalidated(self):
        result = read_registry(ROOT / 'validations/submissions')
        self.assertEqual(result['number_of_submissions'], 0)
        self.assertEqual(result['status'], 'EVIDENCE_INDEX_NOT_CLINICAL_CERTIFICATION')

if __name__ == '__main__':
    unittest.main()
