"""Integration test: actual fake HTTP transport yields research provenance; no paid API."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from clinicaljudge.engine import call_openai_compatible_traced
from clinicaljudge.provenance import read_manifest, provenance_report, framework_code_sha256

class _Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(n))
        assert request["temperature"] == 0
        verdict = {"criteria": [], "limitations": [], "needs_review": False}
        data = json.dumps({"id": "c-test123", "model": "qwen-judge-local", "created": 1792771200,
                           "system_fingerprint": "fp-example",
                           "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                           "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-ID", "req-example")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass

class TestProvenance(unittest.TestCase):
    def test_api_captures_reported_identity_and_time(self):
        server = HTTPServer(("127.0.0.1", 0), _Fake)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            value, provenance = call_openai_compatible_traced(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="qwen-judge-local", messages=[{"role":"system","content":"Judge only"}],
                api_key="SECRET_SHOULD_NOT_LEAK", permit_remote=False)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)
        self.assertEqual(value["criteria"], [])
        self.assertEqual(provenance["api_response_id"], "c-test123")
        self.assertEqual(provenance["api_request_id"], "req-example")
        self.assertEqual(provenance["response_model"], "qwen-judge-local")
        self.assertEqual(provenance["api_system_fingerprint"], "fp-example")
        self.assertIn("2026", provenance["api_created_at_utc"])
        self.assertEqual(provenance["api_finish_reason"], "stop")
        self.assertEqual(provenance["api_usage"]["total_tokens"], 13)
        self.assertEqual(provenance["api_model_identity_status"], "MATCHED_REQUEST")
        self.assertEqual(len(provenance["system_prompt_sha256"]), 64)
        self.assertEqual(len(framework_code_sha256()), 64)
        self.assertNotIn("SECRET_SHOULD_NOT_LEAK", json.dumps(provenance))

    def test_manifest_rejects_alias_as_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"manifest.json"
            manifest = {"schema_version": 1,"served_model":"qwen-j", "backbone_model":"Qwen/Qwen3-32B",
                        "backbone_commit": "main", "inference_runtime":"vllm", "runtime_version": "0.11.0"}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"commit"):
                read_manifest(str(path), served_model="qwen-j", backend="openai-compatible",require_pinned=True)
            manifest["backbone_commit"] = "a"*40
            path.write_text(json.dumps(manifest))
            accepted = read_manifest(str(path), served_model="qwen-j", backend="openai-compatible", require_pinned=True)
            self.assertEqual(accepted["backbone_commit"], "a"*40)
            with self.assertRaisesRegex(ValueError,"served_model"):
                read_manifest(str(path), served_model="wrong",backend="openai-compatible")
            with self.assertRaisesRegex(ValueError,"adapter"):
                read_manifest(str(path), served_model="qwen-j",backend="preceptron")
            manifest.update({"adapter_model":"tbuckley/PrecepTron-32B-CPC", "adapter_commit":"b"*40})
            path.write_text(json.dumps(manifest))
            accepted = read_manifest(str(path), served_model="qwen-j",backend="preceptron",require_pinned=True)
            self.assertEqual(accepted["adapter_commit"], "b"*40)

    def test_report_covers_missing_metadata(self):
        r = provenance_report([{"judge": {"model":"qwen", "backend":"openai-compatible",
            "model_revision":"UNPINNED","transport_provenance":{ "response_model":"qwen-server",
            "request_sent_at_utc":"2026-10-23T18:00:00Z", "system_prompt_sha256":"abc"}}}])
        self.assertEqual(r["n_groups"], 1)
        self.assertEqual(r["groups"][0]["missing_api_created_time"], 1)
        self.assertEqual(r["groups"][0]["api_response_models"], ["qwen-server"])
        self.assertEqual(r["groups"][0]["system_prompt_hashes"], ["abc"])

if __name__ == "__main__": unittest.main()
