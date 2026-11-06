"""The client is tested against a tiny stub HTTP server that speaks Ollama's protocol."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from generation.ollama_client import OllamaClient, OllamaError


class Stub:
    """Mutable behaviour + a record of what the client sent."""
    def __init__(self):
        self.mode = "ok"
        self.payloads: list[dict] = []
        self.models = ["qwen3.5:9b", "llama3:latest"]


@pytest.fixture
def server():
    stub = Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/version":
                self._json(200, {"version": "0.0-test"})
            elif self.path == "/api/tags":
                self._json(200, {"models": [{"name": m} for m in stub.models]})
            else:
                self._json(404, {})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stub.payloads.append(payload)
            if stub.mode == "missing":
                return self._json(404, {"error": "model not found"})
            if stub.mode == "no_think" and "think" in payload:
                body = b'{"error":"model does not support think"}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if stub.mode == "slow":
                time.sleep(1.5)
            if stub.mode == "server_error":
                return self._json(500, {"error": "kaput"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            lines = [{"message": {"content": "Hel"}}, {"message": {"thinking": "hmm"}},
                     {"message": {"content": "lo"}}]
            if stub.mode == "error_chunk":
                lines = [{"message": {"content": "partial"}}, {"error": "boom"}]
            else:
                lines.append({"done": True, "prompt_eval_count": 11, "eval_count": 3})
            for ln in lines:
                self.wfile.write(json.dumps(ln).encode() + b"\n")
                self.wfile.flush()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stub.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield stub
    httpd.shutdown()


def client(server, **kw):
    return OllamaClient(server.url, **{"model": "qwen3.5:9b", "timeout": 5, **kw})


MSG = [{"role": "user", "content": "hi"}]


def test_streamed_reply_is_assembled_and_tokens_are_reported(server):
    seen = []
    r = client(server).chat(MSG, on_token=seen.append)
    assert r.content == "Hello" and seen == ["Hel", "lo"] and r.thinking == "hmm"
    assert (r.prompt_tokens, r.completion_tokens) == (11, 3) and r.model == "qwen3.5:9b" and r.total_seconds >= 0


def test_generation_options_and_schema_are_sent(server):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    client(server, temperature=0.2, num_ctx=4096, num_predict=77, seed=7).chat(MSG, fmt=schema)
    p = server.payloads[-1]
    assert p["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 77, "seed": 7}
    assert p["format"] == schema and p["stream"] is True and p["messages"] == MSG and p["think"] is False


def test_think_option_modes(server):
    client(server, think="on").chat(MSG)
    assert server.payloads[-1]["think"] is True
    client(server, think="auto").chat(MSG)
    assert "think" not in server.payloads[-1]
    client(server).chat(MSG, think="on")                       # per-call override
    assert server.payloads[-1]["think"] is True


def test_a_model_without_think_support_is_retried_and_remembered(server):
    server.mode = "no_think"
    c = client(server)
    assert c.chat(MSG).content == "Hello"
    assert "think" in server.payloads[0] and "think" not in server.payloads[1]
    c.chat(MSG)
    assert "think" not in server.payloads[-1] and len(server.payloads) == 3       # no second failed attempt


def test_missing_model_gives_an_actionable_message(server):
    server.mode = "missing"
    with pytest.raises(OllamaError, match="ollama pull qwen3.5:9b"):
        client(server).chat(MSG)


def test_server_error_and_error_chunk(server):
    server.mode = "server_error"
    with pytest.raises(OllamaError, match="500"):
        client(server).chat(MSG)
    server.mode = "error_chunk"
    with pytest.raises(OllamaError, match="boom"):
        client(server).chat(MSG)


def test_timeout_is_reported(server):
    server.mode = "slow"
    with pytest.raises(OllamaError, match="timed out"):
        client(server, timeout=0.4).chat(MSG)


def test_unreachable_server():
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient("http://127.0.0.1:9", timeout=1).chat(MSG)
    assert OllamaClient("http://127.0.0.1:9").is_up() is False


def test_status_and_model_checks(server):
    c = client(server)
    st = c.status()
    assert st["connected"] and st["model_available"] and st["version"] == "0.0-test"
    assert st["models"] == ["qwen3.5:9b", "llama3:latest"]
    assert c.has_model("llama3") and not c.has_model("mistral")          # bare name matches ':latest'
    assert client(server, model="mistral").status()["model_available"] is False
    down = OllamaClient("http://127.0.0.1:9").status()
    assert down["connected"] is False and down["models"] == [] and down["version"] is None


def test_client_from_settings(monkeypatch):
    from config import get_settings
    monkeypatch.setenv("LEXORA_OLLAMA_MODEL", "tiny:1b")
    monkeypatch.setenv("LEXORA_OLLAMA_NUM_CTX", "2048")
    get_settings.cache_clear()
    c = OllamaClient.from_settings()
    assert c.model == "tiny:1b" and c.num_ctx == 2048 and c.seed == 42 and c.base_url == "http://localhost:11434"
