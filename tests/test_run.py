import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import run

ROOT = Path(__file__).resolve().parents[1]


def test_the_two_commands_use_the_given_host_and_ports():
    api, ui = run.commands("127.0.0.1", 8123, 8456)
    assert api[:3] == [sys.executable, "-m", "uvicorn"] and "main:app" in api
    assert api[api.index("--host") + 1] == "127.0.0.1" and api[api.index("--port") + 1] == "8123"
    assert ui[:4] == [sys.executable, "-m", "streamlit", "run"] and "app.py" in ui
    assert ui[ui.index("--server.port") + 1] == "8456" and ui[ui.index("--server.address") + 1] == "127.0.0.1"
    assert ui[ui.index("--server.headless") + 1] == "true"


def test_the_entry_points_the_commands_name_exist():
    assert (ROOT / "main.py").is_file() and (ROOT / "app.py").is_file()
    from main import app
    assert app.title == "Lexora AI API"


def test_port_in_use():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert run.port_in_use("127.0.0.1", port)
    assert not run.port_in_use("127.0.0.1", port)


def serve(status: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_wait_for_api():
    ok = serve(200)
    assert run.wait_for_api(f"http://127.0.0.1:{ok.server_address[1]}/", timeout=3)
    ok.shutdown()
    broken = serve(503)
    assert not run.wait_for_api(f"http://127.0.0.1:{broken.server_address[1]}/", timeout=0.5)
    broken.shutdown()
    assert not run.wait_for_api("http://127.0.0.1:9/", timeout=0.5)
    assert not run.wait_for_api("http://127.0.0.1:9/", timeout=5, alive=lambda: False)          # the API process died


def test_main_refuses_to_start_when_a_port_is_taken(monkeypatch, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        monkeypatch.setenv("API_PORT", str(s.getsockname()[1]))
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        assert run.main() == 2
    err = capsys.readouterr().err
    assert "API port" in err and "already in use" in err and "API_PORT=" in err


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_makefile_targets_expand_to_the_expected_commands():
    def dry(*args):
        return subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "run.py" in dry("run") and "main.py" in dry("api") and "streamlit run app.py" in dry("ui")
    assert "pytest tests -q" in dry("test")
    assert "scripts.ask_pdf a.pdf b.pdf -q \"What?\"" in dry("ask", 'PDFS=a.pdf b.pdf', "Q=What?")
    setup = dry("setup")
    assert "venv .venv" in setup and "download.pytorch.org/whl/cpu" in setup and "-r requirements.txt" in setup
