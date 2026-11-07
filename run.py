"""One-command start: the API (FastAPI) and the UI (Streamlit) together.     python run.py

Environment (all optional):  API_HOST (127.0.0.1)  API_PORT (8000)  UI_PORT (8501)  NO_BROWSER=1
The API binds to 127.0.0.1 by default, so nothing is reachable from other machines.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def commands(host: str, api_port: int, ui_port: int) -> tuple[list[str], list[str]]:
    """The two commands that make up the application."""
    api = [sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(api_port)]
    ui = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(ui_port),
          "--server.address", host, "--server.headless", "true"]
    return api, ui


def port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_api(url: str, timeout: float = 60.0, alive=lambda: True) -> bool:
    """Poll ``url`` until it answers 200, ``timeout`` seconds pass, or ``alive()`` turns False."""
    deadline = time.time() + timeout
    while time.time() < deadline and alive():
        try:
            if requests.get(url, timeout=1).ok:
                return True
        except requests.RequestException:
            time.sleep(0.3)
    return False


def main() -> int:
    host = os.getenv("API_HOST", "127.0.0.1")
    api_port, ui_port = int(os.getenv("API_PORT", "8000")), int(os.getenv("UI_PORT", "8501"))
    check = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    for name, port in (("API", api_port), ("UI", ui_port)):
        if port_in_use(check, port):
            print(f"{name} port {port} is already in use. Stop the other program or set "
                  f"{'API_PORT' if name == 'API' else 'UI_PORT'}=<another port>.", file=sys.stderr)
            return 2
    env = {**os.environ, "LEXORA_API_URL": f"http://{check}:{api_port}", "PYTHONUTF8": "1"}
    api_cmd, ui_cmd = commands(host, api_port, ui_port)
    api = subprocess.Popen(api_cmd, cwd=ROOT, env=env)
    ui = None
    try:
        if not wait_for_api(f"http://{check}:{api_port}/api/health", alive=lambda: api.poll() is None):
            print("The API did not start; see the messages above.", file=sys.stderr)
            return 1
        ui = subprocess.Popen(ui_cmd, cwd=ROOT, env=env)
        print(f"\nLexora AI is running:\n  UI : http://{check}:{ui_port}\n  API: http://{check}:{api_port}/docs\n"
              "Press Ctrl+C to stop.\n")
        if os.getenv("NO_BROWSER") != "1":
            time.sleep(2)
            webbrowser.open(f"http://{check}:{ui_port}")
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            if p is not None and p.poll() is None:
                p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
