"""The Docker files cannot be built inside the test suite, but they can be kept consistent with the project."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda name: (ROOT / name).read_text(encoding="utf-8")        # noqa: E731


def test_the_docker_torch_pin_matches_requirements():
    req = re.search(r"^torch==([\w.]+)$", read("requirements.txt"), re.M).group(1)
    assert f"pip install torch=={req} --index-url https://download.pytorch.org/whl/cpu" in read("Dockerfile")
    assert f"torch=={req} --index-url" in read("Makefile")


def test_the_image_installs_tesseract_and_starts_the_launcher():
    docker = read("Dockerfile")
    assert "tesseract-ocr" in docker and 'CMD ["python", "run.py"]' in docker and "COPY requirements.txt" in docker
    assert "EXPOSE 8000 8501" in docker and "ENV API_HOST=0.0.0.0" in docker


def test_compose_publishes_ports_on_localhost_only_and_keeps_data_in_volumes():
    compose = read("docker-compose.yml")
    ports = re.findall(r'^\s*-\s*"([^"]*:\d+:\d+)"', compose, re.M)
    assert ports == ["127.0.0.1:8501:8501", "127.0.0.1:8000:8000"]
    assert "host.docker.internal:11434" in compose and "lexora-data:/data" in compose and "hf-cache" in compose


def test_secrets_and_local_data_stay_out_of_the_image_and_the_repository():
    assert {".env", "storage", ".venv"} <= set(read(".dockerignore").split())
    assert {".env", "storage/", ".venv/"} <= set(read(".gitignore").split())
