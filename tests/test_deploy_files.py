"""The Docker files cannot be built inside the test suite, but they can be kept consistent with the project."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda name: (ROOT / name).read_text(encoding="utf-8")        # noqa: E731


def test_the_docker_torch_pin_matches_requirements():
    req = re.search(r"^torch==([\w.]+)$", read("requirements.txt"), re.M).group(1)
    docker = read("Dockerfile")
    assert f"pip install torch=={req} --index-url ${{TORCH_INDEX}}" in docker
    assert "ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu" in docker       # CPU-only build by default
    assert f"torch=={req} --index-url" in read("Makefile")


def test_the_image_installs_tesseract_and_starts_the_launcher():
    docker = read("Dockerfile")
    assert "tesseract-ocr tesseract-ocr-eng" in docker and 'CMD ["python", "run.py"]' in docker
    assert "COPY requirements.txt" in docker and "EXPOSE 8000 8501" in docker and "API_HOST=0.0.0.0" in docker


def test_the_container_runs_as_an_unprivileged_user_that_owns_the_data_folder():
    docker = read("Dockerfile")
    assert "useradd" in docker and "USER lexora" in docker
    assert docker.index("chown lexora /data") < docker.index("VOLUME /data")      # the volume starts out writable
    assert docker.index("USER lexora") < docker.index("CMD [")


def test_the_image_bakes_in_the_configured_models_and_has_a_healthcheck():
    docker = read("Dockerfile")
    assert "s.embedding_model" in docker and "s.reranker_model" in docker          # names come from config.py, not copied
    assert "|| echo" in docker                                                      # an offline build still succeeds
    assert "HEALTHCHECK" in docker and "/api/health" in docker
    assert "LEXORA_STORAGE_DIR=/data" in docker and "VOLUME /data" in docker
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in docker


def _service(compose: str, name: str) -> str:
    m = re.search(rf"^  {name}:[^\n]*\n(.*?)(?=^  \w|^\w|\Z)", compose, re.S | re.M)
    return m.group(1)


def test_compose_publishes_ports_on_localhost_only_and_keeps_data_in_volumes():
    compose = read("docker-compose.yml")
    ports = lambda block: re.findall(r'^\s*-\s*"([^"]*:\d+:\d+)"', block, re.M)     # noqa: E731
    assert ports(_service(compose, "lexora")) == ["127.0.0.1:8501:8501", "127.0.0.1:8000:8000"]
    assert all(p.startswith("127.0.0.1:") for p in ports(compose))                  # nothing is exposed to the network
    assert "host.docker.internal:11434" in compose and "lexora-data:/data" in compose and "hf-cache" in compose
    assert "extra_hosts" in compose and "host-gateway" in compose


def test_compose_offers_ollama_as_an_optional_profile_and_passes_the_local_env_file():
    compose = read("docker-compose.yml")
    assert 'profiles: ["ollama"]' in _service(compose, "ollama")
    assert "ollama/ollama" in compose and "ollama-models" in compose
    assert "path: .env" in compose and "required: false" in compose
    env_block = _service(compose, "lexora").split("environment:")[1]
    assert "LEXORA_STORAGE_DIR: /data" in env_block         # environment: beats env_file:, so .env cannot move the data


def test_secrets_and_local_data_stay_out_of_the_image_and_the_repository():
    assert {".env", "storage", ".venv"} <= set(read(".dockerignore").split())
    assert {".env", "storage/", ".venv/"} <= set(read(".gitignore").split())


def test_the_makefile_has_the_docker_targets():
    make = read("Makefile")
    for target in ("docker-build:", "docker-up:", "docker-down:", "docker-smoke:"):
        assert target in make
    assert "docker compose up -d --build" in make and "scripts.smoke_test" in make
