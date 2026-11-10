# Docker verification

This page says what was actually run and what was not. Nothing here is claimed without a command behind it.

## Run it

```bash
ollama pull qwen3.5:9b                        # Ollama runs on the host, as without Docker
docker compose up -d --build                  # first build: about 10 minutes (CPU PyTorch and the two small models are downloaded)
python -m scripts.smoke_test --with-llm       # checks the running container end to end (add --pdf <file> to use another PDF)
```

Open <http://127.0.0.1:8501>. `docker compose down` stops it; the data volume is kept (`docker compose down -v` deletes it).
`make docker-up`, `make docker-down` and `make docker-smoke` do the same.

Without Ollama on the host: `docker compose --profile ollama up -d --build`, set `LEXORA_OLLAMA_BASE_URL=http://ollama:11434`
(in `.env` or the shell), then `docker compose exec ollama ollama pull qwen3.5:9b`.

What the image contains: Python 3.12, Tesseract with English data, CPU-only PyTorch, the pinned requirements, the application,
and (when the build machine is online) the two small Hugging Face models. It runs as an unprivileged user (`lexora`),
keeps its data in the volume `/data`, has a health check on `/api/health`, and publishes its ports on `127.0.0.1` only.

## Verified in a Linux sandbox (Docker Engine 29.9.2, Compose 5.6.1)

| Step | Result |
|---|---|
| `docker compose config` on `docker-compose.yml` | valid; ports bound to `127.0.0.1`, `LEXORA_STORAGE_DIR=/data` pinned, `ollama` service only with the profile |
| `docker build` of the Dockerfile | succeeded (see the differences below) |
| Container start with `run.py` | API and UI both up; `docker ps` reports `healthy` |
| UI | HTTP 200 on port 8501 |
| Process owner | `lexora`, not root; `/data` owned by `lexora` and writable |
| `python -m scripts.smoke_test` against the container | API health, upload, indexing (11-page PDF, 21 headings detected), page image, refusal of an unanswerable question: all PASS |
| Grounded answer through the container (`--with-llm`) | PASS: answered, 1 of 1 claims VERIFIED |
| `docker restart` | the case and its index were still there (volume works) |

The sandbox could not do everything the real build does, so the test image differed from the Dockerfile in these ways:

* PyTorch came from PyPI (the CUDA build, 7 GB image) because `download.pytorch.org` was not reachable. The Dockerfile's
  default `TORCH_INDEX` is the CPU index; the `--build-arg TORCH_INDEX=...` mechanism itself was exercised.
* `apt-get` was blocked by the sandbox proxy, so Tesseract was **not installed** in the test image. Consequently the
  smoke test's OCR check FAILED there, as it should.
* Hugging Face was not reachable, so the model download step took its documented fallback branch (a warning, the build
  continues). The container was started with two tiny stand-in models from a mounted folder instead of MiniLM.
* The language model was a stand-in HTTP server speaking Ollama's protocol, not Ollama and not `qwen3.5:9b`.
* Linux host networking was used for the run (`--network host`), so the `host.docker.internal` mapping was not exercised.

## Not verified in Docker (run these yourself and fill in the table below)

* the CPU-only PyTorch install and the Tesseract install inside the real image
* the model download at build time
* answers from the real model through the container, and `host.docker.internal` reaching Ollama on the host
* Docker Desktop on Windows (the development machine)
* the optional `ollama` profile

## Your verification run

Run on: _date, machine, Docker version_

```text
paste the output of:  python -m scripts.smoke_test --with-llm
```

| Item | Result |
|---|---|
| `docker compose up -d --build` finished, image size | |
| `docker ps` health | |
| Smoke test: all checks PASS (including OCR and the grounded answer) | |
| Time of the first grounded answer through the container | |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama reachable` fails | Ollama must be running on the host. On Linux it must listen on all interfaces: `OLLAMA_HOST=0.0.0.0 ollama serve` (the container reaches the host through `host.docker.internal`). |
| `Model available` fails | `ollama pull qwen3.5:9b` on the host (or `docker compose exec ollama ollama pull ...` with the profile). |
| Port already in use | Stop the program using 8000 or 8501, or change the published ports in `docker-compose.yml`. |
| Build is slow or fails downloading PyTorch | Use a mirror: `docker build --build-arg TORCH_INDEX=<index url> .` |
| First question is slow | The model is loaded into memory on first use; keep Ollama running (`OLLAMA_KEEP_ALIVE=30m`). |
| Start fresh | `docker compose down -v` removes the data volume (uploaded PDFs, indexes, database). |
