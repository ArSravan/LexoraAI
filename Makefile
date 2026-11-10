# Lexora AI - common tasks.   `make help` lists them.
# (Without make, run the commands by hand: they are all one-liners, see the README.)

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python
else
  VENV_PY := .venv/bin/python
endif
# use the virtual environment when it exists, otherwise whatever `python` is on the PATH
PY ?= $(if $(wildcard .venv),$(VENV_PY),python)

.PHONY: help setup run api ui test ask eval extended report check docker-build docker-up docker-down docker-smoke
help:  ## show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:  ## create .venv and install the pinned dependencies (CPU-only PyTorch)
	python -m venv .venv
	$(VENV_PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(VENV_PY) -m pip install -r requirements.txt

run:  ## start the API and the UI together and open the browser
	$(PY) run.py

api:  ## start only the API (http://127.0.0.1:8000/docs)
	$(PY) main.py

ui:  ## start only the UI (the API must be running)
	$(PY) -m streamlit run app.py

test:  ## run the test suite
	$(PY) -m pytest tests -q

ask:  ## ask from the command line:  make ask PDFS="a.pdf b.pdf" Q="your question"
	$(PY) -m scripts.ask_pdf $(PDFS) -q "$(Q)"

docker-build:  ## build the Docker image (about 10 minutes the first time)
	docker compose build

docker-up:  ## start the application in Docker (Ollama must run on the host), then open http://127.0.0.1:8501
	docker compose up -d --build

docker-down:  ## stop the container (the data volume is kept)
	docker compose down

docker-smoke:  ## test the running container end to end (add ARGS=--with-llm to also ask a grounded question)
	$(PY) -m scripts.smoke_test $(ARGS)

eval:  ## the main evaluation: retrieval and answers incl. the verifier ablation (needs Ollama, about an hour; resumable)
	$(PY) -m evaluation.evaluate sample_docs/*.pdf --gen-configs no_retrieval hybrid hybrid_noverify --refresh-citations --export-grading

extended:  ## the extended-feature evaluations (no LLM: OCR, tables, autoencoder, table ablation, gate diagnostics, cache benchmark)
	$(PY) -m evaluation.ocr_eval sample_docs/*.pdf
	$(PY) -m evaluation.table_eval sample_docs/*.pdf
	$(PY) -m evaluation.page_anomaly_eval sample_docs/*.pdf
	$(PY) -m evaluation.ablation_tables
	$(PY) -m evaluation.diagnostics sample_docs/*.pdf
	$(PY) -m evaluation.bench_indexing sample_docs/ePrivacy_Directive_2002_58.pdf
	$(PY) -m evaluation.latency_report

report:  ## write docs/RESULTS.md and the results block of the README from evaluation/results
	$(PY) -m evaluation.report

check:  ## is the repository ready to hand in? (pins, placeholders, results, unpushed commits)
	$(PY) -m scripts.check_submission
