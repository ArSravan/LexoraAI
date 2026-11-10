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
