"""The project runs from ANY folder name, with plain module imports and no environment setup."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import config

ROOT = config.ROOT
SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "storage", "__pycache__", ".pytest_cache", "*.pyc", "sample_docs")


def test_the_project_root_is_found_and_storage_lives_inside_it():
    assert (ROOT / "app.py").is_file() and (ROOT / "main.py").is_file() and (ROOT / "requirements.txt").is_file()
    assert config.Settings().storage_dir == ROOT / "storage"          # storage/ and .env belong INSIDE the project


def test_every_import_is_a_plain_module_import():
    """`from config import ...`, not `from LexoraAI.config import ...`: the folder name must not matter."""
    offenders = []
    for py in ROOT.rglob("*.py"):
        if any(part in (".git", ".venv", "venv", "storage", "__pycache__", "sample_docs") for part in py.relative_to(ROOT).parts):
            continue
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(LexoraAI|_bootstrap)\b", line):
                offenders.append(f"{py.relative_to(ROOT)}:{n}: {line.strip()}")
    assert offenders == []
    assert not (ROOT / "_bootstrap.py").exists() and not (ROOT / "__init__.py").exists()


def run_in(folder: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}            # no help from the environment
    env["HF_HUB_OFFLINE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=folder, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("folder_name", ["LexoraAI-main", "My Project Copy"])
def test_scripts_and_the_api_start_from_a_folder_with_any_name(tmp_path, folder_name):
    """A ZIP download is called LexoraAI-main; people also rename folders. None of it may matter."""
    copy = tmp_path / folder_name
    shutil.copytree(ROOT, copy, ignore=SKIP)
    r = run_in(copy, "-m", "scripts.search_pdf", "--help")
    assert r.returncode == 0 and "usage" in r.stdout, r.stderr[-500:]
    r = run_in(copy, "-m", "evaluation.evaluate", "--help")
    assert r.returncode == 0 and "usage" in r.stdout, r.stderr[-500:]
    r = run_in(copy, "-c", "import main; print(main.app.title)")
    assert r.returncode == 0 and r.stdout.strip() == "Lexora AI API", r.stderr[-500:]
    r = run_in(copy, "-c", "from config import ROOT; print(ROOT)")
    assert r.returncode == 0 and Path(r.stdout.strip()).resolve() == copy.resolve(), r.stderr[-500:]


def test_the_readme_documents_the_evaluation_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("python -m evaluation.evaluate", "python -m evaluation.calibrate", "--apply-grades", "--export-grading",
                   "python -m scripts.download_sample_docs", "gen_cache.jsonl"):
        assert needle in readme, needle
    assert "evaluation/results/gen_cache.jsonl" in (ROOT / ".gitignore").read_text()
    assert "_bootstrap" not in readme and "LexoraAI.config" not in readme           # the README describes the code as it is
    assert "plain name" in readme and "refusal_stage" in readme
