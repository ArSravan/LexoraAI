"""LexoraAI must work whether the project folder CONTAINS the package or IS the package (flat layout)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import _bootstrap
import config

SKIP = shutil.ignore_patterns(".git", ".venv", "venv", "storage", "__pycache__", ".pytest_cache", "*.pyc", "sample_docs")


def test_nothing_changes_when_the_project_folder_contains_the_package(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)
    assert _bootstrap.ensure_importable(tmp_path) is False and sys.path == before


def test_a_flat_project_folder_puts_its_parent_on_the_path_once(tmp_path, monkeypatch):
    project = tmp_path / "LexoraAI"
    project.mkdir()
    (project / "__init__.py").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert _bootstrap.ensure_importable(project) is True and sys.path[0] == str(tmp_path)
    assert _bootstrap.ensure_importable(project) is False and sys.path.count(str(tmp_path)) == 1


def test_a_flat_project_folder_with_the_wrong_name_is_explained(tmp_path):
    project = tmp_path / "Lexora AI"
    project.mkdir()
    (project / "__init__.py").write_text("")
    with pytest.raises(ImportError, match="Rename the folder to 'LexoraAI'"):
        _bootstrap.ensure_importable(project)


def test_the_project_root_is_found_in_either_layout():
    root = config.ROOT
    assert (root / "app.py").is_file() and (root / "main.py").is_file() and (root / "requirements.txt").is_file()
    assert config.Settings().storage_dir == root / "storage"          # storage/ and .env belong INSIDE the project


def flat_copy(tmp_path: Path) -> Path:
    """The whole project laid out the flat way: a folder named LexoraAI whose modules sit next to app.py."""
    dest = tmp_path / "LexoraAI"

    def skip(directory, names):
        ignored = set(SKIP(directory, names))
        if Path(directory) == config.ROOT:
            ignored.add("LexoraAI")                                  # a nested package is merged in below
        return ignored

    shutil.copytree(config.ROOT, dest, ignore=skip)
    nested = config.ROOT / "LexoraAI"
    if nested.is_dir():
        shutil.copytree(nested, dest, ignore=SKIP, dirs_exist_ok=True)
    (dest / "__init__.py").touch()
    return dest


def run_in(folder: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}            # no help from the environment
    env["HF_HUB_OFFLINE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=folder, env=env, capture_output=True, text=True, timeout=120)


def test_scripts_and_the_api_start_from_a_flat_folder_without_pythonpath(tmp_path):
    flat = flat_copy(tmp_path)
    assert not (flat / "LexoraAI").exists() and (flat / "config.py").is_file() and (flat / "app.py").is_file()
    r = run_in(flat, "-m", "scripts.search_pdf", "--help")
    assert r.returncode == 0 and "usage" in r.stdout, r.stderr[-500:]
    r = run_in(flat, "-c", "import main; print(main.app.title)")
    assert r.returncode == 0 and r.stdout.strip() == "Lexora AI API", r.stderr[-500:]
    r = run_in(flat, "-c", "import _bootstrap; from LexoraAI.config import ROOT; print(ROOT)")
    assert r.returncode == 0 and Path(r.stdout.strip()).resolve() == flat.resolve(), r.stderr[-500:]
