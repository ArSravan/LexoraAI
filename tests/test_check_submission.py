import subprocess
from pathlib import Path

import pytest

from evaluation import report
from scripts import check_submission as cs

SUMMARY = "Run 2026-11-08\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
NOTES = "## Notes\n\nSee the documents in docs/ for the design and the results.\n"


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"], "HOME": str(root)})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "evaluation" / "results").mkdir(parents=True)
    (r / "docs").mkdir()
    (r / "requirements.txt").write_text("# pinned\nrequests==2.34.2\nnumpy==2.5.4\n")
    (r / "config.py").write_text("class Settings:\n    seed: int = 42\n")
    (r / "run.py").write_text("")
    (r / "Makefile").write_text("run:\n\tpython run.py\n")
    (r / "Dockerfile").write_text("FROM python\n")
    (r / "docker-compose.yml").write_text("services: {}\n")
    (r / "docs" / "DOCKER_VERIFICATION.md").write_text("Run on: 2026-11-10, Windows 11, Docker 29\n")
    res = r / "evaluation" / "results"
    for n in ("summary.json", "retrieval_by_question.csv", "generation_by_question.csv", "manual_grades.csv", "meta.json"):
        (res / n).write_text("{}")
    (res / "summary.md").write_text(SUMMARY)
    (r / "README.md").write_text(f"# Project\n\nSee [the results](docs/RESULTS.md).\n\n{report.START}\n{report.END}\n\n" + NOTES)
    report.update_readme(r / "README.md", res)
    (r / "docs" / "RESULTS.md").write_text(report.build(res))
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "x")
    return r


def levels(root):
    return {i.name: i.level for i in cs.run_checks(root)}


def test_a_ready_repository_has_no_failure(repo):
    lv = levels(repo)
    assert "FAIL" not in lv.values()
    assert lv["requirements are pinned"] == lv["seed is fixed"] == "PASS"
    assert lv["working tree is clean"] == "PASS" and lv["incremental history"] == "WARN" and lv["extended-feature results exist"] == "WARN"


def test_an_unpinned_dependency_and_a_missing_seed_fail(repo):
    (repo / "requirements.txt").write_text("requests>=2\nnumpy==2.5.4\n")
    (repo / "config.py").write_text("x = 1\n")
    lv = levels(repo)
    assert lv["requirements are pinned"] == "FAIL" and lv["seed is fixed"] == "FAIL"


def test_missing_one_command_files_fail(repo):
    (repo / "run.py").unlink()
    (repo / "Makefile").write_text("all:\n")
    lv = levels(repo)
    assert lv["python run.py exists"] == "FAIL" and lv["make run exists"] == "FAIL"


@pytest.mark.parametrize("readme_tail, name", [
    ("\n## Notes\n\n" + "See the documents in docs/ for the design. " * 2 + "\nTODO check\n", "README has no placeholder"),
    ("\n" + NOTES + "\nSee [nothing](docs/NOPE.md).\n", "README links resolve"),
])
def test_readme_problems_are_found(repo, readme_tail, name):
    (repo / "README.md").write_text(f"# P\n\n[results](docs/RESULTS.md)\n\n{report.START}\n{report.END}\n" + readme_tail)
    report.update_readme(repo / "README.md", repo / "evaluation" / "results")
    assert levels(repo)[name] == "FAIL"


def test_stale_results_in_the_readme_or_the_results_page_fail(repo):
    (repo / "evaluation" / "results" / "summary.md").write_text(SUMMARY + "| 3 | 4 |\n")
    lv = levels(repo)
    assert lv["README shows the committed summary"] == "FAIL" and lv["docs/RESULTS.md matches the result files"] == "FAIL"


def test_a_readme_without_the_markers_fails_with_a_hint(repo):
    (repo / "README.md").write_text("# P\n\n[r](docs/RESULTS.md)\n\n" + NOTES)
    item = next(i for i in cs.run_checks(repo) if i.name == "README shows the committed summary")
    assert item.level == "FAIL" and "RESULTS:START" in item.detail


def test_missing_core_results_fail(repo):
    (repo / "evaluation" / "results" / "manual_grades.csv").unlink()
    assert levels(repo)["core evaluation results exist"] == "FAIL"


def test_the_docker_verification_template_is_a_warning_not_a_failure(repo):
    (repo / "docs" / "DOCKER_VERIFICATION.md").write_text("paste the output of: python -m scripts.smoke_test\n")
    assert levels(repo)["Docker verification recorded"] == "WARN"
    (repo / "docs" / "DOCKER_VERIFICATION.md").unlink()
    assert levels(repo)["Docker verification recorded"] == "WARN"


def test_dirty_trees_and_unpushed_commits_fail(repo, tmp_path):
    (repo / "new.txt").write_text("x")
    assert levels(repo)["working tree is clean"] == "FAIL"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "y")
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "origin", str(bare))
    git(repo, "push", "-q", "-u", "origin", "main")
    assert levels(repo)["everything is pushed"] == "PASS"
    (repo / "more.txt").write_text("z")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "z")
    assert levels(repo)["everything is pushed"] == "FAIL"


def test_not_a_git_repository_is_a_warning(repo):
    import os
    import shutil
    import stat

    def make_writable_and_retry(func, path, _exc):      # git's object files are read-only; Windows won't delete those
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(repo / ".git", onexc=make_writable_and_retry)
    assert levels(repo)["git state"] == "WARN"
    
def test_main_exit_code_follows_failures(monkeypatch, capsys):
    monkeypatch.setattr(cs, "run_checks", lambda: [cs.Item("PASS", "a"), cs.Item("WARN", "b", "d")])
    assert cs.main() == 0 and "1 passed, 1 warning(s), 0 failure(s)" in capsys.readouterr().out
    monkeypatch.setattr(cs, "run_checks", lambda: [cs.Item("FAIL", "a", "bad")])
    assert cs.main() == 1 and "1 failure" in capsys.readouterr().out
