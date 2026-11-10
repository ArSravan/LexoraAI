"""Is the repository ready to hand in?   python -m scripts.check_submission

It checks what a reviewer would check and what is easy to forget, and prints one line per item. It changes nothing.
Exit code 0 only if there is no FAIL (a WARN is something to look at, not a blocker).

  * every dependency in requirements.txt is pinned with ==; the default seed is fixed
  * the one-command run exists (run.py, Makefile target, docker compose file)
  * the README has no leftover placeholder, links to its documents, and has a section on the use of AI assistance that is filled in
  * the evaluation results exist, the README shows the committed summary, and docs/RESULTS.md matches the result files
  * docs/DOCKER_VERIFICATION.md has a verification run filled in (otherwise Docker is still unverified on this machine)
  * the working tree is clean and everything is pushed
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from evaluation import report

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = ("TODO", "FIXME", "TBD", "_date, machine, Docker version_", "paste the output of")


@dataclass
class Item:
    level: str          # PASS | WARN | FAIL
    name: str
    detail: str = ""


def _read(root: Path, name: str) -> str:
    f = root / name
    return f.read_text(encoding="utf-8") if f.is_file() else ""


def check_pins(root: Path) -> list[Item]:
    lines = [l.strip() for l in _read(root, "requirements.txt").splitlines() if l.strip() and not l.strip().startswith("#")]
    loose = [l for l in lines if "==" not in l]
    out = [Item("FAIL", "requirements are pinned", "not pinned: " + ", ".join(loose)) if loose
           else Item("PASS", "requirements are pinned", f"{len(lines)} packages, all with ==")]
    seed = re.search(r"^\s*seed: int = (\d+)", _read(root, "config.py"), re.M)
    out.append(Item("PASS", "seed is fixed", f"default seed {seed.group(1)}") if seed else Item("FAIL", "seed is fixed", "no default seed in config.py"))
    return out


def check_one_command(root: Path) -> list[Item]:
    out = []
    out.append(Item("PASS" if (root / "run.py").is_file() else "FAIL", "python run.py exists"))
    make = _read(root, "Makefile")
    out.append(Item("PASS" if re.search(r"^run:", make, re.M) else "FAIL", "make run exists"))
    compose, docker = _read(root, "docker-compose.yml"), _read(root, "Dockerfile")
    out.append(Item("PASS" if compose and docker else "FAIL", "docker compose up --build files exist"))
    return out


def check_readme(root: Path) -> list[Item]:
    readme = _read(root, "README.md")
    out = []
    left = [p for p in PLACEHOLDERS if p in readme]
    out.append(Item("FAIL", "README has no placeholder", "found: " + ", ".join(left)) if left else Item("PASS", "README has no placeholder"))
    m = re.search(r"^## AI assistance\s*\n(.*?)(?=^## |\Z)", readme, re.S | re.M)
    body = m.group(1).strip() if m else ""
    if not m:
        out.append(Item("FAIL", "README states how AI assistance was used", "no '## AI assistance' section"))
    elif len(body) < 80 or "[Describe" in body:
        out.append(Item("FAIL", "README states how AI assistance was used", "the section is empty or still the template"))
    else:
        out.append(Item("PASS", "README states how AI assistance was used"))
    broken = [t for t in re.findall(r"\]\(([^)#\s]+)\)", readme) if not t.startswith(("http://", "https://", "mailto:")) and not (root / t).exists()]
    out.append(Item("FAIL", "README links resolve", "missing: " + ", ".join(broken)) if broken else Item("PASS", "README links resolve"))
    return out


def check_results(root: Path) -> list[Item]:
    res = root / "evaluation" / "results"
    out = []
    need = ["summary.md", "summary.json", "retrieval_by_question.csv", "generation_by_question.csv", "manual_grades.csv", "meta.json"]
    gone = [n for n in need if not (res / n).is_file()]
    out.append(Item("FAIL", "core evaluation results exist", "missing: " + ", ".join(gone)) if gone else Item("PASS", "core evaluation results exist"))
    extended = ["ocr_eval.md", "table_eval.md", "page_anomaly_eval.md", "ablation_tables.md", "latency_report.md", "bench_indexing.md", "diagnostics.md"]
    gone = [n for n in extended if not (res / n).is_file()]
    out.append(Item("WARN", "extended-feature results exist", "not run yet: " + ", ".join(gone)) if gone else Item("PASS", "extended-feature results exist"))
    readme_path = root / "README.md"
    if not (res / "summary.md").is_file():
        out.append(Item("FAIL", "README shows the committed summary", "no summary.md"))
    elif report.START not in _read(root, "README.md"):
        out.append(Item("FAIL", "README shows the committed summary", f"the README has no {report.START} block: add the two marker comments and run python -m evaluation.report"))
    elif not report.readme_is_current(readme_path, res):
        out.append(Item("FAIL", "README shows the committed summary", "the results block is out of date: run python -m evaluation.report"))
    else:
        out.append(Item("PASS", "README shows the committed summary"))
    current = _read(root, "docs/RESULTS.md") == report.build(res)
    out.append(Item("PASS", "docs/RESULTS.md matches the result files") if current
               else Item("FAIL", "docs/RESULTS.md matches the result files", "run: python -m evaluation.report"))
    return out


def check_docker_verification(root: Path) -> list[Item]:
    text = _read(root, "docs/DOCKER_VERIFICATION.md")
    if not text:
        return [Item("WARN", "Docker verification recorded", "docs/DOCKER_VERIFICATION.md is missing")]
    filled = "paste the output of" not in text and "_date, machine, Docker version_" not in text
    return [Item("PASS", "Docker verification recorded") if filled
            else Item("WARN", "Docker verification recorded", "the 'Your verification run' section is still the template: run it and fill it in")]


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def check_git(root: Path) -> list[Item]:
    dirty = _git(root, "status", "--porcelain")
    if dirty is None:
        return [Item("WARN", "git state", "not a git repository, or git is not installed")]
    out = [Item("FAIL", "working tree is clean", f"{len(dirty.splitlines())} changed or untracked file(s): commit them") if dirty
           else Item("PASS", "working tree is clean")]
    ahead = _git(root, "rev-list", "--count", "@{upstream}..HEAD")
    if ahead is None:
        out.append(Item("WARN", "everything is pushed", "no upstream branch to compare with"))
    else:
        out.append(Item("FAIL", "everything is pushed", f"{ahead} commit(s) not pushed") if ahead != "0" else Item("PASS", "everything is pushed"))
    n = _git(root, "rev-list", "--count", "HEAD")
    out.append(Item("PASS" if n and int(n) >= 20 else "WARN", "incremental history", f"{n} commits"))
    return out


def run_checks(root: Path = ROOT) -> list[Item]:
    return [i for fn in (check_pins, check_one_command, check_readme, check_results, check_docker_verification, check_git) for i in fn(root)]


def main(argv: list[str] | None = None) -> int:
    items = run_checks()
    for i in items:
        print(f"{i.level:<5} {i.name}" + (f"  -  {i.detail}" if i.detail else ""))
    fails, warns = sum(i.level == "FAIL" for i in items), sum(i.level == "WARN" for i in items)
    print(f"\n{len(items) - fails - warns} passed, {warns} warning(s), {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
