"""The documentation must stay consistent with the repository: diagrams in sync, links valid, README not mangled."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NAMES = ("overview", "indexing", "question")


def _mermaid_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.findall(r"```mermaid\n(.*?)```", text, re.S)]


def test_every_diagram_has_source_and_picture():
    for n in NAMES:
        assert (DOCS / f"{n}.mmd").is_file(), n
        assert (DOCS / f"{n}.png").stat().st_size > 10_000, n


def test_architecture_page_embeds_exactly_the_diagram_sources():
    blocks = _mermaid_blocks((DOCS / "architecture.md").read_text(encoding="utf-8"))
    sources = [(DOCS / f"{n}.mmd").read_text(encoding="utf-8").strip() for n in NAMES]
    assert blocks == sources


def test_diagram_text_is_well_formed():
    for n in NAMES:
        src = (DOCS / f"{n}.mmd").read_text(encoding="utf-8")
        assert src.startswith("flowchart ")
        for a, b in ("[]", "()", "{}"):
            assert src.count(a) == src.count(b), (n, a)


def test_modules_named_in_the_architecture_table_exist():
    text = (DOCS / "architecture.md").read_text(encoding="utf-8")
    table = text[text.index("## Where each step lives"):text.index("## Design decisions")]
    paths = re.findall(r"`([a-z_]+/[a-z_]+\.py)`", table)
    assert len(paths) >= 12
    for p in paths:
        assert (ROOT / p).is_file(), p


def test_relative_links_and_images_in_markdown_files_resolve():
    for md in [ROOT / "README.md", *DOCS.glob("*.md")]:
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (md.parent / target).exists(), f"{md.name}: broken link {target}"


def test_readme_results_section_is_not_duplicated():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("### Results") == 1
    for heading in ("#### Retrieval (answerable questions)", "#### Generation", "#### Manual grades", "#### Reading the numbers"):
        assert text.count(heading) == 1, heading
    assert text.count("### Limits of the evaluation") == 1


def test_readme_results_match_the_committed_summary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = (ROOT / "evaluation" / "results" / "summary.md").read_text(encoding="utf-8")
    first_line = summary.splitlines()[0]
    assert first_line in readme, "paste the current evaluation/results/summary.md into the README"
    for row in re.findall(r"^\| .*\|$", summary, re.M):
        assert row in readme, row


# ------------------------------------------------------------------------------ the README against the repository
def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_path_in_the_project_layout_exists():
    block = re.search(r"## Project layout.*?```text\n(.*?)```", _readme(), re.S).group(1)
    paths, stack = [], []
    for line in block.splitlines():
        m = re.match(r"^((?:│   |    )*)(?:├|└)── (\S+)", line)
        if not m:
            continue
        depth = len(m.group(1)) // 4
        stack[depth:] = [m.group(2).rstrip("/")]
        if stack[0] != "storage" or len(stack) == 1:  # storage/ contents are created at runtime
            paths.append("/".join(stack))
    assert len(paths) >= 40
    for p in paths:
        assert (ROOT / p).exists(), f"README layout lists {p}, which does not exist"


def test_every_python_module_run_in_the_readme_exists():
    modules = set(re.findall(r"python -m ([a-z_]+\.[a-z_]+)", _readme()))              # "scripts.<name>" placeholders do not match
    assert {"evaluation.evaluate", "evaluation.report", "scripts.smoke_test", "scripts.check_submission"} <= modules
    for m in modules:
        assert (ROOT / (m.replace(".", "/") + ".py")).is_file(), m


def test_the_makefile_targets_named_in_the_readme_exist():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("setup", "run", "test", "eval", "extended", "report", "check", "docker-up"):
        assert re.search(rf"^{target}:", make, re.M), target


def test_the_feature_table_in_the_readme_agrees_with_the_extended_features_page():
    def statuses(text: str) -> dict[int, bool]:
        return {int(m.group(1)): "not implemented" in m.group(2) for m in re.finditer(r"^\| (\d+) \| [^|]+\|([^|]+)\|", text, re.M)}
    readme, page = statuses(_readme()), statuses((DOCS / "EXTENDED_FEATURES.md").read_text(encoding="utf-8"))
    assert set(readme) == set(page) == set(range(1, 12))
    assert readme == page
    assert f"{11 - sum(readme.values())} of the eleven optional features" in _readme().replace("Eight", "8")


def test_the_results_page_and_the_readme_block_are_what_the_result_files_say():
    from evaluation import report
    assert report.main(["--check"]) == 0, "run python -m evaluation.report"


def test_documents_that_the_readme_promises_exist():
    for name in ("architecture.md", "RESULTS.md", "EXTENDED_FEATURES.md", "EVALUATION_ANALYSIS.md", "DOCKER_VERIFICATION.md", "UI_VERIFICATION.md"):
        assert (DOCS / name).is_file(), name
        assert f"docs/{name}" in _readme()


def test_the_readme_has_a_placeholder_only_in_the_ai_section():
    text = _readme()
    before = text.split("## AI assistance")[0]
    assert "TODO" not in before and "[Describe" not in before
