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
