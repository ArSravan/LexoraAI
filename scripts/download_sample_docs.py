"""Download the public EU legislation used for the evaluation into sample_docs/  (EU law is free to reuse).

    python -m scripts.download_sample_docs

eur-lex.europa.eu blocks scripted clients, so this uses the Publications Office endpoint, which serves the
official PDF of a document (``Accept: application/pdf``). Files that are already there are kept.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "sample_docs"
FILES = {
    # the official Official Journal PDF/A of the GDPR (a Cellar item)
    "GDPR_Regulation_2016_679.pdf": "https://publications.europa.eu/resource/cellar/3e485e15-11bd-11e6-ba9a-01aa75ed71a1.0006.01/DOC_1",
    "ePrivacy_Directive_2002_58.pdf": "https://publications.europa.eu/resource/celex/32002L0058",
    "Data_Act_2023_2854.pdf": "https://publications.europa.eu/resource/celex/32023R2854",
}
MIN_BYTES = 50_000                 # a smaller file is an error page or a truncated download, not a regulation


def download(files: dict[str, str], out: Path, timeout: float = 180) -> list[str]:
    """Fetch every missing file; returns a line per file for the user. Nothing is written for a bad response."""
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for name, url in files.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size > MIN_BYTES:
            report.append(f"have {name}")
            continue
        r = requests.get(url, headers={"Accept": "application/pdf", "Accept-Language": "en", "User-Agent": "Mozilla/5.0"},
                         timeout=timeout)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError(f"{name}: the response is not a PDF")
        if len(r.content) <= MIN_BYTES:
            raise RuntimeError(f"{name}: the response is only {len(r.content)} bytes, too small for the real document")
        dest.write_bytes(r.content)
        report.append(f"downloaded {name} ({len(r.content) / 1e6:.1f} MB)")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help=f"target folder (default: {OUT})")
    args = ap.parse_args(argv)
    for line in download(FILES, args.out):
        print(line)


if __name__ == "__main__":
    main()
