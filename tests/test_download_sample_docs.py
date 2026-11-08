import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from scripts import download_sample_docs as dl

PDF = b"%PDF-1.7\n" + b"x" * 60_000


@pytest.fixture
def server():
    seen = []
    routes = {"/good": (200, PDF), "/html": (200, b"<html>" + b"x" * 60_000), "/tiny": (200, b"%PDF-1.7 tiny"),
              "/gone": (404, b"not found")}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Accept"), self.headers.get("Accept-Language")))
            code, body = routes.get(self.path, (404, b""))
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    srv.seen = seen
    yield srv
    srv.shutdown()


def test_a_missing_file_is_downloaded_and_an_existing_one_is_kept(server, tmp_path):
    files = {"A.pdf": f"{server.url}/good", "B.pdf": f"{server.url}/good"}
    (tmp_path / "B.pdf").write_bytes(PDF)
    report = dl.download(files, tmp_path)
    assert report[0].startswith("downloaded A.pdf") and report[1] == "have B.pdf"
    assert (tmp_path / "A.pdf").read_bytes() == PDF and [p for p, *_ in server.seen] == ["/good"]      # B was not fetched
    assert server.seen[0][1:] == ("application/pdf", "en")


def test_the_output_folder_is_created(server, tmp_path):
    dl.download({"A.pdf": f"{server.url}/good"}, tmp_path / "new" / "sample_docs")
    assert (tmp_path / "new" / "sample_docs" / "A.pdf").is_file()


@pytest.mark.parametrize("path,error", [("/html", "not a PDF"), ("/tiny", "too small")])
def test_bad_responses_are_refused_and_nothing_is_written(server, tmp_path, path, error):
    with pytest.raises(RuntimeError, match=error):
        dl.download({"A.pdf": f"{server.url}{path}"}, tmp_path)
    assert not (tmp_path / "A.pdf").exists()


def test_an_http_error_is_raised(server, tmp_path):
    with pytest.raises(requests.HTTPError):
        dl.download({"A.pdf": f"{server.url}/gone"}, tmp_path)


def test_a_truncated_earlier_download_is_fetched_again(server, tmp_path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.7 half")
    assert dl.download({"A.pdf": f"{server.url}/good"}, tmp_path)[0].startswith("downloaded")


def test_the_real_bundle_is_the_three_eu_documents_the_questions_refer_to():
    assert set(dl.FILES) == {"GDPR_Regulation_2016_679.pdf", "ePrivacy_Directive_2002_58.pdf", "Data_Act_2023_2854.pdf"}
    assert all(u.startswith("https://publications.europa.eu/") for u in dl.FILES.values())
    from evaluation.dataset import DEV_SET, TEST_SET, load_dataset
    used = {e.document for p in (TEST_SET, DEV_SET) for q in load_dataset(p).questions for e in q.expected_evidence}
    assert used <= set(dl.FILES)


def test_help_prints_usage_and_downloads_nothing(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "download", lambda *a, **k: pytest.fail("--help must not download anything"))
    with pytest.raises(SystemExit) as e:
        dl.main(["--help"])
    assert e.value.code == 0 and "--out" in capsys.readouterr().out


def test_the_target_folder_can_be_chosen(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dl, "FILES", {"A.pdf": f"{server.url}/good"})
    dl.main(["--out", str(tmp_path / "docs")])
    assert (tmp_path / "docs" / "A.pdf").is_file() and "downloaded A.pdf" in capsys.readouterr().out
