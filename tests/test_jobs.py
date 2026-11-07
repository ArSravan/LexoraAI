import threading

import pytest

from services import jobs


def test_a_job_runs_in_the_background_and_reports_done():
    seen = {}

    def work(progress):
        progress(0.5, "halfway")
        seen["thread"] = threading.current_thread().name

    jid = jobs.start("index", work, {"case_id": "c1"})
    done = jobs.wait(jid)
    assert done["status"] == "done" and done["progress"] == 1.0 and done["message"] == "Done"
    assert done["kind"] == "index" and done["meta"] == {"case_id": "c1"} and done["finished"] >= done["started"]
    assert seen["thread"] == f"job-{jid}" and seen["thread"] != threading.current_thread().name


def test_progress_is_visible_while_the_job_runs():
    gate, reached = threading.Event(), threading.Event()

    def work(progress):
        progress(0.4, "extracting page 4/10")
        reached.set()
        gate.wait(5)

    jid = jobs.start("index", work)
    assert reached.wait(5)
    mid = jobs.get(jid)
    assert mid["status"] == "running" and mid["progress"] == 0.4 and mid["message"] == "extracting page 4/10"
    gate.set()
    assert jobs.wait(jid)["status"] == "done"


def test_progress_is_clamped_to_zero_one():
    seen, ready = {}, threading.Event()

    def work(progress):
        ready.wait(5)                                   # the job id is only known once start() has returned
        progress(7, "too much")
        seen["high"] = jobs.get(seen["jid"])["progress"]
        progress(-3, "too little")
        seen["low"] = jobs.get(seen["jid"])["progress"]

    seen["jid"] = jobs.start("t", work)
    ready.set()
    assert jobs.wait(seen["jid"])["status"] == "done"
    assert seen["high"] == 1.0 and seen["low"] == 0.0


def test_a_failing_job_reports_the_error():
    def boom(progress):
        raise RuntimeError("PDF is encrypted")
    done = jobs.wait(jobs.start("index", boom))
    assert done["status"] == "failed" and done["error"] == "PDF is encrypted" and "Failed: PDF is encrypted" in done["message"]
    assert done["finished"] is not None


def test_unknown_job_and_snapshots_are_copies():
    assert jobs.get("nope") is None
    jid = jobs.start("t", lambda p: None, {"a": 1})
    jobs.wait(jid)
    snap = jobs.get(jid)
    snap["status"] = "hacked"
    snap["meta"]["a"] = 2
    assert jobs.get(jid)["status"] == "done" and jobs.get(jid)["meta"] == {"a": 1}


def test_wait_times_out_for_a_job_that_never_finishes():
    gate = threading.Event()
    jid = jobs.start("t", lambda p: gate.wait(5))
    with pytest.raises(TimeoutError):
        jobs.wait(jid, timeout=0.1)
    gate.set()
    jobs.wait(jid)
