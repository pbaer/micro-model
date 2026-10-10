"""WorkerClient: a worker that dies must surface as an error, never as a request that waits forever (the chat page sat
on "loading" for days on 2026-10-09: the worker had died at spawn and the load call blocked in recv)."""

import threading
import time

import pytest

from slm.portal.services.worker import WorkerClient, WorkerDied, describe_exit


def _run_with_deadline(fn, seconds=30):
    out = {}

    def go():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"the call was still waiting after {seconds}s (a dead worker must not hang a request)"
    return out


def test_a_worker_that_dies_at_spawn_fails_the_call():
    import _worker_stub

    w = WorkerClient("unused", target=_worker_stub.dies_at_once)
    out = _run_with_deadline(lambda: w.call("load", slot="A"))
    assert isinstance(out.get("error"), WorkerDied) and "exit code 7" in str(out["error"])
    assert not w.alive() and w.proc is None, "reset, so the next request starts a fresh worker"


def test_a_worker_that_dies_mid_request_fails_the_call_and_the_stream():
    import _worker_stub

    w = WorkerClient("unused", target=_worker_stub.dies_mid_request)
    out = _run_with_deadline(lambda: w.call("load"))
    assert isinstance(out.get("error"), WorkerDied) and "exit code 3" in str(out["error"])
    out = _run_with_deadline(lambda: list(w.stream("generate")))
    assert out["value"] and out["value"][-1]["event"] == "error" and "exit code 3" in out["value"][-1]["error"]


def test_a_healthy_worker_still_answers_and_the_client_recovers_after_a_death():
    import _worker_stub

    w = WorkerClient("unused", target=_worker_stub.dies_at_once)
    _run_with_deadline(lambda: w.call("status"))
    w.target = _worker_stub.echoes  # the next request starts a fresh worker
    out = _run_with_deadline(lambda: w.call("status"))
    assert out.get("value") == {"echo": "status"}
    w.stop()


def test_idle_stop_never_runs_while_a_request_holds_the_lock():
    import _worker_stub

    w = WorkerClient("unused", idle_timeout_s=0.0, target=_worker_stub.echoes)
    w.call("status")
    time.sleep(0.05)
    with w.lock:  # a request in flight
        assert w.maybe_idle_stop() is False and w.alive()
    assert w.maybe_idle_stop() is True and not w.alive()


@pytest.mark.parametrize("code,needle", [(0xC0000142 - (1 << 32), "console that has since closed"), (7, "exit code 7"), (None, "still running")])
def test_describe_exit(code, needle):
    assert needle in describe_exit(code)
