"""The in-process scheduler must survive a failed first download and keep refreshing."""

from __future__ import annotations

import time
from types import SimpleNamespace

from backend.app import scheduler as sched
from pipeline import jobs


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_first_setup_is_retried_until_it_works_then_refreshes_on_a_timer(monkeypatch):
    calls = {"boot": 0, "refresh": 0}

    def boot(if_empty=False):
        calls["boot"] += 1
        if calls["boot"] < 3:
            raise RuntimeError("network down")

    def refresh():
        calls["refresh"] += 1
        if calls["refresh"] == 1:
            raise RuntimeError("one bad tick must not kill the loop")

    monkeypatch.setattr(jobs, "bootstrap", boot)
    monkeypatch.setattr(jobs, "refresh", refresh)
    monkeypatch.setattr(sched, "RETRY_SECONDS", 0.01)
    monkeypatch.setattr(sched, "get_settings", lambda: SimpleNamespace(refresh_minutes=0.0003))

    s = sched.Scheduler()
    s.start()
    try:
        assert wait_for(lambda: calls["boot"] == 3 and calls["refresh"] >= 2)
    finally:
        s.stop()
    assert calls["boot"] == 3  # stopped retrying once it succeeded
