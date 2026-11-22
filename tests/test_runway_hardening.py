"""Runway submission/billing hardening and the JavaScript bridge allowlist.

Covers lost responses, ambiguous server replies, persistence failures, restart recovery,
conservative (unknown) charges, budget exhaustion, explicit resolution, and bridge exposure.
All HTTP goes through httpx.MockTransport; nothing contacts Runway.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

from avatar_v2 import cloud_ledger, desktop, render_queue, secret_store
from avatar_v2.composer import build_job
from avatar_v2.desktop import DesktopAPI
from avatar_v2.errors import classify_exception
from avatar_v2.js_bridge import JS_API_METHODS, JsBridge
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.providers import runway as runway_api
from avatar_v2.providers.runway import (
    RunwayAPIError,
    RunwayBudgetError,
    RunwayPersistenceError,
    RunwayProvider,
    RunwayUncertainSubmission,
)
from avatar_v2.render_queue import QueuedRender, RenderQueue

KEY = "rw_test_key_0123456789abcdef"
TASK_ID = "11111111-2222-3333-4444-555555555555"
CREATE = ("POST", "/v1/text_to_image")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(secret_store, "SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    monkeypatch.setenv("RUNWAYML_API_SECRET", KEY)


def _png(path: Path) -> Path:
    Image.new("RGB", (96, 128), (180, 120, 90)).save(path)
    return path


def _job(tmp_path, *, cap=150):
    avatar = AvatarManifest.model_validate(
        {"avatar_id": "p", "display_name": "P", "subjects": [{"id": "p", "kind": "synthetic"}],
         "identity_refs": [str(_png(tmp_path / "id.png"))]}
    )
    job = build_job(avatar, ShotSpec.model_validate({"shot_id": "s", "user_prompt": "smile", "engine_preference": "runway"}))
    spec = {"version": 1, "model": "gen4_image", "params": {"prompt": "@presenter smiles", "ratio": "1080:1440", "seed": 1},
            "media": {"references": [{"asset": "IDENTITY_REF_1", "tag": "presenter"}]},
            "estimate": {"credits": 8}, "max_credits": cap}
    job.asset_map["RUNWAY_SPEC"] = json.dumps(spec)
    return job


class Server:
    """Scripted Runway: per (method, path) a list of responses, an exception, or a callable."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, request):
        key = (request.method, request.url.path) if request.url.host == "api.dev.runwayml.com" else ("GET", request.url.host)
        self.calls.append(key)
        handler = self.routes.get(key)
        if isinstance(handler, list):
            handler = handler.pop(0) if len(handler) > 1 else handler[0]
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(request)
        return handler or httpx.Response(404)

    def count(self, key):
        return self.calls.count(key)


def _provider(tmp_path, server, **hooks):
    config = ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE, output_dir=str(tmp_path / "out"), timeout_s=600)
    provider = RunwayProvider(config, transport=httpx.MockTransport(server), sleep=lambda _s: None)
    for name, hook in hooks.items():
        setattr(provider, name, hook)
    return provider


def _succeeding_task(cost=8):
    body = {"id": TASK_ID, "status": "SUCCEEDED", "output": ["https://cdn.example/o.png"]}
    if cost is not None:
        body["cost"] = {"credits": cost}
    return {
        ("GET", f"/v1/tasks/{TASK_ID}"): httpx.Response(200, json=body),
        ("GET", "cdn.example"): httpx.Response(200, content=b"img", headers={"content-type": "image/png"}),
    }


# --------------------------------------------------------------------------- lost and ambiguous responses


@pytest.mark.parametrize(
    "reply",
    [
        httpx.ReadTimeout("response lost"),
        httpx.RemoteProtocolError("connection dropped mid-response"),
        httpx.Response(502, text="bad gateway"),
        httpx.Response(500, json={"error": "internal"}),
        httpx.Response(200, json={"estimatedCost": {"credits": 8}}),  # success without an id
        httpx.Response(200, text="<html>not json</html>"),
    ],
)
def test_ambiguous_create_outcomes_are_uncertain_never_unbilled_and_never_retried(tmp_path, reply):
    server = Server({CREATE: reply})
    with pytest.raises(RunwayUncertainSubmission) as caught:
        _provider(tmp_path, server).render("w", _job(tmp_path))
    receipt = caught.value.receipt
    assert receipt["submission"]["state"] == "uncertain"
    assert receipt["charge_status"] == "unknown" and receipt["credits_charged"] is None
    assert server.count(CREATE) == 1  # no automatic resubmission
    assert classify_exception(caught.value).category == "cloud_submission_uncertain"


@pytest.mark.parametrize(
    "reply,expected_state",
    [
        (httpx.ConnectError("refused"), "not_sent"),
        (httpx.ConnectTimeout("no route"), "not_sent"),
        (httpx.Response(422, json={"error": "Validation failed"}), "rejected"),
        (httpx.Response(401, json={"error": "bad key"}), "rejected"),
    ],
)
def test_provably_failed_creates_are_definitely_unbilled(tmp_path, reply, expected_state):
    server = Server({CREATE: reply})
    with pytest.raises(RunwayAPIError) as caught:
        _provider(tmp_path, server).render("w", _job(tmp_path))
    receipt = caught.value.receipt
    assert receipt["submission"]["state"] == expected_state
    assert receipt["charge_status"] == "none" and receipt["credits_charged"] == 0
    assert server.count(CREATE) == 1


def test_rate_limits_are_retried_because_no_task_exists_then_reported_as_rejected(tmp_path):
    server = Server({CREATE: [httpx.Response(429), httpx.Response(429), httpx.Response(429), httpx.Response(429)]})
    with pytest.raises(RunwayAPIError) as caught:
        _provider(tmp_path, server).render("w", _job(tmp_path))
    assert server.count(CREATE) == 4 and caught.value.receipt["submission"]["state"] == "rejected"


# --------------------------------------------------------------------------- persistence failures


def test_submission_is_not_sent_when_its_intent_cannot_be_recorded(tmp_path):
    server = Server({CREATE: httpx.Response(200, json={"id": TASK_ID})})

    def failing(_receipt):
        raise OSError("disk full")

    with pytest.raises(RunwayPersistenceError) as caught:
        _provider(tmp_path, server, on_submission_started=failing).render("w", _job(tmp_path))
    assert server.count(CREATE) == 0
    assert caught.value.receipt["submission"]["state"] == "not_sent"
    assert caught.value.receipt["charge_status"] == "none"


def test_unpersistable_task_id_is_cancelled_and_its_charge_kept_unknown(tmp_path):
    server = Server({CREATE: httpx.Response(200, json={"id": TASK_ID, "estimatedCost": {"credits": 8}}),
                     ("DELETE", f"/v1/tasks/{TASK_ID}"): httpx.Response(204)})

    def failing(_task_id, _receipt):
        raise OSError("read-only filesystem")

    with pytest.raises(RunwayPersistenceError) as caught:
        _provider(tmp_path, server, on_task_created=failing).render("w", _job(tmp_path))
    receipt = caught.value.receipt
    assert server.count(("DELETE", f"/v1/tasks/{TASK_ID}")) == 1
    assert receipt["submission"]["state"] == "cancelled_unpersisted"
    assert receipt["charge_status"] == "unknown" and receipt["task_id"] == TASK_ID
    assert classify_exception(caught.value).category == "cloud_persistence"


def test_unpersistable_task_that_cannot_be_cancelled_is_finished_in_memory(tmp_path):
    routes = {CREATE: httpx.Response(200, json={"id": TASK_ID}),
              ("DELETE", f"/v1/tasks/{TASK_ID}"): httpx.Response(500), **_succeeding_task(cost=8)}
    server = Server(routes)

    def failing(_task_id, _receipt):
        raise OSError("read-only filesystem")

    result = _provider(tmp_path, server, on_task_created=failing).render("w", _job(tmp_path))
    assert result["cloud_receipt"]["submission"]["state"] == "accepted_unpersisted"
    assert "persistence_warning" in result["cloud_receipt"] and Path(result["outputs"][0]).is_file()


def test_charges_are_actual_only_when_runway_reports_them(tmp_path):
    with_cost = _provider(tmp_path, Server({CREATE: httpx.Response(200, json={"id": TASK_ID}), **_succeeding_task(cost=8)}))
    assert with_cost.render("w", _job(tmp_path))["cloud_receipt"]["charge_status"] == "actual"
    without_cost = _provider(tmp_path, Server({CREATE: httpx.Response(200, json={"id": TASK_ID}), **_succeeding_task(cost=None)}))
    receipt = without_cost.render("w", _job(tmp_path))["cloud_receipt"]
    assert receipt["charge_status"] == "unknown" and receipt["credits_charged"] is None


def test_budget_cancelled_task_keeps_its_estimate_reserved(tmp_path):
    server = Server({CREATE: httpx.Response(200, json={"id": TASK_ID, "estimatedCost": {"credits": 500}}),
                     ("DELETE", f"/v1/tasks/{TASK_ID}"): httpx.Response(204)})
    with pytest.raises(RunwayBudgetError) as caught:
        _provider(tmp_path, server).render("w", _job(tmp_path, cap=100))
    assert caught.value.receipt["charge_status"] == "unknown" and caught.value.receipt["credits_charged"] is None


# --------------------------------------------------------------------------- queue integration and restart recovery


class Memory:
    def maybe_cleanup(self, *_a, **k):
        return k

    def start_trace(self, _l):
        pass

    def stop_trace(self):
        return {}

    def update_settings(self, **_k):
        pass


def _shell(tmp_path, items=()):
    queue = RenderQueue.__new__(RenderQueue)
    queue.work_dir = tmp_path / "queue"
    queue.work_dir.mkdir(parents=True, exist_ok=True)
    queue.history_dir = tmp_path / "history"
    queue.history_dir.mkdir(parents=True, exist_ok=True)
    queue._ledger_path = tmp_path / "ledger.jsonl"
    queue._items = list(items)
    queue._lock = threading.RLock()
    queue._wake = threading.Event()
    queue._stop = threading.Event()
    queue._active_provider = None
    queue._active_id = None
    queue.candidate_qa_runner = None
    queue.memory_manager = Memory()
    return queue


def _item(tmp_path, **updates):
    data = {"job": _job(tmp_path), "workflow": "runway:still:gen4_image", "profile": "cloud_runway",
            "provider_config": ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE, output_dir=str(tmp_path / "out")),
            "runtime": {"estimate_credits": 8, "model": "gen4_image"}, "label": "master"}
    data.update(updates)
    return QueuedRender(**data)


def _run_worker(queue, item, timeout=10):
    worker = threading.Thread(target=queue._worker, daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while item.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.05)
    queue._stop.set()
    queue._wake.set()
    worker.join(timeout=5)


def test_worker_records_lost_responses_as_uncertain_reserves_and_never_requeues(tmp_path, monkeypatch):
    seen_on_disk = {}

    class Provider(RunwayProvider):
        def __init__(self, config):
            server = Server({CREATE: httpx.ReadTimeout("lost")})
            super().__init__(config, transport=httpx.MockTransport(server), sleep=lambda _s: None)

    original = render_queue.RenderQueue._record_cloud_submission

    def spy(self, item, receipt):
        original(self, item, receipt)
        seen_on_disk.update(json.loads((self.work_dir / "items" / f"{item.id}.json").read_text()))

    monkeypatch.setattr(render_queue, "RunwayProvider", Provider)
    monkeypatch.setattr(render_queue.RenderQueue, "_record_cloud_submission", spy)
    item = _item(tmp_path)
    queue = _shell(tmp_path, [item])
    _run_worker(queue, item)
    assert seen_on_disk["cloud_receipt"]["submission"]["state"] == "sending"  # durable before the request left
    assert item.status == "error" and item.error_category == "cloud_submission_uncertain"
    assert [i.status for i in queue._items] == ["error"]  # nothing re-queued
    unresolved = cloud_ledger.unresolved(queue._ledger_path)
    assert len(unresolved) == 1 and unresolved[0]["reserved_credits"] == 8
    assert unresolved[0]["key"].startswith("req:") and unresolved[0]["status"] == "UNCERTAIN"


def test_restart_after_sending_is_uncertain_logged_once_and_retired(tmp_path):
    queue = _shell(tmp_path)
    sending = _item(tmp_path, status="running", cloud_receipt={"submission": {"state": "sending", "request_key": "req:abc"}})
    items_dir = queue.work_dir / "items"
    items_dir.mkdir()
    (items_dir / f"{sending.id}.json").write_text(json.dumps(queue._item_to_dict(sending)))
    queue._restore()
    restored = queue._items[0]
    assert restored.error_category == "cloud_submission_uncertain"
    assert not (items_dir / f"{sending.id}.json").exists()  # stale file retired; it cannot reappear
    assert (queue.history_dir / f"{sending.id}.json").is_file()
    assert cloud_ledger.task_states(queue._ledger_path)["req:abc"]["status"] == "UNCERTAIN"
    queue._items = []
    queue._restore()
    assert queue._items == []


def test_durable_record_failure_propagates_so_the_send_is_aborted(tmp_path, monkeypatch):
    queue = _shell(tmp_path)
    item = _item(tmp_path, status="running")

    def broken(_path, _text):
        raise OSError("no space left on device")

    monkeypatch.setattr(render_queue, "write_durable", broken)
    with pytest.raises(OSError):
        queue._record_cloud_submission(item, {"submission": {"state": "sending"}})
    with pytest.raises(OSError):
        queue._record_cloud_task(item, TASK_ID, {"model": "gen4_image"})


def test_durable_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "items" / "x.json"
    render_queue.write_durable(target, '{"a": 1}')
    monkeypatch.setattr(render_queue.os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("crash during rename")))
    with pytest.raises(OSError):
        render_queue.write_durable(target, '{"a": 2}')
    assert json.loads(target.read_text()) == {"a": 1}  # never a half-written record


# --------------------------------------------------------------------------- uncertain billing and budgets


def test_ledger_counts_unknown_charges_at_their_estimate_until_resolved(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    cloud_ledger.record(ledger, {"request_key": "req:1", "status": "UNCERTAIN", "charge_status": "unknown", "estimate_credits": 8})
    cloud_ledger.record(ledger, {"task_id": "t-cancel", "status": "CREATED", "api_estimated_credits": 60})
    cloud_ledger.record(ledger, {"task_id": "t-cancel", "status": "CANCELLED", "charge_status": "unknown", "credits_charged": None})
    cloud_ledger.record(ledger, {"task_id": "t-fail", "status": "FAILED", "charge_status": "actual", "credits_charged": 8})
    summary = cloud_ledger.spend_summary(ledger)
    assert summary["today_charged"] == 8 and summary["today_unconfirmed"] == 68 and summary["unresolved"] == 2
    cloud_ledger.record(ledger, {"request_key": "req:1", "status": "RESOLVED", "charge_status": "none", "credits_charged": 0})
    cloud_ledger.record(ledger, {"task_id": "t-cancel", "status": "RESOLVED", "charge_status": "actual", "credits_charged": 2})
    summary = cloud_ledger.spend_summary(ledger)
    assert summary["today_unconfirmed"] == 0 and summary["today_charged"] == 10 and summary["unresolved"] == 0


@pytest.fixture
def api(tmp_path, monkeypatch):
    for name, value in {"CONFIG_PATH": tmp_path / "desktop.json", "RUNS_DIR": tmp_path / "runs",
                        "HISTORY_DIR": tmp_path / "hist", "CLOUD_LEDGER_PATH": tmp_path / "cloud_ledger.jsonl",
                        "RUNWAY_AVATARS_PATH": tmp_path / "ra.json", "LESSONS_PATH": tmp_path / "lessons.json",
                        "VOICEOVER_DIR": tmp_path / "vo"}.items():
        monkeypatch.setattr(desktop, name, value)
    (tmp_path / "hist").mkdir()
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)
    instance = DesktopAPI.__new__(DesktopAPI)

    class Queue:
        def __init__(self):
            self.items, self.memory_manager = [], Memory()

        def add_many(self, items):
            self.items.extend(items)
            return [i.id for i in items]

        def list(self):
            return [i.public() for i in self.items]

    instance.queue = Queue()
    instance.save_settings({"runway_enabled": True, "runway_upload_consent": True, "runway_daily_credit_cap": 20})
    instance.runway_save_key(KEY)
    return instance


def _payload(tmp_path):
    return {
        "settings": {},
        "avatar": {"avatar_id": "p", "display_name": "P", "sex": "female", "apparent_age_years": 26, "subject_kind": "synthetic",
                   "age_verified_18_plus": True, "consent_confirmed": False, "identity_refs": [str(_png(tmp_path / "f.png"))]},
        "shot": {"shot_id": "s", "content_class": "general", "user_prompt": "smile", "seed": 41001, "engine_preference": "runway",
                 "camera": {}, "wind": {}, "references": {}},
        "runway": {"task": "still", "ratio": "1080:1440"},
        "confirm_cloud_upload": True, "confirmed_estimate_credits": 8,
    }


def test_unresolved_charges_exhaust_budget_and_require_explicit_acknowledgement(api, tmp_path):
    cloud_ledger.record(desktop.CLOUD_LEDGER_PATH, {"request_key": "req:lost", "status": "UNCERTAIN", "charge_status": "unknown",
                                                    "estimate_credits": 8, "label": "lost master"})
    estimate = api.runway_estimate(_payload(tmp_path))
    assert estimate["unresolved"] == {"count": 1, "credits": 8.0}
    refused = api.queue_runway(_payload(tmp_path))
    assert refused["ok"] is False and "unknown charges" in refused["error"]
    queued = api.queue_runway({**_payload(tmp_path), "acknowledge_unresolved": True})
    assert queued["ok"] is True  # 8 unconfirmed + 8 new <= 20
    exhausted = api.queue_runway({**_payload(tmp_path), "acknowledge_unresolved": True})
    assert exhausted["ok"] is False and "daily cap of 20" in exhausted["error"]  # 8 + 8 queued + 8 > 20


def test_resolution_settles_unknown_charges_without_resubmitting(api, tmp_path):
    cloud_ledger.record(desktop.CLOUD_LEDGER_PATH, {"request_key": "req:lost", "status": "UNCERTAIN", "charge_status": "unknown",
                                                    "estimate_credits": 8})
    assert [i["key"] for i in api.runway_unresolved()["items"]] == ["req:lost"]
    assert api.runway_resolve_charge({"key": "req:lost", "resolution": "billed"})["ok"] is False
    assert api.runway_resolve_charge({"key": "req:lost", "resolution": "billed", "credits": 8})["credits_charged"] == 8
    assert api.runway_unresolved()["items"] == [] and api.queue.items == []
    assert api.runway_resolve_charge({"key": "req:lost", "resolution": "not_billed"})["ok"] is False  # already settled


def test_reconcile_downloads_an_accepted_task_found_later(api, tmp_path, monkeypatch):
    cloud_ledger.record(desktop.CLOUD_LEDGER_PATH, {"request_key": "req:lost", "status": "UNCERTAIN", "charge_status": "unknown",
                                                    "estimate_credits": 8, "label": "lost master"})
    server = Server(_succeeding_task(cost=8))
    original_init = RunwayProvider.__init__

    def patched(self, config, **kwargs):
        original_init(self, config, transport=httpx.MockTransport(server), sleep=lambda _s: None)

    monkeypatch.setattr(RunwayProvider, "__init__", patched)
    result = api.runway_reconcile_task({"key": "req:lost", "task_id": TASK_ID})
    assert result["ok"] is True and result["status"] == "SUCCEEDED" and Path(result["outputs"][0]).is_file()
    assert ("POST", "/v1/text_to_image") not in server.calls  # nothing resubmitted
    assert api.runway_unresolved()["items"] == []
    assert cloud_ledger.spend_summary(desktop.CLOUD_LEDGER_PATH)["today_charged"] == 8


def test_lessons_never_requeue_segments_with_uncertain_submissions(api, tmp_path):
    lesson = {"id": "intro", "title": "Intro", "runway_avatar_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
              "segments": [{"id": "s1", "mode": "script", "script": "Welcome."}]}
    api.lesson_save({"lesson": lesson})
    store = api._lessons()
    input_hash = store.plan("intro", {})[0]["input_hash"]
    store.record_render("intro", "s1", {"status": "queued", "queue_id": "q1", "input_hash": input_hash})
    (desktop.HISTORY_DIR / "q1.json").write_text(json.dumps({"status": "error", "error_category": "cloud_submission_uncertain"}))
    plan = api.lesson_plan("intro")
    assert plan["rows"][0]["state"] == "unresolved" and plan["to_render"] == []


# --------------------------------------------------------------------------- JavaScript bridge


def _pywebview_exposure(obj, base="", seen=None):
    """Mirror of pywebview 5.x ``get_functions`` (nested in webview.util.inject_pywebview)."""
    import inspect

    seen = [] if seen is None else seen
    if any(item is obj for item in seen):
        return set()
    seen.append(obj)
    found = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name)
        full = f"{base}.{name}" if base else name
        if inspect.ismethod(attr):
            found.add(full)
        elif inspect.isclass(attr) or (isinstance(attr, object) and not callable(attr) and hasattr(attr, "__module__")):
            found |= _pywebview_exposure(attr, full, seen)
    return found


def test_mirror_matches_the_installed_pywebview_rules():
    import webview.util

    source = Path(webview.util.__file__).read_text(encoding="utf-8")
    for rule in ("for name in dir(obj):", "if name.startswith('_'):", "if inspect.ismethod(attr):",
                 "not callable(attr) and hasattr(attr, \"__module__\")"):
        assert rule in source, f"pywebview exposure rule changed: {rule}"


def test_bridge_exposes_exactly_the_allowlist_and_no_internal_objects(tmp_path):
    class Full(DesktopAPI):
        def __init__(self):
            self.work_dir = tmp_path  # stands in for queue/library internals that hold Path objects

    leaky = _pywebview_exposure(Full())
    assert "work_dir.unlink" in leaky and "provider_profile_delete" in leaky  # what passing DesktopAPI exposed
    exposed = _pywebview_exposure(JsBridge(Full()))
    assert exposed == set(JS_API_METHODS)
    assert not any("." in name for name in exposed)


def test_bridge_allowlist_matches_every_ui_call_and_forwards_arguments():
    html = Path(desktop.__file__).with_name("ui").joinpath("index.html").read_text(encoding="utf-8")
    used = set(re.findall(r"pywebview\.api\.([a-z_][a-z0-9_]*)", html)) | set(re.findall(r"\bapi\.([a-z_][a-z0-9_]*)\(", html))
    for block in re.findall(r"for\(const name of \[([^\]]*)\]\)await probe", html):
        used |= {n.strip().strip("'\"") for n in block.split(",") if n.strip()}
    assert used == set(JS_API_METHODS)

    class Recorder:
        def __getattr__(self, name):
            return lambda *args: (name, args)

    bridge = JsBridge(Recorder())
    assert bridge.queue_runway({"a": 1}) == ("queue_runway", ({"a": 1},))
    with pytest.raises(AttributeError):
        bridge.queue  # internals are not reachable
    with pytest.raises(AttributeError):
        JsBridge(object())  # a bridge over an incomplete API fails fast
