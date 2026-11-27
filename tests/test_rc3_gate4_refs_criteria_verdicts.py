"""Adversarial RC3 Gate 4 tests: explicit paid provider choice, locked multi-ref integrity on video,
multi-clause/multi-sentence requirement extraction, truthful per-output inspection labels,
end-to-end character profile asset_fingerprint preservation, durable staged pins, and zero-network idle UI.

Directly reproduces Astra RC2 independent recheck findings:
- RC-04: studio_price / studio_go("move") missing engine silently defaulting to paid Runway
- RC-06: Wan video with starter + locked back_body dropping locked ref while claiming it in verification_record
- RC-07: "Exactly two cups and three bowls." bleeding count; multi-sentence prompt dropping second sentence
- RC-09: Uninspected alternate output inheriting job-level "Requirements verified" in UI & backend
- RC-19: UI dropping asset_fingerprint on Character Profile load/submit; durable staged pins & release on finish
- RC-22: Zero automatic stRefreshBalance() on active-job transitions
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import avatar_v2.desktop as desktop
from avatar_v2.intent_interpreter import interpret_natural_language
from avatar_v2.reference_picker import LocalReferenceManager


def _make_png(path: Path, color: tuple[int, int, int] = (120, 150, 190)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=color).save(path, format="PNG")
    return path


def _isolated_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[desktop.DesktopAPI, Path]:
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    return desktop.DesktopAPI(), app_home


# ---------------------------------------------------------------------------
# RC-04: Missing engine on studio_price / studio_go("move") MUST refuse paid fallback
# ---------------------------------------------------------------------------


def _stub_local_engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    import avatar_v2.local_engine as le

    monkeypatch.setattr(
        le,
        "inspect",
        lambda settings: {
            "installed": True,
            "missing_models": [],
            "comfyui_dir": "/tmp/fake_comfy",
            "workflow_present": True,
            "video_models": {
                "a14b": {"id": "a14b", "label": "Wan 2.2 A14B", "minutes": 6, "missing_models": [], "installed": True},
                "5b": {"id": "5b", "label": "Wan 2.2 5B", "minutes": 5, "missing_models": [], "installed": True},
            },
        },
    )
    monkeypatch.setattr(
        le,
        "video_model_info",
        lambda settings: {
            "a14b": {"id": "a14b", "label": "Wan 2.2 A14B", "minutes": 6, "missing_models": [], "installed": True},
            "5b": {"id": "5b", "label": "Wan 2.2 5B", "minutes": 5, "missing_models": [], "installed": True},
        },
    )


def test_rc04_move_without_explicit_engine_refuses_paid_runway_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_local_engine_installed(monkeypatch)
    api, _ = _isolated_api(monkeypatch, tmp_path)
    try:
        starter = _make_png(tmp_path / "starter.png")
        # Reproduces Astra RC-04 counterexample: omitted engine/quality MUST NOT default to paid Runway!
        p_res = api.studio_price({"action": "move", "image": str(starter), "motion": "slow camera pan"})
        assert p_res["ok"] is False
        assert "engine" in p_res["error"].lower()

        g_res = api.studio_go({"action": "move", "image": str(starter), "motion": "slow camera pan"})
        assert g_res["ok"] is False
        assert "engine" in g_res["error"].lower()

        # Unknown engine also fails closed
        bad_res = api.studio_price({"action": "move", "image": str(starter), "engine": "unknown_cloud"})
        assert bad_res["ok"] is False
        assert "unknown" in bad_res["error"].lower()

        # Explicit local engine works cleanly at 0 credits
        loc_res = api.studio_price({"action": "move", "image": str(starter), "engine": "local"})
        assert loc_res["ok"] is True
        assert loc_res["credits"] == 0 and loc_res["local"] is True
    finally:
        api.queue.shutdown()


# ---------------------------------------------------------------------------
# RC-06: Wan video refuses locked supplemental refs & keeps verification_record truthful
# ---------------------------------------------------------------------------


def test_rc06_wan_video_refuses_locked_back_body_and_never_claims_unconditioned_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-06 counterexample:
    Starter image + locked back_body on Wan scene MUST fail closed instead of queueing
    a single-ref Wan job while claiming back_body in verification_record.
    """
    _stub_local_engine_installed(monkeypatch)
    api, _ = _isolated_api(monkeypatch, tmp_path)
    try:
        starter = _make_png(tmp_path / "starter.png", (100, 110, 120))
        rear = _make_png(tmp_path / "rear.png", (200, 90, 80))

        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "ready", "message": "ready"})

        # 1. Locked back_body on direct Wan video -> REFUSED cleanly
        res_locked = api.forge_go(
            {
                "action": "scene",
                "image": str(starter),
                "prompt": "She slowly turns around in the studio.",
                "subject_references": {
                    "references": [{"role": "back_body", "path": str(rear), "locked": True}],
                },
            }
        )
        assert res_locked["ok"] is False
        assert "locked" in res_locked["error"].lower() or "video workflow" in res_locked["error"].lower()
        assert api.queue.list() == []

        # 2. Unlocked back_body on direct Wan video -> queued with warning, AND verification_record
        # MUST NOT claim back_body was conditioned in the Wan graph!
        res_unlocked = api.forge_go(
            {
                "action": "scene",
                "image": str(starter),
                "prompt": "She slowly turns around in the studio.",
                "subject_references": {
                    "references": [{"role": "back_body", "path": str(rear), "locked": False}],
                },
            }
        )
        assert res_unlocked["ok"] is True
        queued_items = list(api.queue._items)
        assert len(queued_items) == 1
        ver = queued_items[0].runtime.get("verification_record") or {}
        ref_roles = [r.get("role") for r in (ver.get("reference_requirements") or [])]
        assert "back_body" not in ref_roles, (
            f"Wan verification_record falsely claimed unconditioned back_body reference: {ref_roles}"
        )
    finally:
        api.queue.shutdown()


# ---------------------------------------------------------------------------
# RC-07: Multi-count extraction & multi-sentence requirement preservation
# ---------------------------------------------------------------------------


def test_rc07_multi_count_and_multi_sentence_requirements_preserved() -> None:
    """Reproduces BOTH Astra RC-07 counterexamples verbatim."""
    # Counterexample A: "Exactly two cups and three bowls."
    res_a = interpret_natural_language("Exactly two cups and three bowls.", requested_action="photo", use_llm=False)
    descs_a = [r["description"].lower() for r in res_a.mandatory_requirements]
    assert any("2 cups" in d and "and three" not in d for d in descs_a), f"Corrupted first count: {descs_a}"
    assert any("3 bowls" in d for d in descs_a), f"Missed second count: {descs_a}"

    # Counterexample B: "Two ceramic cups. Keep the wooden box closed."
    res_b = interpret_natural_language(
        "Two ceramic cups. Keep the wooden box closed.", requested_action="photo", use_llm=False
    )
    descs_b = [r["description"].lower() for r in res_b.mandatory_requirements]
    assert any("2 ceramic cups" in d for d in descs_b), f"Missed count requirement: {descs_b}"
    assert any("wooden box" in d and "closed" in d for d in descs_b), (
        f"Dropped second sentence object-state requirement: {descs_b}"
    )

    # Counterexample C: general uncovered second sentence preserved too
    res_c = interpret_natural_language(
        "Three red apples on a plate. Soft morning light reflects off the brass kettle.",
        requested_action="photo",
        use_llm=False,
    )
    descs_c = [r["description"].lower() for r in res_c.mandatory_requirements]
    assert any("3 red apples" in d for d in descs_c)
    assert any("brass kettle" in d for d in descs_c), f"Uncovered second sentence was dropped: {descs_c}"


# ---------------------------------------------------------------------------
# RC-09 + RC-19 + RC-22: UI JS behavioral verification via real Node.js runtime
# ---------------------------------------------------------------------------


def _extract_ui_script() -> str:
    html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    matches = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL)
    assert matches, "Could not locate <script> block in ui/index.html"
    return "\n".join(matches)


def test_rc09_rc19_rc22_ui_js_invariants_in_node() -> None:
    """Execute the ACTUAL functions from avatar_v2/ui/index.html in Node.js to verify:
    - RC-09: Uninspected alternate output on a verified job shows 'Not inspected' (NEVER 'Requirements verified'!).
    - RC-19: charProfileLoad() -> forgeSubjectRefsPayload() preserves asset_fingerprint end-to-end!
    - RC-22: stLoad() positive-to-zero active job transition does NOT call stRefreshBalance()!
    """
    script = _extract_ui_script()
    harness = f"""
const vm = require('vm');
let balanceCalls = 0;
const elements = {{}};
const makeEl = () => ({{
  value: '', checked: true, disabled: false, innerHTML: '', textContent: '',
  style: {{}}, dataset: {{}}, classList: {{ contains: () => false, add: () => {{}}, remove: () => {{}} }},
  options: [], appendChild: () => {{}}, querySelector: () => null, querySelectorAll: () => [],
  addEventListener: () => {{}},
}});
const sandbox = {{
  console,
  Map, Set, Array, Object, Date, Math, JSON, Promise, Number, String, Boolean, RegExp,
  setTimeout: () => 0,
  clearTimeout: () => {{}},
  setInterval: () => 0,
  IntersectionObserver: class {{ observe() {{}} unobserve() {{}} disconnect() {{}} }},
  document: {{
    getElementById: (id) => (elements[id] = elements[id] || makeEl()),
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => makeEl(),
    addEventListener: () => {{}},
  }},
  window: {{
    addEventListener: () => {{}},
    pywebview: {{ api: {{}} }},
  }},
}};
vm.createContext(sandbox);
const safeSrc = {json.dumps(script)}.replace(/window\\.addEventListener\\('pywebviewready'[\\s\\S]*$/, '');
vm.runInContext(safeSrc + `
(async () => {{
  const fakeResult = {{
    label: 'Test render',
    finished_at: 1700000000,
    outputs: [
      {{ path: '/tmp/verified_a.png', kind: 'image' }},
      {{ path: '/tmp/uninspected_b.png', kind: 'image' }},
    ],
    inspection: {{ ui_status: 'verified', ui_status_label: 'Requirements verified' }},
    correction: {{ ui_status: 'verified', ui_status_label: 'Requirements verified' }},
    output_inspections: {{
      '/tmp/verified_a.png': {{ ui_status: 'verified', ui_status_label: 'Requirements verified', inspected: true, is_best: true }},
    }},
  }};
  const viewerOut = mediaViewerItems([
    {{ path: '/tmp/verified_a.png', kind: 'image', result: fakeResult }},
    {{ path: '/tmp/uninspected_b.png', kind: 'image', result: fakeResult }},
  ]);
  const resolvedA = _resolveOutputInspection(fakeResult, '/tmp/verified_a.png');
  const resolvedB = _resolveOutputInspection(fakeResult, '/tmp/uninspected_b.png');

  st.forge.profiles = [{{
    profile_id: 'prof-1',
    name: 'Elena',
    reference_set: {{
      references: [{{
        role: 'front_body',
        path: '/tmp/elena_front.png',
        subject_id: 'primary',
        locked: true,
        asset_fingerprint: 'deadbeef0123456789abcdef',
      }}],
      preserve_identity: true,
      preserve_appearance: true,
      preserve_wardrobe: true,
    }},
  }}];
  subjRender = () => {{}};
  forgePriceSoon = () => {{}};
  charProfileLoad('prof-1');
  const payload = forgeSubjectRefsPayload();

  stRefreshBalance = () => {{ globalThis.__balanceCalls = (globalThis.__balanceCalls || 0) + 1; }};
  stRender = () => {{}};
  st.data = {{ jobs: [{{ id: 'j1' }}] }};
  st.lastJobs = 1;
  window.pywebview.api.studio_state = async () => ({{ ok: true, jobs: [] }});
  await stLoad();
  console.log(JSON.stringify({{
    viewerA: viewerOut[0].meta,
    viewerB: viewerOut[1].meta,
    resolvedA: resolvedA.statusLbl,
    resolvedB: resolvedB.statusLbl,
    fingerprint: payload.references[0].asset_fingerprint,
    balanceCalls: globalThis.__balanceCalls || 0,
  }}));
}})();
`, sandbox);
"""
    proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    assert proc.returncode == 0, f"Node UI test failed: {proc.stderr}"
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["resolvedA"] == "Requirements verified"
    assert data["resolvedB"] == "Not inspected"
    assert "Requirements verified" not in data["viewerB"]
    assert "Not inspected" in data["viewerB"]
    assert data["fingerprint"] == "deadbeef0123456789abcdef", f"UI dropped asset_fingerprint: {data}"
    assert data["balanceCalls"] == 0, f"stLoad() auto-called stRefreshBalance() on job completion: {data}"


# ---------------------------------------------------------------------------
# RC-19: Durable staged pins survive manager reload & unpin cleanly on finish
# ---------------------------------------------------------------------------


def test_rc19_staged_pins_persist_across_restart_and_release_on_unpin(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staged"
    src_dir = tmp_path / "src"
    mgr1 = LocalReferenceManager(staging_dir=staging_dir, max_staged_files=8)

    pinned_img = _make_png(src_dir / "pinned_0.png", (10, 20, 30))
    staged0 = mgr1.stage_for_job(pinned_img, workflow="multi_reference", require_authorization=False)
    pinned_path = Path(staged0["staged_path"])
    mgr1.pin_staged(pinned_path, "active-job-1")
    assert (staging_dir / ".pins.json").is_file()

    # Simulate process restart with a brand-new LocalReferenceManager loading .pins.json
    mgr2 = LocalReferenceManager(staging_dir=staging_dir, max_staged_files=8)
    for i in range(1, 14):
        other = _make_png(src_dir / f"other_{i}.png", (i * 15 % 255, 50, 90))
        mgr2.stage_for_job(other, workflow="multi_reference", require_authorization=False)

    assert pinned_path.is_file(), "Pinned staged reference was evicted after restart!"

    # Now unpin active-job-1 and stage one more file -> pinned file is now eligible for eviction
    mgr2.unpin_job("active-job-1")
    pins_after = json.loads((staging_dir / ".pins.json").read_text(encoding="utf-8"))
    assert pins_after == {}
