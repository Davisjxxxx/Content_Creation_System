"""Host-side GenerationBroker enforcing limits, policy, capability checks, and RenderQueue routing (Sprint 4).

Listens on a private Unix domain socket (mode 0600) bound into the Bubblewrap sandbox so
sandboxed scripts with ``--unshare-net`` can submit generation requests ONLY through Avatar V2's
central validation, compilation, ResourcePlanner, and RenderQueue pipeline.

Host-enforced invariants (cannot be bypassed by concurrent script threads or subprocesses):
- Per-run generation cap (default 10 items; deliberately authorized extended ceiling).
- Concurrent active/queued job cap (default 1, configurable up to 2).
- Local-only by default (Runway/cloud requests blocked unless explicitly host-authorized).
- Policy enforcement (`depicts_minor`, `StructuredPromptSpec.detect_conflicts()`).
- Model capability validation (`ModelRegistry` + `compile_prompt_for_model`).
- Reference asset allowlist (only authorized run references or staged paths accepted).
- Safe-boundary Pause / Resume / Stop preserving all completed outputs.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .model_registry import DEFAULT_REGISTRY, ModelRegistry
from .ownership import write_json_atomic
from .policy import depicts_minor
from .prompt_compilers import compile_prompt_for_model
from .prompt_schema import StructuredPromptSpec
from .reference_picker import LocalReferenceManager, validate_local_image
from .resource_planner import ExecutionPlan, ResourcePlanner, WorkloadRequest
from .sdk_client import SDK_CONTRACT_VERSION

RunState = Literal["idle", "running", "paused", "stopping", "stopped", "completed", "error"]

DEFAULT_MAX_ITEMS = 10
HARD_MAX_AUTHORIZED_ITEMS = 200
DEFAULT_MAX_CONCURRENT = 1
MAX_BROKER_CONNECTIONS = 16

# Only models with real executable builders in StudioMixin._script_submit_job (Astra RC-15).
# Deferred specialists (such as wan22_funcontrol) are excluded until installed & wired end-to-end.
EXECUTABLE_SCRIPT_MODELS: frozenset[str] = frozenset({"flux2_4b", "flux2_9b", "wan22_a14b", "wan22_5b"})

_BUDGET_FILE_LOCK = threading.RLock()


def charge_durable_script_budget(
    runs_dir: Path,
    run_id: str,
    parent_job_id: str,
    *,
    note: str = "",
) -> tuple[bool, str]:
    """Fail-closed durable run-budget charge surviving broker detach or app restart (Astra RC-11)."""
    budget_path = Path(runs_dir) / f"{run_id}.budget.json"
    with _BUDGET_FILE_LOCK:
        if not budget_path.is_file():
            return False, (
                f"Durable script run budget ledger missing for run '{run_id}'; "
                f"refusing detached corrective generation for {parent_job_id} (fail-closed)."
            )
        try:
            data = json.loads(budget_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"Unreadable script budget ledger for '{run_id}': {exc}"
        if str(data.get("state") or "") in {"stopping", "stopped"}:
            return False, f"Script run '{run_id}' was stopped; refusing corrective generation."
        used = int(data.get("submitted", 0))
        limit = int(data.get("max_items", 0))
        if limit <= 0 or used >= limit:
            return False, (
                f"Durable script run budget exhausted ({used}/{limit}); "
                f"refusing corrective attempt for {parent_job_id}."
            )
        data["submitted"] = used + 1
        charges = list(data.get("corrective_charges") or [])
        charges.append({"parent_job_id": parent_job_id, "note": note, "at": time.time()})
        data["corrective_charges"] = charges[-50:]
        write_json_atomic(budget_path, data)
        return True, f"Charged durable script budget ({used + 1}/{limit}) {note}".strip()


@dataclass
class BrokerSubmitContext:
    """Validated payload handed from GenerationBroker to the host queue submitter."""

    run_id: str
    item_index: int
    spec: StructuredPromptSpec
    model_id: str
    compiled_positive: str
    compiled_negative: str
    compiled_warnings: list[str]
    execution_plan: ExecutionPlan
    staged_references: list[str]
    label: str


@dataclass
class BrokerJobRecord:
    job_id: str
    model_id: str
    status: str = "queued"
    outputs: list[str] = field(default_factory=list)
    compiled_prompt: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "outputs": list(self.outputs),
            "compiled_prompt": self.compiled_prompt,
            "warnings": list(self.warnings),
            "error": self.error,
            "metadata": dict(self.metadata),
            "quality_inspection": None,
            "correction_hint": None,
        }


class GenerationBroker:
    """Thread-safe host broker authenticating and enforcing every SDK request."""

    def __init__(
        self,
        *,
        run_id: str,
        submit_fn: Callable[[BrokerSubmitContext], str],
        status_fn: Callable[[str], dict[str, Any]],
        cancel_fn: Callable[[str], bool] | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        extended_authorized: bool = False,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        allow_cloud: bool = False,
        authorized_references: dict[str, str] | None = None,
        reference_manager: LocalReferenceManager | None = None,
        planner: ResourcePlanner | None = None,
        registry: ModelRegistry = DEFAULT_REGISTRY,
        token: str | None = None,
        snapshot_override: Any = None,
        storage_check_override: Any = None,
        budget_path: Path | None = None,
        comfyui_dir: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.token = token or secrets.token_urlsafe(32)
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._cancel_fn = cancel_fn
        self.registry = registry
        self.planner = planner or ResourcePlanner(registry=registry)
        self.reference_manager = reference_manager
        self._snapshot_override = snapshot_override
        self._storage_check_override = storage_check_override
        self._budget_path = Path(budget_path) if budget_path is not None else None
        self._comfyui_dir = Path(comfyui_dir) if comfyui_dir is not None else None
        self._conn_sem = threading.BoundedSemaphore(MAX_BROKER_CONNECTIONS)

        cap_ceiling = HARD_MAX_AUTHORIZED_ITEMS if extended_authorized else DEFAULT_MAX_ITEMS
        self.max_items = max(1, min(int(max_items), cap_ceiling))
        self.extended_authorized = bool(extended_authorized)
        self.max_concurrent = max(1, min(int(max_concurrent), 2))
        self.allow_cloud = bool(allow_cloud)

        # Maps sandbox reference alias (e.g. "/workspace/refs/ref_0.png" or basename) -> host real path.
        self._ref_map: dict[str, str] = dict(authorized_references or {})

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._state: RunState = "running"
        self._submitted_count: int = 0
        self._jobs: dict[str, BrokerJobRecord] = {}
        self._job_order: list[str] = []

        self._sock_path: Path | None = None
        self._server_sock: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._serving = False
        self._sync_budget_file_locked()

    def _sync_budget_file_locked(self) -> None:
        if self._budget_path is None:
            return
        with _BUDGET_FILE_LOCK:
            try:
                existing = json.loads(self._budget_path.read_text(encoding="utf-8")) if self._budget_path.is_file() else {}
            except (OSError, json.JSONDecodeError):
                existing = {}
            payload = {
                "run_id": self.run_id,
                "state": self._state,
                "submitted": max(self._submitted_count, int(existing.get("submitted") or 0)),
                "max_items": self.max_items,
                "corrective_charges": list(existing.get("corrective_charges") or []),
            }
            self._submitted_count = payload["submitted"]
            try:
                write_json_atomic(self._budget_path, payload)
            except OSError:
                pass

    # ------------------------------------------------------------------ lifecycle & control

    @property
    def state(self) -> RunState:
        with self._lock:
            return self._state

    def pause(self) -> None:
        """Pause at safe generation boundaries (in-flight job finishes; new submissions wait)."""
        with self._cond:
            if self._state == "running":
                self._state = "paused"
                self._cond.notify_all()

    def resume(self) -> None:
        with self._cond:
            if self._state == "paused":
                self._state = "running"
                self._cond.notify_all()

    def stop(self, *, cancel_pending: bool = True) -> None:
        """Stop accepting new jobs and optionally cancel still-queued jobs, preserving completed outputs."""
        to_cancel: list[str] = []
        with self._cond:
            if self._state not in {"stopped", "completed"}:
                self._state = "stopped"
            self._sync_budget_file_locked()
            if cancel_pending and self._cancel_fn is not None:
                for jid in self._job_order:
                    rec = self._jobs[jid]
                    if rec.status in {"queued", "running"}:
                        to_cancel.append(jid)
            self._cond.notify_all()
        if self._cancel_fn is not None:
            for jid in to_cancel:
                try:
                    self._cancel_fn(jid)
                except Exception:
                    pass

    def mark_completed(self) -> None:
        with self._cond:
            if self._state in {"running", "paused"}:
                self._state = "completed"
                self._cond.notify_all()

    def charge_corrective_attempt(self, parent_job_id: str, *, note: str = "") -> tuple[bool, str]:
        """Charge a corrective generation attempt against the SAME host-enforced run budget."""
        with self._lock:
            self._sync_budget_file_locked()
            if self._state in {"stopping", "stopped"}:
                return False, "Script run was stopped; refusing additional corrective generation."
            if self._submitted_count >= self.max_items:
                return False, (
                    f"Shared script run budget exhausted ({self._submitted_count}/{self.max_items}); "
                    f"cannot start corrective attempt for {parent_job_id}."
                )
            self._submitted_count += 1
            self._sync_budget_file_locked()
            self._sync_budget_file_locked()
            return True, f"Charged corrective attempt ({self._submitted_count}/{self.max_items}) {note}".strip()

    def snapshot_status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_jobs_locked()
            completed = sum(1 for j in self._jobs.values() if j.status == "done")
            failed = sum(1 for j in self._jobs.values() if j.status == "error")
            cancelled = sum(1 for j in self._jobs.values() if j.status == "cancelled")
            active = sum(1 for j in self._jobs.values() if j.status in {"queued", "running"})
            outputs: list[str] = []
            for jid in self._job_order:
                for out in self._jobs[jid].outputs:
                    if out not in outputs:
                        outputs.append(out)
            return {
                "run_id": self.run_id,
                "state": self._state,
                "submitted": self._submitted_count,
                "max_items": self.max_items,
                "remaining": max(0, self.max_items - self._submitted_count),
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "active": active,
                "max_concurrent": self.max_concurrent,
                "local_only": not self.allow_cloud,
                "outputs": outputs,
                "jobs": [self._jobs[jid].public() for jid in self._job_order],
            }

    # ------------------------------------------------------------------ Unix socket server

    def start_socket_server(self, socket_path: Path) -> Path:
        socket_path = Path(socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            socket_path.unlink(missing_ok=True)
        except OSError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        srv.listen(8)
        srv.settimeout(0.25)
        self._sock_path = socket_path
        self._server_sock = srv
        self._serving = True
        self._server_thread = threading.Thread(
            target=self._accept_loop,
            name=f"avatar-broker-{self.run_id[:8]}",
            daemon=True,
        )
        self._server_thread.start()
        return socket_path

    def close_socket_server(self) -> None:
        self._serving = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        if self._sock_path is not None:
            try:
                self._sock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while self._serving:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._conn_sem.acquire(blocking=False):
                try:
                    conn.sendall(
                        b'{"ok":false,"code":"broker_busy","error":"Too many concurrent script broker connections."}\n'
                    )
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            threading.Thread(
                target=self._serve_connection_bounded,
                args=(conn,),
                daemon=True,
            ).start()

    def _serve_connection_bounded(self, conn: socket.socket) -> None:
        try:
            self._serve_connection(conn)
        finally:
            self._conn_sem.release()

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(30.0)
            chunks: list[bytes] = []
            try:
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk or sum(len(c) for c in chunks) > 1024 * 1024:
                        break
            except OSError:
                return
            raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            if not raw:
                return
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                resp = {"ok": False, "code": "invalid_json", "error": f"Invalid JSON request: {exc}"}
            else:
                resp = self.handle_request(request)
            try:
                conn.sendall((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
            except OSError:
                return

    # ------------------------------------------------------------------ request dispatch

    def handle_request(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            return {"ok": False, "code": "bad_request", "error": "Request must be a JSON object."}
        if not secrets.compare_digest(str(envelope.get("token") or ""), self.token):
            return {"ok": False, "code": "unauthorized", "error": "Invalid broker authentication token."}
        version = str(envelope.get("contract_version") or "")
        if version != SDK_CONTRACT_VERSION:
            return {
                "ok": False,
                "code": "version_mismatch",
                "error": f"Unsupported SDK contract version '{version}'; expected '{SDK_CONTRACT_VERSION}'.",
            }
        op = str(envelope.get("op") or "")
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}

        if op == "capabilities":
            return self._op_capabilities()
        if op == "run_status":
            return {"ok": True, **self.snapshot_status()}
        if op == "list_references":
            return {
                "ok": True,
                "references": [
                    {"alias": alias, "name": Path(host_path).name}
                    for alias, host_path in self._ref_map.items()
                ],
            }
        if op == "submit":
            return self._op_submit(payload)
        if op == "job_status":
            return self._op_job_status(payload)
        if op == "cancel_job":
            return self._op_cancel_job(payload)
        return {"ok": False, "code": "unknown_op", "error": f"Unsupported broker operation '{op}'."}

    def _is_model_installed(self, model_spec: Any) -> tuple[bool, list[str]]:
        if model_spec.model_id not in EXECUTABLE_SCRIPT_MODELS:
            return False, [f"Model '{model_spec.model_id}' is deferred/not executable in Script mode."]
        if self._comfyui_dir is None:
            return True, []
        missing = [c.filename for c in model_spec.components if not c.exists_in(self._comfyui_dir)]
        return (not missing), missing

    def _op_capabilities(self) -> dict[str, Any]:
        models = []
        for m in self.registry.list_models():
            installed, _missing = self._is_model_installed(m)
            if not installed:
                continue
            models.append(
                {
                    "model_id": m.model_id,
                    "family": m.family,
                    "label": m.label,
                    "tasks": list(m.tasks),
                    "default_width": m.default_width,
                    "default_height": m.default_height,
                    "default_frames": m.default_frames,
                    "requires_start_image": m.requires_start_image,
                    "supports_edit": m.supports_edit,
                    "supports_motion": m.supports_motion,
                    "supports_camera_movement": m.supports_camera_movement,
                    "installed": True,
                }
            )
        return {
            "ok": True,
            "contract_version": SDK_CONTRACT_VERSION,
            "local_only": not self.allow_cloud,
            "max_items": self.max_items,
            "max_concurrent": self.max_concurrent,
            "models": models,
        }

    def _refresh_jobs_locked(self) -> None:
        for rec in self._jobs.values():
            if rec.status in {"done", "error", "cancelled"}:
                continue
            try:
                live = self._status_fn(rec.job_id) or {}
            except Exception as exc:
                rec.status = "error"
                rec.error = str(exc)
                continue
            if live.get("status"):
                rec.status = str(live["status"])
            if live.get("outputs"):
                rec.outputs = list(live["outputs"])
            if live.get("error"):
                rec.error = str(live["error"])

    def _resolve_reference_path(self, ref_path: str) -> str:
        raw = str(ref_path or "").strip()
        if not raw:
            raise ValueError("Reference path cannot be empty.")
        if raw in self._ref_map:
            host_path = self._ref_map[raw]
        elif Path(raw).name in self._ref_map:
            host_path = self._ref_map[Path(raw).name]
        else:
            # Deny arbitrary host filesystem paths submitted by untrusted scripts!
            raise PermissionError(
                f"Reference asset '{raw}' is not in the authorized reference list for this script run."
            )
        if self.reference_manager is not None:
            staged = self.reference_manager.stage_for_job(host_path, workflow="script", require_authorization=False)
            return str(staged["staged_path"])
        checked = validate_local_image(host_path, workflow="script")
        if not checked.ok:
            raise ValueError(checked.error)
        return checked.path

    def _op_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 1. Wait cleanly at pause boundary; reject immediately if stopped.
        with self._cond:
            while self._state == "paused":
                self._cond.wait(timeout=0.5)
            if self._state in {"stopped", "stopping", "error"}:
                return {"ok": False, "code": "run_stopped", "error": "Script run has been stopped."}

            self._refresh_jobs_locked()
            if self._submitted_count >= self.max_items:
                return {
                    "ok": False,
                    "code": "run_limit_reached",
                    "error": (
                        f"Host-enforced generation limit reached ({self._submitted_count}/{self.max_items} items). "
                        "Authorize an extended run to generate more."
                    ),
                }
            active_count = sum(1 for j in self._jobs.values() if j.status in {"queued", "running"})
            if active_count >= self.max_concurrent:
                return {
                    "ok": False,
                    "code": "concurrency_limit",
                    "error": (
                        f"Concurrent job limit reached ({active_count}/{self.max_concurrent} active). "
                        "Wait for the current job to finish before submitting another."
                    ),
                }

        # 2. Schema validation (StructuredPromptSpec v1.0).
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict):
            return {"ok": False, "code": "invalid_spec", "error": "Missing or invalid StructuredPromptSpec object."}
        try:
            spec = StructuredPromptSpec.model_validate(raw_spec)
        except Exception as exc:
            return {"ok": False, "code": "invalid_spec", "error": f"StructuredPromptSpec validation failed: {exc}"}

        # 3. Local-only cloud lock (cannot be bypassed by script).
        if not spec.constraints.local_only and not self.allow_cloud:
            return {
                "ok": False,
                "code": "cloud_blocked",
                "error": "Script mode is local-only; cloud/Runway generation requires explicit host authorization.",
            }
        pref = spec.constraints.model_preference if spec.constraints.model_preference != "auto" else ""
        requested_model = str(payload.get("model_id") or pref or "").strip()
        if requested_model.lower().startswith("runway") or requested_model.lower().startswith("gen4"):
            if not self.allow_cloud:
                return {
                    "ok": False,
                    "code": "cloud_blocked",
                    "error": f"Cloud model '{requested_model}' is blocked in local-only Script mode.",
                }

        # 4. Model selection & capability validation.
        if not requested_model:
            requested_model = "wan22_a14b" if spec.intent.task in {"video_generate", "video_scene"} else "flux2_4b"
        model_spec = self.registry.get(requested_model)
        if model_spec is None:
            return {
                "ok": False,
                "code": "unknown_model",
                "error": f"Model '{requested_model}' is not an approved local model in Avatar V2's ModelRegistry.",
            }
        installed_ok, missing_files = self._is_model_installed(model_spec)
        if not installed_ok:
            return {
                "ok": False,
                "code": "unsupported_model",
                "error": (
                    f"Model '{requested_model}' is unavailable in Script mode: "
                    + (", ".join(missing_files) if missing_files else "not installed")
                ),
            }

        # 5. Reference asset authorization & staging.
        staged_refs: list[str] = []
        for ref in spec.reference_assets:
            try:
                staged_refs.append(self._resolve_reference_path(ref.path))
            except PermissionError as exc:
                return {"ok": False, "code": "unauthorized_path", "error": str(exc)}
            except ValueError as exc:
                return {"ok": False, "code": "invalid_reference", "error": str(exc)}

        if model_spec.requires_start_image and not staged_refs:
            return {
                "ok": False,
                "code": "missing_reference",
                "error": f"{model_spec.label} requires an authorized start reference image.",
            }

        # 6. Policy enforcement, model capability check & deterministic prompt compilation.
        compiled = compile_prompt_for_model(spec, model_spec.model_id, registry=self.registry)
        phrase = depicts_minor(compiled.positive)
        if phrase:
            return {
                "ok": False,
                "code": "policy_blocked",
                "error": f'Adult-only policy violation: remove "{phrase}" from the prompt.',
            }
        policy_errors = [
            c["message"] for c in compiled.conflicts
            if c.get("severity") == "error" and "minor" in str(c.get("code") or "").lower()
        ]
        if policy_errors:
            return {"ok": False, "code": "policy_blocked", "error": policy_errors[0]}
        if compiled.unsupported_capabilities:
            return {
                "ok": False,
                "code": "unsupported_capability",
                "error": compiled.unsupported_capabilities[0],
            }
        other_errors = [c["message"] for c in compiled.conflicts if c.get("severity") == "error"]
        if other_errors:
            return {"ok": False, "code": "spec_conflict", "error": other_errors[0]}
        if not compiled.positive.strip():
            return {"ok": False, "code": "empty_prompt", "error": "Compiled prompt is empty."}

        # 7. ResourcePlanner admission check (storage, thermal, RAM/swap protection).
        req = WorkloadRequest(
            model_id=model_spec.model_id,
            workflow=model_spec.workflow,
            width=int(spec.constraints.width or model_spec.default_width),
            height=int(spec.constraints.height or model_spec.default_height),
            frames=int(spec.constraints.frames or model_spec.default_frames),
            local_only=not self.allow_cloud,
            has_reference_assets=bool(staged_refs),
        )
        plan = self.planner.plan_execution(
            req,
            snapshot=self._snapshot_override,
            storage_check=self._storage_check_override,
        )
        if not plan.admitted:
            return {"ok": False, "code": "resource_blocked", "error": plan.admission_reason}

        # 8. Atomic reservation under lock (prevents concurrent-request race bypassing max_items / max_concurrent).
        with self._cond:
            self._sync_budget_file_locked()
            if self._state in {"stopped", "stopping", "error"}:
                return {"ok": False, "code": "run_stopped", "error": "Script run has been stopped."}
            self._refresh_jobs_locked()
            if self._submitted_count >= self.max_items:
                return {
                    "ok": False,
                    "code": "run_limit_reached",
                    "error": f"Host-enforced generation limit reached ({self._submitted_count}/{self.max_items} items).",
                }
            active_count = sum(1 for j in self._jobs.values() if j.status in {"queued", "running"})
            if active_count >= self.max_concurrent:
                return {
                    "ok": False,
                    "code": "concurrency_limit",
                    "error": f"Concurrent job limit reached ({active_count}/{self.max_concurrent} active).",
                }
            item_index = self._submitted_count + 1
            label = str(payload.get("label") or "").strip() or f"Script item {item_index} of {self.max_items} · {model_spec.label}"
            ctx = BrokerSubmitContext(
                run_id=self.run_id,
                item_index=item_index,
                spec=spec,
                model_id=model_spec.model_id,
                compiled_positive=compiled.positive,
                compiled_negative=compiled.negative,
                compiled_warnings=list(compiled.warnings),
                execution_plan=plan,
                staged_references=staged_refs,
                label=label,
            )
            try:
                job_id = str(self._submit_fn(ctx))
            except Exception as exc:
                return {"ok": False, "code": "submit_failed", "error": str(exc)}

            self._submitted_count += 1
            self._sync_budget_file_locked()
            rec = BrokerJobRecord(
                job_id=job_id,
                model_id=model_spec.model_id,
                status="queued",
                compiled_prompt=compiled.positive,
                warnings=list(compiled.warnings),
                metadata={
                    "run_id": self.run_id,
                    "item_index": item_index,
                    "execution_plan": plan.public(),
                    "provenance": compiled.provenance,
                },
            )
            self._jobs[job_id] = rec
            self._job_order.append(job_id)
            self._refresh_jobs_locked()
            return {"ok": True, "job": rec.public()}

    def _op_job_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return {"ok": False, "code": "unknown_job", "error": f"Job '{job_id}' does not belong to this run."}
            self._refresh_jobs_locked()
            return {"ok": True, "job": rec.public()}

    def _op_cancel_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return {"ok": False, "code": "unknown_job", "error": f"Job '{job_id}' does not belong to this run."}
        cancelled = False
        if self._cancel_fn is not None:
            try:
                cancelled = bool(self._cancel_fn(job_id))
            except Exception:
                cancelled = False
        with self._lock:
            if cancelled:
                rec.status = "cancelled"
            return {"ok": True, "cancelled": cancelled, "job": rec.public()}
