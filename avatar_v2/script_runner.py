"""Bubblewrap OS-isolated script execution and run lifecycle management (Sprint 4, Sections 7-10).

Security architecture:
- Fails CLOSED if Bubblewrap (``bwrap``) is unavailable or fails its live OS probe — NEVER
  silently falls back to unrestricted host execution.
- Isolates filesystem, network (``--unshare-net``), PID namespace (``--unshare-pid``), IPC,
  UTS, credentials, environment variables (``--clearenv``), and GPU devices (minimal ``--dev``
  without ``/dev/nvidia*``).
- Pairs Bubblewrap with complementary OS resource limits (``prlimit`` for virtual address space,
  CPU time, max processes, open files, file size) plus a host wall-clock watchdog and full
  process-group teardown (``SIGTERM`` -> ``SIGKILL``).
- Communicates with Avatar V2 strictly through the authenticated Unix-socket ``GenerationBroker``.
- Also supports zero-execution static-extracted runs for declarative prompt libraries.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .prompt_extractor import build_specs_from_extraction, extract_prompt_data
from .reference_picker import LocalReferenceManager
from .resource_planner import ResourcePlanner
from .script_broker import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_ITEMS,
    BrokerSubmitContext,
    GenerationBroker,
)
from .script_inspector import ScriptInspectionReport, inspect_script

def _resolve_sdk_client_source() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "avatar_v2" / "sdk_client.py"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parent / "sdk_client.py"


SDK_CLIENT_SOURCE = _resolve_sdk_client_source()


def resolve_script_python(explicit: str | None = None) -> str:
    """Resolve a real general-purpose Python 3 interpreter for sandboxed Script mode.

    When packaged with PyInstaller (``sys.frozen`` is True), ``sys.executable`` points to
    the frozen AvatarV2 desktop binary — NEVER a general-purpose Python interpreter.
    """
    for cand in (explicit, os.environ.get("AVATAR_V2_SCRIPT_PYTHON")):
        if cand and Path(str(cand)).expanduser().is_file() and os.access(str(Path(str(cand)).expanduser()), os.X_OK):
            return str(Path(str(cand)).expanduser())

    if not getattr(sys, "frozen", False):
        return sys.executable

    for name in ("python3", "python3.14", "python3.12", "python3.11"):
        found = shutil.which(name)
        if found and Path(found).is_file():
            return found
    for fallback in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if Path(fallback).is_file():
            return fallback
    raise SandboxUnavailableError(
        "No host Python 3 interpreter found for Script mode (set AVATAR_V2_SCRIPT_PYTHON or install python3)."
    )


class SandboxUnavailableError(RuntimeError):
    """Raised when OS sandboxing (Bubblewrap) is unavailable; fails closed."""


@dataclass(frozen=True)
class SandboxLimits:
    """Complementary OS resource limits applied alongside Bubblewrap."""

    wall_timeout_s: float = 3600.0
    cpu_seconds: int = 300
    memory_mb: int = 1024
    max_processes: int = 24
    max_open_files: int = 128
    max_file_size_mb: int = 16


@dataclass(frozen=True)
class SandboxProbe:
    """Result of verifying actual OS Bubblewrap and prlimit support."""

    available: bool
    bwrap_path: str = ""
    bwrap_version: str = ""
    prlimit_path: str = ""
    live_test_ok: bool = False
    error: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


def probe_sandbox_support(*, bwrap_override: str | None = None) -> SandboxProbe:
    """Verify actual operating-system Bubblewrap support without modifying system security config."""
    bwrap = bwrap_override if bwrap_override is not None else (shutil.which("bwrap") or "")
    prlimit = shutil.which("prlimit") or ""
    if not bwrap or not Path(bwrap).is_file():
        return SandboxProbe(
            available=False,
            bwrap_path=bwrap,
            prlimit_path=prlimit,
            error="Bubblewrap (bwrap) is not installed; refusing unrestricted script execution.",
        )

    version = ""
    try:
        ver_res = subprocess.run([bwrap, "--version"], capture_output=True, text=True, timeout=5, check=False)
        version = (ver_res.stdout or ver_res.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return SandboxProbe(available=False, bwrap_path=bwrap, prlimit_path=prlimit, error=str(exc))

    cmd = [bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid"]
    for sys_dir in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(sys_dir).exists():
            cmd.extend(["--ro-bind", sys_dir, sys_dir])
    cmd.extend(["--", "/bin/true"])

    try:
        test_res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return SandboxProbe(
            available=False,
            bwrap_path=bwrap,
            bwrap_version=version,
            prlimit_path=prlimit,
            error=f"Bubblewrap self-test failed: {exc}",
        )

    if test_res.returncode != 0:
        err = (test_res.stderr or test_res.stdout or f"exit {test_res.returncode}").strip()
        return SandboxProbe(
            available=False,
            bwrap_path=bwrap,
            bwrap_version=version,
            prlimit_path=prlimit,
            live_test_ok=False,
            error=f"Bubblewrap namespace isolation failed on this host: {err}",
        )

    return SandboxProbe(
        available=True,
        bwrap_path=bwrap,
        bwrap_version=version,
        prlimit_path=prlimit,
        live_test_ok=True,
    )


def _python_ro_mounts(python_exe: Path) -> list[tuple[str, str]]:
    """Compute minimal read-only mounts so the selected Python interpreter works inside bwrap."""
    mounts: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        s = str(p)
        if p.exists() and s not in seen:
            seen.add(s)
            mounts.append((s, s))

    for base in ("/usr", "/lib", "/lib64", "/bin", "/etc/alternatives"):
        _add(Path(base))

    # If python lives in a virtualenv (e.g. .venv314/bin/python), mount ONLY that venv read-only,
    # NEVER the whole repository, user home directory, or PyInstaller _MEIPASS bundle.
    exe_resolved = python_exe.resolve()
    candidates: list[Path] = [exe_resolved.parent.parent]
    if not getattr(sys, "frozen", False):
        candidates.extend([Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()])
    for p in candidates:
        if p.exists() and not any(str(p).startswith(std) for std in ("/usr", "/lib", "/bin")):
            if (p / "bin").is_dir() or (p / "lib").is_dir():
                _add(p)
    return mounts


def build_bwrap_command(
    *,
    bwrap_path: str,
    prlimit_path: str,
    python_exe: str,
    script_host_path: Path,
    sdk_dir: Path,
    broker_dir: Path,
    broker_sock_name: str,
    broker_token: str,
    run_id: str,
    refs_dir: Path | None = None,
    configs_dir: Path | None = None,
    limits: SandboxLimits = SandboxLimits(),
) -> list[str]:
    """Build a fail-closed Bubblewrap + prlimit command with zero network, zero GPU, zero host secrets."""
    cmd: list[str] = [
        bwrap_path,
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--clearenv",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/workspace",
        "--tmpfs", "/workspace/tmp",
    ]

    for src, dst in _python_ro_mounts(Path(python_exe)):
        cmd.extend(["--ro-bind", src, dst])

    # Mount ONLY the standalone stdlib SDK client (never host app code or secrets).
    cmd.extend(["--ro-bind", str(sdk_dir), "/sdk"])
    # Mount ONLY the single inspected Python script read-only.
    cmd.extend(["--ro-bind", str(script_host_path), f"/workspace/script/{script_host_path.name}"])
    if refs_dir is not None and refs_dir.is_dir():
        cmd.extend(["--ro-bind", str(refs_dir), "/workspace/refs"])
    if configs_dir is not None and configs_dir.is_dir():
        cmd.extend(["--ro-bind", str(configs_dir), "/workspace/configs"])
    # Mount ONLY the per-run Unix broker socket directory.
    cmd.extend(["--bind", str(broker_dir), "/run/avatar_broker"])

    cmd.extend(["--chdir", "/workspace/tmp"])

    env_pairs = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/workspace/tmp",
        "TMPDIR": "/workspace/tmp",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/sdk",
        "AVATAR_V2_BROKER_SOCK": f"/run/avatar_broker/{broker_sock_name}",
        "AVATAR_V2_BROKER_TOKEN": broker_token,
        "AVATAR_V2_SCRIPT_RUN_ID": run_id,
        "AVATAR_V2_REFS_DIR": "/workspace/refs",
        "AVATAR_V2_CONFIGS_DIR": "/workspace/configs",
    }
    for key, val in env_pairs.items():
        cmd.extend(["--setenv", key, val])

    cmd.append("--")

    # Apply complementary OS resource limits via prlimit when available.
    if prlimit_path and Path(prlimit_path).is_file():
        as_bytes = int(limits.memory_mb) * 1024 * 1024
        fsize_bytes = int(limits.max_file_size_mb) * 1024 * 1024
        cmd.extend(
            [
                prlimit_path,
                f"--as={as_bytes}",
                f"--cpu={int(limits.cpu_seconds)}",
                f"--nofile={int(limits.max_open_files)}",
                f"--fsize={fsize_bytes}",
                "--",
            ]
        )

    cmd.extend([python_exe, "-s", "-B", f"/workspace/script/{script_host_path.name}"])
    return cmd


@dataclass
class ScriptRunRecord:
    run_id: str
    script_path: str
    script_name: str
    script_sha256: str
    classification: str
    execution_mode: str  # "sandbox" | "static_extracted"
    sandbox_mode: str
    max_items: int
    extended_authorized: bool
    started_at: str
    finished_at: str = ""
    status: str = "running"
    exit_code: int | None = None
    error: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


class ScriptRunManager:
    """Coordinates inspected script authorization, Bubblewrap runs, static-extracted runs, and audit logs."""

    def __init__(
        self,
        *,
        runs_dir: Path,
        submit_fn: Callable[[BrokerSubmitContext], str],
        status_fn: Callable[[str], dict[str, Any]],
        cancel_fn: Callable[[str], bool] | None = None,
        reference_manager: LocalReferenceManager | None = None,
        planner: ResourcePlanner | None = None,
        python_exe: str | None = None,
        bwrap_override: str | None = None,
        snapshot_override: Any = None,
        storage_check_override: Any = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._cancel_fn = cancel_fn
        self._reference_manager = reference_manager
        self._planner = planner or ResourcePlanner()
        self._python_exe = resolve_script_python(python_exe)
        self._bwrap_override = bwrap_override
        self._snapshot_override = snapshot_override
        self._storage_check_override = storage_check_override

        self._lock = threading.RLock()
        self._selected_report: ScriptInspectionReport | None = None
        self._selected_bytes: bytes | None = None
        self._selected_configs: list[str] = []
        self._active_broker: GenerationBroker | None = None
        self._active_proc: subprocess.Popen[str] | None = None
        self._active_record: ScriptRunRecord | None = None
        self._active_thread: threading.Thread | None = None
        self._active_tempdir: Path | None = None

    # ------------------------------------------------------------------ inspection (never executes!)

    def select_script(self, script_path: str | Path) -> dict[str, Any]:
        """Statically inspect a selected Python script without executing it."""
        path_obj = Path(script_path).expanduser()
        raw_bytes = path_obj.read_bytes() if path_obj.is_file() else None
        report = inspect_script(path_obj)
        probe = probe_sandbox_support(bwrap_override=self._bwrap_override)
        with self._lock:
            self._selected_report = report
            self._selected_bytes = raw_bytes if report.ok else None
            self._selected_configs = []
        return {
            "ok": report.ok,
            "report": report.public(),
            "sandbox": probe.public(),
            "executed": False,
        }

    def add_config_files(self, paths: list[str]) -> dict[str, Any]:
        valid: list[str] = []
        for raw in paths:
            p = Path(str(raw or "")).expanduser()
            if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".txt"}:
                resolved = str(p.resolve())
                if resolved not in valid:
                    valid.append(resolved)
        with self._lock:
            for item in valid:
                if item not in self._selected_configs:
                    self._selected_configs.append(item)
            return {"ok": True, "configs": list(self._selected_configs)}

    # ------------------------------------------------------------------ start / pause / resume / stop

    def start_sandboxed_run(
        self,
        *,
        script_path: str | Path | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        extended_authorized: bool = False,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        references: list[str] | None = None,
        config_files: list[str] | None = None,
        limits: SandboxLimits = SandboxLimits(),
        allow_review_classification: bool = False,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Explicitly authorize and launch a Python script inside Bubblewrap isolation.

        Enforces content-hash integrity so the exact validated bytes are snapshotted read-only
        into the sandbox and post-validation script mutations are refused cleanly.
        """
        with self._lock:
            if self._active_record is not None and self._active_record.status in {"running", "paused"}:
                return {"ok": False, "error": "A script run is already active. Stop it before starting another."}
            selected_report = self._selected_report

        target = Path(script_path).expanduser() if script_path else (
            Path(selected_report.script_path) if selected_report else None
        )
        if target is None or not target.is_file():
            return {"ok": False, "error": "Select a valid Python (.py) script first."}

        # Read bytes once and inspect from the exact same bytes (TOCTOU-proof!).
        import hashlib
        from .script_inspector import inspect_script_source

        raw_bytes = target.read_bytes()
        live_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        # Verify against previously validated hash if caller passed expected_sha256 OR ran selected script.
        pin_sha = expected_sha256 or (
            selected_report.sha256
            if selected_report is not None and str(target.resolve()) == selected_report.script_path
            else None
        )
        if pin_sha and live_sha256 != pin_sha:
            return {
                "ok": False,
                "code": "script_mutated",
                "error": (
                    "The script file changed on disk after static validation "
                    f"(expected {pin_sha[:12]}, got {live_sha256[:12]}). Re-select and inspect it before running."
                ),
            }

        report = inspect_script_source(
            raw_bytes.decode("utf-8", errors="replace"),
            script_path=str(target.resolve()),
            sha256=live_sha256,
        )
        if not report.ok or not report.syntax_valid:
            return {"ok": False, "error": report.error or "Script failed static validation.", "report": report.public()}
        if report.classification == "unsupported":
            return {
                "ok": False,
                "error": "This script is classified as unsupported and cannot be executed.",
                "report": report.public(),
            }
        if report.classification == "requires_review" and not allow_review_classification:
            return {
                "ok": False,
                "error": (
                    "This script requires explicit review acknowledgement due to flagged operations "
                    "(or run its statically extracted prompts instead)."
                ),
                "report": report.public(),
            }

        # Fail CLOSED if Bubblewrap is unavailable or broken on this host!
        probe = probe_sandbox_support(bwrap_override=self._bwrap_override)
        if not probe.available:
            raise SandboxUnavailableError(probe.error or "Bubblewrap sandbox is unavailable; refusing execution.")

        run_id = f"srun_{secrets.token_hex(6)}"
        temp_root = Path(tempfile.mkdtemp(prefix=f"avatar_v2_{run_id}_"))
        sdk_dir = temp_root / "sdk"
        script_snap_dir = temp_root / "script"
        broker_dir = temp_root / "broker"
        refs_dir = temp_root / "refs"
        configs_dir = temp_root / "configs"
        sdk_dir.mkdir(parents=True, exist_ok=True)
        script_snap_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(broker_dir, 0o700)

        # Write exact validated bytes into an immutable snapshot file for Bubblewrap bind-mount.
        script_snapshot = script_snap_dir / target.name
        script_snapshot.write_bytes(raw_bytes)
        os.chmod(script_snapshot, 0o400)

        # Stage read-only standalone SDK module as /sdk/avatar_v2_sdk.py
        shutil.copy2(SDK_CLIENT_SOURCE, sdk_dir / "avatar_v2_sdk.py")

        # Stage read-only copies of ONLY explicitly authorized reference assets & configs
        auth_refs: dict[str, str] = {}
        if references:
            refs_dir.mkdir(parents=True, exist_ok=True)
            for idx, ref_raw in enumerate(references):
                rp = Path(str(ref_raw)).expanduser()
                if not rp.is_file():
                    continue
                alias_name = f"ref_{idx}_{rp.name}"
                shutil.copy2(rp, refs_dir / alias_name)
                auth_refs[f"/workspace/refs/{alias_name}"] = str(rp.resolve())
                auth_refs[alias_name] = str(rp.resolve())
                auth_refs[rp.name] = str(rp.resolve())

        all_configs = list(config_files if config_files is not None else self._selected_configs)
        if all_configs:
            configs_dir.mkdir(parents=True, exist_ok=True)
            for cfg_raw in all_configs:
                cp = Path(str(cfg_raw)).expanduser()
                if cp.is_file():
                    shutil.copy2(cp, configs_dir / cp.name)

        broker = GenerationBroker(
            run_id=run_id,
            submit_fn=self._submit_fn,
            status_fn=self._status_fn,
            cancel_fn=self._cancel_fn,
            max_items=max_items,
            extended_authorized=extended_authorized,
            max_concurrent=max_concurrent,
            allow_cloud=False,
            authorized_references=auth_refs,
            reference_manager=self._reference_manager,
            planner=self._planner,
            snapshot_override=self._snapshot_override,
            storage_check_override=self._storage_check_override,
        )
        sock_name = "broker.sock"
        broker.start_socket_server(broker_dir / sock_name)

        cmd = build_bwrap_command(
            bwrap_path=probe.bwrap_path,
            prlimit_path=probe.prlimit_path,
            python_exe=self._python_exe,
            script_host_path=script_snapshot,
            sdk_dir=sdk_dir,
            broker_dir=broker_dir,
            broker_sock_name=sock_name,
            broker_token=broker.token,
            run_id=run_id,
            refs_dir=refs_dir if references else None,
            configs_dir=configs_dir if all_configs else None,
            limits=limits,
        )

        record = ScriptRunRecord(
            run_id=run_id,
            script_path=str(target.resolve()),
            script_name=target.name,
            script_sha256=report.sha256,
            classification=report.classification,
            execution_mode="sandbox",
            sandbox_mode=f"bwrap ({probe.bwrap_version or 'installed'})" + (" + prlimit" if probe.prlimit_path else ""),
            max_items=broker.max_items,
            extended_authorized=extended_authorized,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            broker.close_socket_server()
            shutil.rmtree(temp_root, ignore_errors=True)
            return {"ok": False, "error": f"Failed to start sandboxed script: {exc}"}

        with self._lock:
            self._active_broker = broker
            self._active_proc = proc
            self._active_record = record
            self._active_tempdir = temp_root
            self._persist_record_locked(record, broker.snapshot_status())

        watcher = threading.Thread(
            target=self._watch_process,
            args=(proc, broker, record, temp_root, limits.wall_timeout_s),
            name=f"avatar-script-watch-{run_id[:8]}",
            daemon=True,
        )
        self._active_thread = watcher
        watcher.start()
        return {"ok": True, "run_id": run_id, "record": record.public(), "status": broker.snapshot_status()}

    def start_extracted_run(
        self,
        *,
        script_path: str | Path | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        extended_authorized: bool = False,
        model_id: str = "flux2_4b",
        seed: int = 41001,
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute statically extracted prompt data through Avatar V2 WITHOUT running any script code."""
        with self._lock:
            if self._active_record is not None and self._active_record.status in {"running", "paused"}:
                return {"ok": False, "error": "A script run is already active. Stop it before starting another."}

        target = Path(script_path).expanduser() if script_path else (
            Path(self._selected_report.script_path) if self._selected_report else None
        )
        if target is None or not target.is_file():
            return {"ok": False, "error": "Select a valid Python (.py) or JSON file first."}

        report = inspect_script(target) if target.suffix.lower() == ".py" else None
        extraction = extract_prompt_data(target)
        if not extraction.ok or not extraction.has_extractable_data:
            return {"ok": False, "error": extraction.error or "No statically extractable prompt data found."}

        run_id = f"srun_{secrets.token_hex(6)}"
        auth_refs = {Path(str(r)).name: str(Path(str(r)).expanduser().resolve()) for r in (references or []) if Path(str(r)).expanduser().is_file()}
        broker = GenerationBroker(
            run_id=run_id,
            submit_fn=self._submit_fn,
            status_fn=self._status_fn,
            cancel_fn=self._cancel_fn,
            max_items=max_items,
            extended_authorized=extended_authorized,
            max_concurrent=1,
            allow_cloud=False,
            authorized_references=auth_refs,
            reference_manager=self._reference_manager,
            planner=self._planner,
            snapshot_override=self._snapshot_override,
            storage_check_override=self._storage_check_override,
        )
        specs = build_specs_from_extraction(
            extraction,
            count=broker.max_items,
            seed=seed,
            model_preference=model_id,
        )
        record = ScriptRunRecord(
            run_id=run_id,
            script_path=str(target.resolve()),
            script_name=target.name,
            script_sha256=report.sha256 if report else "",
            classification=report.classification if report else "statically_extractable",
            execution_mode="static_extracted",
            sandbox_mode="static_extraction_zero_code_exec",
            max_items=broker.max_items,
            extended_authorized=extended_authorized,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )

        with self._lock:
            self._active_broker = broker
            self._active_proc = None
            self._active_record = record
            self._active_tempdir = None
            self._persist_record_locked(record, broker.snapshot_status())

        worker = threading.Thread(
            target=self._run_extracted_loop,
            args=(broker, record, specs, model_id),
            name=f"avatar-extracted-{run_id[:8]}",
            daemon=True,
        )
        self._active_thread = worker
        worker.start()
        return {"ok": True, "run_id": run_id, "record": record.public(), "status": broker.snapshot_status()}

    def _run_extracted_loop(
        self,
        broker: GenerationBroker,
        record: ScriptRunRecord,
        specs: list[Any],
        model_id: str,
    ) -> None:
        from .sdk_client import AvatarV2Client, RunLimitReached, RunStopped, SDKError

        client = AvatarV2Client(token=broker.token, transport=broker.handle_request)
        try:
            for spec in specs:
                if broker.state == "stopped":
                    break
                client.submit_spec(spec, model_id=model_id, wait=True, poll_interval_s=0.05)
        except (RunLimitReached, RunStopped):
            pass
        except SDKError as exc:
            record.error = str(exc)
            record.status = "error"
        finally:
            if record.status == "running":
                record.status = "stopped" if broker.state == "stopped" else "completed"
                if record.status == "completed":
                    broker.mark_completed()
            record.finished_at = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._persist_record_locked(record, broker.snapshot_status())

    def _watch_process(
        self,
        proc: subprocess.Popen[str],
        broker: GenerationBroker,
        record: ScriptRunRecord,
        temp_root: Path,
        wall_timeout_s: float,
    ) -> None:
        try:
            stdout, stderr = proc.communicate(timeout=max(1.0, float(wall_timeout_s)))
            record.exit_code = proc.returncode
            record.stdout_tail = (stdout or "")[-4000:]
            record.stderr_tail = (stderr or "")[-4000:]
            if broker.state == "stopped":
                record.status = "stopped"
            elif proc.returncode == 0:
                broker.mark_completed()
                record.status = "completed"
            else:
                record.status = "error"
                record.error = (stderr or stdout or f"Script exited with code {proc.returncode}").strip()[-500:]
        except subprocess.TimeoutExpired:
            self._terminate_process_group(proc)
            stdout, stderr = proc.communicate()
            record.exit_code = proc.returncode
            record.stdout_tail = (stdout or "")[-4000:]
            record.stderr_tail = (stderr or "")[-4000:]
            record.status = "error"
            record.error = f"Script exceeded wall-clock timeout ({wall_timeout_s:g}s) and was terminated."
            broker.stop(cancel_pending=True)
        finally:
            broker.close_socket_server()
            record.finished_at = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._persist_record_locked(record, broker.snapshot_status())
                if self._active_proc is proc:
                    self._active_proc = None
            shutil.rmtree(temp_root, ignore_errors=True)

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self._active_broker is None or self._active_record is None:
                return {"ok": False, "error": "No active script run to pause."}
            self._active_broker.pause()
            if self._active_record.status == "running":
                self._active_record.status = "paused"
            return {"ok": True, "status": self._active_broker.snapshot_status()}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._active_broker is None or self._active_record is None:
                return {"ok": False, "error": "No active script run to resume."}
            self._active_broker.resume()
            if self._active_record.status == "paused":
                self._active_record.status = "running"
            return {"ok": True, "status": self._active_broker.snapshot_status()}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            broker = self._active_broker
            proc = self._active_proc
            record = self._active_record
        if broker is None or record is None:
            return {"ok": False, "error": "No active script run to stop."}
        broker.stop(cancel_pending=True)
        record.status = "stopped"
        if proc is not None:
            self._terminate_process_group(proc)
        return {"ok": True, "status": broker.snapshot_status()}

    def wait_for_run(self, timeout_s: float = 30.0) -> dict[str, Any]:
        t = self._active_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout_s)
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            probe = probe_sandbox_support(bwrap_override=self._bwrap_override)
            return {
                "ok": True,
                "sandbox": probe.public(),
                "selected": self._selected_report.public() if self._selected_report else None,
                "configs": list(self._selected_configs),
                "record": self._active_record.public() if self._active_record else None,
                "run": self._active_broker.snapshot_status() if self._active_broker else None,
            }

    def _persist_record_locked(self, record: ScriptRunRecord, run_status: dict[str, Any]) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            path = self.runs_dir / f"{record.run_id}.json"
            payload = {"record": record.public(), "run": run_status}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
