"""Explicit Process Ownership & Managed Media Ownership Guards (Astra RC1 P0-A / P0-B).

Provides fail-closed ownership verification so Avatar V2:
1. NEVER signals (SIGTERM/SIGKILL) or disrupts (global /free or /interrupt) a process it did not
   launch or whose OS process identity cannot be verified (protecting against PID reuse, stale
   PID files, host reboots, and shared/external renderers).
2. NEVER unlinks an externally owned original file, symlink-to-external file, or hard-linked
   external file (separating metadata detach from verified managed-archive physical deletion).
3. Persists ownership/state files atomically with unique temp paths and fsync.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EngineOwnership = Literal["owned_private", "shared_or_external", "stale_or_reused", "stopped"]
MediaOwnershipClass = Literal[
    "managed_output",
    "managed_import",
    "ephemeral_staging",
    "external_original",
    "external_symlink",
    "hardlinked_shared",
    "missing",
]


# --------------------------------------------------------------------------- Atomic Durable JSON


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write JSON via unique temp file + flush + fsync + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    text = json.dumps(data, indent=2)
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- P0-A: Process Ownership


def read_host_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_proc_stat(stat_text: str) -> tuple[str, int] | None:
    """Parse (state, starttime_ticks) from Linux /proc/<pid>/stat safely even if comm contains spaces/parens."""
    rparen = stat_text.rfind(")")
    if rparen < 0:
        return None
    fields = stat_text[rparen + 1 :].strip().split()
    # After '(comm)', index 0 is state (field 3), index 19 is starttime (field 22).
    if len(fields) < 20:
        return None
    state = fields[0]
    try:
        starttime = int(fields[19])
    except ValueError:
        return None
    return state, starttime


def inspect_process_identity(pid: int | None) -> dict[str, Any] | None:
    """Inspect a live Linux process identity from /proc; returns None if missing or zombie."""
    if not pid or int(pid) <= 0:
        return None
    pid_i = int(pid)
    proc_dir = Path(f"/proc/{pid_i}")
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = parse_proc_stat(stat_text)
    if parsed is None:
        return None
    state, start_ticks = parsed
    if state == "Z":
        return None

    exe_realpath = ""
    try:
        exe_realpath = os.readlink(proc_dir / "exe")
    except OSError:
        exe_realpath = ""

    cmdline: list[str] = []
    try:
        raw_cmd = (proc_dir / "cmdline").read_bytes()
        cmdline = [tok.decode("utf-8", errors="replace") for tok in raw_cmd.split(b"\x00") if tok]
    except OSError:
        cmdline = []

    return {
        "pid": pid_i,
        "state": state,
        "start_ticks": start_ticks,
        "boot_id": read_host_boot_id(),
        "exe_realpath": exe_realpath,
        "cmdline": cmdline,
    }


@dataclass(frozen=True)
class ProcessOwnershipRecord:
    """Durable ownership record binding an app-launched child to its OS identity."""

    pid: int
    start_ticks: int | None = None
    boot_id: str = ""
    exe_realpath: str = ""
    cmdline: list[str] = field(default_factory=list)
    port: int | None = None
    owned_by_app: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> ProcessOwnershipRecord | None:
        if not isinstance(raw, dict):
            return None
        try:
            pid = int(raw.get("pid") or 0)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return None
        raw_ticks = raw.get("start_ticks")
        ticks = int(raw_ticks) if isinstance(raw_ticks, int) and not isinstance(raw_ticks, bool) else None
        raw_port = raw.get("port")
        port = int(raw_port) if isinstance(raw_port, int) and not isinstance(raw_port, bool) else None
        cmd = [str(x) for x in (raw.get("cmdline") or []) if isinstance(x, str)]
        return cls(
            pid=pid,
            start_ticks=ticks,
            boot_id=str(raw.get("boot_id") or ""),
            exe_realpath=str(raw.get("exe_realpath") or ""),
            cmdline=cmd,
            port=port,
            owned_by_app=bool(raw.get("owned_by_app", False)),
            created_at=str(raw.get("created_at") or ""),
        )

    @classmethod
    def capture_for_pid(
        cls,
        pid: int,
        *,
        port: int | None = None,
        fallback_cmdline: list[str] | None = None,
    ) -> ProcessOwnershipRecord:
        ident = inspect_process_identity(pid)
        return cls(
            pid=int(pid),
            start_ticks=ident["start_ticks"] if ident else None,
            boot_id=ident["boot_id"] if ident else read_host_boot_id(),
            exe_realpath=ident["exe_realpath"] if ident else "",
            cmdline=(ident["cmdline"] if (ident and ident["cmdline"]) else list(fallback_cmdline or [])),
            port=port,
            owned_by_app=True,
        )


def verify_process_ownership(record: ProcessOwnershipRecord | None) -> tuple[bool, str]:
    """Fail-closed verification that ``record`` still identifies the exact app-owned OS process."""
    if record is None:
        return False, "no_ownership_record"
    if not record.owned_by_app:
        return False, "not_owned_by_app"
    if record.pid <= 0:
        return False, "invalid_pid"
    if record.start_ticks is None:
        # Never trust a PID without start-time identity (eliminates bare-PID reuse risk!).
        return False, "missing_start_ticks"

    live = inspect_process_identity(record.pid)
    if live is None:
        return False, "process_exited"
    if record.boot_id and live["boot_id"] and record.boot_id != live["boot_id"]:
        return False, "boot_id_mismatch"
    if live["start_ticks"] != record.start_ticks:
        return False, "pid_reused"
    if record.exe_realpath and live["exe_realpath"] and record.exe_realpath != live["exe_realpath"]:
        return False, "exe_mismatch"
    if record.cmdline and live["cmdline"]:
        # Verify at least script/entry tokens match (e.g. main.py / port) to catch exec replacement.
        rec_tokens = set(record.cmdline[:6])
        live_tokens = set(live["cmdline"][:6])
        if record.cmdline[0] != live["cmdline"][0] and not (rec_tokens & live_tokens):
            return False, "cmdline_mismatch"
        if record.port is not None and str(record.port) in record.cmdline and str(record.port) not in live["cmdline"]:
            return False, "port_mismatch"
    return True, "verified_owned"


class OwnedProcessHandle:
    """In-memory handle for a process launched by Avatar V2, using Linux pidfd when available."""

    def __init__(self, proc: subprocess.Popen[Any], record: ProcessOwnershipRecord) -> None:
        self.proc = proc
        self.record = record
        self._pidfd: int | None = None
        if hasattr(os, "pidfd_open"):
            try:
                self._pidfd = os.pidfd_open(proc.pid, 0)
            except OSError:
                self._pidfd = None

    @property
    def pid(self) -> int:
        return self.record.pid

    def is_alive(self) -> bool:
        if self.proc.poll() is not None:
            return False
        ok, _ = verify_process_ownership(self.record)
        return ok

    def send_signal(self, sig: int) -> tuple[bool, str]:
        """Deliver a signal strictly to this owned child (kernel pidfd_send_signal prevents PID-reuse races)."""
        if self.proc.poll() is not None:
            return False, "process_exited"
        if self._pidfd is not None and hasattr(signal, "pidfd_send_signal"):
            try:
                signal.pidfd_send_signal(self._pidfd, sig)
                return True, "signalled_via_pidfd"
            except ProcessLookupError:
                return False, "process_exited"
            except OSError as exc:
                return False, f"pidfd_error:{exc}"
        ok, reason = verify_process_ownership(self.record)
        if not ok:
            return False, reason
        try:
            os.kill(self.record.pid, sig)
            return True, "signalled_verified_pid"
        except ProcessLookupError:
            return False, "process_exited"
        except OSError as exc:
            return False, f"kill_error:{exc}"

    def close(self) -> None:
        if self._pidfd is not None:
            try:
                os.close(self._pidfd)
            except OSError:
                pass
            self._pidfd = None


def send_signal_if_owned(
    record: ProcessOwnershipRecord | None,
    sig: int,
    *,
    handle: OwnedProcessHandle | None = None,
) -> tuple[bool, str]:
    """Signal a process ONLY if ownership is verified via live handle or OS identity check."""
    if handle is not None and record is not None and handle.pid == record.pid:
        return handle.send_signal(sig)
    ok, reason = verify_process_ownership(record)
    if not ok or record is None:
        return False, reason
    try:
        os.kill(record.pid, sig)
        return True, "signalled_verified_record"
    except ProcessLookupError:
        return False, "process_exited"
    except OSError as exc:
        return False, f"kill_error:{exc}"


def load_ownership_record(path: Path) -> ProcessOwnershipRecord | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return ProcessOwnershipRecord.from_dict(raw)


def save_ownership_record(path: Path, record: ProcessOwnershipRecord) -> None:
    write_json_atomic(path, record.to_dict())


# --------------------------------------------------------------------------- P0-B: Media Ownership


def _is_canonically_within(target: Path, root: Path) -> bool:
    try:
        real_target = Path(os.path.realpath(target))
        real_root = Path(os.path.realpath(root))
        return real_target == real_root or real_target.is_relative_to(real_root)
    except OSError:
        return False


@dataclass(frozen=True)
class MediaOwnershipResult:
    classification: MediaOwnershipClass
    deletable: bool
    canonical_path: str
    reason: str


def classify_media_ownership(
    raw_path: str | Path,
    *,
    managed_output_roots: list[Path],
    managed_import_roots: list[Path] | None = None,
    staging_roots: list[Path] | None = None,
) -> MediaOwnershipResult:
    """Classify a media file path strictly before any unlink/cleanup operation.

    Fail-closed invariants:
    - Any symlink pointing outside managed roots is NEVER deletable.
    - Any hard-linked file (st_nlink > 1) is NEVER deletable (prevents silently dropping a link to an external original).
    - Any file outside explicit app-owned managed directories is ALWAYS classified ``external_original`` and NOT deletable.
    """
    text = str(raw_path or "").strip()
    if not text:
        return MediaOwnershipResult("missing", False, "", "Empty file path.")

    candidate = Path(text).expanduser()
    try:
        lst = os.lstat(candidate)
    except FileNotFoundError:
        return MediaOwnershipResult("missing", False, str(candidate), "File does not exist.")
    except OSError as exc:
        return MediaOwnershipResult("external_original", False, str(candidate), f"Cannot stat path: {exc}")

    canonical = Path(os.path.realpath(candidate))
    out_roots = [Path(r).expanduser() for r in managed_output_roots if r]
    imp_roots = [Path(r).expanduser() for r in (managed_import_roots or []) if r]
    stg_roots = [Path(r).expanduser() for r in (staging_roots or []) if r]

    # Reject symlinks whose target escapes managed directories (or symlinks altogether for delete safety).
    if candidate.is_symlink():
        if not any(_is_canonically_within(canonical, r) for r in (*out_roots, *imp_roots, *stg_roots)):
            return MediaOwnershipResult(
                "external_symlink",
                False,
                str(canonical),
                "Symlink points outside Avatar V2 managed directories; refusing deletion.",
            )
        return MediaOwnershipResult(
            "external_symlink",
            False,
            str(canonical),
            "Symlinks are detached rather than unlinked to protect source targets.",
        )

    # Reject directories or non-regular files.
    if not candidate.is_file():
        return MediaOwnershipResult(
            "external_original",
            False,
            str(canonical),
            "Target is not a regular file.",
        )

    # Protect hard-linked files (st_nlink > 1 could be hard-linked to an external user original).
    if lst.st_nlink > 1:
        return MediaOwnershipResult(
            "hardlinked_shared",
            False,
            str(canonical),
            f"File has multiple hard links (nlink={lst.st_nlink}); refusing physical deletion.",
        )

    # Check canonical containment inside explicit app-owned directories ONLY.
    if any(_is_canonically_within(canonical, r) for r in stg_roots):
        return MediaOwnershipResult("ephemeral_staging", True, str(canonical), "App-owned ephemeral staging cache file.")
    if any(_is_canonically_within(canonical, r) for r in out_roots):
        return MediaOwnershipResult("managed_output", True, str(canonical), "App-owned managed output file.")
    if any(_is_canonically_within(canonical, r) for r in imp_roots):
        return MediaOwnershipResult("managed_import", True, str(canonical), "App-owned managed imported copy.")

    return MediaOwnershipResult(
        "external_original",
        False,
        str(canonical),
        "File resides outside Avatar V2 managed directories and is externally owned; refusing physical deletion.",
    )
