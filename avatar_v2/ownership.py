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

import hashlib
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
from urllib.parse import urlparse

EngineOwnership = Literal["owned_private", "shared_or_external", "stale_or_reused", "stopped"]
MediaOwnershipClass = Literal[
    "generated_output",
    "managed_archive_copy",
    "managed_output",
    "managed_import_copy",
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


def read_json_safe(
    path: Path,
    default: Any,
    *,
    quarantine: bool = True,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    """Read JSON safely; quarantine corrupt files instead of silently overwriting them later."""
    target = Path(path)
    if not target.is_file():
        return default
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if expected_type is not None and not isinstance(data, expected_type):
            raise ValueError(f"Unexpected JSON root type {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError):
        if quarantine:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            quarantined = target.with_name(f"{target.name}.corrupt.{stamp}")
            try:
                os.replace(target, quarantined)
            except OSError:
                pass
        return default
    except OSError:
        return default


class DirectoryWriterLock:
    """Non-blocking POSIX flock guarding single-writer ownership of a state directory."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"Another Avatar V2 instance already holds the writer lock at {self.lock_path}."
            )
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __del__(self) -> None:
        self.release()


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


_LOOPBACK_ENDPOINT_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def parse_endpoint_host_port(base_url: str) -> tuple[str, int] | None:
    """Parse (host, port) from an HTTP(S) renderer URL; returns None if malformed."""
    raw = str(base_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname.lower(), int(port)


def pid_listens_on_port(pid: int, port: int) -> bool | None:
    """Return True if ``pid`` owns a listening TCP socket on ``port``, False if verified not, None if unreadable."""
    if pid <= 0 or port <= 0:
        return False
    fd_dir = Path(f"/proc/{pid}/fd")
    if not fd_dir.is_dir():
        return None
    socket_inodes: set[str] = set()
    try:
        for entry in fd_dir.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                socket_inodes.add(target[8:-1])
    except OSError:
        return None

    listening_inodes: set[str] = set()
    any_proc_net = False
    port_hex = f"{int(port):04X}"
    for net_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = net_path.read_text(encoding="utf-8").splitlines()[1:]
            any_proc_net = True
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            # 0A == TCP_LISTEN
            if state != "0A":
                continue
            _, _, lport = local_addr.rpartition(":")
            if lport.upper() == port_hex:
                listening_inodes.add(inode)
    if not any_proc_net:
        return None
    return bool(socket_inodes & listening_inodes)


def verify_endpoint_ownership(
    record: ProcessOwnershipRecord | None,
    base_url: str,
    *,
    require_listening: bool = False,
) -> tuple[bool, str]:
    """Fail-closed binding between an owned ComfyUI process record and a target HTTP endpoint (RC-01)."""
    ok, reason = verify_process_ownership(record)
    if not ok or record is None:
        return False, reason
    parsed = parse_endpoint_host_port(base_url)
    if parsed is None:
        return False, "invalid_endpoint_url"
    host, port = parsed
    if host not in _LOOPBACK_ENDPOINT_HOSTS:
        return False, f"non_loopback_endpoint:{host}"
    if record.port is None:
        return False, "record_missing_port"
    if int(record.port) != int(port):
        return False, f"endpoint_port_mismatch:owned_{record.port}_vs_target_{port}"
    if require_listening:
        listening = pid_listens_on_port(record.pid, port)
        if listening is False:
            return False, f"owned_pid_not_listening_on_port:{port}"
    return True, "verified_owned_endpoint"


def send_signal_if_owned(
    record: ProcessOwnershipRecord | None,
    sig: int,
    *,
    handle: OwnedProcessHandle | None = None,
) -> tuple[bool, str]:
    """Signal a process ONLY if ownership is verified via atomic kernel pidfd (zero PID-reuse TOCTOU!)."""
    if handle is not None and record is not None and handle.pid == record.pid:
        return handle.send_signal(sig)
    if record is None or record.pid <= 0:
        return False, "no_ownership_record"

    # RC-01 Gate 2: On Linux with pidfd_open + pidfd_send_signal, pin the kernel task FIRST,
    # THEN verify OS start_ticks/boot_id/exe/cmdline/port while holding pidfd open, THEN signal pidfd!
    # Eliminates verify-then-numeric-kill PID reuse TOCTOU after application restart.
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            pidfd = os.pidfd_open(record.pid, 0)
        except ProcessLookupError:
            return False, "process_exited"
        except OSError as exc:
            return False, f"pidfd_open_error:{exc}"
        try:
            ok, reason = verify_process_ownership(record)
            if not ok:
                return False, reason
            try:
                signal.pidfd_send_signal(pidfd, sig)
                return True, "signalled_via_reopened_pidfd"
            except ProcessLookupError:
                return False, "process_exited"
            except OSError as exc:
                return False, f"pidfd_signal_error:{exc}"
        finally:
            try:
                os.close(pidfd)
            except OSError:
                pass

    ok, reason = verify_process_ownership(record)
    if not ok:
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


# --------------------------------------------------------------------------- Gate 1 (RC-02): Durable Explicit Managed Asset Registry

_MANAGED_ASSET_ORIGINS = frozenset(
    {"generated_output", "managed_archive_copy", "managed_import_copy", "ephemeral_staging"}
)


def _sha256_file_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ManagedAssetRecord:
    """Explicit durable proof that Avatar V2 created or copied this exact asset on disk."""

    canonical_path: str
    sha256: str
    size_bytes: int
    inode: int
    device: int
    origin: Literal["generated_output", "managed_archive_copy", "managed_import_copy", "ephemeral_staging"]
    job_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> ManagedAssetRecord | None:
        if not isinstance(raw, dict):
            return None
        origin = str(raw.get("origin") or "")
        if origin not in _MANAGED_ASSET_ORIGINS:
            return None
        cpath = str(raw.get("canonical_path") or "").strip()
        digest = str(raw.get("sha256") or "").strip()
        if not cpath or len(digest) != 64:
            return None
        try:
            size_b = int(raw["size_bytes"])
            ino = int(raw["inode"])
            dev = int(raw["device"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            canonical_path=cpath,
            sha256=digest,
            size_bytes=size_b,
            inode=ino,
            device=dev,
            origin=origin,  # type: ignore[arg-type]
            job_id=str(raw.get("job_id") or ""),
            created_at=str(raw.get("created_at") or ""),
        )


def _load_managed_asset_map(registry_path: Path) -> dict[str, dict[str, Any]]:
    data = read_json_safe(Path(registry_path), {}, quarantine=True, expected_type=dict)
    return data if isinstance(data, dict) else {}


def register_managed_asset(
    registry_path: Path,
    file_path: str | Path,
    *,
    origin: Literal["generated_output", "managed_archive_copy", "managed_import_copy", "ephemeral_staging"],
    job_id: str = "",
) -> ManagedAssetRecord | None:
    """Durably register an asset generated or copied by Avatar V2. Refuses symlinks & hardlinks."""
    if origin not in _MANAGED_ASSET_ORIGINS:
        raise ValueError(f"Unsupported managed asset origin: {origin!r}")
    candidate = Path(str(file_path)).expanduser()
    try:
        lst = os.lstat(candidate)
    except OSError:
        return None
    if candidate.is_symlink() or not candidate.is_file() or lst.st_nlink > 1:
        return None
    canonical = str(Path(os.path.realpath(candidate)))
    try:
        digest = _sha256_file_bytes(Path(canonical))
    except OSError:
        return None
    rec = ManagedAssetRecord(
        canonical_path=canonical,
        sha256=digest,
        size_bytes=int(lst.st_size),
        inode=int(lst.st_ino),
        device=int(lst.st_dev),
        origin=origin,
        job_id=str(job_id or ""),
    )
    reg = Path(registry_path)
    lock = DirectoryWriterLock(reg.with_name(f".{reg.name}.lock"))
    try:
        lock.acquire()
    except Exception:
        lock = None  # fallback if lock dir is read-only; atomic write still runs
    try:
        mapping = _load_managed_asset_map(reg)
        mapping[canonical] = rec.to_dict()
        write_json_atomic(reg, mapping)
    finally:
        if lock is not None:
            lock.release()
    return rec


def unregister_managed_asset(registry_path: Path, raw_path: str | Path) -> bool:
    """Remove a managed asset record after verified deletion."""
    reg = Path(registry_path)
    if not reg.is_file():
        return False
    canonical = str(Path(os.path.realpath(Path(str(raw_path)).expanduser())))
    lock = DirectoryWriterLock(reg.with_name(f".{reg.name}.lock"))
    try:
        lock.acquire()
    except Exception:
        lock = None
    try:
        mapping = _load_managed_asset_map(reg)
        removed = mapping.pop(canonical, None) is not None
        if removed:
            write_json_atomic(reg, mapping)
        return removed
    finally:
        if lock is not None:
            lock.release()


def verify_managed_asset_ownership(
    registry_path: Path,
    raw_path: str | Path,
) -> tuple[bool, MediaOwnershipResult]:
    """Fail-closed proof that Avatar V2 owns the exact file on disk (RC-02 Gate 1).

    Directory membership, filename, history presence, master status, and file extension
    NEVER authorize physical deletion without a matching durable ownership record whose
    canonical path, device, inode, size, and SHA-256 all match the live file!
    """
    text = str(raw_path or "").strip()
    if not text:
        return False, MediaOwnershipResult("missing", False, "", "Empty file path.")
    candidate = Path(text).expanduser()
    try:
        lst = os.lstat(candidate)
    except FileNotFoundError:
        return False, MediaOwnershipResult("missing", False, str(candidate), "File does not exist.")
    except OSError as exc:
        return False, MediaOwnershipResult("external_original", False, str(candidate), f"Cannot stat path: {exc}")

    canonical = str(Path(os.path.realpath(candidate)))
    if candidate.is_symlink():
        return False, MediaOwnershipResult(
            "external_symlink",
            False,
            canonical,
            "Symlinks are detached rather than unlinked to protect source targets.",
        )
    if not candidate.is_file():
        return False, MediaOwnershipResult("external_original", False, canonical, "Target is not a regular file.")
    if lst.st_nlink > 1:
        return False, MediaOwnershipResult(
            "hardlinked_shared",
            False,
            canonical,
            f"File has multiple hard links (nlink={lst.st_nlink}); refusing physical deletion.",
        )

    mapping = _load_managed_asset_map(Path(registry_path))
    rec = ManagedAssetRecord.from_dict(mapping.get(canonical))
    if rec is None:
        return False, MediaOwnershipResult(
            "external_original",
            False,
            canonical,
            "No verified Avatar V2 managed-asset ownership record exists for this file; refusing physical deletion.",
        )

    if int(lst.st_dev) != rec.device or int(lst.st_ino) != rec.inode:
        return False, MediaOwnershipResult(
            "external_original",
            False,
            canonical,
            "File inode/device no longer matches Avatar V2's managed-asset record (replaced or moved); refusing deletion.",
        )
    if int(lst.st_size) != rec.size_bytes:
        return False, MediaOwnershipResult(
            "external_original",
            False,
            canonical,
            "File size no longer matches Avatar V2's managed-asset record; refusing deletion.",
        )
    try:
        live_sha = _sha256_file_bytes(Path(canonical))
    except OSError as exc:
        return False, MediaOwnershipResult("external_original", False, canonical, f"Cannot hash file: {exc}")
    if live_sha != rec.sha256:
        return False, MediaOwnershipResult(
            "external_original",
            False,
            canonical,
            "File SHA-256 content no longer matches Avatar V2's managed-asset record; refusing deletion.",
        )

    return True, MediaOwnershipResult(
        rec.origin,
        True,
        canonical,
        f"Verified Avatar V2 managed asset ({rec.origin}, job={rec.job_id or 'managed'}).",
    )
