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
import ipaddress
import json
import os
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    _fcntl = None

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
            if os.name == "posix":
                raise
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
    """Fail-closed verification that ``record`` still identifies the exact app-owned OS process (F01 / RC-01)."""
    if record is None:
        return False, "no_ownership_record"
    if not record.owned_by_app:
        return False, "not_owned_by_app"
    if record.pid <= 0:
        return False, "invalid_pid"
    if record.start_ticks is None:
        # Never trust a PID without start-time identity (eliminates bare-PID reuse risk!).
        return False, "missing_start_ticks"
    if sys.platform.startswith("linux") and not str(record.boot_id or "").strip():
        # F01: Missing boot identity on Linux MUST fail closed!
        return False, "missing_record_boot_id"

    live = inspect_process_identity(record.pid)
    if live is None:
        return False, "process_exited"
    if sys.platform.startswith("linux"):
        live_boot = str(live.get("boot_id") or "").strip()
        if not live_boot:
            return False, "missing_live_boot_id"
        if str(record.boot_id).strip() != live_boot:
            return False, "boot_id_mismatch"
    elif record.boot_id and live["boot_id"] and record.boot_id != live["boot_id"]:
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
        """Deliver a signal strictly to this owned child via kernel pidfd (zero PID-reuse race)."""
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
        if sys.platform.startswith("linux"):
            # F01: On Linux, require a stable kernel pidfd handle rather than falling back to numeric os.kill.
            return send_signal_if_owned(self.record, sig, handle=None)
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
    ident = parse_endpoint_identity(base_url)
    if ident is None:
        return None
    _scheme, host, port, _family = ident
    return host, port


def parse_endpoint_identity(base_url: str) -> tuple[str, str, int, int] | None:
    """Parse strict (scheme, host_literal, port, af_family_or_0) without DNS normalization (F01 / RC-01).

    Rejects credentials, query/fragment, non-http(s) schemes, and non-loopback hosts.
    """
    raw = str(base_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if port <= 0 or port > 65535:
        return None
    host = parsed.hostname.lower()
    if host == "127.0.0.1":
        return parsed.scheme, "127.0.0.1", int(port), int(socket.AF_INET)
    if host == "::1":
        return parsed.scheme, "::1", int(port), int(socket.AF_INET6)
    if host == "localhost":
        return parsed.scheme, "localhost", int(port), 0
    return parsed.scheme, host, int(port), -1


def _parse_proc_net_ip(hex_ip: str, *, is_v6: bool) -> str | None:
    """Decode Linux /proc/net/tcp or /proc/net/tcp6 local IP hex string into canonical IP literal."""
    try:
        if not is_v6:
            if len(hex_ip) != 8:
                return None
            raw = int(hex_ip, 16).to_bytes(4, "little")
            return str(ipaddress.IPv4Address(raw))
        if len(hex_ip) != 32:
            return None
        chunks = [int(hex_ip[i : i + 8], 16).to_bytes(4, "little") for i in range(0, 32, 8)]
        return str(ipaddress.IPv6Address(b"".join(chunks)))
    except ValueError:
        return None


def pid_listening_endpoints(pid: int) -> tuple[bool, set[tuple[int, str, int]], str]:
    """Return (telemetry_ok, {(af_family, ip_str, port)}, error) for TCP_LISTEN sockets owned by ``pid``.

    Strictly fail-closed (F01 / RC-01): unreadable /proc/<pid>/fd or /proc/net/tcp returns telemetry_ok=False.
    """
    if pid <= 0:
        return False, set(), "invalid_pid"
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        entries = list(fd_dir.iterdir())
    except OSError as exc:
        return False, set(), f"unreadable_proc_fd:{exc}"
    socket_inodes: set[str] = set()
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            socket_inodes.add(target[8:-1])

    endpoints: set[tuple[int, str, int]] = set()
    readable_tables = 0
    for net_path, is_v6, family in (
        (Path("/proc/net/tcp"), False, int(socket.AF_INET)),
        (Path("/proc/net/tcp6"), True, int(socket.AF_INET6)),
    ):
        try:
            lines = net_path.read_text(encoding="utf-8").splitlines()[1:]
            readable_tables += 1
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if state != "0A" or inode not in socket_inodes:
                continue
            hex_ip, _, lport_hex = local_addr.rpartition(":")
            ip_str = _parse_proc_net_ip(hex_ip, is_v6=is_v6)
            if ip_str is None:
                continue
            try:
                lport = int(lport_hex, 16)
            except ValueError:
                continue
            endpoints.add((family, ip_str, lport))
    if readable_tables == 0:
        return False, set(), "unreadable_proc_net_tcp"
    return True, endpoints, "ok"


def pid_listens_on_port(pid: int, port: int) -> bool:
    """Fail-closed check whether ``pid`` owns ANY TCP_LISTEN socket on ``port`` (unreadable telemetry -> False)."""
    if pid <= 0 or port <= 0:
        return False
    ok, endpoints, _err = pid_listening_endpoints(pid)
    if not ok:
        return False
    return any(p == int(port) for _fam, _ip, p in endpoints)


def authorize_owned_endpoint(
    record: ProcessOwnershipRecord | None,
    base_url: str,
    *,
    require_listening: bool = True,
) -> tuple[bool, str, str]:
    """Fail-closed binding between an owned process record and an exact concrete loopback endpoint (F01 / RC-01).

    Returns ``(allowed, pinned_concrete_url, reason)``:
    - Enforces exact address family (IPv4 listener NEVER authorizes IPv6 ``[::1]`` and vice versa).
    - Fails closed on missing boot_id or unreadable ``/proc`` listener evidence.
    - Resolves ``localhost`` ONLY to the exact bound IP literal of the proven listener (zero DNS ambiguity).
    """
    ok, reason = verify_process_ownership(record)
    if not ok or record is None:
        return False, "", reason
    ident = parse_endpoint_identity(base_url)
    if ident is None:
        return False, "", "invalid_endpoint_url"
    scheme, host, port, family = ident
    if host not in _LOOPBACK_ENDPOINT_HOSTS or family < 0:
        return False, "", f"non_loopback_endpoint:{host}"
    if record.port is None:
        return False, "", "record_missing_port"
    if int(record.port) != int(port):
        return False, "", f"endpoint_port_mismatch:owned_{record.port}_vs_target_{port}"

    if require_listening:
        tel_ok, endpoints, tel_err = pid_listening_endpoints(record.pid)
        if not tel_ok:
            return False, "", f"unreadable_listener_telemetry:{tel_err}"
        v4_match = any(
            fam == int(socket.AF_INET) and p == port and ip in {"127.0.0.1", "0.0.0.0"}
            for fam, ip, p in endpoints
        )
        v6_match = any(
            fam == int(socket.AF_INET6) and p == port and ip in {"::1", "::"}
            for fam, ip, p in endpoints
        )
        if family == int(socket.AF_INET):
            if not v4_match:
                return False, "", f"ipv4_listener_not_owned_on_port:{port}"
            return True, f"{scheme}://127.0.0.1:{port}", "verified_owned_ipv4_endpoint"
        if family == int(socket.AF_INET6):
            if not v6_match:
                return False, "", f"ipv6_listener_not_owned_on_port:{port}"
            return True, f"{scheme}://[::1]:{port}", "verified_owned_ipv6_endpoint"
        # host == "localhost": pin strictly to the concrete family actually bound by the owned PID.
        if v4_match and not v6_match:
            return True, f"{scheme}://127.0.0.1:{port}", "verified_owned_localhost_pinned_ipv4"
        if v6_match and not v4_match:
            return True, f"{scheme}://[::1]:{port}", "verified_owned_localhost_pinned_ipv6"
        if v4_match and v6_match:
            return True, f"{scheme}://127.0.0.1:{port}", "verified_owned_localhost_dualstack_pinned_ipv4"
        return False, "", f"owned_pid_not_listening_on_port:{port}"

    concrete = "[::1]" if host == "::1" else ("127.0.0.1" if host == "localhost" else host)
    return True, f"{scheme}://{concrete}:{port}", "verified_owned_starting_endpoint"


def owns_connected_peer(record: ProcessOwnershipRecord, client: tuple, server: tuple) -> bool:
    """Confirm that the server end of this established loopback connection belongs to the owned PID."""
    v6 = ":" in server[0]
    table = Path("/proc/net/tcp6" if v6 else "/proc/net/tcp")
    # connect() may return just before the renderer calls accept().
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            for line in table.read_text().splitlines()[1:]:
                row = line.split()
                if len(row) < 10 or row[3] != "01":
                    continue
                local, remote = row[1].split(":"), row[2].split(":")
                if (int(local[1], 16) != server[1] or int(remote[1], 16) != client[1]
                        or _parse_proc_net_ip(local[0], is_v6=v6) != server[0]
                        or _parse_proc_net_ip(remote[0], is_v6=v6) != client[0]):
                    continue
                if row[9] != "0":
                    for fd in Path(f"/proc/{record.pid}/fd").iterdir():
                        try:
                            if os.readlink(fd) == f"socket:[{row[9]}]":
                                return verify_process_ownership(record)[0]
                        except FileNotFoundError:
                            continue
        except (OSError, ValueError, IndexError):
            return False
        time.sleep(0.01)
    return False


def verify_endpoint_ownership(
    record: ProcessOwnershipRecord | None,
    base_url: str,
    *,
    require_listening: bool = False,
) -> tuple[bool, str]:
    """Fail-closed binding between an owned ComfyUI process record and a target HTTP endpoint (F01 / RC-01)."""
    ok, _pinned, reason = authorize_owned_endpoint(record, base_url, require_listening=require_listening)
    return ok, reason


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

    # F01 / RC-01: On Linux, pin the kernel task via pidfd_open FIRST, verify full OS identity
    # (start_ticks + boot_id + exe + cmdline + port) while holding pidfd open, then deliver signal via pidfd.
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

    if sys.platform.startswith("linux"):
        return False, "pidfd_unavailable_fail_closed"

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


class RegistryLockError(OSError):
    """Raised when the managed-asset registry exclusive transaction lock cannot be acquired (F02 / RC-02)."""


_REGISTRY_THREAD_LOCKS_GUARD = threading.Lock()
_REGISTRY_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock_for(lock_path: Path) -> threading.RLock:
    key = str(lock_path.resolve(strict=False))
    with _REGISTRY_THREAD_LOCKS_GUARD:
        lk = _REGISTRY_THREAD_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _REGISTRY_THREAD_LOCKS[key] = lk
        return lk


class _RegistryTransactionLock:
    """Fail-closed thread + cross-process exclusive lock covering an entire registry transaction (F02 / RC-02)."""

    def __init__(self, registry_path: Path, *, timeout_s: float = 2.0) -> None:
        reg = Path(registry_path)
        self.lock_path = reg.with_name(f".{reg.name}.lock")
        self.timeout_s = max(0.0, float(timeout_s))
        self._tlock = _thread_lock_for(self.lock_path)
        self._fd: int | None = None

    def __enter__(self) -> _RegistryTransactionLock:
        deadline = time.monotonic() + self.timeout_s
        if not self._tlock.acquire(timeout=max(0.01, self.timeout_s)):
            raise RegistryLockError(f"Timed out acquiring in-process lock for {self.lock_path}")
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    if _fcntl is not None:
                        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    self._fd = fd
                    return self
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        raise RegistryLockError(f"Managed-asset registry lock is held by another writer: {self.lock_path}")
                    time.sleep(0.01)
                except BaseException:
                    os.close(fd)
                    raise
        except BaseException:
            self._tlock.release()
            raise

    def __exit__(self, *args: Any) -> None:
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                if _fcntl is not None:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        self._tlock.release()


def _sha256_fd(fd: int) -> str:
    h = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()


def allocate_exclusive_path(dest_dir: Path, filename: str) -> tuple[int, Path]:
    """Exclusively create a brand-new file inside ``dest_dir`` using O_CREAT|O_EXCL|O_NOFOLLOW (F02 / RC-02).

    NEVER overwrites, truncates, or replaces any pre-existing file or symlink.
    If ``dest_dir / filename`` already exists, allocates a collision-free sibling name instead.
    """
    target_dir = Path(dest_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(str(filename or "output.bin")).name
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "output.bin"
    base = Path(safe_name)
    stem, suffix = base.stem or "output", base.suffix
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    for attempt in range(64):
        name = safe_name if attempt == 0 else f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        candidate = target_dir / name
        try:
            fd = os.open(str(candidate), flags, 0o644)
            return fd, candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate exclusive destination in {target_dir} for {safe_name}")


def reserve_fresh_output_path(dest_dir: Path, filename: str) -> Path:
    """Return a non-existing path inside ``dest_dir`` without overwriting any pre-existing file (F02 / RC-02)."""
    target_dir = Path(dest_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(str(filename or "output.bin")).name
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "output.bin"
    base = Path(safe_name)
    stem, suffix = base.stem or "output", base.suffix
    for attempt in range(64):
        name = safe_name if attempt == 0 else f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        candidate = target_dir / name
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise FileExistsError(f"Could not reserve fresh output path in {target_dir} for {safe_name}")


def copy_file_exclusive(src: Path, dest_dir: Path, filename: str | None = None) -> Path:
    """Copy ``src`` into a freshly created exclusive destination in ``dest_dir`` and verify SHA-256 (F02 / RC-02).

    - NEVER overwrites an existing file at ``dest_dir / filename``.
    - On interrupted write or hash mismatch, removes ONLY the newly created destination and raises.
    """
    from .safe_output import publish_output

    source = Path(src).expanduser()
    target = dest_dir / Path(filename or source.name).name

    def copy_into(staged: Path) -> None:
        digest = hashlib.sha256()
        with source.open("rb") as reader, staged.open("xb") as writer:
            while chunk := reader.read(1 << 20):
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(writer.fileno(), view)
                    if written <= 0:
                        raise OSError("Short write during exclusive archive copy")
                    view = view[written:]
        if _sha256_file_bytes(staged) != digest.hexdigest():
            raise OSError("Exclusive copy SHA-256 verification failed")

    return publish_output(target, copy_into)


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
        job_id = str(raw.get("job_id") or "").strip()
        if not cpath or len(digest) != 64 or not job_id:
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
            job_id=job_id,
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
    allowed_roots: tuple[Path, ...] | list[Path] | None = None,
    lock_timeout_s: float = 2.0,
) -> ManagedAssetRecord | None:
    """Durably register an asset exclusively created/acquired by Avatar V2 under a fail-closed exclusive lock (F02).

    - Rejects empty ``job_id``, symlinks, hardlinks (nlink > 1), non-regular files, and paths outside ``allowed_roots``.
    - Raises ``RegistryLockError`` if the registry lock cannot be acquired (NEVER mutates unlocked!).
    """
    if origin not in _MANAGED_ASSET_ORIGINS:
        raise ValueError(f"Unsupported managed asset origin: {origin!r}")
    clean_job_id = str(job_id or "").strip()
    if not clean_job_id:
        raise ValueError("register_managed_asset requires a non-empty producing job_id")

    candidate = Path(str(file_path)).expanduser()
    reg = Path(registry_path)

    with _RegistryTransactionLock(reg, timeout_s=lock_timeout_s):
        flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0))
        try:
            fd = os.open(str(candidate), flags)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
                return None
            canonical = str(Path(os.path.realpath(candidate)))
            if allowed_roots is not None:
                canon_p = Path(canonical)
                in_root = False
                for r in allowed_roots:
                    try:
                        canon_p.relative_to(Path(r).expanduser().resolve())
                        in_root = True
                        break
                    except ValueError:
                        continue
                if not in_root:
                    return None
            digest = _sha256_fd(fd)
        finally:
            os.close(fd)

        rec = ManagedAssetRecord(
            canonical_path=canonical,
            sha256=digest,
            size_bytes=int(st.st_size),
            inode=int(st.st_ino),
            device=int(st.st_dev),
            origin=origin,
            job_id=clean_job_id,
        )
        mapping = _load_managed_asset_map(reg)
        mapping[canonical] = rec.to_dict()
        write_json_atomic(reg, mapping)
        return rec


def unregister_managed_asset(
    registry_path: Path,
    raw_path: str | Path,
    *,
    lock_timeout_s: float = 2.0,
) -> bool:
    """Remove a managed asset record under the fail-closed exclusive registry lock (F02 / RC-02)."""
    reg = Path(registry_path)
    if not reg.is_file():
        return False
    canonical = str(Path(os.path.realpath(Path(str(raw_path)).expanduser())))
    with _RegistryTransactionLock(reg, timeout_s=lock_timeout_s):
        mapping = _load_managed_asset_map(reg)
        removed = mapping.pop(canonical, None) is not None
        if removed:
            write_json_atomic(reg, mapping)
        return removed


def delete_managed_asset_if_owned(
    registry_path: Path,
    raw_path: str | Path,
    *,
    lock_timeout_s: float = 2.0,
) -> tuple[bool, MediaOwnershipResult]:
    """Atomically verify ownership via open fd + dir_fd, unlink, and remove registry entry under one lock (F02 / RC-02).

    Eliminates check-then-unlink pathname replacement races and concurrent register/delete conflicts.
    """
    text = str(raw_path or "").strip()
    if not text:
        return False, MediaOwnershipResult("missing", False, "", "Empty file path.")
    candidate = Path(text).expanduser()
    reg = Path(registry_path)

    with _RegistryTransactionLock(reg, timeout_s=lock_timeout_s):
        try:
            lst = os.lstat(candidate)
        except FileNotFoundError:
            return False, MediaOwnershipResult("missing", False, str(candidate), "File does not exist.")
        except OSError as exc:
            return False, MediaOwnershipResult("external_original", False, str(candidate), f"Cannot stat path: {exc}")

        canonical = str(Path(os.path.realpath(candidate)))
        if stat.S_ISLNK(lst.st_mode):
            return False, MediaOwnershipResult(
                "external_symlink",
                False,
                canonical,
                "Symlinks are detached rather than unlinked to protect source targets.",
            )
        if not stat.S_ISREG(lst.st_mode):
            return False, MediaOwnershipResult("external_original", False, canonical, "Target is not a regular file.")
        if lst.st_nlink > 1:
            return False, MediaOwnershipResult(
                "hardlinked_shared",
                False,
                canonical,
                f"File has multiple hard links (nlink={lst.st_nlink}); refusing physical deletion.",
            )

        mapping = _load_managed_asset_map(reg)
        rec = ManagedAssetRecord.from_dict(mapping.get(canonical))
        if rec is None:
            return False, MediaOwnershipResult(
                "external_original",
                False,
                canonical,
                "No verified Avatar V2 managed-asset ownership record exists for this file; refusing physical deletion.",
            )

        canon_path = Path(canonical)
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            dir_fd = os.open(str(canon_path.parent), dir_flags)
        except OSError as exc:
            return False, MediaOwnershipResult("external_original", False, canonical, f"Cannot open parent dir: {exc}")
        try:
            file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_fd = os.open(canon_path.name, file_flags, dir_fd=dir_fd)
            except OSError as exc:
                return False, MediaOwnershipResult("external_original", False, canonical, f"Cannot open file safely: {exc}")
            try:
                fst = os.fstat(file_fd)
                if not stat.S_ISREG(fst.st_mode) or fst.st_nlink > 1:
                    return False, MediaOwnershipResult(
                        "hardlinked_shared" if fst.st_nlink > 1 else "external_original",
                        False,
                        canonical,
                        "File type or link count changed; refusing physical deletion.",
                    )
                if int(fst.st_dev) != rec.device or int(fst.st_ino) != rec.inode:
                    return False, MediaOwnershipResult(
                        "external_original",
                        False,
                        canonical,
                        "File inode/device no longer matches Avatar V2's managed-asset record (replaced or moved); refusing deletion.",
                    )
                if int(fst.st_size) != rec.size_bytes:
                    return False, MediaOwnershipResult(
                        "external_original",
                        False,
                        canonical,
                        "File size no longer matches Avatar V2's managed-asset record; refusing deletion.",
                    )
                live_sha = _sha256_fd(file_fd)
                if live_sha != rec.sha256:
                    return False, MediaOwnershipResult(
                        "external_original",
                        False,
                        canonical,
                        "File SHA-256 content no longer matches Avatar V2's managed-asset record; refusing deletion.",
                    )
                # Final dir_fd lstat identity check immediately before dir_fd unlink!
                pre_unlink = os.lstat(canon_path.name, dir_fd=dir_fd)
                if (
                    stat.S_ISLNK(pre_unlink.st_mode)
                    or int(pre_unlink.st_dev) != rec.device
                    or int(pre_unlink.st_ino) != rec.inode
                    or pre_unlink.st_nlink > 1
                ):
                    return False, MediaOwnershipResult(
                        "external_original",
                        False,
                        canonical,
                        "File identity changed immediately before unlink; refusing deletion.",
                    )
                os.unlink(canon_path.name, dir_fd=dir_fd)
            finally:
                os.close(file_fd)
        finally:
            os.close(dir_fd)

        mapping.pop(canonical, None)
        write_json_atomic(reg, mapping)
        return True, MediaOwnershipResult(
            rec.origin,
            True,
            canonical,
            f"Verified and deleted Avatar V2 managed asset ({rec.origin}, job={rec.job_id}).",
        )


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
