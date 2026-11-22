"""Append-only local ledger of cloud generation cost, used for history and budget limits.

Each entry is keyed by its Runway ``task_id``, or by a ``request_key`` when a submission's outcome
is unknown and no task id was ever received. The latest entry per key wins.

Charges are conservative:

* ``charge_status == "actual"``: Runway reported the credits charged.
* ``charge_status == "none"``: the request provably never created a task (never sent, or rejected).
* anything else on a finished, cancelled, failed or uncertain entry: the charge is **unknown**, so its
  estimate counts as *unconfirmed* spend until the owner resolves it.

In-flight tasks count at their estimate as *reserved*. Unconfirmed and reserved credits both count
against daily and monthly budgets.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
SETTLED_STATUSES = FINAL_STATUSES | {"UNCERTAIN", "RESOLVED"}
_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def entry_key(entry: dict[str, Any]) -> str:
    return str(entry.get("task_id") or entry.get("request_key") or "")


def record(path: Path | None, entry: dict[str, Any]) -> None:
    """Append one entry durably (flush + fsync). A missing path disables the ledger (unit-test shells)."""
    if path is None:
        return
    line = {"ts": _now().isoformat(), **entry}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    output: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            output.append(item)
    return output


def task_states(path: Path | None) -> dict[str, dict[str, Any]]:
    """Latest merged entry per key, keeping the first-seen timestamp as ``created_ts``."""
    states: dict[str, dict[str, Any]] = {}
    for item in entries(path):
        key = entry_key(item)
        if not key:
            continue
        created = states.get(key, {}).get("created_ts") or item.get("ts")
        states[key] = {**states.get(key, {}), **item, "created_ts": created, "key": key}
    return states


def _estimate(state: dict[str, Any]) -> float:
    for name in ("api_estimated_credits", "estimate_credits"):
        value = state.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def classify(state: dict[str, Any]) -> tuple[float, str]:
    """(credits, bucket) where bucket is 'charged', 'unconfirmed' or 'reserved'."""
    status = str(state.get("status") or "")
    charge = state.get("charge_status")
    charged = state.get("credits_charged")
    if status in SETTLED_STATUSES:
        if charge == "none":
            return 0.0, "charged"
        if charge == "actual" or (charge is None and status in FINAL_STATUSES and isinstance(charged, (int, float))):
            return float(charged or 0), "charged"
        return _estimate(state), "unconfirmed"
    return _estimate(state), "reserved"


def unresolved(path: Path | None) -> list[dict[str, Any]]:
    """Entries whose charge is still unknown (counted at their estimate until resolved)."""
    items = []
    for state in task_states(path).values():
        credits, bucket = classify(state)
        if bucket == "unconfirmed":
            items.append({**state, "reserved_credits": credits})
    return sorted(items, key=lambda item: item.get("created_ts", ""), reverse=True)


def spend_summary(path: Path | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Charged, unconfirmed and reserved credits for today and this calendar month (UTC)."""
    current = now or _now()
    today = current.date().isoformat()
    month = current.strftime("%Y-%m")
    summary: dict[str, Any] = {"tasks": 0, "by_model": {}, "all_time_charged": 0.0, "unresolved": 0}
    for scope in ("today", "month"):
        for bucket in ("charged", "unconfirmed", "reserved"):
            summary[f"{scope}_{bucket}"] = 0.0
    for state in task_states(path).values():
        credits, bucket = classify(state)
        created = str(state.get("created_ts") or "")
        summary["tasks"] += 1
        if bucket == "charged":
            summary["all_time_charged"] += credits
            model = str(state.get("model") or "unknown")
            summary["by_model"][model] = summary["by_model"].get(model, 0.0) + credits
        if bucket == "unconfirmed":
            summary["unresolved"] += 1
        if created.startswith(today):
            summary[f"today_{bucket}"] += credits
        if created.startswith(month):
            summary[f"month_{bucket}"] += credits
    for scope in ("today", "month"):
        summary[f"{scope}_total"] = sum(summary[f"{scope}_{bucket}"] for bucket in ("charged", "unconfirmed", "reserved"))
    return summary
