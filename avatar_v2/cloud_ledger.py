"""Append-only local ledger of cloud generation cost, used for history and budget limits.

Every Runway task appends an entry when it is created (Runway's own maximum estimate) and
when it finishes (the credits Runway actually charged). The latest entry per task wins.
Unfinished tasks count against budgets at their estimate, so a burst of queued jobs cannot
overshoot a daily or monthly cap.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record(path: Path | None, entry: dict[str, Any]) -> None:
    """Append one ledger entry. A missing path disables the ledger (used by unit-test shells)."""
    if path is None:
        return
    line = {"ts": _now().isoformat(), **entry}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


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
    """Latest entry per task id, keeping the first-seen timestamp as ``created_ts``."""
    states: dict[str, dict[str, Any]] = {}
    for item in entries(path):
        task_id = str(item.get("task_id") or "")
        if not task_id:
            continue
        created = states.get(task_id, {}).get("created_ts") or item.get("ts")
        states[task_id] = {**states.get(task_id, {}), **item, "created_ts": created}
    return states


def _credits(state: dict[str, Any]) -> tuple[float, bool]:
    """(credits, final) for one task state."""
    status = str(state.get("status") or "")
    if status in FINAL_STATUSES:
        charged = state.get("credits_charged")
        return (float(charged) if isinstance(charged, (int, float)) else 0.0), True
    for key in ("api_estimated_credits", "estimate_credits"):
        value = state.get(key)
        if isinstance(value, (int, float)):
            return float(value), False
    return 0.0, False


def spend_summary(path: Path | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Charged and reserved credits for today and this calendar month (UTC)."""
    current = now or _now()
    today = current.date().isoformat()
    month = current.strftime("%Y-%m")
    summary: dict[str, Any] = {
        "today_charged": 0.0,
        "today_reserved": 0.0,
        "month_charged": 0.0,
        "month_reserved": 0.0,
        "all_time_charged": 0.0,
        "tasks": 0,
        "by_model": {},
    }
    for state in task_states(path).values():
        credits, final = _credits(state)
        created = str(state.get("created_ts") or "")
        summary["tasks"] += 1
        if final:
            summary["all_time_charged"] += credits
            model = str(state.get("model") or "unknown")
            summary["by_model"][model] = summary["by_model"].get(model, 0.0) + credits
        bucket = "charged" if final else "reserved"
        if created.startswith(today):
            summary[f"today_{bucket}"] += credits
        if created.startswith(month):
            summary[f"month_{bucket}"] += credits
    summary["today_total"] = summary["today_charged"] + summary["today_reserved"]
    summary["month_total"] = summary["month_charged"] + summary["month_reserved"]
    return summary
