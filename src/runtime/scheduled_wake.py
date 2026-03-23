"""Persistent scheduled wakes: ISO-8601 time + context in ``scheduled_wakes.json``."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NotRequired, TypedDict

_LOCK = threading.Lock()
FILENAME = "scheduled_wakes.json"


class ScheduledWakeRecord(TypedDict):
    id: str
    at: str  # ISO 8601 (UTC stored)
    context: str
    telegram_chat_id: NotRequired[int]  # replies route here if set (Telegram session when scheduled)


def _store_path(sandbox_root: Path) -> Path:
    return Path(sandbox_root).resolve() / FILENAME


def _load(sandbox_root: Path) -> list[ScheduledWakeRecord]:
    p = _store_path(sandbox_root)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[ScheduledWakeRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        iid = str(item.get("id", "")).strip()
        at = str(item.get("at", "")).strip()
        ctx = str(item.get("context", ""))
        if iid and at:
            row: ScheduledWakeRecord = {"id": iid, "at": at, "context": ctx}
            tc = item.get("telegram_chat_id")
            if tc is not None:
                try:
                    row["telegram_chat_id"] = int(tc)
                except (TypeError, ValueError):
                    pass
            out.append(row)
    return out


def _save(sandbox_root: Path, rows: list[ScheduledWakeRecord]) -> None:
    p = _store_path(sandbox_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def parse_when_iso(when_iso: str) -> datetime:
    """Parse ISO 8601; naive times use the local system timezone."""
    s = when_iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        if local_tz is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc)


def schedule_wake_add(
    sandbox_root: Path,
    when_iso: str,
    context: str,
    *,
    telegram_chat_id: int | None = None,
) -> tuple[bool, str]:
    """Validate future UTC instant, append record. Returns (ok, json message or error text)."""
    context = context.strip()
    if not context:
        return False, "context must be non-empty."
    try:
        when = parse_when_iso(when_iso)
    except (ValueError, TypeError) as e:
        return False, f"Invalid datetime (use ISO 8601, e.g. 2026-03-23T14:30:00-07:00 or ...Z): {e}"
    now = datetime.now(timezone.utc)
    if when <= now:
        return False, "Scheduled time must be strictly in the future."
    rec: ScheduledWakeRecord = {
        "id": str(uuid.uuid4()),
        "at": when.isoformat(),
        "context": context[:8000],
    }
    if telegram_chat_id is not None:
        rec["telegram_chat_id"] = int(telegram_chat_id)
    with _LOCK:
        rows = _load(sandbox_root)
        rows.append(rec)
        _save(sandbox_root, rows)
    payload = {"ok": True, "id": rec["id"], "at_utc": rec["at"], "message": "Wake will run at that time with your saved context."}
    return True, json.dumps(payload, indent=2)


def delete_wake(sandbox_root: Path, wake_id: str) -> None:
    with _LOCK:
        rows = [r for r in _load(sandbox_root) if r["id"] != wake_id]
        _save(sandbox_root, rows)


def list_due_wakes(sandbox_root: Path) -> list[ScheduledWakeRecord]:
    """Records whose ``at`` is <= now (UTC), oldest due time first."""
    now = datetime.now(timezone.utc)
    due: list[tuple[datetime, ScheduledWakeRecord]] = []
    for r in _load(sandbox_root):
        try:
            raw = r["at"].strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            at = datetime.fromisoformat(raw)
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            at = at.astimezone(timezone.utc)
        except (ValueError, KeyError):
            continue
        if at <= now:
            due.append((at, r))
    due.sort(key=lambda x: x[0])
    return [r for _, r in due]
