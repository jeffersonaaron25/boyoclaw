"""Save Telegram file payloads into the agent sandbox workspace."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path


UPLOAD_SUBDIR = "telegram_uploads"


def _safe_filename(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    base = Path(str(name).replace("\\", "/")).name.strip()
    if not base or base in (".", ".."):
        return fallback
    return base[:200]


async def download_telegram_file(
    bot: object,
    *,
    agent_home: Path,
    file_id: str,
    suggested_name: str,
) -> str:
    """Download a Telegram file into the agent workspace (``.agent-home``). Returns a path relative to that root."""
    root = Path(agent_home).resolve()
    day = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    dest_dir = root / UPLOAD_SUBDIR / day
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(suggested_name, "file.bin")
    unique = f"{uuid.uuid4().hex[:10]}_{safe}"
    dest = dest_dir / unique

    tg_file = await bot.get_file(file_id)
    await tg_file.download_to_drive(custom_path=dest)

    rel = dest.relative_to(root)
    return rel.as_posix()
