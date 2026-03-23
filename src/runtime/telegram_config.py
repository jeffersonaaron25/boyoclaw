"""Telegram integration settings stored in the workspace (JSON)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "telegram.json"


@dataclass
class TelegramSettings:
    bot_token: str
    allowed_chat_ids: list[int]

    def to_json_dict(self) -> dict[str, Any]:
        return {"bot_token": self.bot_token, "allowed_chat_ids": self.allowed_chat_ids}


def telegram_config_path(workspace_root: Path) -> Path:
    return Path(workspace_root).resolve() / CONFIG_FILENAME


def load_telegram_settings(workspace_root: Path) -> TelegramSettings | None:
    path = telegram_config_path(workspace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = (data.get("bot_token") or "").strip()
    raw_ids = data.get("allowed_chat_ids")
    if not token or not isinstance(raw_ids, list):
        return None
    ids: list[int] = []
    for x in raw_ids:
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            continue
    if not ids:
        return None
    return TelegramSettings(bot_token=token, allowed_chat_ids=ids)


def save_telegram_settings(workspace_root: Path, settings: TelegramSettings) -> Path:
    workspace_root = Path(workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    path = telegram_config_path(workspace_root)
    path.write_text(
        json.dumps(asdict(settings), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def mask_token(token: str) -> str:
    t = token.strip()
    if len(t) <= 8:
        return "****"
    return f"{t[:4]}…{t[-4:]}"
