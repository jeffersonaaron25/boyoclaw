"""Interactive Telegram setup (``python -m runtime telegram configure``)."""

from __future__ import annotations

import getpass
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from runtime.telegram_config import (
    TelegramSettings,
    load_telegram_settings,
    mask_token,
    save_telegram_settings,
    telegram_config_path,
)


def _get_me(token: str) -> dict | None:
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 — Telegram API
            data = json.loads(resp.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None
    if not data.get("ok"):
        return None
    return data.get("result")


def _parse_chat_ids(line: str) -> list[int]:
    out: list[int] = []
    for part in line.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def run_configure(workspace_root: Path) -> int:
    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    path = telegram_config_path(workspace_root)

    print("BoyoClaw — Telegram setup")
    print(f"Config file: {path}")
    print()
    print("1. Open Telegram, find @BotFather, send /newbot (or use an existing bot).")
    print("2. Copy the HTTP API token BotFather gives you.")
    print()

    existing = load_telegram_settings(workspace_root)
    token_default = existing.bot_token if existing else ""
    prompt = "Bot token"
    if token_default:
        prompt += f" [leave empty to keep {mask_token(token_default)}]"
    prompt += ": "

    token = getpass.getpass(prompt).strip() or token_default
    if not token:
        print("Error: bot token is required.", file=sys.stderr)
        return 1

    print("Checking token with Telegram…")
    me = _get_me(token)
    if not me:
        print("Error: invalid token or Telegram unreachable.", file=sys.stderr)
        return 1
    username = me.get("username", "?")
    print(f"OK — bot @{username}")

    ids_line = ""
    if existing and existing.allowed_chat_ids:
        ids_line = " ".join(str(i) for i in existing.allowed_chat_ids)
    print()
    print(
        "Allowed chat IDs (only these Telegram chats can use the bot). "
        "Message @userinfobot on Telegram to see your numeric id. "
        "Separate multiple ids with spaces or commas.",
    )
    raw = input(f"Chat id(s) [{ids_line}]: ").strip()
    if raw:
        chat_ids = _parse_chat_ids(raw)
    else:
        chat_ids = list(existing.allowed_chat_ids) if existing else []

    if not chat_ids:
        print("Error: at least one allowed chat id is required.", file=sys.stderr)
        return 1

    settings = TelegramSettings(bot_token=token, allowed_chat_ids=chat_ids)
    out = save_telegram_settings(workspace_root, settings)
    print()
    print(f"Saved {out}")
    print("Start the runtime with:  python -m runtime --telegram")
    return 0


def run_show(workspace_root: Path) -> int:
    workspace_root = workspace_root.resolve()
    path = telegram_config_path(workspace_root)
    if not path.is_file():
        print(f"No Telegram config at {path}")
        print("Run:  python -m runtime telegram configure")
        return 1
    s = load_telegram_settings(workspace_root)
    if not s:
        print(f"{path} exists but is invalid or incomplete.")
        return 1
    print(f"Config: {path}")
    print(f"  Bot token: {mask_token(s.bot_token)}")
    print(f"  Allowed chat ids: {', '.join(str(x) for x in s.allowed_chat_ids)}")
    return 0
