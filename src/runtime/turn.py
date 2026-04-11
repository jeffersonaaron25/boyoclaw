"""Queued work for the agent worker (CLI or Telegram)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedTurn:
    """One user utterance: terminal line and/or Telegram caption plus optional saved uploads."""

    text: str
    telegram_chat_id: int | None = None
    uploaded_workspace_paths: tuple[str, ...] = ()
    # Telegram display name / @username; injected as a per-turn system message, not into inbox text.
    telegram_sender_label: str | None = None
    # SQLite id from :meth:`MessageInbox.add_human` (unread until the worker starts ``run_async``).
    human_message_id: str | None = None
