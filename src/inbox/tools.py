"""LangChain tools: inbox + acknowledge + reply (SPEC)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
import time
from typing import Annotated, Any

from langchain_core.tools import tool

from .store import MessageInbox

ReplyCallback = Callable[[str], None]
TelegramFileTrySend = Callable[[str, str], tuple[bool, str]]


def build_inbox_tools(
    inbox: MessageInbox,
    *,
    sandbox_root: Path,
    on_reply: ReplyCallback | None = None,
    delivery: Any | None = None,
    telegram_file_try_send: TelegramFileTrySend | None = None,
    current_telegram_chat_id: Callable[[], int | None] | None = None,
) -> list:
    """Inbox search/fetch plus acknowledge and send_message for Rich CLI delivery."""

    @tool
    def fetch_unread_messages(
        mark_as_read: Annotated[
            bool,
            "If true (default), mark fetched human messages as read after returning them.",
        ] = True,
    ) -> str:
        """Fetch unread human messages from your inbox."""
        rows = inbox.fetch_unread_human(mark_as_read=mark_as_read)
        return json.dumps({"unread": rows, "count": len(rows)}, indent=2)

    @tool
    def read_recent_messages(
        limit: Annotated[
            int,
            "Number of latest messages to return (human + assistant), 1–25. Default 5; oldest first in the list.",
        ] = 5,
    ) -> str:
        """Read the most recent inbox messages (human and assistant) for quick context.

        Not the same as unread messages: this is a fixed tail of the thread. Human **unread**
        state is unchanged here so lines queued while another turn runs stay unread until
        ``fetch_unread_messages`` or until that line’s turn reaches the model.
        """
        lim = max(1, min(limit, 25))
        rows = inbox.fetch_recent_messages(limit=lim, mark_read_in_window=False)
        return json.dumps({"messages": rows, "count": len(rows)}, indent=2)

    @tool
    def search_messages(
        query: Annotated[str, "Natural language or keywords to find in stored messages."],
        limit: Annotated[int, "Maximum results (1–50). Default 10."] = 10,
    ) -> str:
        """Search the inbox using semantic vector search."""
        rows = inbox.search_semantic(query, limit)
        return json.dumps({"results": rows, "count": len(rows)}, indent=2)

    @tool
    def schedule_wake(
        when_iso: Annotated[
            str,
            "When the wake fires: ISO 8601 datetime strictly in the future. "
            "Always include a timezone: offset (e.g. -07:00 for US Pacific) or Z for UTC—never naive local time without offset. "
            "The stored instant is absolute; 09:00-07:00 and 16:00Z are different clocks. "
            "If the user says 'morning', use an actual morning hour in their zone (e.g. 08:00 or 09:00), not 15:00. "
            "If their timezone is unclear, infer from context (travel, city) or ask once; do not assume UTC. "
            "Examples: 2026-03-24T08:30:00-07:00, 2026-03-24T15:30:00Z.",
        ],
        context: Annotated[
            str,
            "Why this wake exists; you will see it again when the alarm fires (keep it actionable).",
        ],
    ) -> str:
        """Schedule a future Wake Up with saved context.

        Times must be unambiguous (offset or Z). Align prose with the ISO instant—do not
        call a 15:00 local wake 'morning'."""
        from runtime.scheduled_wake import schedule_wake_add

        tc = current_telegram_chat_id() if current_telegram_chat_id else None
        ok, body = schedule_wake_add(
            sandbox_root,
            when_iso,
            context,
            telegram_chat_id=tc,
        )
        if not ok:
            return body
        return body

    @tool
    def send_message(
        message: Annotated[str, "Message to send to the user."],
    ) -> str:
        """Send a response to the user.

        For interactive turns that use a skill/long-running work, send one short acknowledgement first,
        then send one final reply when done.
        """
        inbox.add_assistant(message)
        if delivery is not None:
            delivery.reply_via_tool = True
        if on_reply:
            on_reply(message)
        return "Reply delivered to the user and saved to the inbox."

    @tool
    def wait(
        seconds: Annotated[int, "Number of seconds to wait before continuing."],
    ) -> str:
        """Wait for a specified number of seconds before continuing."""
        time.sleep(seconds)
        return f"Waited for {seconds} seconds."

    tools: list = [
        fetch_unread_messages,
        read_recent_messages,
        search_messages,
        schedule_wake,
        send_message,
        wait,
    ]

    if telegram_file_try_send is not None:

        @tool
        def send_file_to_user(
            workspace_relative_path: Annotated[
                str,
                "Path relative to the workspace root; use forward slashes (e.g. outputs/report.pdf).",
            ],
            caption: Annotated[
                str,
                "Optional short caption shown with the file on Telegram (max ~1024 chars).",
            ] = "",
        ) -> str:
            """Send a file from the workspace to the user via Telegram (same chat they wrote from).

            Use when the user asked for a document, export, or artifact and they are on Telegram.
            For terminal-only sessions this will not deliver a file—use send_message with the path instead.
            Telegram bot uploads are limited in size (order of tens of MB).
            """
            _ok, msg = telegram_file_try_send(workspace_relative_path.strip(), caption.strip())
            return msg

        tools.append(send_file_to_user)

    return tools
