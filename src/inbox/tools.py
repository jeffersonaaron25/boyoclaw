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
        """Fetch unread human messages from the inbox."""
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

        Not the same as unread messages: this is a fixed tail of the thread. Any human message
        in this window that was still unread is marked read so it will not appear again in
        fetch_unread_messages.
        """
        lim = max(1, min(limit, 25))
        rows = inbox.fetch_recent_messages(limit=lim)
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
            "When to wake you: ISO 8601 datetime strictly in the future "
            "(e.g. 2026-03-24T09:00:00-07:00 or 2026-03-24T16:00:00Z).",
        ],
        context: Annotated[
            str,
            "Why this wake exists; you will see it again when the alarm fires (keep it actionable).",
        ],
    ) -> str:
        """Schedule a future Wake Up with saved context. The record is removed after that wake runs."""
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
        # Remove Markdown formatting: asterisks, underscores, backticks, and tildes only when used in Markdown constructs—not inside words or numbers.
        import re
        def _remove_markdown_formatting(msg: str) -> str:
            # Remove **bold**, *italic*, __under__, _under_, ***word***, ~~strike~~, and `inline code`.
            # Handles multi-line, block, and leading list/heading markdown (e.g. "- **foo**").
            # Remove list/heading markdown tokens at line start
            msg = re.sub(r'^[\s>*-]+', '', msg, flags=re.MULTILINE)
            # Remove bold/italic/underline combos: ***, ___, **, __, *, _
            msg = re.sub(r'(\*\*\*|___)(.*?)\1', r'\2', msg, flags=re.DOTALL)
            msg = re.sub(r'(\*\*|__)(.*?)\1', r'\2', msg, flags=re.DOTALL)
            msg = re.sub(r'(\*|_)(.*?)\1', r'\2', msg, flags=re.DOTALL)
            # Inline code: `code`
            msg = re.sub(r'`([^`]*)`', r'\1', msg)
            # Strikethrough: ~~strike~~
            msg = re.sub(r'~~([^~]*)~~', r'\1', msg)
            # Remove bold/italic applied only to part of line (catch leftovers)
            msg = re.sub(r'(\*{1,3}|_{1,3})(\S.*?)\1', r'\2', msg)
            # Remove excess whitespace, leading/trailing spacing
            msg = re.sub(r' +', ' ', msg)
            return msg.strip()

        message = _remove_markdown_formatting(message)
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
