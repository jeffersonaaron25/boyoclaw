"""Async BoyoClaw runtime: background agent worker + Rich CLI for human messages."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent import (
    SandboxedAssistant,
    agent_home_directory,
    default_sandbox_directory,
    is_loop_control_message,
    project_root,
)
from inbox.store import MessageInbox
from inbox.tools import build_inbox_tools
from runtime.todos import (
    clear_completed_todos,
    ensure_memory_placeholder,
    has_non_placeholder_pending_todos,
    non_placeholder_open_todos_fingerprint,
    prepend_wake_todos,
)
from runtime.turn import QueuedTurn
from datetime import datetime

logger = logging.getLogger(__name__)

# Default: periodic Wake Up on this interval (first fire after one full interval from process start).
PERIODIC_WAKE_INTERVAL_SEC = 3 * 3600

# Appended to system context only for timer-driven periodic wakes (not user chat).
PERIODIC_WAKE_SYSTEM_APPENDIX = """
## Periodic wake (this turn only — runtime timer, not interactive chat)
This wake runs on a fixed interval. The user is **not** waiting at the screen unless you notify them.
- **Quickly** read **`TODOS.md`** and decide if anything needs doing. Use **`fetch_unread_messages`** only when the checklist or obvious context requires it.
- **Do not** call **`send_message`** unless something **requires** the human (deadline, blocker, risk, or a decision only they can make).
- Do quiet work (edit todos, files, memory) without narrating. When there is **nothing** the user must see, do **not** use **`send_message`** and your **final** assistant message must be exactly the single word **`SILENT`** (uppercase). The runtime will not show **`SILENT`** to the user or store it as a user-visible reply.
- If you used **`send_message`** because you had to alert them, you may still end your final turn with **`SILENT`** if there is no further text needed (or leave a minimal closing line — prefer **`SILENT`** alone when done).
"""


def _periodic_output_suppressed(text: str | None) -> bool:
    """True when periodic wake should emit nothing to inbox/terminal/Telegram."""
    if text is None:
        return True
    s = text.strip()
    if not s:
        return True
    # Single-token SILENT (allow trailing punctuation from sloppy models)
    return s.casefold().rstrip(".!") == "silent"


def _agent_pause_command_kind(text: str) -> Literal["pause", "resume"] | None:
    """Terminal/Telegram slash commands for global agent pause (no wakes, no replies)."""
    s = text.strip().lower()
    if s in ("/agent-pause", "/agent_pause"):
        return "pause"
    if s in ("/agent-resume", "/agent_resume"):
        return "resume"
    return None


def _user_message_is_wake_up(user_message: str) -> bool:
    """True when this turn is a Wake Up run (internal or user-typed), including Wake + Telegram upload suffix."""
    s = user_message.strip()
    if not s:
        return False
    first = s.split("\n", 1)[0].strip()
    norm = " ".join(first.split()).casefold()
    return norm == "wake up"


BOYOCLAW_RUNTIME_PROMPT = """
Current date and time: {current_date_time}

## Agent workspace vs system files
- Your **file and shell tools** are confined to **``.agent-home``** (that directory is your workspace root for paths like ``MEMORY.md``, ``TODOS.md``, ``skills/``, etc.).
- **Outside** that folder (same parent directory) the runtime keeps **system data** you must **not** assume you can list or open: ``inbox/`` (message store), ``telegram.json``, ``scheduled_wakes.json``.
- Inbox content is only through **fetch_unread_messages**, **read_recent_messages**, and **search_messages**.

## Skills paths: file tools vs ``execute``
- File tools (``read_file``, ``glob``, filesystem ``ls``) should use skills paths under ``/skills/project/...``. Do not rewrite skill paths to host absolute paths like ``/Users/...``.
- ``execute`` is a real shell. In Docker, ``$HOME`` is the **inner** agent copy at ``/mnt/workspace/.agent-home`` (synced with the user-facing host ``.agent-home``). Use shell paths like ``$HOME/skills/project/...``.
- Do not pass shell-only paths (for example ``$HOME/...`` or ``/mnt/workspace/.agent-home/...``) into ``read_file``. Use ``/skills/...`` with file tools, and use ``$HOME`` paths only with ``execute``.
- To locate files for shell commands, search under ``$HOME``. Do not run ``find /`` (slow, often times out).

## Apple Calendar scripting guardrails
- For Calendar events in AppleScript, use ``start date`` and ``end date`` fields. Avoid ``date of event``.
- Prefer bounded queries like ``every event whose start date is greater than or equal to t0 and start date is less than or equal to t1``.
- Do not hide Calendar errors behind blanket ``try/on error`` blocks. Surface stderr and simplify to one minimal diagnostic script when results are unexpectedly empty.

## Turn context (interactive messages only — not the full conversation)
- When the **human has just sent a message** via the terminal or Telegram, the **message list** may include up to **three prior** human/assistant turns **before** that message. That is a **short tail only**—not a complete transcript. **Scheduled wakes, startup wakes, and other internal wakes do not include this tail.**
- In those interactive turns, a **Semantic snippets** system section may list up to three excerpts from older messages (similarity-ranked). Those snippets are **not** exhaustive and may be out of order—use **read_recent_messages** or **search_messages** when you need reliable or complete history.
- A **periodic Wake Up** runs every **3 hours** by default (runtime timer). On that turn you get extra instructions: quick **TODOS.md** check, **`send_message`** only if truly needed; otherwise end with exactly **`SILENT`** so the user sees nothing.

## BoyoClaw interaction rules
- Interactive terminal: the user already sees what they typed. Answer that line directly in a natural chat tone.
  Do not describe their lines as "unread inbox messages" or re-list them as mail unless they explicitly ask about the inbox.
- Use `fetch_unread_messages` only for "Wake Up" processing or when the user asks about unread/inbox mail—not on every reply.
- Use `search_messages` when you need to recall past conversation or facts (e.g. a name they mentioned before). You must use this 
  if user is following up on a previous conversation or fact and you have some context to search for.
- Use `read_recent_messages` to get the context of the recent conversation when you have no idea what the user is referring to.
- Use `schedule_wake` when the user wants a **future** reminder: pass an ISO 8601 time **in the future** and a short **context** explaining why. The runtime fires a normal Wake at that time with that context - so ensure you have sufficient context to perform the task.

## Default wake TODOs — never user-facing
`TODOS.md` is auto-prepended with three **system** lines (same idea every run): see unread / acknowledge / update TODOS.md.
These are **runtime housekeeping**, not the user's personal task list.

- In normal chat (e.g. "what's up?", "got any todos?", small talk): **do not** list, summarize, number, or offer to work through these three defaults. Treat them as invisible in replies.
- If asked broadly whether they have todos: mention **only** other, non-default items you see in **`TODOS.md`** after **`read_file`**; if there are none, say they have no extra todos (you may briefly note the system keeps an internal checklist—one short clause—without enumerating it).
- Only discuss those three items in detail if the user explicitly asks about the wake checklist, inbox workflow, or editing that section of **`TODOS.md`**.

- On receiving "Wake Up" message (or from the user on terminal/Telegram, **not** the periodic timer):
    1. Optionally read **`TODOS.md`** for context; do not recite the default three to the user.
    2. Use `fetch_unread_messages` and update todos as required by the workflow.
    3. Use `send_message` for a concise status if needed; then continue real work. This is mandatory when you use long running tools, skills or perform complex tasks.
    4. **Acknowledgements (strict):** Send **at most one** short acknowledgement before heavy work (multi-step tools, skills, or >~30s of work): one sentence—receipt + what you will do. **Do not** acknowledge simple questions, one-shot answers, or trivia. **Do not** send “Got it / On it / Working on it” twice in the same turn. After work finishes, send **one** final `send_message` with the outcome (or stay silent if there is nothing user-facing). On **terminal** interactive lines, the user already sees their text—skip redundant preamble unless you are about to run slow tools.
- On receiving "Go to Sleep" message:
    1. Update todos to mark only incomplete tasks as remaining.
    2. Finish the response appropriately (do not treat as a normal user message).
- When completing a todo, mark it as completed in **`TODOS.md`** using `[x]` notation (via your normal file tools).
- Do not treat "Wake Up" or "Go to Sleep" as user mail; they are control messages (not stored in the inbox).
- Do not repeat the same substantive answer twice in one turn (avoid duplicate `send_message` content).
- Some messages arrive from **Telegram** (authorized chats). Replies still go through `send_message`; they are delivered to that Telegram chat. Do not tell the user to "check the terminal" when they wrote from Telegram.
- Files users send via Telegram are saved under **`telegram_uploads/<date>/`** inside your **agent workspace** (``.agent-home``). By default they are attached to the **next text message**; **voice uploads** start a turn immediately with the saved file path attached. The composed user message lists relative paths—read or process them when relevant. 
- When the user is on **Telegram** and needs a file you created, call `send_file_to_user` with a **path relative to the agent workspace** (and optional caption). For **terminal-only** users, do not use that tool—give them the path in `send_message` instead. Large files may exceed Telegram limits; say so if send fails.
- **Audio:** Use **kokoro-tts-telegram** (TTS) and **audio-stt-faster-whisper** (STT); see those skills.
- **Skills / long tasks:** If the turn needs a skill or clearly multi-step work, you may send **one** acknowledgement line first (see rules above), then execute, then **one** closing `send_message`. If the task is quick or obvious, **skip** the acknowledgement and answer once.
- **Simple chat:** For normal Q&A or short requests, use **a single** `send_message` with the answer—no separate acknowledgement paragraph.

## MEMORY.md (strict)
- The **full text of `MEMORY.md`** is appended after these rules. Treat it as **durable workspace memory**; if the user contradicts a stored fact, follow the user and **update** the file.
- **Update without being asked** only when the user **states** something worth persisting: name/preferences, standing instructions, recurring schedule facts, important corrections, or explicit “remember this”. **Do not** add speculative psychology, play-by-play chat, one-off task detail, or duplicate facts already in MEMORY.
- **How to write:** One line per fact with date; no essays. Prefer **append** or **edit in place**. If the file grows long or you see a **size warning**, **compress** to essential bullets on the next edit.
- **When not to touch MEMORY:** Pure small talk, one-turn questions, or items that belong only in **`TODOS.md`** or another project file—use the right place instead.
- **Routine tasks / reminders:** If the user asks for recurring work, capture what you need in MEMORY and/or **`TODOS.md`** and confirm intervals with them before promising automation.
- Let MEMORY inform tone and consistency; do not recite MEMORY back unless they ask what is stored.

## Workspace hygiene (strict)
- **Scope:** Your writable tree is **``.agent-home``** (and paths the user clearly treats as theirs under it). Never delete or edit **inbox/**, **telegram.json**, **scheduled_wakes.json**, or anything outside what your tools can open.
- **After a task:** Remove **your** scratch dirs, duplicate downloads, superseded exports, and obsolete **``outputs/``** or **``telegram_uploads/``** files you created or that are clearly safe to drop (e.g. old dated uploads you already processed and no longer need). **Do not** delete the only copy of something the user might want, **MEMORY.md**, **TODOS.md**, or user-authored notes unless they asked or agreed.
- **One canonical artifact:** If you regenerate a deliverable, **delete or overwrite** the old version so the workspace does not accumulate numbered copies unless the user wants history.
- **Large or sensitive files:** Do not hoard big binaries; if an intermediate file is no longer needed for the next step, remove it in the same session when reasonable.

## Shell ``execute`` (sandbox policy)
- Host shell commands are **filtered** before they run: **sudo/doas**, **system-wide rm**, **disk erase**, **shutdown/reboot**, **curl/wget piped to shell**, and similar patterns are **blocked**. You will see a clear message and exit code 126 when blocked.
- Do **not** ask the user to turn off ``BOYOCLAW_SHELL_POLICY`` unless they explicitly want unsafe mode and understand the risk.
- This does **not** replace the user's responsibility: the policy is best-effort pattern matching, not a full security guarantee.

## Response Formatting
- Whenever you have a message from user (fetch_unread_messages or directly from telegram), you should acknowledge the message with a reply if it includes some work on your end (like reading files, using skills, etc.).
- Do not use markdown formatting, use appropriate formatting for clear readability on telegram and terminal which do not support markdown.
- Do not use *bold* or other formatting.
"""


@dataclass
class DeliveryState:
    reply_via_tool: bool = False


def _extract_assistant_text(result: dict[str, Any]) -> str:
    msgs: list[Any] = result.get("messages", [])
    for m in reversed(msgs):
        if not isinstance(m, AIMessage):
            continue
        c = m.content
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts: list[str] = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
            if parts:
                return "\n".join(parts).strip()
    return ""


class BoyoClawRuntime:
    """Queue human lines, run the agent in the asyncio event loop, render with Rich."""

    def __init__(
        self,
        *,
        sandbox_root: Path | None = None,
        model: str | BaseChatModel | None = None,
        system_prompt: str | None = None,
        ollama_base_url: str | None = None,
        recursion_limit: int = 300,
    ) -> None:
        logger.info("Initializing BoyoClawRuntime with sandbox_root: %s", sandbox_root)
        self.sandbox_root = (sandbox_root or default_sandbox_directory()).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.agent_home = agent_home_directory(self.sandbox_root)
        self.agent_home.mkdir(parents=True, exist_ok=True)
        self._recursion_limit = recursion_limit
        self._human_queue: asyncio.Queue[QueuedTurn] = asyncio.Queue()
        self._shutdown = asyncio.Event()
        self._cli_prompt_allowed = asyncio.Event()
        self._cli_prompt_allowed.set()
        self._console = None
        self._print_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._telegram_app: Any = None
        self._pending_telegram_chat_id: int | None = None
        # Telegram: file-only messages save here until the user sends a text line (same chat).
        self._telegram_pending_uploads: dict[int, list[tuple[str, str]]] = {}
        # After a Wake driven by non-placeholder todos, same todo set → same fingerprint → no repeat Wake.
        self._last_todo_wake_fingerprint: str = ""
        # When True: no _run_agent_turn (wakes, replies, scheduled, periodic). Telegram still polls; /agent-resume clears.
        self._agent_paused: bool = False

        self.inbox = MessageInbox(
            self.sandbox_root / "inbox",
            ollama_base_url=ollama_base_url,
        )
        self._delivery = DeliveryState()

        combined = (system_prompt or "").strip()
        if combined:
            combined = combined + "\n\n" + BOYOCLAW_RUNTIME_PROMPT.format(current_date_time=datetime.now().isoformat())
        else:
            combined = BOYOCLAW_RUNTIME_PROMPT.strip()

        tools = build_inbox_tools(
            self.inbox,
            sandbox_root=self.sandbox_root,
            on_reply=self._sync_reply_ui,
            delivery=self._delivery,
            telegram_file_try_send=self._try_schedule_telegram_file,
            current_telegram_chat_id=lambda: self._pending_telegram_chat_id,
        )
        self._assistant = SandboxedAssistant(
            sandbox_root=self.sandbox_root,
            model=model,
            system_prompt=combined,
            extra_tools=tools,
            debug=True,
            prefer_docker_isolation=True,
        )

    def _ensure_console(self) -> None:
        if self._console is None:
            from rich.console import Console

            self._console = Console(stderr=False)

    def _sync_ack(self, message: str) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._async_print_ack(message), self._loop)

    def _sync_reply_ui(self, message: str) -> None:
        """Deliver send_message to Rich and/or Telegram (inbox row already added in tool)."""
        if self._loop is None:
            return
        cid = self._pending_telegram_chat_id
        if cid is not None and self._telegram_app is not None:
            from runtime.telegram_bot import send_plain_text

            asyncio.run_coroutine_threadsafe(
                send_plain_text(self._telegram_app.bot, cid, message),
                self._loop,
            )
            return
        asyncio.run_coroutine_threadsafe(self._async_print_reply(message), self._loop)

    def _try_schedule_telegram_file(self, rel: str, caption: str) -> tuple[bool, str]:
        """Schedule a workspace file for Telegram; capture chat/app while the agent turn is active."""
        cid = self._pending_telegram_chat_id
        app = self._telegram_app
        if cid is None:
            return (
                False,
                "The user is not on Telegram in this turn. Tell them the workspace-relative path "
                "to the file in send_message (they are using the terminal).",
            )
        if app is None:
            return (
                False,
                "Telegram is not connected (--telegram not used or bot not running). "
                "Give the user the file path in send_message instead.",
            )
        if self._loop is None:
            return False, "Runtime event loop is not ready; cannot send file."
        asyncio.run_coroutine_threadsafe(
            self._telegram_deliver_workspace_file(cid, app, rel, caption),
            self._loop,
        )
        return True, "Started uploading the file to the user's Telegram chat. You can still summarize in send_message."

    async def _telegram_deliver_workspace_file(
        self,
        chat_id: int,
        app: Any,
        rel: str,
        caption: str,
    ) -> None:
        from runtime.telegram_bot import send_plain_text, send_workspace_file

        root = self.agent_home.resolve()
        cleaned = rel.strip().replace("\\", "/")
        if not cleaned or ".." in Path(cleaned).parts:
            await send_plain_text(app.bot, chat_id, f"Invalid path for send_file_to_user: {rel!r}")
            return
        if Path(cleaned).is_absolute():
            await send_plain_text(app.bot, chat_id, f"Use a workspace-relative path, not absolute: {rel!r}")
            return

        path = (root / cleaned).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            await send_plain_text(app.bot, chat_id, f"Path escapes workspace: {rel!r}")
            return
        if not path.is_file():
            await send_plain_text(app.bot, chat_id, f"Not a file or missing in workspace: {cleaned}")
            return

        try:
            await send_workspace_file(app.bot, chat_id, path, caption=caption or None)
        except OSError as e:
            logger.exception("Telegram file read failed")
            await send_plain_text(app.bot, chat_id, f"Could not read file: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Telegram file send failed")
            await send_plain_text(
                app.bot,
                chat_id,
                f"Could not send file (size/type limits or network): {e}",
            )
            return

        note = f"[Sent file to Telegram: {cleaned}]"
        if caption:
            note += f" — {caption[:200]}"
        self.inbox.add_assistant(note)
        async with self._print_lock:
            self._ensure_console()
            self._console.print(f"[dim]Telegram file sent:[/dim] {cleaned}")

    async def _async_print_ack(self, message: str) -> None:
        from rich.panel import Panel

        async with self._print_lock:
            self._ensure_console()
            self._console.print(
                Panel(message, title="[cyan]Ack[/cyan]", border_style="cyan"),
            )

    async def _async_print_reply(self, message: str) -> None:
        from rich.panel import Panel

        async with self._print_lock:
            self._ensure_console()
            self._console.print(
                Panel(message, title="[green]Assistant[/green]", border_style="green"),
            )

    async def _print_status(self, text: str) -> None:
        async with self._print_lock:
            self._ensure_console()
            self._console.print(f"[dim]{text}[/dim]")

    @property
    def shutdown(self) -> asyncio.Event:
        return self._shutdown

    def agent_paused(self) -> bool:
        return self._agent_paused

    def set_agent_paused(self, paused: bool) -> None:
        self._agent_paused = paused

    def attach_telegram_application(self, app: Any) -> None:
        self._telegram_app = app

    async def enqueue_turn(self, turn: QueuedTurn) -> None:
        await self._human_queue.put(turn)

    def record_telegram_pending_upload(self, chat_id: int, workspace_relative: str, caption: str) -> None:
        """Stack a saved file for this chat until the user sends a text message."""
        self._telegram_pending_uploads.setdefault(chat_id, []).append(
            (workspace_relative, caption.strip()),
        )

    def pop_telegram_pending_uploads(self, chat_id: int) -> tuple[tuple[str, ...], str]:
        """Drain queued uploads for this chat; returns (paths, optional caption prefix for the user line)."""
        items = self._telegram_pending_uploads.pop(chat_id, [])
        if not items:
            return (), ""
        paths = tuple(r for r, _ in items)
        lines: list[str] = []
        for rel, cap in items:
            if cap:
                lines.append(f"Caption for `{rel}`: {cap}")
        prefix = "\n".join(lines)
        if prefix:
            prefix = prefix + "\n\n"
        return paths, prefix

    async def _telegram_send_text(self, chat_id: int, text: str) -> None:
        if self._telegram_app is None:
            return
        from runtime.telegram_bot import send_plain_text

        await send_plain_text(self._telegram_app.bot, chat_id, text)

    async def _agent_error_user_visible(self, detail: str) -> None:
        await self._print_status(f"[red]{detail}[/red]")
        cid = self._pending_telegram_chat_id
        if cid is not None:
            await self._telegram_send_text(cid, detail)

    def _compose_user_message(self, turn: QueuedTurn) -> str:
        """Merge caption/text with Telegram upload path hints for the model and inbox."""
        text = turn.text.strip()
        paths = turn.uploaded_workspace_paths
        if not paths:
            return text
        block = "[User uploaded via Telegram — files saved under the workspace]\n" + "\n".join(
            f"- {p}" for p in paths
        )
        if text:
            return f"{text}\n\n{block}"
        return f"The user sent file(s) from Telegram without a text message.\n\n{block}"

    def _telegram_system_appendix(self, turn: QueuedTurn) -> str | None:
        """Per-turn system text: Telegram display name (not written to inbox)."""
        if turn.telegram_chat_id is None:
            return None
        label = (turn.telegram_sender_label or "").strip()
        if not label:
            return None
        return (
            "The human is messaging you from Telegram.\n"
            f"Their Telegram display identity is: {label}.\n"
            "Use their name when it fits the reply. If the user has a preferred name in your memory, use that instead of the Telegram display name."
        )

    def _inbox_rows_to_lc(self, rows: list[dict[str, Any]]) -> list[HumanMessage | AIMessage]:
        out: list[HumanMessage | AIMessage] = []
        for r in rows:
            if r.get("type") == "human":
                out.append(HumanMessage(content=str(r.get("content", ""))))
            else:
                out.append(AIMessage(content=str(r.get("content", ""))))
        return out

    def _retrieval_system_block(self, query: str, exclude_message_id: str | None) -> str:
        """Up to 3 semantically similar past messages; labeled so the model does not treat as full history."""
        q = (query or "").strip()
        if len(q) > 8000:
            q = q[:8000]
        if not q:
            q = "."
        try:
            rows = self.inbox.search_semantic(q, limit=8)
        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic retrieval for turn failed: %s", e)
            return ""
        if exclude_message_id:
            rows = [r for r in rows if r.get("id") != exclude_message_id]
        rows = rows[:3]
        if not rows:
            return ""
        rows.sort(key=lambda r: str(r.get("created_at", "")))
        lines = [
            "## Semantic snippets from past messages (NOT the full conversation)",
            "",
            "Up to three excerpts chosen by similarity to the current message, then **sorted oldest → newest** by "
            "timestamp. They may be incomplete or omit relevant context. This is not a transcript—use "
            "`read_recent_messages` or `search_messages` when you need reliable history. This is intended to give you a "
            "starting point to search for messages if required. ",
            "Do not assume you don't have information until you have searched with `search_messages` and/or `read_recent_messages`."
            "",
        ]
        for i, r in enumerate(rows, 1):
            role = "Human" if r.get("type") == "human" else "Assistant"
            ts = str(r.get("created_at", ""))[:19]
            content = str(r.get("content") or "").strip()
            if len(content) > 1500:
                content = content[:1500] + "…"
            lines.append(f"### Snippet {i} ({role}, {ts})")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    async def _process_one(self, turn: QueuedTurn) -> None:
        self._pending_telegram_chat_id = turn.telegram_chat_id
        try:
            raw = turn.text
            text = raw.strip()
            if not text and not turn.uploaded_workspace_paths:
                self._cli_prompt_allowed.set()
                return
            if text and text.lower() in ("/quit", "/exit", ":q"):
                self._shutdown.set()
                self._cli_prompt_allowed.set()
                return

            ap = _agent_pause_command_kind(text or "")
            if ap == "pause":
                self.set_agent_paused(True)
                await self._print_status(
                    "[dim]Agent paused — no wakes or agent replies. Send /agent-resume to continue.[/dim]",
                )
                self._cli_prompt_allowed.set()
                return
            if ap == "resume":
                self.set_agent_paused(False)
                await self._print_status("[dim]Agent resumed.[/dim]")
                self._cli_prompt_allowed.set()
                return

            if self._agent_paused:
                await self._print_status(
                    "[dim]Agent is paused. Send /agent-resume or /agent_resume to continue.[/dim]",
                )
                self._cli_prompt_allowed.set()
                return

            await self._print_status("Agent running…")
            self._cli_prompt_allowed.set()

            composed = self._compose_user_message(turn)

            if is_loop_control_message(text):
                await self._run_agent_turn(
                    composed,
                    announce=False,
                    system_appendix=self._telegram_system_appendix(turn),
                    interaction_turn=True,
                    tail_mode="no_new_human",
                )
                return

            new_id = self.inbox.add_human(composed, unread=False)
            await self._run_agent_turn(
                composed,
                announce=False,
                system_appendix=self._telegram_system_appendix(turn),
                interaction_turn=True,
                tail_mode="after_new_human",
                new_human_message_id=new_id,
            )
        finally:
            self._pending_telegram_chat_id = None

    async def _run_agent_turn(
        self,
        user_message: str,
        *,
        announce: bool = True,
        system_appendix: str | None = None,
        interaction_turn: bool = False,
        tail_mode: Literal["after_new_human", "no_new_human"] = "no_new_human",
        new_human_message_id: str | None = None,
        periodic_wake: bool = False,
    ) -> bool:
        async with self._agent_turn_lock:
            if self._assistant is None:
                return False
            if self._agent_paused:
                return False
            if _user_message_is_wake_up(user_message):
                self._sync_todos_wake_file_prep()
            self._delivery.reply_via_tool = False
            prior_lc: list[HumanMessage | AIMessage] = []
            retrieval_block: str | None = None
            if interaction_turn:
                if tail_mode == "after_new_human":
                    prior_rows = self.inbox.fetch_messages_before_newest(count=3)
                else:
                    prior_rows = self.inbox.fetch_recent_messages(
                        limit=3,
                        mark_read_in_window=False,
                    )
                prior_lc = self._inbox_rows_to_lc(prior_rows)
                exclude_id = new_human_message_id if tail_mode == "after_new_human" else None
                retrieval_block = self._retrieval_system_block(user_message, exclude_id) or None

            effective_appendix = system_appendix
            if periodic_wake:
                extra = PERIODIC_WAKE_SYSTEM_APPENDIX.strip()
                if effective_appendix and effective_appendix.strip():
                    effective_appendix = f"{effective_appendix.strip()}\n\n{extra}"
                else:
                    effective_appendix = extra
            try:
                if announce:
                    await self._print_status("Agent running…")
                result = await self._assistant.run_async(
                    user_message,
                    recursion_limit=self._recursion_limit,
                    system_appendix=effective_appendix,
                    prior_messages=prior_lc if interaction_turn else None,
                    retrieval_system_block=retrieval_block,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("Agent run failed")
                await self._agent_error_user_visible(f"Error: {e}")
                return False

            if not self._delivery.reply_via_tool:
                fallback = _extract_assistant_text(result)
                if periodic_wake and _periodic_output_suppressed(fallback):
                    return True
                if fallback:
                    self.inbox.add_assistant(fallback)
                    cid = self._pending_telegram_chat_id
                    if cid is not None:
                        await self._telegram_send_text(cid, fallback)
                    else:
                        await self._async_print_reply(fallback)
            return True

    async def _maybe_wake_for_pending_todos(self) -> None:
        """If TODOS.md has real open work and the set changed since last todo-Wake, run Wake Up once."""
        todos_path = self.agent_home / "TODOS.md"
        if not has_non_placeholder_pending_todos(todos_path):
            return
        fp = non_placeholder_open_todos_fingerprint(todos_path)
        if not fp or fp == self._last_todo_wake_fingerprint:
            return
        ok = await self._run_agent_turn("Wake Up", announce=True)
        if ok:
            self._last_todo_wake_fingerprint = non_placeholder_open_todos_fingerprint(todos_path)

    async def _agent_worker(self) -> None:
        # When idle, avoid hammering TODOS.md on every short queue timeout.
        idle_polls_before_todo_check = 0
        while not self._shutdown.is_set():
            try:
                line = await asyncio.wait_for(self._human_queue.get(), timeout=0.35)
            except asyncio.TimeoutError:
                idle_polls_before_todo_check += 1
                if idle_polls_before_todo_check >= 15:
                    idle_polls_before_todo_check = 0
                    await self._maybe_wake_for_pending_todos()
                continue
            idle_polls_before_todo_check = 0
            await self._process_one(line)

    async def _input_loop(self) -> None:
        self._ensure_console()
        await self._print_status(
            "Messages are queued to the background agent. Commands: [bold]/quit[/bold]",
        )
        while not self._shutdown.is_set():
            await self._cli_prompt_allowed.wait()
            self._cli_prompt_allowed.clear()
            async with self._print_lock:
                self._ensure_console()
                self._console.print("[bold magenta]›[/bold magenta] ", end="")
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except (EOFError, KeyboardInterrupt):
                self._shutdown.set()
                break
            if not line:
                self._shutdown.set()
                break
            stripped = line.rstrip("\n\r").strip()
            if stripped.lower() in ("/quit", "/exit", ":q"):
                # Handle here so we exit the input loop immediately. If we only handled
                # quit in _process_one, this coroutine would block on the next readline()
                # while gather() waits forever for _input_loop to finish.
                self._shutdown.set()
                break
            await self._human_queue.put(QueuedTurn(text=line.rstrip("\n\r")))

    def _sync_todos_wake_file_prep(self) -> None:
        """Remove completed lines, ensure default wake todo lines exist, MEMORY placeholder — run before every Wake Up."""
        todos_path = self.agent_home / "TODOS.md"
        clear_completed_todos(todos_path)
        prepend_wake_todos(todos_path)
        ensure_memory_placeholder(self.agent_home / "MEMORY.md")

    async def _wake_maintenance(self) -> tuple[bool, Path]:
        """Clear completed todos, prepend default wake todos, ensure MEMORY. Returns (should_run_wake_agent, todos_path)."""
        todos_path = self.agent_home / "TODOS.md"
        self._sync_todos_wake_file_prep()
        should_run = self.inbox.has_unread_human() or has_non_placeholder_pending_todos(
            todos_path,
        )
        return should_run, todos_path

    async def _headless_stdin_placeholder(self) -> None:
        """Keep the process alive without reading stdin (launchd has no TTY)."""
        await self._shutdown.wait()

    async def _scheduled_wake_scheduler(self) -> None:
        """Fire due scheduled wakes (from ``schedule_wake`` tool); delete each record after a successful turn."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=10.0)
                return
            except asyncio.TimeoutError:
                await self._drain_due_scheduled_wakes()

    async def _drain_due_scheduled_wakes(self) -> None:
        from runtime.scheduled_wake import delete_wake, list_due_wakes

        for rec in list_due_wakes(self.sandbox_root):
            if self._shutdown.is_set():
                return
            msg = (
                "Wake Up\n\n"
                "[Scheduled wake — context saved when this alarm was set]\n"
                f"{rec['context']}"
            )
            tc = rec.get("telegram_chat_id")
            prev = self._pending_telegram_chat_id
            if isinstance(tc, int):
                self._pending_telegram_chat_id = tc
            try:
                ok = await self._run_agent_turn(msg, announce=True)
            finally:
                self._pending_telegram_chat_id = prev
            if ok:
                delete_wake(self.sandbox_root, rec["id"])

    async def _periodic_three_hour_wake(self) -> None:
        """Timer wake: quick todo pass; user-visible output only via send_message or SILENT."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=PERIODIC_WAKE_INTERVAL_SEC)
                return
            except asyncio.TimeoutError:
                pass
            if self._shutdown.is_set():
                return
            msg = (
                "Wake Up\n\n"
                "[Periodic wake — quick TODOS.md check; notify user only if necessary. "
                "Otherwise end with SILENT.]"
            )
            await self._run_agent_turn(
                msg,
                announce=False,
                interaction_turn=False,
                periodic_wake=True,
            )

    async def run_forever(self, *, enable_telegram: bool = False, headless: bool = False) -> None:
        self._loop = asyncio.get_running_loop()
        self._agent_turn_lock = asyncio.Lock()

        await self._print_status(
            "[bold]BoyoClaw[/bold] — agent workspace: %s [dim](system root: %s)[/dim]"
            % (self.agent_home, self.sandbox_root),
        )
        if headless:
            await self._print_status(
                "[dim]Headless mode: no local stdin (Telegram or other queues only).[/dim]",
            )

        should_wake, todos_path = await self._wake_maintenance()
        if should_wake:
            await self._run_agent_turn("Wake Up")
        else:
            await self._print_status(
                "[dim]Idle: no unread inbox messages and no open todos beyond the default "
                "wake placeholders — Wake Up skipped; send a message to run the agent.[/dim]",
            )
        self._last_todo_wake_fingerprint = non_placeholder_open_todos_fingerprint(todos_path)

        tg_task: asyncio.Task[None] | None = None
        if enable_telegram:
            from runtime.telegram_config import load_telegram_settings

            tg = load_telegram_settings(self.sandbox_root)
            if not tg:
                await self._print_status(
                    "[red]Telegram enabled but config missing. Run: "
                    "python -m runtime telegram configure[/red]",
                )
            else:
                try:
                    from runtime.telegram_bot import run_telegram_polling
                except ImportError:
                    await self._print_status(
                        "[red]Install Telegram support: pip install python-telegram-bot[/red]",
                    )
                else:

                    async def _telegram_guard() -> None:
                        try:
                            await run_telegram_polling(self, tg)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception("Telegram bot stopped")

                    await self._print_status(
                        "[dim]Telegram bot polling (authorized chats only).[/dim]",
                    )
                    tg_task = asyncio.create_task(_telegram_guard())

        sched_task = asyncio.create_task(self._scheduled_wake_scheduler())
        periodic_task = asyncio.create_task(self._periodic_three_hour_wake())
        extras: list[asyncio.Task[Any]] = [sched_task, periodic_task]
        if tg_task is not None:
            extras.append(tg_task)
        await self._print_status(
            f"[dim]Periodic wake every {PERIODIC_WAKE_INTERVAL_SEC // 3600}h (first after one full interval).[/dim]",
        )

        if headless:
            await asyncio.gather(
                self._headless_stdin_placeholder(),
                self._agent_worker(),
                *extras,
            )
        else:
            await asyncio.gather(
                self._input_loop(),
                self._agent_worker(),
                *extras,
            )


async def async_main(*, enable_telegram: bool = False, headless: bool = False) -> None:
    logging.basicConfig(level=logging.WARNING)
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="minimax-m2.7:cloud")
    rt = BoyoClawRuntime(
        sandbox_root=project_root() / ".sandbox" / "workspace",
        model=model,
        system_prompt="You are BoyoClaw, a capable, autonomous, long running background agent. Follow the BoyoClaw interaction rules.",
    )
    await rt.run_forever(enable_telegram=enable_telegram, headless=headless)


async def main() -> None:
    await async_main(enable_telegram=False)


if __name__ == "__main__":
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    asyncio.run(main())
