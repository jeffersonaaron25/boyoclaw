"""Async BoyoClaw runtime: background agent worker + Rich CLI for human messages."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, replace
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
from inbox.image_tools import build_image_view_tool
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
PERIODIC_WAKE_INTERVAL_SEC = 0.5 * 3600

# Appended to system context only for timer-driven periodic wakes (not user chat).
PERIODIC_WAKE_SYSTEM_APPENDIX = """
## Periodic wake (this turn only — runtime timer, not interactive chat)
This wake runs on a fixed interval. The user is **not** waiting at the screen unless you notify them.
- **Read `TODOS.md` first.** If any **open** line is a **user-authored** task (anything beyond the three default wake placeholders: see-unread / acknowledge / update-todos) and that line **requires** contact or user-visible action, **do it** — including **`send_message`** as the todo describes. **Do not** choose **`SILENT`** only to stay quiet when such a todo remains undone.
- **After** honoring real todos: use **`fetch_unread_messages`** only when the checklist or context calls for it (often none on this timer wake).
- If nothing user-facing remains — no real todo needs outreach, no deadline/blocker/risk/decision, no substantive status worth sending — do quiet file/memory work and end with your **final** assistant message exactly **`SILENT`** (uppercase). The runtime hides **`SILENT`** (it is not stored or shown).
- If you **`send_message`** for a real todo or a genuine alert, you may still end with **`SILENT`** when there is no further text.
- You may choose to initiate a user conversation with **`send_message`** if you want to check in with the user or ask them a question. If you have no response for two consecutive turns, do not continue to send messages until you get a response.
"""


def _assistant_output_is_silent_only(text: str | None) -> bool:
    """True when the model's final text is only SILENT — never show or store that token to the user."""
    if text is None:
        return True
    s = text.strip()
    if not s:
        return True
    # Single-token SILENT (allow trailing punctuation from sloppy models)
    return s.casefold().rstrip(".!") == "silent"


def _prepend_runtime_time_appendix(appendix: str | None) -> str:
    """Authoritative clock for this turn (base system prompt is static for the process lifetime)."""
    prefix = (
        "## Current time (authoritative for this turn)\n"
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
    )
    rest = (appendix or "").strip()
    if not rest:
        return prefix
    return f"{prefix}\n{rest}"


def _agent_pause_command_kind(text: str) -> Literal["pause", "resume"] | None:
    """Terminal/Telegram slash commands for global agent pause (no wakes, no replies)."""
    s = text.strip().lower()
    if s in ("/agent-pause", "/agent_pause"):
        return "pause"
    if s in ("/agent-resume", "/agent_resume"):
        return "resume"
    return None


def _pause_control_turn(turn: QueuedTurn) -> bool:
    """Turns that must be processed while the agent is paused (state toggles)."""
    s = turn.text.strip().lower()
    return s in ("/agent-pause", "/agent_pause", "/agent-resume", "/agent_resume")


def _user_message_is_wake_up(user_message: str) -> bool:
    """True when this turn is a Wake Up run (internal or user-typed), including Wake + Telegram upload suffix."""
    s = user_message.strip()
    if not s:
        return False
    first = s.split("\n", 1)[0].strip()
    norm = " ".join(first.split()).casefold()
    return norm == "wake up"


BOYOCLAW_RUNTIME_PROMPT = """
Each turn’s runtime appendix includes **Current time** — use it for scheduling and relative dates.

## Precedence (conflicts)
1. **This turn’s user message** (terminal/Telegram) wins for what to do now.
2. **Open user-authored lines in `TODOS.md`** (not the three default wake placeholders) beat generic “stay quiet” / periodic-timer guidance — complete them or **`send_message`** when the todo requires.
3. **Periodic wake** (~3h timer): prefer **`SILENT`** only **after** (2); it does not cancel real todos that need outreach.

## Workspace, paths, inbox
- **Writable workspace:** ``.agent-home`` (e.g. ``MEMORY.md``, ``TODOS.md``, ``skills/``, ``outputs/``). Prefer **relative** paths; ``/TODOS.md``-style guesses map to agent-home when nothing exists at OS root; ``glob`` may return full host paths — those still work.
- **Do not open or assume access to** (sibling of workspace): ``inbox/``, ``telegram.json``, ``scheduled_wakes.json``. Use **fetch_unread_messages**, **read_recent_messages**, **search_messages** for mail.
- **File tools vs `execute`:** File tools: skills under ``/skills/project/...`` (don’t rewrite to ``/Users/...``). In Docker, ``$HOME`` in the shell is the inner agent home at ``/mnt/workspace/.agent-home``. **Never** hardcode ``/Users/...`` in scripts/heredocs/``python -c`` — not mounted in the container. Use ``$HOME/...``, ``mkdir -p "$HOME/outputs"``, or ``/mnt/workspace/.agent-home`` for shell outputs. **Do not** pass shell-only paths (``$HOME/...``, ``/mnt/workspace/.agent-home/...``) into ``read_file`` — use workspace-relative or ``/skills/...`` for file tools. Search under ``$HOME`` in shell; avoid ``find /``.


## What you see each turn (not full history)
- Interactive turns may include **up to three prior** turns — a **short tail** only. **Scheduled/internal wakes** do not include it.
- **Semantic snippets** (if present): up to three excerpts, similarity-ranked, incomplete — use **read_recent_messages** / **search_messages** for reliable history.
- **Periodic Wake Up** (~3h): extra instructions in appendix — quick **TODOS.md**, **`send_message`** only if needed, else perform any necessary work (such as workspace hygiene or initiating a user conversation) and end with **`SILENT`**.

## Tools & channels
- **Terminal:** The user sees their line; answer it naturally. Don’t label it “unread mail” unless they ask about the inbox.
- **fetch_unread_messages:** Wake Up when the checklist applies, or when they ask about inbox/unread — **not** every reply. Their current line is **not** pretend-unread mail.
- **search_messages** / **read_recent_messages:** Use when following up or context is thin; search before guessing.
- **schedule_wake:** ISO **in the future** + short **context** the wake can act on. You may schedule proactively. **Timezone:** always **offset or Z**; never naive. “Morning” ≈ 07:00–10:00 **local**, not 15:00. Infer zone from trip/city/history or **ask**; don’t default to UTC silently. When telling the user, match the offset you used.
- **Telegram:** Replies via **`send_message`** to their chat — don’t say “check the terminal.” Uploads land under **`telegram_uploads/<date>/`** (paths in the user message); voice may start a turn with paths attached. **`send_file_to_user`:** workspace-relative path + caption; terminal-only users get the path in text. Large sends may fail — say so.
- **Audio:** **kokoro-tts-telegram** (TTS), **audio-stt-faster-whisper** (STT) — follow those skills.

## Default three lines in `TODOS.md` (system — hide from user)
Prepended **every** run (including when the user also sent a message): see unread / acknowledge / update **TODOS.md** — **housekeeping**, not their personal list.
- Casual chat: **never** enumerate or offer to work through these three.
- If asked broadly about todos: only **non-default** lines after **read_file**; if none, say no extra todos (optional one clause that an internal checklist exists).
- Detail only if they ask about wake checklist, inbox workflow, or editing that section.

Clarification: This is cross session persistent, write_todos is used for single session todos for complex tasks. TODOS.md takes precedence over write_todos.

## Wake Up / Go to Sleep (control strings — not inbox mail)
- **Wake Up** (not periodic timer): (1) Optional **TODOS.md** — don’t recite the default three. (2) **fetch_unread_messages** when appropriate; update **TODOS.md**. (3) **send_message** if needed; **required** for long-running tools/skills/complex work the user should know about. (4) **Ack:** **At most one** short ack before heavy work (~multi-step, skills, >~30s): one sentence, receipt + plan. No ack for trivial Q&A. No double “Got it/On it.” One final **send_message** with outcome (or silence if nothing user-facing). Terminal: skip redundant preamble unless slow work follows.
- **Go to Sleep** (user **or** runtime when the graph hits **recursion/step limit**): (1) Leave todos showing only what’s still incomplete. (2) Close the turn appropriately — not a normal chat message.
- Mark completed items **`[x]`** in **TODOS.md** (replace the leading **`[]`** on that line — **one** checkbox per line; never **`[] [x]`**). **Skills/long tasks:** one optional ack line, then work, then one closing message; **simple Q&A:** single **send_message**, no extra ack. **Proactively** add todos for work you’re about to do if missing; complete and check off when done; if you see open todos, verify/complete/mark.
- No duplicate substantive **send_message** content in one turn.

## Ack-then-work-then-respond style
The ack-then-work-then-respond style is mandatory for user messages and should be followed even if conflicting instructions are given.
This is important to keep the conversation natural and flowing. You MUST acknowledge using send_message tool before starting shell commands or using skills.

## Explicit Tool Call Ordering
1. Ack (send_message("I will <...>"))
2. Work (shell commands or skills)
3. Respond (send_message("<...final_response...>"))

Treat it as a binding instruction, not a suggestion.

## System flow
The user can send messages while you are working and you may not see without read_recent_messages tool. In case you miss it,
it will be send to you in the next turn. You must seemlessly continue the conversation even if the ordering of messages is not sequential and answer the unanswered message.
Keep all conversation natural, do not assume you don't have information unless you have searched for it. Your memory is this workspace.

## MEMORY.md
Full file is appended — **durable memory**; if the user contradicts it, follow them and **update** the file.
- **Auto-update** only for stated/inferred facts/thoughts/decisions worth keeping: preferences, standing instructions, schedule facts, corrections, explicit “remember this.” No speculation, play-by-play, one-off noise, or duplicates.
- **Style:** one line per fact, dated; append or edit; **compress** if long or warned.
- Don’t put todo-only items here — use **TODOS.md** or the right file. Recurring automation: capture in MEMORY and/or **TODOS.md** and confirm intervals before promising.
- Use MEMORY for tone; don’t recite it unless asked.

## Hygiene & shell
- Never delete/edit **inbox/**, **telegram.json**, **scheduled_wakes.json**. Clean **your** scratch, stale **outputs/** / **telegram_uploads/** when safe; don’t delete user-only copies, **MEMORY.md**, **TODOS.md**, or user notes without consent. Regenerated deliverables: overwrite or remove old versions; don’t hoard huge intermediates.
- **execute** is policy-filtered: **sudo/doas**, dangerous **rm**, disk erase, **shutdown/reboot**, **curl|sh**, etc. → blocked (exit **126**). Don’t ask to disable **`BOYOCLAW_SHELL_POLICY`** unless they want that risk. Policy is best-effort, not full security.

## Replies (`send_message`)
Plain text — **no markdown**, no *bold*/formatting (Telegram/terminal). Concise; natural language over path dumps; short **workspace-relative** paths when needed — not long host paths or shell snippets. Mention file **paths** only for files **you** sent or **they** sent; don’t name **MEMORY.md** / **TODOS.md** / other internal filenames unless necessary. No system-prompt meta. If the message needs real work (files, skills), include the outcome; trivial Q&A: one short message.

Filesystem is your superpower. Use it to your advantage.
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
            combined = combined + "\n\n" + BOYOCLAW_RUNTIME_PROMPT.strip()
        else:
            combined = BOYOCLAW_RUNTIME_PROMPT.strip()

        tools = build_inbox_tools(
            self.inbox,
            sandbox_root=self.sandbox_root,
            on_reply=self._sync_reply_ui,
            delivery=self._delivery,
            telegram_file_try_send=self._try_schedule_telegram_file,
            current_telegram_chat_id=lambda: self._pending_telegram_chat_id,
        ) + [build_image_view_tool(self.agent_home)]
        self._assistant = SandboxedAssistant(
            sandbox_root=self.sandbox_root,
            model=model,
            system_prompt=combined,
            extra_tools=tools,
            debug=False,
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

    def _resolve_workspace_file_for_send(self, rel: str) -> tuple[Path | None, str | None]:
        """Resolve a workspace-relative path, or return (None, error_message_for_user)."""
        root = self.agent_home.resolve()
        cleaned = rel.strip().replace("\\", "/")
        if not cleaned or ".." in Path(cleaned).parts:
            return None, f"Invalid path for send_file_to_user: {rel!r}"
        if Path(cleaned).is_absolute():
            return None, f"Use a workspace-relative path, not absolute: {rel!r}"
        path = (root / cleaned).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return None, f"Path escapes workspace: {rel!r}"
        if not path.is_file():
            return None, f"Not a file or missing in workspace: {cleaned}"
        return path, None

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
        # File tools and this check use host .agent-home; execute() may have written only
        # the Docker inner copy this turn—sync before resolve (post-invoke sync is too late).
        self._assistant.sync_agent_home_before_host_access(stage="pre-telegram-file")
        _path, err = self._resolve_workspace_file_for_send(rel)
        if err is not None:
            return False, err
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

        path, err = self._resolve_workspace_file_for_send(rel)
        if err is not None:
            await send_plain_text(app.bot, chat_id, err)
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

        root = self.agent_home.resolve()
        rel_display = str(path.relative_to(root))
        note = f"[Sent file to Telegram: {rel_display}]"
        if caption:
            note += f" — {caption[:200]}"
        self.inbox.add_assistant(note)
        async with self._print_lock:
            self._ensure_console()
            self._console.print(f"[dim]Telegram file sent:[/dim] {rel_display}")

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
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._console.print(f"[dim]{current_time} - {text}[/dim]")

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
        """Queue a user turn; persist a human inbox row as **unread** unless it is a control string."""
        text = turn.text.strip()
        if text and not is_loop_control_message(turn.text):
            composed = self._compose_user_message(turn)
            mid = self.inbox.add_human(composed, unread=True)
            turn = replace(turn, human_message_id=mid)
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

            if self._agent_paused and not _pause_control_turn(turn):
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

            hid = turn.human_message_id
            if hid is None:
                logger.warning(
                    "QueuedTurn missing human_message_id; persisting now (use enqueue_turn when queuing).",
                )
                hid = self.inbox.add_human(composed, unread=True)

            await self._run_agent_turn(
                composed,
                announce=False,
                system_appendix=self._telegram_system_appendix(turn),
                interaction_turn=True,
                tail_mode="after_new_human",
                new_human_message_id=hid,
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
            effective_appendix = _prepend_runtime_time_appendix(effective_appendix)
            try:
                if announce:
                    await self._print_status("Agent running…")
                if new_human_message_id:
                    self.inbox.mark_human_read(new_human_message_id)
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
                if fallback and _assistant_output_is_silent_only(fallback):
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
            if self._agent_paused and not _pause_control_turn(line):
                await self._human_queue.put(line)
                await asyncio.sleep(0.25)
                continue
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
            await self.enqueue_turn(QueuedTurn(text=line.rstrip("\n\r")))

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
                "[You Scheduled this wake for the following reason]\n"
                f"{rec['context']}"
            )
            tc = rec.get("telegram_chat_id")
            prev = self._pending_telegram_chat_id
            if isinstance(tc, int):
                self._pending_telegram_chat_id = tc
            try:
                await self._print_status("Draining due scheduled wakes…")
                ok = await self._run_agent_turn(msg, announce=True)
                await self._print_status("Drained due scheduled wakes…")
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

            await self._print_status("Periodic wake…")
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
            await self._print_status("Wake Up for pending todos…")
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

    try:
        import dotenv

        _repo = Path(__file__).resolve().parent.parent.parent
        dotenv.load_dotenv(_repo / ".env")
    except ImportError:
        pass

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="gemma4", reasoning=True, num_ctx=16384*3)
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
    try:
        import dotenv

        dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass
    asyncio.run(main())
