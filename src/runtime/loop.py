"""Async BoyoClaw runtime: background agent worker + Rich CLI for human messages."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent import SandboxedAssistant, default_sandbox_directory, is_loop_control_message, project_root
from inbox.store import MessageInbox
from inbox.tools import build_inbox_tools
from runtime.todos import (
    clear_completed_todos,
    ensure_memory_placeholder,
    has_non_placeholder_pending_todos,
    prepend_wake_todos,
)
from runtime.turn import QueuedTurn

logger = logging.getLogger(__name__)

MYCLAW_RUNTIME_PROMPT = """
## BoyoClaw interaction rules
- Interactive terminal: the user already sees what they typed. Answer that line directly in a natural chat tone.
  Do not describe their lines as "unread inbox messages" or re-list them as mail unless they explicitly ask about the inbox.
- Use `fetch_unread_messages` only for "Wake Up" processing or when the user asks about unread/inbox mail—not on every reply.
- Use `search_messages` when you need to recall past conversation or facts (e.g. a name they mentioned before). You must use this 
  if user is following up on a previous conversation or fact and you have some context to search for.
- Use `read_recent_messages` to get the context of the recent conversation when you have no idea what the user is referring to.

## Default wake TODOs — never user-facing
`TODOS.md` is auto-prepended with three **system** lines (same idea every run): see unread / acknowledge / update TODOS.md.
These are **runtime housekeeping**, not the user's personal task list.

- In normal chat (e.g. "what's up?", "got any todos?", small talk): **do not** list, summarize, number, or offer to work through these three defaults. Treat them as invisible in replies.
- If asked broadly whether they have todos: mention **only** other, non-default items you see in `TODOS.md` after `read_file`; if there are none, say they have no extra todos (you may briefly note the system keeps an internal checklist—one short clause—without enumerating it).
- Only discuss those three items in detail if the user explicitly asks about the wake checklist, inbox workflow, or editing that section of `TODOS.md`.

- On receiving "Wake Up" message:
    1. Optionally read `TODOS.md` for context; do not recite the default three to the user.
    2. Use `fetch_unread_messages` and update todos as required by the workflow.
    3. Use `reply_to_human` for a concise status if needed; then continue real work.
- On receiving "Go to Sleep" message:
    1. Update todos to mark only incomplete tasks as remaining.
    2. Finish the response appropriately (do not treat as a normal user message).
- For each normal user message in the terminal, prefer a single clear `reply_to_human` with your answer.
  Use a short receipt first only if you are about to run tools that take noticeable time; then send one final `reply_to_human`.
- When completing a todo, mark it as completed in TODOS.md using [x] notation.
- Do not treat "Wake Up" or "Go to Sleep" as user mail; they are control messages (not stored in the inbox).
- Do not repeat the same substantive answer twice in one turn (avoid duplicate `reply_to_human` content).
- Some messages arrive from **Telegram** (authorized chats). Replies still go through `reply_to_human`; they are delivered to that Telegram chat. Do not tell the user to "check the terminal" when they wrote from Telegram.
- Files users send via Telegram are saved under `telegram_uploads/<date>/` and are attached to their **next text message** (uploads alone do not run the agent). The composed user message lists relative paths—read or process them when relevant.
- When the user is on **Telegram** and needs a file you created, call `send_file_to_user` with a **workspace-relative path** (and optional caption). For **terminal-only** users, do not use that tool—give them the path in `reply_to_human` instead. Large files may exceed Telegram limits; say so if send fails.

## MEMORY.md
- The **full text of `MEMORY.md`** is appended to your system instructions (after these rules). Treat it as long-term workspace memory unless the user contradicts it. Prefer **short, durable** entries; append or tighten rather than pasting huge transcripts.
- If you see a **size warning** above the MEMORY block, prioritize **rewriting `MEMORY.md` into a compact form** (essential facts only, compact notation) when you touch that file.
- If user asks you to do routine tasks, you must add to memory and ensure you add them to your TODOS.md and execute them at the interval of user choice. Ensure you get the relevant information from user before adding to memory.
- Save user preferences, your learnings and experiences in MEMORY.md if it is relevant to your long term memory.

## Workspace hygiene
- Keep the sandbox lean: remove **scratch**, **temporary**, and **duplicate** outputs you no longer need (old exports, repeated downloads, abandoned build dirs) once a task is done or superseded.
- Do **not** delete the user's inbox, long-term notes, `Memory/`, or project source unless they clearly asked or you confirmed it is disposable.
- Prefer one canonical artifact over many copies; if you regenerate a file, delete or replace the obsolete version.

## Response Formatting
- Do not use markdown formatting, use appropriate formatting for clear readability on telegram and terminal which do not support markdown.
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
        self.sandbox_root = (sandbox_root or default_sandbox_directory()).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
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

        self.inbox = MessageInbox(
            self.sandbox_root / "inbox",
            ollama_base_url=ollama_base_url,
        )
        self._delivery = DeliveryState()

        combined = (system_prompt or "").strip()
        if combined:
            combined = combined + "\n\n" + MYCLAW_RUNTIME_PROMPT
        else:
            combined = MYCLAW_RUNTIME_PROMPT.strip()

        tools = build_inbox_tools(
            self.inbox,
            on_reply=self._sync_reply_ui,
            delivery=self._delivery,
            telegram_file_try_send=self._try_schedule_telegram_file,
        )
        self._assistant = SandboxedAssistant(
            sandbox_root=self.sandbox_root,
            model=model,
            system_prompt=combined,
            extra_tools=tools,
            debug=True,
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
        """Deliver reply_to_human to Rich and/or Telegram (inbox row already added in tool)."""
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
                "to the file in reply_to_human (they are using the terminal).",
            )
        if app is None:
            return (
                False,
                "Telegram is not connected (--telegram not used or bot not running). "
                "Give the user the file path in reply_to_human instead.",
            )
        if self._loop is None:
            return False, "Runtime event loop is not ready; cannot send file."
        asyncio.run_coroutine_threadsafe(
            self._telegram_deliver_workspace_file(cid, app, rel, caption),
            self._loop,
        )
        return True, "Started uploading the file to the user's Telegram chat. You can still summarize in reply_to_human."

    async def _telegram_deliver_workspace_file(
        self,
        chat_id: int,
        app: Any,
        rel: str,
        caption: str,
    ) -> None:
        from runtime.telegram_bot import send_plain_text, send_workspace_file

        root = self.sandbox_root.resolve()
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

            await self._print_status("Agent running…")
            self._cli_prompt_allowed.set()

            composed = self._compose_user_message(turn)

            if is_loop_control_message(text):
                await self._run_agent_turn(
                    composed,
                    announce=False,
                    system_appendix=self._telegram_system_appendix(turn),
                )
                return

            self.inbox.add_human(composed, unread=False)
            await self._run_agent_turn(
                composed,
                announce=False,
                system_appendix=self._telegram_system_appendix(turn),
            )
        finally:
            self._pending_telegram_chat_id = None

    async def _run_agent_turn(
        self,
        user_message: str,
        *,
        announce: bool = True,
        system_appendix: str | None = None,
    ) -> None:
        if self._assistant is None:
            return
        self._delivery.reply_via_tool = False
        try:
            if announce:
                await self._print_status("Agent running…")
            result = await self._assistant.run_async(
                user_message,
                recursion_limit=self._recursion_limit,
                system_appendix=system_appendix,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent run failed")
            await self._agent_error_user_visible(f"Error: {e}")
            return

        if not self._delivery.reply_via_tool:
            fallback = _extract_assistant_text(result)
            if fallback:
                self.inbox.add_assistant(fallback)
                cid = self._pending_telegram_chat_id
                if cid is not None:
                    await self._telegram_send_text(cid, fallback)
                else:
                    await self._async_print_reply(fallback)

    async def _agent_worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = await asyncio.wait_for(self._human_queue.get(), timeout=0.35)
            except asyncio.TimeoutError:
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
            await self._human_queue.put(QueuedTurn(text=line.rstrip("\n\r")))

    async def _wake_maintenance(self) -> tuple[bool, Path]:
        """Clear completed todos, prepend default wake todos, ensure MEMORY. Returns (should_run_wake_agent, todos_path)."""
        todos_path = self.sandbox_root / "TODOS.md"
        clear_completed_todos(todos_path)
        prepend_wake_todos(todos_path)
        ensure_memory_placeholder(self.sandbox_root / "MEMORY.md")
        should_run = self.inbox.has_unread_human() or has_non_placeholder_pending_todos(
            todos_path,
        )
        return should_run, todos_path

    async def run_forever(self, *, enable_telegram: bool = False) -> None:
        self._loop = asyncio.get_running_loop()

        await self._print_status(
            "[bold]BoyoClaw[/bold] — workspace: %s" % self.sandbox_root,
        )

        should_wake, _todos = await self._wake_maintenance()
        if should_wake:
            await self._run_agent_turn("Wake Up")
        else:
            await self._print_status(
                "[dim]Idle: no unread inbox messages and no open todos beyond the default "
                "wake placeholders — Wake Up skipped; send a message to run the agent.[/dim]",
            )

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

        await asyncio.gather(
            self._input_loop(),
            self._agent_worker(),
            *([] if tg_task is None else [tg_task]),
        )


async def async_main(*, enable_telegram: bool = False) -> None:
    logging.basicConfig(level=logging.WARNING)
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from langchain_ollama import ChatOllama

    model = ChatOllama(model="minimax-m2.7:cloud")
    rt = BoyoClawRuntime(
        sandbox_root=project_root() / ".sandbox" / "workspace",
        model=model,
        system_prompt="You are BoyoClaw, a capable agent. Follow the BoyoClaw interaction rules.",
    )
    await rt.run_forever(enable_telegram=enable_telegram)


async def main() -> None:
    await async_main(enable_telegram=False)


if __name__ == "__main__":
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    asyncio.run(main())
