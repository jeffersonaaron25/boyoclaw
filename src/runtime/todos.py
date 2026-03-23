"""TODOS.md wake prep (SPEC)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Lines to prepend on wake / runtime start (dedupe by title key).
WAKE_TODO_LINES: list[str] = [
    "[] See unread messages | Use fetch_unread_messages tool to fetch unread messages from the inbox. | High | todo | ASAP",
    "[] Acknowledge unread messages | Use acknowledge(message: str) to acknowledge unread messages. | High | todo | ASAP",
    "[] Update TODOS.md | Update TODOS.md with the new TODOs. | High | todo | ASAP",
]

# Open todo: [] or [ ] at line start; completed: [x] / [X]
_OPEN_TODO = re.compile(r"^\[\s*\]\s")
_COMPLETED_TODO = re.compile(r"^\[[xX]\]\s*")


def _body_after_checkbox(line: str) -> str:
    s = line.strip()
    idx = s.find("]")
    if idx < 0:
        return ""
    return s[idx + 1 :].lstrip()


def _todo_title_key(line: str) -> str:
    """Title key for dedupe; supports both open [] and completed [x] lines."""
    line = line.strip()
    if not (_COMPLETED_TODO.match(line) or _OPEN_TODO.match(line)):
        return ""
    rest = _body_after_checkbox(line)
    if "|" not in rest:
        return rest[:80]
    return rest.split("|", 1)[0].strip()


# Title keys of WAKE_TODO_LINES — open todos with only these keys are default placeholders.
_WAKE_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    k for k in (_todo_title_key(ln) for ln in WAKE_TODO_LINES) if k
)


def clear_completed_todos(todos_path: Path) -> None:
    """Remove completed todo lines ([x] / [X]) from TODOS.md (SPEC: clear on wake)."""
    todos_path = Path(todos_path)
    if not todos_path.is_file():
        return
    lines = todos_path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not _is_completed_todo_line(ln)]
    text = "\n".join(kept)
    if text.strip():
        todos_path.write_text(text + "\n", encoding="utf-8")
    else:
        todos_path.write_text("", encoding="utf-8")


def _is_completed_todo_line(line: str) -> bool:
    return bool(_COMPLETED_TODO.match(line.strip()))


def has_pending_todos(todos_path: Path) -> bool:
    """True if there is at least one open (incomplete) todo line."""
    todos_path = Path(todos_path)
    if not todos_path.is_file():
        return False
    for line in todos_path.read_text(encoding="utf-8").splitlines():
        if _OPEN_TODO.match(line.strip()):
            return True
    return False


def has_non_placeholder_pending_todos(todos_path: Path) -> bool:
    """True if any open todo is not one of the default wake template lines (by title key)."""
    todos_path = Path(todos_path)
    if not todos_path.is_file():
        return False
    for line in todos_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not _OPEN_TODO.match(stripped):
            continue
        key = _todo_title_key(line)
        if not key or key not in _WAKE_PLACEHOLDER_KEYS:
            return True
    return False


def non_placeholder_open_todos_fingerprint(todos_path: Path) -> str:
    """SHA-256 of sorted open non-placeholder todo title keys — for wake deduplication.

    Same todo set → same fingerprint. When the set changes (complete/add/edit open item),
    the fingerprint changes so a new Wake is allowed.
    """
    todos_path = Path(todos_path)
    if not todos_path.is_file():
        return ""
    keys: list[str] = []
    for line in todos_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not _OPEN_TODO.match(stripped):
            continue
        key = _todo_title_key(line)
        if not key or key in _WAKE_PLACEHOLDER_KEYS:
            continue
        keys.append(key)
    keys.sort()
    if not keys:
        return ""
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepend_wake_todos(todos_path: Path) -> None:
    """Prepend SPEC wake TODOs at the top without duplicating the same titles."""
    todos_path = Path(todos_path)
    todos_path.parent.mkdir(parents=True, exist_ok=True)
    existing = todos_path.read_text(encoding="utf-8") if todos_path.exists() else ""
    existing_keys = {_todo_title_key(ln) for ln in existing.splitlines() if ln.strip()}
    existing_keys.discard("")
    to_prepend: list[str] = []
    for line in WAKE_TODO_LINES:
        key = _todo_title_key(line)
        if key and key not in existing_keys:
            to_prepend.append(line)
            existing_keys.add(key)
    if not to_prepend:
        return
    body = "\n".join(to_prepend)
    if existing.strip():
        todos_path.write_text(body + "\n\n" + existing.lstrip(), encoding="utf-8")
    else:
        todos_path.write_text(body + "\n", encoding="utf-8")


def ensure_memory_placeholder(memory_path: Path) -> None:
    """Ensure MEMORY.md exists (empty or one-line header)."""
    memory_path = Path(memory_path)
    if memory_path.exists():
        return
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "# MEMORY\n\n[Format: <Date>] <Type: Experience|Thought|Learning>:<Content>\n",
        encoding="utf-8",
    )


# Rough English-heavy estimate (~4 characters per token) for prompt-size guidance.
MEMORY_TOKEN_WARN_THRESHOLD = 10_000


def _approx_token_count_text(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def memory_system_prompt_appendix(memory_path: Path) -> str:
    """Load ``MEMORY.md`` for appending to the agent system prompt.

    When estimated size is at or above ``MEMORY_TOKEN_WARN_THRESHOLD``, injects an instruction
    to compress the file on the next edit.
    """
    memory_path = Path(memory_path)
    ensure_memory_placeholder(memory_path)
    try:
        raw = memory_path.read_text(encoding="utf-8")
    except OSError:
        raw = ""
    body = raw.strip()
    approx = _approx_token_count_text(body)

    lines = [
        "## MEMORY.md (workspace long-term memory, full contents below)",
        "",
    ]
    if approx >= MEMORY_TOKEN_WARN_THRESHOLD:
        lines.extend(
            [
                "**MEMORY.md is very large** (about {:,} estimated tokens; threshold {:,}). "
                "When you next **edit MEMORY.md** with your tools, **compress it**: keep only the most important "
                "durable facts, use compact notation (bullets, ISO dates, abbreviations, tables where helpful), "
                "and remove redundancy and stale entries. Prefer a short canonical summary plus pointers, not full logs."
                .format(approx, MEMORY_TOKEN_WARN_THRESHOLD),
                "",
            ]
        )
    lines.append("-----BEGIN MEMORY.md-----")
    lines.append(body if body else "(MEMORY.md has no content yet.)")
    lines.append("-----END MEMORY.md-----")
    return "\n".join(lines)
