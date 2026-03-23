"""SQLite message inbox with optional FAISS semantic search."""

from .store import MessageInbox
from .tools import build_inbox_tools

__all__ = ["MessageInbox", "build_inbox_tools"]
