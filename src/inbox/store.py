"""SQLite inbox + FAISS vector index (Ollama nomic-embed-text + faiss-cpu)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

MessageType = Literal["human", "assistant"]

try:
    import faiss  # type: ignore[import-untyped]
    import numpy as np

    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HAS_FAISS = False

try:
    from langchain_ollama import OllamaEmbeddings

    _HAS_OLLAMA_EMB = True
except ImportError:
    OllamaEmbeddings = None  # type: ignore[assignment, misc]
    _HAS_OLLAMA_EMB = False


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _l2_normalize_rows(arr: Any) -> Any:
    if np is None:
        raise RuntimeError("numpy required")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class MessageInbox:
    """Message store: SQLite + optional FAISS (Ollama ``nomic-embed-text`` + faiss-cpu)."""

    def __init__(
        self,
        base_dir: Path,
        *,
        ollama_embed_model: str = "nomic-embed-text",
        ollama_base_url: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ollama_model = ollama_embed_model
        self._ollama_base_url = ollama_base_url
        self._embeddings: Any = None
        self.db_path = self.base_dir / "messages.sqlite"
        self._index_path = self.base_dir / "faiss.index"
        self._ids_path = self.base_dir / "faiss_ids.json"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._faiss_ids: list[str] = []
        self._index: Any = None
        self._dim: int | None = None
        self._load_or_build_index()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('human', 'assistant')),
                content TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0 CHECK (is_read IN (0, 1))
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_unread ON messages(is_read, type)"
        )
        self._conn.commit()

    def _lazy_embeddings(self) -> Any:
        if not _HAS_OLLAMA_EMB or OllamaEmbeddings is None:
            raise RuntimeError(
                "langchain-ollama is required for Ollama embeddings. "
                "Install with `pip install langchain-ollama` and run `ollama pull nomic-embed-text`.",
            )
        if self._embeddings is None:
            kwargs: dict[str, Any] = {"model": self._ollama_model}
            if self._ollama_base_url:
                kwargs["base_url"] = self._ollama_base_url
            self._embeddings = OllamaEmbeddings(**kwargs)
        return self._embeddings

    def _encode(self, texts: list[str]) -> Any:
        if np is None:
            raise RuntimeError("numpy required for FAISS")
        emb = self._lazy_embeddings().embed_documents(texts)
        arr = np.array(emb, dtype=np.float32)
        return _l2_normalize_rows(arr)

    def _embedding_dim(self) -> int:
        """Dimension of the active Ollama embedding model (probe once)."""
        if self._dim is not None:
            return self._dim
        v = self._encode(["."])
        self._dim = int(v.shape[1])
        return self._dim

    def _load_or_build_index(self) -> None:
        if not _HAS_FAISS or faiss is None or np is None:
            logger.warning(
                "faiss/numpy not available; search_messages will use SQL fallback. "
                "Install with `pip install faiss-cpu numpy` for vector search.",
            )
            return
        if not _HAS_OLLAMA_EMB:
            logger.warning(
                "langchain-ollama not available; search_messages will use SQL fallback.",
            )
            return

        if self._index_path.is_file() and self._ids_path.is_file():
            try:
                self._index = faiss.read_index(str(self._index_path))
                self._faiss_ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
                self._dim = self._index.d
                if self._index.ntotal != len(self._faiss_ids):
                    logger.warning("FAISS row count mismatch; rebuilding index.")
                    self._rebuild_index()
                    return
                try:
                    probe_dim = self._encode(["probe"]).shape[1]
                    if probe_dim != self._index.d:
                        logger.warning(
                            "Embedding dimension changed (%s vs %s); rebuilding FAISS index.",
                            self._index.d,
                            probe_dim,
                        )
                        self._rebuild_index()
                except RuntimeError as e:
                    logger.warning("Could not probe embeddings (%s); rebuilding index.", e)
                    self._rebuild_index()
                return
            except (OSError, ValueError, json.JSONDecodeError) as e:
                logger.warning("Failed to load FAISS index (%s); rebuilding.", e)

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not _HAS_FAISS or faiss is None or np is None:
            return
        rows = self._conn.execute(
            "SELECT id, content FROM messages ORDER BY created_at ASC",
        ).fetchall()
        self._faiss_ids = [str(r[0]) for r in rows]
        if not rows:
            try:
                dim = self._embedding_dim()
            except RuntimeError as e:
                logger.warning("Cannot probe Ollama embeddings (%s); vector index disabled.", e)
                self._index = None
                return
            self._index = faiss.IndexFlatIP(dim)
            self._persist_index()
            return

        texts = [str(r[1]) for r in rows]
        try:
            emb = self._encode(texts)
        except RuntimeError as e:
            logger.warning("Embedding failed (%s); vector index disabled.", e)
            self._index = None
            return
        self._dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(emb)
        self._persist_index()

    def _persist_index(self) -> None:
        if self._index is None or not _HAS_FAISS:
            return
        faiss.write_index(self._index, str(self._index_path))
        self._ids_path.write_text(json.dumps(self._faiss_ids), encoding="utf-8")

    def add_human(self, content: str, *, unread: bool = True) -> str:
        """Store a human message. When ``unread`` is false (e.g. live REPL input), it is not counted as inbox mail."""
        mid = str(uuid.uuid4())
        is_read = 0 if unread else 1
        self._conn.execute(
            """
            INSERT INTO messages (id, created_at, type, content, is_read)
            VALUES (?, ?, 'human', ?, ?)
            """,
            (mid, _utc_now_iso(), content, is_read),
        )
        self._conn.commit()
        self._append_vector(mid, content)
        return mid

    def add_assistant(self, content: str) -> str:
        mid = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO messages (id, created_at, type, content, is_read)
            VALUES (?, ?, 'assistant', ?, 1)
            """,
            (mid, _utc_now_iso(), content),
        )
        self._conn.commit()
        self._append_vector(mid, content)
        return mid

    def _append_vector(self, mid: str, content: str) -> None:
        if not _HAS_FAISS or faiss is None or self._index is None:
            return
        try:
            emb = self._encode([content])
            self._index.add(emb)
            self._faiss_ids.append(mid)
            self._persist_index()
        except RuntimeError as e:
            logger.warning("Skipping vector append: %s", e)

    def has_unread_human(self) -> bool:
        """True if any human message is still unread."""
        row = self._conn.execute(
            "SELECT 1 FROM messages WHERE is_read = 0 AND type = 'human' LIMIT 1",
        ).fetchone()
        return row is not None

    def fetch_unread_human(self, *, mark_as_read: bool) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, created_at, type, content, is_read
            FROM messages
            WHERE is_read = 0 AND type = 'human'
            ORDER BY created_at ASC
            """
        ).fetchall()
        out = [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "type": r["type"],
                "content": r["content"],
                "is_read": bool(r["is_read"]),
            }
            for r in rows
        ]
        if mark_as_read and out:
            ids = [r["id"] for r in out]
            self._conn.executemany(
                "UPDATE messages SET is_read = 1 WHERE id = ?",
                [(i,) for i in ids],
            )
            self._conn.commit()
        return out

    def fetch_recent_messages(self, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return the latest ``limit`` rows (human and assistant), oldest-first.

        Any **human** message in this slice that was unread is marked read (does not affect
        other unread rows outside the window). Separate from :meth:`fetch_unread_human`.
        """
        limit = max(1, min(limit, 50))
        rows = list(
            self._conn.execute(
                """
                SELECT id, created_at, type, content, is_read
                FROM messages
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall(),
        )
        rows.reverse()

        unread_human_ids: list[Any] = []
        out: list[dict[str, Any]] = []
        for r in rows:
            if r["type"] == "human" and int(r["is_read"]) == 0:
                unread_human_ids.append(r["id"])
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "type": r["type"],
                    "content": r["content"],
                    "is_read": bool(r["is_read"]),
                },
            )

        if unread_human_ids:
            self._conn.executemany(
                "UPDATE messages SET is_read = 1 WHERE id = ?",
                [(i,) for i in unread_human_ids],
            )
            self._conn.commit()
            marked = set(unread_human_ids)
            for item in out:
                if item["id"] in marked:
                    item["is_read"] = True

        return out

    def search_semantic(self, query: str, limit: int) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        if (
            _HAS_FAISS
            and _HAS_OLLAMA_EMB
            and self._index is not None
            and self._index.ntotal > 0
        ):
            try:
                q = self._encode([query])
                scores, indices = self._index.search(q, min(limit, self._index.ntotal))
                results: list[dict[str, Any]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self._faiss_ids):
                        continue
                    mid = self._faiss_ids[idx]
                    row = self._conn.execute(
                        "SELECT id, created_at, type, content, is_read FROM messages WHERE id = ?",
                        (mid,),
                    ).fetchone()
                    if row:
                        results.append(
                            {
                                "id": row["id"],
                                "created_at": row["created_at"],
                                "type": row["type"],
                                "content": row["content"],
                                "is_read": bool(row["is_read"]),
                                "score": float(score),
                            }
                        )
                return results
            except RuntimeError as e:
                logger.warning("Semantic search failed (%s); using SQL fallback.", e)

        return self._search_sql_fallback(query, limit)

    def _search_sql_fallback(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 2]
        if not tokens:
            rows = self._conn.execute(
                """
                SELECT id, created_at, type, content, is_read FROM messages
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            where = " AND ".join(["content LIKE ?" for _ in tokens])
            args = [f"%{t}%" for t in tokens]
            args.append(limit)
            rows = self._conn.execute(
                f"""
                SELECT id, created_at, type, content, is_read FROM messages
                WHERE {where}
                ORDER BY created_at DESC LIMIT ?
                """,
                args,
            ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "type": r["type"],
                "content": r["content"],
                "is_read": bool(r["is_read"]),
                "score": None,
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
