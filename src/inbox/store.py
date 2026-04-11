"""SQLite inbox + FAISS vector index (Ollama embeddings + faiss-cpu).

Corpus vectors use ``embed_documents``; search uses ``embed_query`` (LangChain's
retrieval split). For Nomic models, ``search_document:`` / ``search_query:``
prefixes are applied. Vectors are L2-normalized; ``IndexFlatIP`` is cosine similarity.
"""

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

# Persisted index files (v2: embed_query for queries + Nomic prefixes when applicable).
_FAISS_INDEX_NAME = "faiss_l2ip_v2.index"
_FAISS_IDS_NAME = "faiss_l2ip_v2_ids.json"

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


# MMR: trade off query relevance vs. diversity so near-duplicate assistant lines
# (e.g. repeated packing replies) do not fill the result list.
_DEFAULT_MMR_LAMBDA = 0.55
_DEFAULT_MMR_FETCH_MULT = 6
_DEFAULT_MMR_FETCH_MIN_EXTRA = 20


def _faiss_mmr_indices(
    query_vec_flat: Any,
    candidate_faiss_indices: list[int],
    candidate_scores: list[float],
    index: Any,
    k: int,
    *,
    lambda_mult: float = _DEFAULT_MMR_LAMBDA,
) -> list[int]:
    """Pick up to ``k`` FAISS row ids using maximal marginal relevance.

    ``candidate_*`` must be parallel lists (same length). ``query_vec_flat`` is a single
    L2-normalized embedding (1D). Vectors in the index are assumed L2-normalized so
    dot products equal cosine similarity.
    """
    if np is None:
        raise RuntimeError("numpy required")
    if not candidate_faiss_indices or k <= 0:
        return []
    k = min(k, len(candidate_faiss_indices))
    if k == 1:
        best_i = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        return [candidate_faiss_indices[best_i]]

    vecs: list[Any] = []
    kept_positions: list[int] = []
    for pos, fi in enumerate(candidate_faiss_indices):
        try:
            v = index.reconstruct(int(fi))
        except Exception:
            logger.debug("FAISS reconstruct failed for row %s", fi, exc_info=True)
            continue
        vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        kept_positions.append(pos)

    if not vecs:
        return candidate_faiss_indices[:k]

    D = np.stack(vecs, axis=0)
    rel = np.array([candidate_scores[p] for p in kept_positions], dtype=np.float32)
    faiss_ids = [candidate_faiss_indices[p] for p in kept_positions]
    n = D.shape[0]
    k = min(k, n)
    sim = D @ D.T

    first = int(np.argmax(rel))
    selected_local: list[int] = [first]
    selected_set = {first}

    while len(selected_local) < k:
        best_j = -1
        best_mmr = -np.inf
        for j in range(n):
            if j in selected_set:
                continue
            max_sim_to_sel = max(sim[j, s] for s in selected_local)
            mmr = lambda_mult * rel[j] - (1.0 - lambda_mult) * max_sim_to_sel
            if mmr > best_mmr:
                best_mmr = mmr
                best_j = j
        if best_j < 0:
            break
        selected_local.append(best_j)
        selected_set.add(best_j)

    return [faiss_ids[i] for i in selected_local]


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
        self._index_path = self.base_dir / _FAISS_INDEX_NAME
        self._ids_path = self.base_dir / _FAISS_IDS_NAME
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

    def _use_nomic_asymmetric_prefixes(self) -> bool:
        """Nomic models are trained with search_query vs search_document prefixes for retrieval."""
        return "nomic" in self._ollama_model.casefold()

    def _text_for_document_embedding(self, raw: str) -> str:
        if self._use_nomic_asymmetric_prefixes():
            return f"search_document: {raw}"
        return raw

    def _text_for_query_embedding(self, raw: str) -> str:
        s = raw.strip() if raw else ""
        if not s:
            s = " "
        if self._use_nomic_asymmetric_prefixes():
            return f"search_query: {s}"
        return s

    def _encode_documents(self, texts: list[str]) -> Any:
        """Batch embeddings for stored messages (corpus side of retrieval)."""
        if np is None:
            raise RuntimeError("numpy required for FAISS")
        prepared = [self._text_for_document_embedding(t) for t in texts]
        emb = self._lazy_embeddings().embed_documents(prepared)
        arr = np.array(emb, dtype=np.float32)
        return _l2_normalize_rows(arr)

    def _encode_query_vector(self, query: str) -> Any:
        """Single query vector for search — uses ``embed_query`` (LangChain retrieval convention)."""
        if np is None:
            raise RuntimeError("numpy required for FAISS")
        text = self._text_for_query_embedding(query)
        emb = self._lazy_embeddings().embed_query(text)
        arr = np.array([emb], dtype=np.float32)
        return _l2_normalize_rows(arr)

    def _embedding_dim(self) -> int:
        """Dimension of the active Ollama embedding model (probe once)."""
        if self._dim is not None:
            return self._dim
        v = self._encode_documents(["."])
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
                    probe_dim = self._encode_documents(["probe"]).shape[1]
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
            emb = self._encode_documents(texts)
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

    def mark_human_read(self, message_id: str) -> None:
        """Set ``is_read`` for a human row (no-op if already read or missing)."""
        self._conn.execute(
            "UPDATE messages SET is_read = 1 WHERE id = ? AND type = 'human'",
            (message_id,),
        )
        self._conn.commit()

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
            emb = self._encode_documents([content])
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

    def fetch_messages_before_newest(self, *, count: int = 3) -> list[dict[str, Any]]:
        """The ``count`` messages strictly before the newest row (by ``created_at``). Oldest-first.

        Used to prepend recent history **excluding** the current user message (which is already the newest row).
        """
        count = max(1, min(count, 50))
        rows = self._conn.execute(
            """
            SELECT id, created_at, type, content, is_read
            FROM messages
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (count + 1,),
        ).fetchall()
        if len(rows) <= 1:
            return []
        prior = list(rows[1 : 1 + count])
        prior.reverse()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "type": r["type"],
                "content": r["content"],
                "is_read": bool(r["is_read"]),
            }
            for r in prior
        ]

    def fetch_recent_messages(
        self,
        *,
        limit: int = 5,
        mark_read_in_window: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the latest ``limit`` rows (human and assistant), oldest-first.

        Any **human** message in this slice that was unread is marked read when
        ``mark_read_in_window`` is true (does not affect other unread rows outside the window).
        Set to false when loading context for the model without affecting unread state.
        Separate from :meth:`fetch_unread_human`.
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

        if mark_read_in_window and unread_human_ids:
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

    def search_semantic(
        self,
        query: str,
        limit: int,
        *,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        if (
            _HAS_FAISS
            and _HAS_OLLAMA_EMB
            and self._index is not None
            and self._index.ntotal > 0
        ):
            try:
                q = self._encode_query_vector(query)
                ntotal = int(self._index.ntotal)
                lam = (
                    float(mmr_lambda)
                    if mmr_lambda is not None
                    else _DEFAULT_MMR_LAMBDA
                )
                lam = max(0.0, min(1.0, lam))

                if use_mmr and ntotal > 1 and limit > 1:
                    fetch_k = min(
                        ntotal,
                        max(limit * _DEFAULT_MMR_FETCH_MULT, limit + _DEFAULT_MMR_FETCH_MIN_EXTRA),
                    )
                else:
                    fetch_k = min(limit, ntotal)

                scores, indices = self._index.search(q, fetch_k)
                cand_idx: list[int] = []
                cand_score: list[float] = []
                seen_mid: set[str] = set()
                for score, idx in zip(scores[0], indices[0]):
                    ii = int(idx)
                    if ii < 0 or ii >= len(self._faiss_ids):
                        continue
                    mid = self._faiss_ids[ii]
                    if mid in seen_mid:
                        continue
                    seen_mid.add(mid)
                    cand_idx.append(ii)
                    cand_score.append(float(score))

                if use_mmr and len(cand_idx) > 1 and limit > 1:
                    chosen = _faiss_mmr_indices(
                        q.reshape(-1),
                        cand_idx,
                        cand_score,
                        self._index,
                        limit,
                        lambda_mult=lam,
                    )
                else:
                    chosen = cand_idx[:limit]

                score_by_faiss = dict(zip(cand_idx, cand_score))
                results: list[dict[str, Any]] = []
                for idx in chosen:
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
                                "score": float(score_by_faiss.get(idx, 0.0)),
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
