from __future__ import annotations

import base64
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

DEFAULT_IMAGE_MODEL = "gemma4"
IMAGE_VIEW_RESULT_CACHE_SIZE = 5
# Vision payload size guard (read_file / Telegram uploads stay well below this).
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_SOURCE_SEP = "\x1f"  # separates path and mtime in lru_cache key (not valid in paths)


def _truncate(text: str, *, max_chars: int = 12000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated] ..."


def _coerce_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _build_payload(
    *,
    prompt: str,
    image_url: Optional[str],
    mime_type: str,
    file_id: Optional[str],
) -> list[Dict[str, Any]]:
    provided_count = sum(
        1
        for value in (image_url, file_id)
        if isinstance(value, str) and value.strip()
    )
    if provided_count != 1:
        raise ValueError("exactly one of image_url or file_id must be provided")

    text = (prompt or "").strip() or "Describe this image."
    content: list[Dict[str, Any]] = [{"type": "text", "text": text}]

    if image_url and image_url.strip():
        url = image_url.strip()
        content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    content.append(
        {
            "type": "file",
            "file_id": file_id.strip(),
            "mime_type": (mime_type or "image/jpeg").strip(),
        }
    )
    return content


def _guess_image_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    suf = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suf, "image/jpeg")


def _safe_resolve_under_agent_home(raw: str, agent_home: Path) -> tuple[Path, int]:
    """Resolve a user path to a file under ``agent_home``; returns (path, mtime_ns)."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("file_path is empty")
    if ".." in Path(s.replace("\\", "/")).parts:
        raise ValueError("path must not contain '..'")
    agent_home = agent_home.resolve()
    p = Path(s)
    resolved = (p.resolve() if p.is_absolute() else (agent_home / p).resolve())
    try:
        resolved.relative_to(agent_home)
    except ValueError as e:
        raise ValueError("file_path must be under the agent workspace (.agent-home)") from e
    if not resolved.is_file():
        raise ValueError(f"not a file or missing: {resolved}")
    st = resolved.stat()
    if st.st_size > _MAX_IMAGE_BYTES:
        raise ValueError(f"image too large (max {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB)")
    return resolved, int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))


def _content_from_local_file(*, prompt: str, path: Path, mime_hint: str) -> list[Dict[str, Any]]:
    data = path.read_bytes()
    mime = (mime_hint or "").strip() or _guess_image_mime(path)
    if not mime.startswith("image/"):
        mime = _guess_image_mime(path)
    b64 = base64.standard_b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    text = (prompt or "").strip() or "Describe this image."
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


@lru_cache(maxsize=10)
def _get_image_llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, reasoning=True, num_ctx=16384 * 3)


@lru_cache(maxsize=IMAGE_VIEW_RESULT_CACHE_SIZE)
def _image_view_cached_result(
    model: str,
    prompt: str,
    source_kind: str,
    source_payload: str,
    mime_type: str,
) -> str:
    """source_kind: ``url`` | ``file_id`` | ``file`` (payload is url, id, or path+mtime key)."""
    llm = _get_image_llm(model)
    if source_kind == "file":
        path_str, _, mtime_s = source_payload.partition(_SOURCE_SEP)
        path = Path(path_str)
        mtime_ns = int(mtime_s) if mtime_s.isdigit() else 0
        if not path.is_file():
            return "image_view error: file no longer exists"
        st = path.stat()
        cur = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        if mtime_ns and cur != mtime_ns:
            return "image_view error: file changed since lookup; call image_view again"
        if st.st_size > _MAX_IMAGE_BYTES:
            return "image_view error: image too large"
        content = _content_from_local_file(prompt=prompt, path=path, mime_hint=mime_type)
    elif source_kind == "url":
        content = _build_payload(
            prompt=prompt,
            image_url=source_payload,
            mime_type=mime_type,
            file_id=None,
        )
    elif source_kind == "file_id":
        content = _build_payload(
            prompt=prompt,
            image_url=None,
            mime_type=mime_type,
            file_id=source_payload,
        )
    else:
        return f"image_view error: invalid source_kind {source_kind!r}"

    message = HumanMessage(content=content)
    response = llm.invoke([message])
    return _truncate(_coerce_content_text(getattr(response, "content", "")) or "(no response)")


def build_image_view_tool(agent_home: Path):
    """Factory: ``file_path`` is resolved under ``agent_home`` (``.agent-home``)."""
    root = Path(agent_home).resolve()

    @tool
    def image_view(
        prompt: str,
        image_url: Optional[str] = None,
        file_id: Optional[str] = None,
        file_path: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Analyze an image and return a concise textual description.

        Exactly one image source must be provided:
        - image_url: http(s) URL or data URL
        - file_id (+ mime_type): uploaded Google GenAI file URI
        - file_path: path relative to .agent-home, or absolute path still inside .agent-home
          (e.g. telegram_uploads/2026-04-11/photo.jpg)

        Use this image to understand what's in images and seemlessly use that to continue conversation.
        """
        try:
            p = (prompt or "").strip() or "Describe this image."
            url = (image_url or "").strip() or None
            fid = (file_id or "").strip() or None
            fp = (file_path or "").strip() or None
            mt = (mime_type or "image/jpeg").strip() or "image/jpeg"
            n = sum(1 for x in (url, fid, fp) if x)
            if n != 1:
                return (
                    "image_view error: provide exactly one of image_url, file_id, or file_path "
                    f"(got {n})."
                )
            if fp:
                path, mtime_ns = _safe_resolve_under_agent_home(fp, root)
                key = f"{path.resolve().as_posix()}{_SOURCE_SEP}{mtime_ns}"
                return _image_view_cached_result(
                    DEFAULT_IMAGE_MODEL.strip(),
                    p,
                    "file",
                    key,
                    mt,
                )
            if url:
                return _image_view_cached_result(
                    DEFAULT_IMAGE_MODEL.strip(),
                    p,
                    "url",
                    url,
                    mt,
                )
            return _image_view_cached_result(
                DEFAULT_IMAGE_MODEL.strip(),
                p,
                "file_id",
                fid or "",
                mt,
            )
        except Exception as exc:
            return f"image_view error: {exc}"

    return image_view
