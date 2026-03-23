"""Telegram long-polling sidecar (``--telegram``)."""

from __future__ import annotations

import logging
from pathlib import Path

from runtime.telegram_config import TelegramSettings

logger = logging.getLogger(__name__)

TELEGRAM_CHAR_LIMIT = 4096


def format_telegram_sender_label(update: object) -> str | None:
    """Human-readable name / @username from a python-telegram-bot ``Update``."""
    user = getattr(update, "effective_user", None)
    if user is None:
        return None
    fn = (getattr(user, "first_name", None) or "").strip()
    ln = (getattr(user, "last_name", None) or "").strip()
    parts = [p for p in (fn, ln) if p]
    name = " ".join(parts).strip()
    un = getattr(user, "username", None)
    un_s = f"@{un}" if un else ""
    if name and un_s:
        return f"{name} ({un_s})"
    if name:
        return name
    if un_s:
        return un_s
    return None


def chunk_telegram_text(text: str, limit: int = TELEGRAM_CHAR_LIMIT) -> list[str]:
    text = text.strip() or "…"
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


async def send_plain_text(bot: object, chat_id: int, text: str) -> None:
    from telegram.error import TelegramError

    for chunk in chunk_telegram_text(text):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk)
        except TelegramError:
            logger.exception("Telegram send_message failed chat_id=%s", chat_id)
            return


def _local_file_upload_input(path: Path) -> object:
    """Build a python-telegram-bot upload object for a path.

    ``FSInputFile`` existed in older releases; current ``python-telegram-bot`` only
    documents ``InputFile``. Do **not** pass a filesystem path as ``str`` to
    ``InputFile`` — strings are encoded as UTF-8 text, not read as files.
    """
    path = Path(path).resolve()
    try:
        from telegram import FSInputFile  # type: ignore[attr-defined,unused-ignore]
    except ImportError:
        FSInputFile = None  # type: ignore[misc,assignment]
    if FSInputFile is not None:
        try:
            return FSInputFile(str(path))
        except TypeError:
            return FSInputFile(path)

    from telegram import InputFile

    return InputFile(path.read_bytes(), filename=path.name)


async def send_workspace_file(
    bot: object,
    chat_id: int,
    path: Path,
    *,
    caption: str | None = None,
) -> None:
    """Send a file from disk to Telegram (photo for common images, otherwise document)."""
    import mimetypes

    from telegram.error import TelegramError

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    cap = (caption or "").strip()[:1024] or None
    mime, _ = mimetypes.guess_type(str(path))
    upload = _local_file_upload_input(path)

    try:
        if mime and mime.startswith("image/"):
            await bot.send_photo(chat_id=chat_id, photo=upload, caption=cap)
        else:
            await bot.send_document(chat_id=chat_id, document=upload, caption=cap)
    except TelegramError:
        logger.exception("Telegram send file failed chat_id=%s path=%s", chat_id, path)
        raise


def _mime_extension(mime: str | None) -> str:
    if not mime or "/" not in mime:
        return "bin"
    sub = mime.split("/")[-1].strip().lower()
    return (sub[:24] or "bin").replace(";", "")


async def run_telegram_polling(runtime: object, settings: TelegramSettings) -> None:
    from telegram import Update
    from telegram.error import TelegramError
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    from runtime.telegram_uploads import download_telegram_file
    from runtime.turn import QueuedTurn

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or update.message is None:
            return
        cid = update.effective_chat.id
        if cid not in settings.allowed_chat_ids:
            await update.message.reply_text("Not authorized.")
            return
        await update.message.reply_text(
            "BoyoClaw Telegram is linked to your workspace.\n"
            "• Send text to run the agent.\n"
            "• You can upload files first (documents, photos, video, audio, voice); they are saved under "
            "`telegram_uploads/` and picked up on your **next text message**.\n"
            "• Voice uploads start the agent immediately (no extra text needed).\n"
            "• /agent_pause — stop all agent wakes and replies (listener stays on). "
            "/agent_resume — resume.\n"
            "Use /help for this message.",
        )

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await cmd_start(update, context)

    async def cmd_agent_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or update.message is None:
            return
        cid = update.effective_chat.id
        if cid not in settings.allowed_chat_ids:
            await update.message.reply_text("Not authorized.")
            return
        if runtime.agent_paused():
            await update.message.reply_text("Agent is already paused.")
            return
        runtime.set_agent_paused(True)
        await update.message.reply_text(
            "Agent paused. No wakes or agent responses until you send /agent_resume.",
        )

    async def cmd_agent_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or update.message is None:
            return
        cid = update.effective_chat.id
        if cid not in settings.allowed_chat_ids:
            await update.message.reply_text("Not authorized.")
            return
        if not runtime.agent_paused():
            await update.message.reply_text("Agent was not paused.")
            return
        runtime.set_agent_paused(False)
        await update.message.reply_text("Agent resumed. You can send messages again.")

    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or update.message is None:
            return
        msg = update.message.text
        if not msg or not msg.strip():
            return
        cid = update.effective_chat.id
        if cid not in settings.allowed_chat_ids:
            await update.message.reply_text("Not authorized.")
            return
        msg_stripped = msg.strip()
        if runtime.agent_paused():
            await update.message.reply_text(
                "Agent is paused. Send /agent_resume or /agent-resume to continue.",
            )
            return
        paths, caption_prefix = runtime.pop_telegram_pending_uploads(cid)
        body = caption_prefix + msg_stripped
        await runtime.enqueue_turn(
            QueuedTurn(
                text=body,
                telegram_chat_id=cid,
                uploaded_workspace_paths=paths,
                telegram_sender_label=format_telegram_sender_label(update),
            ),
        )

    async def _save_upload_pending(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        file_id: str,
        suggested_name: str,
        auto_enqueue_text: str | None = None,
        notify_saved: bool = True,
    ) -> None:
        msg = update.message
        if msg is None or update.effective_chat is None:
            return
        cid = update.effective_chat.id
        if cid not in settings.allowed_chat_ids:
            await msg.reply_text("Not authorized.")
            return
        caption = (msg.caption or "").strip()
        try:
            rel = await download_telegram_file(
                context.bot,
                agent_home=runtime.agent_home,
                file_id=file_id,
                suggested_name=suggested_name,
            )
        except TelegramError as e:
            logger.exception("Telegram file API error")
            await msg.reply_text(f"Could not fetch file from Telegram: {e}")
            return
        except OSError as e:
            logger.exception("Telegram file write failed")
            await msg.reply_text(f"Could not save file to workspace: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Telegram download failed")
            await msg.reply_text(f"Could not save file: {e}")
            return

        if auto_enqueue_text is not None:
            if runtime.agent_paused():
                await msg.reply_text(
                    "Agent is paused. Send /agent_resume or /agent-resume to continue.",
                )
                return
            body = caption if caption else auto_enqueue_text
            await runtime.enqueue_turn(
                QueuedTurn(
                    text=body,
                    telegram_chat_id=cid,
                    uploaded_workspace_paths=(rel,),
                    telegram_sender_label=format_telegram_sender_label(update),
                ),
            )
            return

        runtime.record_telegram_pending_upload(cid, rel, caption)
        if notify_saved:
            await msg.reply_text(
                f"Saved to agent workspace: {rel}\n"
                "Send a text message when you want the agent to use it."
            )

    async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or msg.document is None:
            return
        doc = msg.document
        name = doc.file_name or f"upload.{_mime_extension(doc.mime_type)}"
        await _save_upload_pending(update, context, file_id=doc.file_id, suggested_name=name)

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or not msg.photo:
            return
        photo = msg.photo[-1]
        await _save_upload_pending(update, context, file_id=photo.file_id, suggested_name="photo.jpg")

    async def on_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or msg.video is None:
            return
        v = msg.video
        name = v.file_name or "video.mp4"
        await _save_upload_pending(update, context, file_id=v.file_id, suggested_name=name)

    async def on_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or msg.audio is None:
            return
        a = msg.audio
        name = a.file_name or f"audio.{_mime_extension(a.mime_type)}"
        await _save_upload_pending(update, context, file_id=a.file_id, suggested_name=name)

    async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or msg.voice is None:
            return
        await _save_upload_pending(
            update,
            context,
            file_id=msg.voice.file_id,
            suggested_name="voice.ogg",
            auto_enqueue_text="The user sent a Telegram voice message. Transcribe it and respond to the user.",
            notify_saved=False,
        )

    async def on_animation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or msg.animation is None:
            return
        anim = msg.animation
        name = anim.file_name or "animation.mp4"
        await _save_upload_pending(update, context, file_id=anim.file_id, suggested_name=name)

    application = Application.builder().token(settings.bot_token).build()
    runtime.attach_telegram_application(application)

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("agent_pause", cmd_agent_pause))
    application.add_handler(CommandHandler("agent_resume", cmd_agent_resume))
    application.add_handler(
        MessageHandler(filters.TEXT & filters.Regex(r"(?i)^/agent-pause$"), cmd_agent_pause),
    )
    application.add_handler(
        MessageHandler(filters.TEXT & filters.Regex(r"(?i)^/agent-resume$"), cmd_agent_resume),
    )
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, on_document))
    application.add_handler(MessageHandler(filters.VIDEO, on_video))
    application.add_handler(MessageHandler(filters.AUDIO, on_audio))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(MessageHandler(filters.ANIMATION, on_animation))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    try:
        async with application:
            await application.start()
            if application.updater is None:
                raise RuntimeError("Telegram Application has no updater")
            await application.updater.start_polling(drop_pending_updates=True)
            await runtime.shutdown.wait()
    finally:
        runtime.attach_telegram_application(None)
