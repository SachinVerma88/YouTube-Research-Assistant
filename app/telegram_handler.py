"""
Telegram Handler
────────────────
Django view for Telegram webhooks + all bot command logic.
Uses python-telegram-bot for update parsing and response sending.
Powered by Ollama (local LLM) — zero API cost.
"""

import json
import logging
import asyncio
import re
import threading

from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings

from telegram import Bot, Update
from telegram.constants import ParseMode

from app.session_manager import session_manager, UserSession
from app.transcript_service import (
    validate_youtube_url,
    extract_video_id,
    fetch_transcript,
    get_key_timestamps,
)
from app.embedding_service import process_and_cache, is_cached
from app.summary_engine import generate_summary, generate_deepdive, generate_actionpoints
from app.rag_engine import answer_question
from app.language_service import (
    resolve_language,
    get_supported_languages_text,
    get_language_name,
)
from app.formatter import (
    format_welcome,
    format_processing,
    format_video_ready,
    format_summary,
    format_answer,
    format_deepdive,
    format_actionpoints,
    format_language_changed,
    format_language_error,
    format_reset,
    format_no_video,
    format_error,
    format_invalid_url,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Bot factory — fresh instance per request to avoid event-loop issues
# ──────────────────────────────────────────────


def _get_bot() -> Bot:
    """Create a new Bot instance (avoids stale event-loop references)."""
    return Bot(token=settings.TELEGRAM_BOT_TOKEN)


# ──────────────────────────────────────────────
# Helper to run async code from sync Django views
# ──────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from synchronous Django code."""
    return asyncio.run(coro)


async def _send_message(chat_id: int, text: str, parse_mode: str = ParseMode.HTML) -> None:
    """Send a message, splitting if it exceeds Telegram's limit."""
    bot = _get_bot()
    max_len = 4096
    if len(text) <= max_len:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    else:
        # Split into chunks at newline boundaries
        parts = []
        while text:
            if len(text) <= max_len:
                parts.append(text)
                break
            split_pos = text.rfind("\n", 0, max_len)
            if split_pos == -1:
                split_pos = max_len
            parts.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")
        for part in parts:
            await bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)


async def _send_action(chat_id: int, action: str = "typing") -> None:
    """Send a chat action (typing indicator)."""
    bot = _get_bot()
    await bot.send_chat_action(chat_id=chat_id, action=action)


# ──────────────────────────────────────────────
# Django views
# ──────────────────────────────────────────────

@csrf_exempt
@require_POST
def webhook_view(request: HttpRequest) -> JsonResponse:
    """Receive Telegram webhook updates and dispatch to handlers."""
    try:
        body = json.loads(request.body)
        update = Update.de_json(body, _get_bot())

        if update.message and update.message.text:
            # Process in background thread so we return 200 OK immediately.
            # This prevents Telegram from retrying during long LLM calls.
            thread = threading.Thread(
                target=_run_async,
                args=(_handle_message(update),),
                daemon=True,
            )
            thread.start()

        return JsonResponse({"ok": True})
    except Exception as exc:
        logger.exception("Webhook error: %s", exc)
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


@require_GET
def health_check(request: HttpRequest) -> JsonResponse:
    """Health check endpoint."""
    return JsonResponse({
        "status": "ok",
        "service": "YouTube Research Assistant (Ollama-powered)",
        "active_sessions": session_manager.active_count,
        "llm_model": settings.OLLAMA_LLM_MODEL,
        "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
    })


# ──────────────────────────────────────────────
# Message dispatcher
# ──────────────────────────────────────────────

async def _handle_message(update: Update) -> None:
    """Route incoming messages to the appropriate handler."""
    message = update.message
    text = message.text.strip()
    chat_id = message.chat_id
    user_id = message.from_user.id

    logger.info("Message from user %s: %s", user_id, text[:100])

    # Command routing
    if text.startswith("/"):
        await _handle_command(chat_id, user_id, text)
    elif validate_youtube_url(text):
        await _handle_youtube_url(chat_id, user_id, text)
    elif _extract_url_from_text(text):
        url = _extract_url_from_text(text)
        if validate_youtube_url(url):
            await _handle_youtube_url(chat_id, user_id, url)
        else:
            session = session_manager.get_session(user_id)
            if session.has_video():
                await _handle_question(chat_id, user_id, text)
            else:
                await _send_message(chat_id, format_invalid_url())
    else:
        session = session_manager.get_session(user_id)
        if session.has_video():
            await _handle_question(chat_id, user_id, text)
        else:
            await _send_message(
                chat_id,
                "👋 Send me a YouTube link to get started!\n\nOr type /help for all commands.",
            )


async def _handle_command(chat_id: int, user_id: int, text: str) -> None:
    """Dispatch bot commands."""
    parts = text.split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # handle /command@botname
    args = parts[1].strip() if len(parts) > 1 else ""

    handlers = {
        "/start": lambda: _cmd_start(chat_id),
        "/help": lambda: _cmd_start(chat_id),
        "/summary": lambda: _cmd_summary(chat_id, user_id),
        "/deepdive": lambda: _cmd_deepdive(chat_id, user_id),
        "/actionpoints": lambda: _cmd_actionpoints(chat_id, user_id),
        "/lang": lambda: _cmd_lang(chat_id, user_id, args),
        "/reset": lambda: _cmd_reset(chat_id, user_id),
        "/ask": lambda: _handle_question(chat_id, user_id, args) if args else _send_message(
            chat_id, "❓ Usage: /ask <i>your question about the video</i>"
        ),
    }

    handler = handlers.get(command)
    if handler:
        await handler()
    else:
        await _send_message(chat_id, "❓ Unknown command. Type /help for available commands.")


# ──────────────────────────────────────────────
# Command handlers
# ──────────────────────────────────────────────

async def _cmd_start(chat_id: int) -> None:
    await _send_message(chat_id, format_welcome())


async def _cmd_summary(chat_id: int, user_id: int) -> None:
    session = session_manager.get_session(user_id)
    if not session.has_video():
        await _send_message(chat_id, format_no_video())
        return

    await _send_action(chat_id)

    # Check cache (only valid if same language)
    if session.summary_cache:
        await _send_message(chat_id, format_summary(session.summary_cache))
        return

    try:
        lang_name = get_language_name(session.language)
        await _send_message(
            chat_id,
            f"📝 Generating structured summary in <b>{lang_name}</b>...\n"
            f"🤖 Using local LLM ({settings.OLLAMA_LLM_MODEL})\n"
            "⏳ This may take a moment."
        )
        await _send_action(chat_id)

        timestamps = get_key_timestamps(session.transcript_segments)

        # Generate directly in target language — no separate translation
        summary = generate_summary(
            session.transcript_text,
            session.chunks,
            timestamps,
            language=session.language,
        )

        session.summary_cache = summary
        await _send_message(chat_id, format_summary(summary))
    except Exception as exc:
        logger.exception("Summary generation failed: %s", exc)
        await _send_message(chat_id, format_error(f"Summary generation failed: {exc}"))


async def _cmd_deepdive(chat_id: int, user_id: int) -> None:
    session = session_manager.get_session(user_id)
    if not session.has_video():
        await _send_message(chat_id, format_no_video())
        return

    await _send_action(chat_id)

    if session.deepdive_cache:
        await _send_message(chat_id, format_deepdive(session.deepdive_cache))
        return

    try:
        lang_name = get_language_name(session.language)
        await _send_message(
            chat_id,
            f"🔍 Analyzing strategic implications in <b>{lang_name}</b>...\n"
            f"🤖 Using local LLM ({settings.OLLAMA_LLM_MODEL})\n"
            "⏳ This may take a moment."
        )
        await _send_action(chat_id)

        analysis = generate_deepdive(session.chunks, language=session.language)
        session.deepdive_cache = analysis
        await _send_message(chat_id, format_deepdive(analysis))
    except Exception as exc:
        logger.exception("Deep dive failed: %s", exc)
        await _send_message(chat_id, format_error(f"Deep dive failed: {exc}"))


async def _cmd_actionpoints(chat_id: int, user_id: int) -> None:
    session = session_manager.get_session(user_id)
    if not session.has_video():
        await _send_message(chat_id, format_no_video())
        return

    await _send_action(chat_id)

    if session.actionpoints_cache:
        await _send_message(chat_id, format_actionpoints(session.actionpoints_cache))
        return

    try:
        lang_name = get_language_name(session.language)
        await _send_message(
            chat_id,
            f"🎯 Extracting actionable insights in <b>{lang_name}</b>...\n"
            f"🤖 Using local LLM ({settings.OLLAMA_LLM_MODEL})\n"
            "⏳ This may take a moment."
        )
        await _send_action(chat_id)

        points = generate_actionpoints(session.chunks, language=session.language)
        session.actionpoints_cache = points
        await _send_message(chat_id, format_actionpoints(points))
    except Exception as exc:
        logger.exception("Action points failed: %s", exc)
        await _send_message(chat_id, format_error(f"Action points generation failed: {exc}"))


async def _cmd_lang(chat_id: int, user_id: int, lang_input: str) -> None:
    if not lang_input:
        session = session_manager.get_session(user_id)
        lang_name = get_language_name(session.language)
        supported = get_supported_languages_text()
        await _send_message(
            chat_id,
            f"🌐 <b>Current language:</b> {lang_name}\n\n"
            f"Supported: {supported}\n\n"
            "Usage: /lang hindi",
        )
        return

    result = resolve_language(lang_input)
    if result is None:
        await _send_message(chat_id, format_language_error(get_supported_languages_text()))
        return

    code, display = result
    session_manager.update_language(user_id, code)
    await _send_message(chat_id, format_language_changed(display))


async def _cmd_reset(chat_id: int, user_id: int) -> None:
    session_manager.reset_session(user_id)
    await _send_message(chat_id, format_reset())


# ──────────────────────────────────────────────
# YouTube URL processing
# ──────────────────────────────────────────────

async def _handle_youtube_url(chat_id: int, user_id: int, url: str) -> None:
    """Process a YouTube URL: fetch transcript, build embeddings, confirm ready."""
    video_id = extract_video_id(url)
    if not video_id:
        await _send_message(chat_id, format_invalid_url())
        return

    session = session_manager.get_session(user_id)
    cached = is_cached(video_id)

    await _send_message(chat_id, format_processing(video_id, cached=cached))
    await _send_action(chat_id)

    try:
        # Step 1: Fetch transcript
        result = fetch_transcript(video_id)
        if not result.success:
            await _send_message(chat_id, format_error(result.error))
            return

        # Step 2: Chunk, embed, and index (via Ollama nomic-embed-text)
        chunks, embeddings, index = process_and_cache(video_id, result.full_text)

        # Step 3: Update session
        session.video_id = video_id
        session.video_url = url
        session.transcript_text = result.full_text
        session.transcript_language = result.language
        session.transcript_segments = result.segments
        session.chunks = chunks
        session.embeddings = embeddings
        session.faiss_index = index
        session.history = []
        session.clear_caches()

        await _send_message(
            chat_id,
            format_video_ready(video_id, len(chunks), result.language),
        )

    except Exception as exc:
        logger.exception("Failed to process video %s: %s", video_id, exc)
        await _send_message(chat_id, format_error(f"Failed to process video: {exc}"))


# ──────────────────────────────────────────────
# Question handling (RAG)
# ──────────────────────────────────────────────

async def _handle_question(chat_id: int, user_id: int, question: str) -> None:
    """Answer a question about the loaded video using RAG (local Ollama)."""
    if not question.strip():
        await _send_message(chat_id, "❓ Please type your question about the video.")
        return

    session = session_manager.get_session(user_id)
    if not session.has_video():
        await _send_message(chat_id, format_no_video())
        return

    await _send_action(chat_id)

    try:
        rag_result = answer_question(question, session)
        await _send_message(
            chat_id,
            format_answer(rag_result.answer, rag_result.confidence_pct),
        )
    except Exception as exc:
        logger.exception("Q&A failed: %s", exc)
        await _send_message(chat_id, format_error(f"Failed to answer question: {exc}"))


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def _extract_url_from_text(text: str) -> str | None:
    """Extract a URL from mixed text."""
    url_pattern = re.compile(r"https?://\S+")
    match = url_pattern.search(text)
    return match.group(0) if match else None
