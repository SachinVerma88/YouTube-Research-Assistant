"""
Response Formatter
──────────────────
Rich Telegram formatting with emojis, bold, and clean sections.
Uses Telegram HTML parse mode for reliable formatting.
"""


def format_welcome() -> str:
    """Format the welcome / start message."""
    return (
        "🎥 <b>AI YouTube Research Assistant</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "I analyze YouTube videos and provide:\n"
        "• 📝 Structured summaries\n"
        "• ❓ Context-aware Q&A (RAG)\n"
        "• 🔍 Strategic deep-dives\n"
        "• 🎯 Actionable insights\n"
        "• 🌐 Multi-language support\n\n"
        "<b>🚀 Quick Start:</b>\n"
        "Just send me a YouTube link!\n\n"
        "<b>📋 Commands:</b>\n"
        "/summary — Structured video summary\n"
        "/ask <i>question</i> — Ask about the video\n"
        "/deepdive — Strategic analysis\n"
        "/actionpoints — 5 actionable insights\n"
        "/lang <i>language</i> — Switch language\n"
        "/reset — Clear session\n"
        "/help — Show this message\n"
    )


def format_processing(video_id: str, cached: bool = False) -> str:
    """Format processing status message."""
    if cached:
        return (
            "⚡ <b>Video found in cache!</b>\n"
            f"📹 Video ID: <code>{video_id}</code>\n\n"
            "Loading cached analysis... This will be instant!"
        )
    return (
        "🔄 <b>Processing video...</b>\n"
        f"📹 Video ID: <code>{video_id}</code>\n\n"
        "⏳ Steps:\n"
        "1️⃣ Fetching transcript...\n"
        "2️⃣ Chunking text...\n"
        "3️⃣ Generating embeddings...\n"
        "4️⃣ Building vector index...\n\n"
        "This may take 15-30 seconds."
    )


def format_video_ready(video_id: str, chunk_count: int, lang: str) -> str:
    """Format video-ready confirmation."""
    return (
        "✅ <b>Video analyzed successfully!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📹 Video ID: <code>{video_id}</code>\n"
        f"📊 Chunks: {chunk_count}\n"
        f"🌐 Language: {lang}\n\n"
        "<b>What would you like to do?</b>\n"
        "• /summary — Get structured summary\n"
        "• /deepdive — Strategic analysis\n"
        "• /actionpoints — Actionable insights\n"
        "• Just type a question to ask about the video!"
    )


def format_summary(summary: str) -> str:
    """Format a summary for Telegram."""
    return (
        "📋 <b>VIDEO SUMMARY</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{summary}"
    )


def format_answer(answer_text: str, confidence_pct: int) -> str:
    """Format a RAG answer with confidence score."""
    # Confidence badge
    if confidence_pct >= 80:
        badge = "🟢"
    elif confidence_pct >= 50:
        badge = "🟡"
    else:
        badge = "🔴"

    return (
        "💬 <b>Answer</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{answer_text}\n\n"
        f"{badge} <b>Confidence:</b> {confidence_pct}%"
    )


def format_deepdive(analysis: str) -> str:
    """Format a deep-dive analysis."""
    return (
        "🔍 <b>STRATEGIC DEEP DIVE</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{analysis}"
    )


def format_actionpoints(points: str) -> str:
    """Format actionable insights."""
    return (
        "🎯 <b>ACTIONABLE INSIGHTS</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{points}"
    )


def format_language_changed(lang_display: str) -> str:
    """Format language change confirmation."""
    return (
        f"🌐 <b>Language changed to {lang_display}</b>\n\n"
        "All future responses will be translated.\n"
        "Cached summaries have been cleared — they'll be regenerated in the new language."
    )


def format_language_error(supported: str) -> str:
    """Format unsupported language error."""
    return (
        "❌ <b>Unsupported language</b>\n\n"
        f"Supported languages: {supported}\n\n"
        "Usage: /lang hindi"
    )


def format_reset() -> str:
    """Format session reset confirmation."""
    return (
        "🔄 <b>Session reset!</b>\n\n"
        "All video data and conversation history cleared.\n"
        "Send a new YouTube link to start fresh."
    )


def format_no_video() -> str:
    """Prompt user to send a video link."""
    return (
        "⚠️ <b>No video loaded</b>\n\n"
        "Please send a YouTube link first.\n"
        "Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )


def format_error(message: str) -> str:
    """Format a generic error message."""
    return f"❌ <b>Error</b>\n\n{message}"


def format_invalid_url() -> str:
    """Format invalid URL error."""
    return (
        "❌ <b>Invalid YouTube URL</b>\n\n"
        "Please send a valid YouTube video link.\n\n"
        "Supported formats:\n"
        "• https://www.youtube.com/watch?v=...\n"
        "• https://youtu.be/...\n"
        "• https://www.youtube.com/shorts/..."
    )
