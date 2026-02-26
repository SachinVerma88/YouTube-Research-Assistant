"""
Language Service
────────────────
Multi-language support using the local LLM via Ollama.
No paid translation API — generates directly in the target language.
Supports: English, Hindi, Kannada, Tamil, Telugu, Bengali, Marathi.
"""

import logging
import ollama
from django.conf import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Supported languages
# ──────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": "English",
    "english": "English",
    "hindi": "Hindi",
    "hi": "Hindi",
    "kannada": "Kannada",
    "kn": "Kannada",
    "tamil": "Tamil",
    "ta": "Tamil",
    "telugu": "Telugu",
    "te": "Telugu",
    "bengali": "Bengali",
    "bn": "Bengali",
    "marathi": "Marathi",
    "mr": "Marathi",
}

LANG_TO_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
}

LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
}


def resolve_language(lang_input: str) -> tuple[str, str] | None:
    """
    Resolve a user's language input to (code, display_name).
    Returns None if the language is not supported.
    """
    key = lang_input.strip().lower()
    display = SUPPORTED_LANGUAGES.get(key)
    if display is None:
        return None
    code = LANG_TO_CODE.get(display, "en")
    return code, display


def get_supported_languages_text() -> str:
    """Return a formatted list of supported languages."""
    unique = sorted(set(SUPPORTED_LANGUAGES.values()))
    return ", ".join(unique)


def get_language_name(code: str) -> str:
    """Get display name for a language code."""
    return LANG_CODE_TO_NAME.get(code, code)


def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to the target language using the local LLM via Ollama.
    This is a fallback — the preferred approach is to generate directly
    in the target language via prompt engineering (see summary_engine.py).
    """
    if target_lang in ("en", "English"):
        return text

    lang_name = LANG_CODE_TO_NAME.get(target_lang, target_lang)

    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        response = client.chat(
            model=settings.OLLAMA_LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following text to {lang_name}. "
                    f"Keep all formatting, emojis, and structure intact. "
                    f"Output ONLY the translation, nothing else.\n\n"
                    f"TEXT:\n{text}"
                ),
            }],
            options={
                "temperature": 0.1,
                "num_predict": 2048,
            },
        )
        translated = response.message.content.strip()
        logger.info("Translated %d chars to %s via Ollama", len(text), lang_name)
        return translated
    except Exception as exc:
        logger.warning("Translation to %s failed: %s — returning original", target_lang, exc)
        return text
