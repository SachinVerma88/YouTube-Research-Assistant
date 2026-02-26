"""
Session Manager
───────────────
Per-user session state: video context, embeddings, language, history.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Holds all state for a single user's active session."""
    user_id: int
    video_id: str | None = None
    video_url: str | None = None
    transcript_text: str | None = None
    transcript_language: str = "en"
    transcript_segments: list = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    embeddings: Any = None           # numpy array
    faiss_index: Any = None          # FAISS index
    language: str = "en"             # user's preferred response language
    history: list[dict] = field(default_factory=list)
    summary_cache: str | None = None
    deepdive_cache: str | None = None
    actionpoints_cache: str | None = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def has_video(self) -> bool:
        return self.video_id is not None and self.transcript_text is not None

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        # Keep last 20 messages to control context size
        if len(self.history) > 20:
            self.history = self.history[-20:]
        self.last_active = time.time()

    def clear_caches(self) -> None:
        self.summary_cache = None
        self.deepdive_cache = None
        self.actionpoints_cache = None


class SessionManager:
    """
    In-memory session store keyed by Telegram user ID.
    Thread-safe for the single-process Django dev server.
    """

    def __init__(self) -> None:
        self._sessions: dict[int, UserSession] = {}

    def get_session(self, user_id: int) -> UserSession:
        """Get or create a session for a user."""
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        session = self._sessions[user_id]
        session.last_active = time.time()
        return session

    def reset_session(self, user_id: int) -> UserSession:
        """Reset a user's session completely."""
        self._sessions[user_id] = UserSession(user_id=user_id)
        logger.info("Session reset for user %s", user_id)
        return self._sessions[user_id]

    def update_language(self, user_id: int, lang: str) -> None:
        session = self.get_session(user_id)
        session.language = lang
        session.clear_caches()  # regenerate in new language
        logger.info("Language set to '%s' for user %s", lang, user_id)

    def cleanup_stale(self, max_age_seconds: int = 3600) -> int:
        """Remove sessions older than max_age_seconds. Returns count removed."""
        now = time.time()
        stale = [
            uid
            for uid, s in self._sessions.items()
            if now - s.last_active > max_age_seconds
        ]
        for uid in stale:
            del self._sessions[uid]
        if stale:
            logger.info("Cleaned up %d stale sessions", len(stale))
        return len(stale)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ──────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────
session_manager = SessionManager()
