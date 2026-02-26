"""
Transcript Service
──────────────────
Validates YouTube URLs, extracts video IDs, fetches and cleans transcripts.
Compatible with youtube-transcript-api v1.2.x (instance-based API).
"""

import re
import logging
from dataclasses import dataclass, field

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# URL patterns
# ──────────────────────────────────────────────

YOUTUBE_PATTERNS = [
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"
    ),
]

# Singleton API instance
_ytt_api = YouTubeTranscriptApi()


@dataclass
class TranscriptSegment:
    """A single transcript segment with timestamp."""
    text: str
    start: float
    duration: float


@dataclass
class TranscriptResult:
    """Complete transcript result."""
    video_id: str
    full_text: str
    language: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def validate_youtube_url(url: str) -> bool:
    """Check if a string is a valid YouTube video URL."""
    url = url.strip()
    return any(pattern.match(url) for pattern in YOUTUBE_PATTERNS)


def extract_video_id(url: str) -> str | None:
    """Extract the 11-character video ID from a YouTube URL."""
    url = url.strip()
    for pattern in YOUTUBE_PATTERNS:
        match = pattern.match(url)
        if match:
            return match.group(1)
    return None


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    total = int(seconds)
    hrs, remainder = divmod(total, 3600)
    mins, secs = divmod(remainder, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def fetch_transcript(video_id: str) -> TranscriptResult:
    """
    Fetch transcript for a YouTube video using youtube-transcript-api v1.2.x.

    Strategy:
        1. Try direct fetch with English
        2. List all transcripts and try manually-created ones first
        3. Fall back to any available transcript
    """
    try:
        # Strategy 1: Quick fetch — tries English by default
        try:
            raw = _ytt_api.fetch(video_id, languages=["en"])
            segments = [
                TranscriptSegment(
                    text=snippet.text.strip(),
                    start=snippet.start,
                    duration=snippet.duration,
                )
                for snippet in raw
            ]
            full_text = _clean_text(" ".join(seg.text for seg in segments))
            return TranscriptResult(
                video_id=video_id,
                full_text=full_text,
                language="en",
                segments=segments,
            )
        except NoTranscriptFound:
            pass

        # Strategy 2: List available transcripts and pick the best one
        try:
            transcript_list = _ytt_api.list(video_id)
            language = "en"

            # Try manually created first, then generated
            best_transcript = None
            for transcript in transcript_list:
                if not transcript.is_generated:
                    best_transcript = transcript
                    language = transcript.language_code
                    break

            if best_transcript is None:
                for transcript in transcript_list:
                    best_transcript = transcript
                    language = transcript.language_code
                    break

            if best_transcript is None:
                return TranscriptResult(
                    video_id=video_id,
                    full_text="",
                    language="",
                    error="❌ Transcript not available for this video.",
                )

            raw = best_transcript.fetch()
            segments = [
                TranscriptSegment(
                    text=snippet.text.strip(),
                    start=snippet.start,
                    duration=snippet.duration,
                )
                for snippet in raw
            ]
            full_text = _clean_text(" ".join(seg.text for seg in segments))
            return TranscriptResult(
                video_id=video_id,
                full_text=full_text,
                language=language,
                segments=segments,
            )
        except Exception:
            return TranscriptResult(
                video_id=video_id,
                full_text="",
                language="",
                error="❌ Transcript not available for this video.",
            )

    except TranscriptsDisabled:
        return TranscriptResult(
            video_id=video_id,
            full_text="",
            language="",
            error="❌ Transcripts are disabled for this video.",
        )
    except VideoUnavailable:
        return TranscriptResult(
            video_id=video_id,
            full_text="",
            language="",
            error="❌ This video is unavailable.",
        )
    except Exception as exc:
        logger.exception("Unexpected error fetching transcript for %s", video_id)
        return TranscriptResult(
            video_id=video_id,
            full_text="",
            language="",
            error=f"❌ Failed to fetch transcript: {exc}",
        )


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove artifacts."""
    text = re.sub(r"\[.*?\]", "", text)          # [Music], [Applause], etc.
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # limit newlines
    return text.strip()


def get_key_timestamps(segments: list[TranscriptSegment], interval_seconds: int = 300) -> list[tuple[str, str]]:
    """
    Extract representative timestamps at regular intervals.
    Returns list of (timestamp_str, text_snippet) tuples.
    """
    if not segments:
        return []

    timestamps = []
    next_mark = 0.0
    for seg in segments:
        if seg.start >= next_mark:
            timestamps.append((_format_timestamp(seg.start), seg.text[:80]))
            next_mark = seg.start + interval_seconds
    return timestamps
