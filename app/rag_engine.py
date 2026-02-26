"""
RAG Engine
──────────
Retrieval-Augmented Generation for grounded Q&A.
Uses Ollama (Mistral) for local inference — zero API cost.
Ensures answers come ONLY from the video transcript — no hallucination.
"""

import logging
from dataclasses import dataclass
import ollama
from django.conf import settings

from app.embedding_service import search_similar
from app.session_manager import UserSession

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Language names for prompt instruction
# ──────────────────────────────────────────────

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
}


@dataclass
class RAGAnswer:
    """A grounded answer with confidence and source chunks."""
    answer: str
    confidence: float          # 0.0 – 1.0
    confidence_pct: int        # 0 – 100
    source_chunks: list[str]
    is_grounded: bool          # False if topic not covered


def _get_client() -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_BASE_URL)


RAG_SYSTEM_PROMPT = """You are a precise research assistant answering questions about a YouTube video.
{lang_instruction}

RULES:
1. Answer ONLY from the provided transcript context below.
2. If the answer is not found in the context, respond EXACTLY with:
   "🚫 This topic is not covered in the video."
3. Be specific and cite details from the transcript.
4. Keep answers concise but comprehensive (3-6 sentences).
5. If partially covered, answer what you can and note what's missing.

TRANSCRIPT CONTEXT:
{context}

CONVERSATION HISTORY:
{history}"""


def answer_question(question: str, session: UserSession) -> RAGAnswer:
    """
    Answer a question using RAG over the video transcript.
    Uses Ollama (Mistral) for local inference. Generates directly in user's language.

    Flow:
        1. Embed the question via nomic-embed-text
        2. Retrieve top-K similar chunks from FAISS
        3. Pass ONLY retrieved context to Mistral
        4. Compute confidence from similarity scores
    """
    if not session.has_video() or session.faiss_index is None:
        return RAGAnswer(
            answer="⚠️ Please send a YouTube link first so I can analyze the video.",
            confidence=0.0,
            confidence_pct=0,
            source_chunks=[],
            is_grounded=False,
        )

    # Retrieve relevant chunks
    results = search_similar(
        query=question,
        index=session.faiss_index,
        chunks=session.chunks,
        top_k=settings.RAG_TOP_K,
    )

    if not results:
        return RAGAnswer(
            answer="🚫 Could not find relevant content in the video transcript.",
            confidence=0.0,
            confidence_pct=0,
            source_chunks=[],
            is_grounded=False,
        )

    # Build context from retrieved chunks
    chunks_text = []
    scores = []
    for chunk, score in results:
        chunks_text.append(chunk)
        scores.append(score)

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Relevance: {scores[i]:.2f}]\n{c}"
        for i, c in enumerate(chunks_text)
    )

    # Build conversation history (last 6 messages)
    history_text = "No previous conversation."
    if session.history:
        recent = session.history[-6:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
        )

    # Language instruction
    lang_instruction = ""
    if session.language != "en":
        lang_name = LANGUAGE_NAMES.get(session.language, session.language)
        lang_instruction = f"\n⚠️ IMPORTANT: Generate your ENTIRE response in {lang_name}.\n"

    # LLM call via Ollama with strict grounding
    client = _get_client()
    system_prompt = RAG_SYSTEM_PROMPT.format(
        context=context,
        history=history_text,
        lang_instruction=lang_instruction,
    )

    response = client.chat(
        model=settings.OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 1024,
        },
    )
    answer_text = response.message.content.strip()

    # Calculate confidence from similarity scores
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    # Weighted: 60% max score, 40% average
    confidence = min((0.6 * max_score + 0.4 * avg_score), 1.0)
    confidence = max(confidence, 0.0)
    confidence_pct = int(confidence * 100)

    # Check if LLM said it wasn't covered
    not_covered_phrases = [
        "not covered in the video",
        "not found in the context",
        "not mentioned in the transcript",
        "not discussed in the video",
    ]
    is_grounded = not any(phrase in answer_text.lower() for phrase in not_covered_phrases)

    # Update session history
    session.add_to_history("user", question)
    session.add_to_history("assistant", answer_text)

    logger.info(
        "RAG answer via Ollama: confidence=%d%%, grounded=%s, chunks=%d",
        confidence_pct, is_grounded, len(chunks_text),
    )

    return RAGAnswer(
        answer=answer_text,
        confidence=confidence,
        confidence_pct=confidence_pct,
        source_chunks=chunks_text,
        is_grounded=is_grounded,
    )
