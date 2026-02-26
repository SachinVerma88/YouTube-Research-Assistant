"""
Embedding Service
─────────────────
Text chunking, local embeddings via Ollama (nomic-embed-text), and FAISS vector store.
Zero API cost — all inference runs locally.
"""

import logging
import hashlib
from typing import Any

import numpy as np
import tiktoken
import faiss
import ollama
from django.conf import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Embedding cache (video_id → pre-built data)
# ──────────────────────────────────────────────
_cache: dict[str, dict[str, Any]] = {}


def _get_client() -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_BASE_URL)


def _get_encoder():
    """Get tiktoken encoder for token counting."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Split text into chunks of ~max_tokens with overlap.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
    overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS
    encoder = _get_encoder()

    # Split into sentences first
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoder.encode(sentence))

        # If a single sentence exceeds max, force-split it
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Hard split by tokens
            tokens = encoder.encode(sentence)
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i : i + max_tokens]
                chunks.append(encoder.decode(chunk_tokens))
            continue

        if current_tokens + sentence_tokens > max_tokens:
            # Flush current chunk
            chunks.append(" ".join(current_chunk))

            # Overlap: carry last few sentences
            overlap_chunk: list[str] = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tok = len(encoder.encode(s))
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_tok

            current_chunk = overlap_chunk + [sentence]
            current_tokens = sum(len(encoder.encode(s)) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info("Split text into %d chunks (max %d tokens)", len(chunks), max_tokens)
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────
# Embeddings (Local via Ollama — nomic-embed-text)
# ──────────────────────────────────────────────

def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts via Ollama (nomic-embed-text).
    Processes in batches to handle long documents.
    """
    client = _get_client()
    all_embeddings = []

    for text in texts:
        response = client.embed(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            input=text,
        )
        all_embeddings.append(response.embeddings[0])

    embeddings = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    logger.info("Generated %d embeddings (dim=%d) via Ollama", len(texts), embeddings.shape[1])
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string via Ollama."""
    client = _get_client()
    response = client.embed(
        model=settings.OLLAMA_EMBEDDING_MODEL,
        input=query,
    )
    embedding = np.array(response.embeddings[0], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(embedding)
    return embedding


# ──────────────────────────────────────────────
# FAISS Index
# ──────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index for inner-product (cosine) search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)
    return index


def search_similar(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Search for the most similar chunks to a query.
    Returns list of (chunk_text, similarity_score) tuples.
    """
    top_k = min(top_k or settings.RAG_TOP_K, len(chunks))
    query_emb = embed_query(query)
    scores, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(scores[0][i])))
    return results


# ──────────────────────────────────────────────
# Cache-aware pipeline
# ──────────────────────────────────────────────

def _cache_key(video_id: str) -> str:
    return hashlib.md5(video_id.encode()).hexdigest()


def process_and_cache(video_id: str, text: str) -> tuple[list[str], np.ndarray, faiss.IndexFlatIP]:
    """
    Chunk, embed, and index a transcript. Caches by video_id for instant re-use.
    """
    key = _cache_key(video_id)
    if key in _cache:
        logger.info("Cache hit for video %s", video_id)
        cached = _cache[key]
        return cached["chunks"], cached["embeddings"], cached["index"]

    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)

    _cache[key] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
    }
    logger.info("Cached embeddings for video %s (%d chunks)", video_id, len(chunks))
    return chunks, embeddings, index


def is_cached(video_id: str) -> bool:
    return _cache_key(video_id) in _cache
