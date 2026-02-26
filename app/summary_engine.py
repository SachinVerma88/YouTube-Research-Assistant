"""
Summary Engine
──────────────
Structured, hierarchical summarization of YouTube transcripts.
Uses Ollama (Mistral) for local LLM inference — zero API cost.
Generates directly in the target language for efficiency.
"""

import logging
import ollama
from django.conf import settings

logger = logging.getLogger(__name__)


def _get_client() -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_BASE_URL)


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = """You are a research analyst. Summarize the following section of a YouTube video transcript.
Focus on key facts, insights, and actionable points. Be concise but comprehensive.

TRANSCRIPT SECTION:
{chunk}

SUMMARY (2-4 sentences):"""

FINAL_SUMMARY_PROMPT = """You are a senior research analyst creating an executive briefing from a YouTube video.
{lang_instruction}

Below are summaries of different sections of the video. Synthesize them into ONE structured report.

SECTION SUMMARIES:
{chunk_summaries}

VIDEO TIMESTAMPS (key moments):
{timestamps}

Generate the output in EXACTLY this format (keep the emojis and headers):

🎥 **Title**: (Infer the most fitting title from the content)

📌 **5 Key Insights**
1. (Specific, actionable insight)
2. (Specific, actionable insight)
3. (Specific, actionable insight)
4. (Specific, actionable insight)
5. (Specific, actionable insight)

⏱ **Important Timestamps**
(List 3-5 key moments with timestamps and brief descriptions)

🧠 **Core Takeaway**
(One powerful strategic insight in 2-3 sentences)

🎯 **Action Points**
1. (What to do based on this video)
2. (What to do based on this video)
3. (What to do based on this video)

👥 **Who Should Watch This**
(2-3 sentences on target audience)

💼 **Business Relevance**
(2-3 sentences on business/career implications)

⚠️ **Risk Factors Discussed**
(Any risks, warnings, or caveats mentioned — or "None explicitly discussed")

IMPORTANT: Be specific, not generic. Use actual details from the content."""


DEEPDIVE_PROMPT = """You are a strategic research analyst performing a deep analysis of a YouTube video.
{lang_instruction}

TRANSCRIPT SECTIONS:
{context}

Generate a strategic deep-dive analysis in this format:

🔍 **Strategic Deep Dive**

📊 **Market Analysis**
(What market dynamics or trends are discussed?)

⚔️ **Competitive Landscape**
(Any competitive insights or positioning mentioned?)

🚀 **Growth Opportunities**
(What opportunities are highlighted?)

⚠️ **Risk Assessment**
(What risks, challenges, or threats are discussed?)

🔮 **Future Predictions**
(Any predictions or forward-looking statements?)

💡 **Strategic Implications**
(What does this mean for decision-makers?)

Be specific and reference actual content. Do not make up information."""


ACTIONPOINTS_PROMPT = """You are an executive coach extracting actionable insights from a YouTube video.
{lang_instruction}

TRANSCRIPT SECTIONS:
{context}

Generate exactly 5 actionable insights in this format:

🎯 **5 Actionable Insights**

1️⃣ **[Short Title]**
   → What to do: (specific action)
   → Why it matters: (brief rationale)
   → Priority: 🔴 High / 🟡 Medium / 🟢 Low

2️⃣ **[Short Title]**
   → What to do: (specific action)
   → Why it matters: (brief rationale)
   → Priority: 🔴 High / 🟡 Medium / 🟢 Low

3️⃣ **[Short Title]**
   → What to do: (specific action)
   → Why it matters: (brief rationale)
   → Priority: 🔴 High / 🟡 Medium / 🟢 Low

4️⃣ **[Short Title]**
   → What to do: (specific action)
   → Why it matters: (brief rationale)
   → Priority: 🔴 High / 🟡 Medium / 🟢 Low

5️⃣ **[Short Title]**
   → What to do: (specific action)
   → Why it matters: (brief rationale)
   → Priority: 🔴 High / 🟡 Medium / 🟢 Low

Be specific and reference actual video content. Do not fabricate information."""


# ──────────────────────────────────────────────
# Language instruction builder
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


def _lang_instruction(lang: str) -> str:
    """Build a language instruction for the prompt. Generates directly in target language."""
    if lang == "en":
        return ""
    name = LANGUAGE_NAMES.get(lang, lang)
    return f"\n⚠️ IMPORTANT: Generate your ENTIRE response in {name}. Write all content in {name} language.\n"


# ──────────────────────────────────────────────
# LLM call via Ollama
# ──────────────────────────────────────────────

def _llm_call(prompt: str) -> str:
    """Make a single LLM call via Ollama (Mistral)."""
    client = _get_client()
    response = client.chat(
        model=settings.OLLAMA_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.3,
            "num_predict": 2048,
        },
    )
    return response.message.content.strip()


# ──────────────────────────────────────────────
# Summarization pipeline
# ──────────────────────────────────────────────

def summarize_chunks(chunks: list[str]) -> list[str]:
    """Summarize each chunk individually (stage 1)."""
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = _llm_call(
                CHUNK_SUMMARY_PROMPT.format(chunk=chunk),
            )
            summaries.append(summary)
        except Exception as exc:
            logger.warning("Failed to summarize chunk %d: %s", i, exc)
            summaries.append(chunk[:200] + "...")
    logger.info("Summarized %d chunks via Ollama", len(summaries))
    return summaries


def generate_summary(
    transcript_text: str,
    chunks: list[str],
    timestamps: list[tuple[str, str]] | None = None,
    language: str = "en",
) -> str:
    """
    Generate a full structured summary using hierarchical summarization.
    Generates directly in the target language — no separate translation step.

    Pipeline:
        1. Summarize each chunk → short summaries
        2. Combine chunk summaries + timestamps → structured output
    """
    lang_inst = _lang_instruction(language)

    # Stage 1: chunk summaries
    if len(chunks) <= 3:
        chunk_summaries_text = "\n\n".join(
            f"Section {i+1}:\n{c}" for i, c in enumerate(chunks)
        )
    else:
        chunk_summaries = summarize_chunks(chunks)
        chunk_summaries_text = "\n\n".join(
            f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )

    # Format timestamps
    ts_text = "Not available"
    if timestamps:
        ts_text = "\n".join(f"{ts} — {desc}" for ts, desc in timestamps)

    # Stage 2: final structured summary
    prompt = FINAL_SUMMARY_PROMPT.format(
        chunk_summaries=chunk_summaries_text,
        timestamps=ts_text,
        lang_instruction=lang_inst,
    )
    summary = _llm_call(prompt)
    logger.info("Generated structured summary (%d chars) via Ollama", len(summary))
    return summary


def generate_deepdive(chunks: list[str], language: str = "en") -> str:
    """Generate a strategic deep-dive analysis. Directly in target language."""
    lang_inst = _lang_instruction(language)
    context = "\n\n---\n\n".join(chunks[:10])
    prompt = DEEPDIVE_PROMPT.format(context=context, lang_instruction=lang_inst)
    return _llm_call(prompt)


def generate_actionpoints(chunks: list[str], language: str = "en") -> str:
    """Generate actionable insights from the video. Directly in target language."""
    lang_inst = _lang_instruction(language)
    context = "\n\n---\n\n".join(chunks[:10])
    prompt = ACTIONPOINTS_PROMPT.format(context=context, lang_instruction=lang_inst)
    return _llm_call(prompt)
