# 🎥 AI YouTube Research Assistant for Telegram

A production-grade Telegram bot that analyzes YouTube videos using **RAG (Retrieval-Augmented Generation)** with **fully local AI inference via Ollama** — delivering structured summaries, grounded Q&A, strategic deep-dives, and multi-language support at **zero API cost**.

> This solution avoids paid APIs by using fully local LLM inference through Ollama, ensuring **zero marginal cost per user** and **improved data privacy**. All data stays on your machine.

---

## 🏗 Architecture

```
Telegram User
     │
     ▼
┌─────────────────────────────┐
│   Telegram Bot API          │
│   (Webhook → Django)        │
└──────────┬──────────────────┘
           │
     ┌─────▼─────┐
     │  Django    │
     │  Webhook   │
     │  View      │
     └─────┬─────┘
           │
     ┌─────▼──────────────────────────────────────┐
     │          Processing Pipeline                │
     │                                             │
     │  1. URL Validator (regex)                   │
     │  2. Transcript Fetcher (youtube-transcript) │
     │  3. Text Cleaner & Chunker (tiktoken)       │
     │  4. Embeddings (Ollama: nomic-embed-text)   │
     │  5. Vector Store (FAISS)                    │
     │  6. Summary Engine (Ollama: Mistral)        │
     │  7. Q&A Engine (RAG — no hallucination)     │
     │  8. Multi-Language (direct LLM generation)  │
     │                                             │
     └─────┬──────────────────────────────────────┘
           │
     ┌─────▼─────┐
     │  Response  │
     │  Formatter │
     │  (HTML)    │
     └───────────┘
```

---

## 🧾 Tech Stack (Zero Cost)

| Component | Tool |
|-----------|------|
| Telegram Bot | BotFather + python-telegram-bot |
| Backend | Django |
| LLM (Summary + Q&A) | Ollama → **Mistral** / Llama3 |
| Embeddings | Ollama → **nomic-embed-text** |
| Vector DB | FAISS (in-memory) |
| Transcript | youtube-transcript-api |
| Translation | Direct LLM generation (no API) |

**Total API cost: $0** — All AI inference runs locally on your machine.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 📝 **Structured Summaries** | 5 key insights, timestamps, takeaways, action points, audience, business relevance |
| ❓ **RAG Q&A** | Grounded answers from transcript only — zero hallucination |
| 🔍 **Deep Dive** | Strategic analysis: market, competitive landscape, risks, predictions |
| 🎯 **Action Points** | 5 prioritized actionable insights with rationale |
| 🌐 **Multi-Language** | English, Hindi, Kannada, Tamil, Telugu, Bengali, Marathi |
| ⚡ **Smart Caching** | Transcript + embeddings cached per video — instant re-analysis |
| 📊 **Confidence Score** | RAG answers show confidence % based on retrieval similarity |
| 💬 **Context-Aware** | Follow-up questions use conversation history |
| 👥 **Multi-User** | Independent sessions per Telegram user |
| 🔒 **Privacy** | All data processed locally — nothing sent to external APIs |

---

## 📋 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/summary` | Structured video summary |
| `/ask <question>` | Ask about the video (RAG) |
| `/deepdive` | Strategic deep-dive analysis |
| `/actionpoints` | 5 actionable insights |
| `/lang <language>` | Switch response language |
| `/reset` | Clear session and start fresh |
| `/help` | Show help |

---

## 🛠 Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed
- Telegram Bot Token (from [BotFather](https://t.me/BotFather))

### Step 1: Install Ollama & Pull Models

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Step 2: Setup Project

```bash
cd TelegramBot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add your TELEGRAM_BOT_TOKEN
```

### Step 3: Run

```bash
# Start Django server
python manage.py runserver 0.0.0.0:8000

# In another terminal — expose via ngrok
ngrok http 8000

# Set webhook (replace URL with your ngrok URL)
python manage.py setup_webhook --url https://YOUR-URL.ngrok.io/webhook/
```

---

## 🧠 Design Decisions

### Why Local LLM (Ollama) Instead of OpenAI API?

| Aspect | OpenAI API | Ollama (Local) |
|--------|-----------|----------------|
| Cost per query | $0.003–0.15 | **$0** |
| Data privacy | Sent to cloud | **Stays on machine** |
| Rate limits | Yes | **None** |
| Internet needed | Yes | **No** (after setup) |
| Latency | Network-dependent | **Local only** |

### Why RAG Instead of Full Context?

| Approach | Tokens Used | Accuracy |
|----------|------------|----------|
| Full transcript to LLM | 10,000–50,000 | ⚠️ May exceed context limits, may hallucinate |
| **RAG (top-5 chunks)** | **1,000–3,000** | **✅ Grounded, fits local LLM context** |

RAG is **essential for local LLMs** because models like Mistral have smaller context windows. By retrieving only the most relevant chunks, we ensure accurate answers within token limits.

### Direct-in-Language Generation

Instead of generating in English → translating:
```
❌ Generate in English → Google Translate → Hindi
✅ Prompt: "Generate your entire response in Hindi"
```

This is **cleaner, faster, and cheaper** (no translation API needed). Mistral and Llama3 handle Hindi, Kannada, and other Indian languages natively.

### Hierarchical Summarization for Long Videos

Local LLMs have limited context windows. Our approach:
1. Summarize each chunk individually (~800 tokens)
2. Combine chunk summaries into final structured output

This enables summarizing **2+ hour videos** without token overflow.

---

## 📁 Project Structure

```
TelegramBot/
├── manage.py                     # Django management
├── requirements.txt              # Dependencies (8 packages)
├── .env.example                  # Environment template
├── README.md                     # This file
│
├── youtube_bot/                  # Django project config
│   ├── settings.py               # Settings + Ollama config
│   ├── urls.py                   # URL routing
│   ├── wsgi.py / asgi.py         # Entry points
│
└── app/                          # Main application
    ├── telegram_handler.py       # Webhook + command handlers
    ├── transcript_service.py     # URL validation + transcript fetching
    ├── embedding_service.py      # Chunking + FAISS + Ollama embeddings
    ├── summary_engine.py         # Hierarchical summarization (Ollama)
    ├── rag_engine.py             # RAG Q&A engine (Ollama)
    ├── language_service.py       # Multi-language (LLM-native)
    ├── session_manager.py        # Per-user session state
    ├── formatter.py              # Telegram response formatting
    └── management/commands/
        └── setup_webhook.py      # Webhook management command
```

---

## 📈 Scaling Strategy

| Current (Demo) | Production |
|----------------|------------|
| In-memory sessions | Redis for session store |
| FAISS in-memory | FAISS on disk / ChromaDB |
| Single process | Gunicorn + workers |
| SQLite | PostgreSQL |
| ngrok | Cloud deployment |
| Ollama local | GPU cloud instance + Ollama |

---

## 🔮 Future Improvements

- [ ] Whisper fallback for videos without captions
- [ ] Multi-video comparison
- [ ] PDF export of summaries
- [ ] Inline keyboard buttons
- [ ] User preferences persistence
- [ ] Admin monitoring dashboard
- [ ] Model switching via bot command (/model llama3)
