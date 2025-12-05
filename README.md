# Evrika Briefs - Final Project

# Goal of Project 

This is the final project at the Ironhack AI Engineering Bootcamp. 
The goal of the project is to build a multimodal CahtBot for YouTube Q&A. 
The AI chatbot should be developed as a RAG system. It should combine 
the power of text and audio processing to answer questions about YouTube videos. 
The bot should extract relevant information from the YouTube videos
in order to provide accurate answers to user queries.

# Overview

**Evrika Briefs** is an AI-powered chatbot for YouTube learning.  
Give it a YouTube URL and it will:

1. Ingest and transcribe the video.
2. Store the content in a vector database (Supabase + pgvector).
3. Let you **chat** with the video using a RAG pipeline.
4. Generate a **brief** with models and frameworks, top insights, and quotes.
6. Let you **edit** the brief.
5. Export the brief as a **downloadable PDF**.
7. Supports a **voice interface** through a FastAPI endpoint used by Lovable.

---

## Features

- **YouTube ingestion**
  - Accepts a YouTube URL.
  - Downloads audio via `yt-dlp`.
  - Transcribes with OpenAI Whisper.

- **Chunking & embeddings**
  - Splits transcript into semantically meaningful **chunks**.
  - Creates **embeddings** with `OpenAIEmbeddings`.
  - Stores chunks in **Supabase** with **pgvector**.

- **RAG Q&A Chat**
  - Retrieval using a Supabase remote procedure call RPC `match_documents`.
  - LangChain-based agent with short-term chat history.
  - Answers grounded in the video content (minimized hallucinations).

- **Brief Generation**
  - Structured **brief template** with:
    - Main Idea
    - Relevant Models & Frameworks
    - Top Insights
    - Memorable Quotes
  - PDF export for download.

- **Voice Interface**
  - FastAPI endpoint for speech-to-text + question answering.
  - Designed to be wired into a Lovable front-end with a microphone button.

- **Evaluation (RAGAS)**
  - RAGAS evaluation for:
    - faithfulness
    - answer relevancy
    - context precision

---

## Architecture

High-level flow:

1. **Ingest**
   - User pastes a YouTube URL in the front-end (Lovable).
   - Back-end fetches metadata and audio, runs transcription.
   - Transcript is chunked, embedded, and stored in Supabase.

2. **Query**
   - User asks a question in chat (text or voice).
   - System:
     - Uses `match_documents` RPC in Supabase to retrieve top-k chunks.
     - Optionally filters by `youtube_id`.
     - Sends chunks + question to the LLM via LangChain.

3. **Brief**
   - User clicks “Generate Brief”.
   - System uses the transcript + key chunks to produce a structured brief.
   - Brief is saved as text and exported as a PDF for download.

---

## Tech Stack

**Frontend**

- [Lovable](https://lovable.dev/) no-code UI (chat, buttons, status messages)

**Backend**

- Python 3.11+ / 3.12
- FastAPI (HTTP API)
- Uvicorn (ASGI server)
- LangChain (RAG & agent logic)
- OpenAI API (LLM + embeddings + Whisper / transcription)
- Supabase (Postgres + pgvector)
- `yt-dlp` (YouTube audio download)

---

## Project Structure

## Repository structure

```text
evrika-brief/
├── api.py                  # Main FastAPI HTTP API (chat, ingest, brief) used by Lovable
├── eval_ragas.py           # Script to evaluate the RAG pipeline with RAGAS
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md

├── evrika/                 # Core Evrika Briefs package
│   ├── __init__.py         # Package marker
│   ├── agent.py            # Conversational agent logic (tools, routing, chat history)
│   ├── api_brief.py        # Helper endpoints / functions for brief generation
│   ├── audio_utils.py      # STT/TTS utilities (Whisper + TTS)
│   ├── config.py           # Shared config: OpenAI, Supabase, text splitter, chat memory
│   ├── metadata_tool.py    # Metadata question-answering tool (title, speaker, length, ...)
│   ├── rag_pipeline.py     # Core RAG pipeline: ingest, semantic search, QA, brief, PDF
│   ├── supabase_store.py   # Supabase vector store + RPC helpers (match_documents, etc.)
│   ├── transcripts.py      # YouTube download, transcription, chunking helpers
│   └── voice_api.py        # FastAPI endpoints for voice Q&A (mic input + spoken answers)

├── evaluation/             # Evaluation scripts (RAGAS)
├── legacy/                 # Old experiments / prototypes kept for reference
├── presentation/           # Project presentation
└── tests/                  # PDF Tests 
```

## Setup and Running the Backend in 3 steps

```
# In the final version you only need 3 commands to start the backend:
# (requirements.txt and full_requirements.txt have been installed)
# (environment variables have been set)
# 1. create virtual environment
# 2. start uvicorn
# 3. start ngrok

# source .venv/bin/activate
# uvicorn evrika.voice_api:app --reload --port 8000
# ngrok http 8000
```

### Clone & create virtual environment

```
git clone <your-repo-url> evrika-brief
cd evrika-brief

python -m venv .venv
source .venv/bin/activate   

pip install -r requirements.txt
```

### Environment Variables

```
OPENAI_API_KEY=sk-...
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
SUPABASE_ANON_KEY=...     
OPENAI_MODEL=gpt-4o-mini  
EMBEDDING_MODEL=text-embedding-3-small
```

#### ngrok if you tunnel FastAPI

```
NGROK_AUTH_TOKEN=...
```

### Start the voice API

```
uvicorn evrika.voice_api:app --reload --port 8001
```

### Tunnel it with ngrok for Lovable

```
ngrok http 8000
```

## Evaluation

The retrieval-augmented generation (RAG) pipeline was evaluated on a small set of 10 test questions using RAGAS.

example ingested videos with average scores:

```
Faithfulness: ~0.98

Answer Relevancy: ~0.73

Context Precision: ~0.91
```

**Interpretation:**

- The model is highly faithful to the provided context.

- Context selection is generally good (relevant chunks).

- Answer relevancy leaves some room for improvement.

## License

```
MIT License
```

## Acknowledgements

- Ironhack AI Engineering Bootcamp for the project framework & support.

- OpenAI, LangChain, Supabase, YouTube and yt-dlp for the core building blocks.

- Lovable for making it easy to plug an API into an interactive front-end.
