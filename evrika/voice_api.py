# evrika/voice_api.py
"""
FastAPI endpoints for Evrika voice & text Q&A.

Run locally with:
    uvicorn evrika.voice_api:app --reload --port 8000
"""

import base64
import re

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .audio_utils import transcribe_question_bytes, synthesize_answer_tts
from .rag_pipeline import (
    answer_question_text,
    ingest_youtube,
    generate_brief_text,
    save_brief_as_pdf,
)

app = FastAPI(
    title="Evrika Voice API",
    description="Text and voice endpoints for Evrika Briefs Q&A",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS CONFIG
# ---------------------------------------------------------------------------

# Lovable sends a custom header: ngrok-skip-browser-warning
# We allow it explicitly and support OPTIONS preflight.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: restrict to your Lovable domain
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "ngrok-skip-browser-warning",
        "Authorization",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Requested-With",
    ],
)

# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# HELPERS FOR YOUTUBE URL + INGESTION
# ---------------------------------------------------------------------------

# YouTube URL detector: youtube.com/watch?v=... or youtu.be/...
YOUTUBE_URL_RE = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w\-]+|youtu\.be/[\w\-]+)[^\s]*)",
    re.IGNORECASE,
)


def _parse_ingestion_url_from_prompt(prompt: str) -> str | None:
    """
    Decide whether this prompt should trigger ingestion.

    Rule:
    - ONLY ingest if the *last non-empty line* starts with "Fetch this video"
      (case-insensitive).
    - Then, extract a YouTube URL from anywhere in the prompt.
    """
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if not lines:
        return None

    last_line = lines[-1].lower()
    if not last_line.startswith("fetch this video"):
        # Treat as normal chat, even if the prompt contains a YouTube URL.
        return None

    # We *know* this is an ingestion request. Now find the URL anywhere.

    # Try [Video URL: ...] pattern first (Lovable style)
    m_bracket = re.search(r"\[Video URL:\s*(.+?)\s*\]", prompt, re.IGNORECASE)
    if m_bracket:
        return m_bracket.group(1).strip()

    # Fallback: any YouTube URL in the whole text
    m_url = YOUTUBE_URL_RE.search(prompt)
    if m_url:
        return m_url.group(1).strip()

    return None


def _extract_video_hint_url(prompt: str) -> str | None:
    """
    Extract a YouTube URL from the prompt for *question answering*.

    Unlike the ingestion parser, this does NOT require "Fetch this video".
    Any prompt that includes a YouTube URL will use that as video_hint.
    """
    # Prefer [Video URL: ...] if present
    m_bracket = re.search(r"\[Video URL:\s*(.+?)\s*\]", prompt, re.IGNORECASE)
    if m_bracket:
        return m_bracket.group(1).strip()

    # Otherwise, first YouTube URL we find
    m_url = YOUTUBE_URL_RE.search(prompt)
    if m_url:
        return m_url.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# TEXT ENDPOINT (INGESTION + CHAT)
# ---------------------------------------------------------------------------


@app.post("/text-query")
async def text_query(
    question: str = Form(...),
    video_hint: str = Form("", description="Optional YouTube URL or ID"),
):
    """
    Single text endpoint used by Lovable for BOTH:

    - ingestion messages like:
        Fetch this video: https://www.youtube.com/watch?v=...
    - normal chat questions.

    For normal chat, if `video_hint` is provided by the frontend, we use it
    directly to restrict RAG to that video. Otherwise we try to infer a URL
    from the text.
    """
    print(f"[TEXT] Incoming question: {repr(question)}, video_hint={video_hint!r}")

    # 1) Check if this is an ingestion command (based on the text)
    ingest_url = _parse_ingestion_url_from_prompt(question)

    if ingest_url:
        # ---- Ingestion path ------------------------------------------------
        print(f"[INGEST] Detected ingestion request for URL: {ingest_url}")
        try:
            meta = ingest_youtube(ingest_url)
            title = meta.get("title") or "this video"
            youtube_id = meta.get("youtube_id", "")
            chunk_count = meta.get("chunk_count", 0)

            print(
                f"[INGEST] Done: title={title!r}, id={youtube_id}, chunks={chunk_count}"
            )

            # User-facing message for Lovable
            answer = "Video ingested into Evrika Briefs."
        except Exception as e:
            print(f"[INGEST][ERROR] {e}")
            answer = f"Failed to ingest video '{ingest_url}': {e}"
    else:
        # ---- Normal chat / Q&A --------------------------------------------
        # Prefer explicit video_hint from Lovable, fallback to URL in text.
        video_hint_url = video_hint or _extract_video_hint_url(question)
        if video_hint_url:
            print(f"[QA] Using video_hint URL: {video_hint_url}")
        else:
            print("[QA] No video_hint URL found; searching across all videos.")

        try:
            answer = answer_question_text(question, video_hint=video_hint_url or "")
        except Exception as e:
            print(f"[QA][ERROR] {e}")
            answer = f"Error while answering your question: {e}"

    return {"question": question, "answer": answer}


# ---------------------------------------------------------------------------
# GENERATE BRIEF ENDPOINT (FORM-BASED, OPTIONAL)
# ---------------------------------------------------------------------------


@app.post("/generate-brief")
async def generate_brief_endpoint(
    video_hint: str = Form(..., description="YouTube URL or ID"),
):
    """
    Generate a 1-page Evrika Brief in your Markdown template
    for a given video URL/ID. (Form-based; used by some tools.)
    """
    print(f"[BRIEF] Generating brief for: {video_hint!r}")
    brief_md = generate_brief_text(video_hint)
    return {
        "video_hint": video_hint,
        "brief_markdown": brief_md,
    }


# ---------------------------------------------------------------------------
# BRIEF + PDF ENDPOINTS FOR LOVABLE (JSON-BASED)
# ---------------------------------------------------------------------------


class BriefRequest(BaseModel):
    video_hint: str  # YouTube URL or ID


class BriefResponse(BaseModel):
    brief_markdown: str


class BriefPdfRequest(BaseModel):
    brief_markdown: str


@app.post("/brief", response_model=BriefResponse)
async def create_brief(req: BriefRequest) -> BriefResponse:
    """
    Step 1: Generate a Markdown Evrika Brief from a YouTube URL/ID.

    The frontend should show this text and allow the user to edit it
    before turning it into a PDF.
    """
    print(f"[BRIEF] JSON brief request for: {req.video_hint!r}")
    brief_md = generate_brief_text(req.video_hint)
    return BriefResponse(brief_markdown=brief_md)


@app.post("/brief/pdf")
async def create_brief_pdf(req: BriefPdfRequest):
    """
    Step 2: Take the (possibly edited) Markdown brief from the user
    and return a generated PDF.
    """
    print("[BRIEF] JSON PDF request")
    filename = "evrika_brief.pdf"
    save_brief_as_pdf(req.brief_markdown, filename)
    return FileResponse(
        filename,
        media_type="application/pdf",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# VOICE ENDPOINT
# ---------------------------------------------------------------------------


@app.post("/voice-query")
async def voice_query(
    file: UploadFile = File(...),
    video_hint: str = Form("", description="Optional YouTube URL or ID"),
):
    """
    Voice endpoint for Evrika:

    1. Receives an audio file from the browser.
    2. Transcribes it to a text question.
    3. Runs RAG to get an answer.
    4. Tries to synthesize TTS audio for the answer (best-effort).
    5. Returns text + optional base64-encoded audio.
    """
    # 1. Read raw audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        print(f"[VOICE][ERROR] reading file: {e}")
        return {
            "error": f"Error reading file: {e}",
            "question": None,
            "answer": None,
            "audio_base64": None,
            "audio_mime": None,
        }

    # 2. STT – user speech -> text question
    try:
        question_text = transcribe_question_bytes(audio_bytes)
        print(f"[VOICE] Transcribed question: {question_text!r}, video_hint={video_hint!r}")
    except Exception as e:
        print(f"[VOICE][STT ERROR] {e}")
        return {
            "error": f"STT error: {e}",
            "question": None,
            "answer": None,
            "audio_base64": None,
            "audio_mime": None,
        }

    # 3. RAG – get answer; if frontend passes video_hint, we restrict to that video
    try:
        answer_text = answer_question_text(question_text, video_hint=video_hint or "")
    except Exception as e:
        print(f"[VOICE][QA ERROR] {e}")
        return {
            "error": f"QA error: {e}",
            "question": question_text,
            "answer": None,
            "audio_base64": None,
            "audio_mime": None,
        }

    # 4. TTS – answer -> audio (best-effort; failure should NOT 500)
    audio_base64: str | None = None
    audio_mime: str | None = None
    try:
        answer_audio_bytes, answer_mime_type = synthesize_answer_tts(answer_text)
        audio_base64 = base64.b64encode(answer_audio_bytes).decode("utf-8")
        audio_mime = answer_mime_type
    except Exception as e:
        # Just log; we still return a valid JSON with text answer
        print(f"[VOICE][TTS ERROR] {e}")

    # 5. Return everything
    return {
        "question": question_text,
        "answer": answer_text,
        "audio_base64": audio_base64,
        "audio_mime": audio_mime,
    }
