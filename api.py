# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from evrika.agent import agent_respond
from evrika.rag_pipeline import (
    ingest_youtube,
    generate_brief_text,
    save_brief_as_pdf,
)

app = FastAPI(
    title="Evrika Briefs API",
    version="0.1.0",
)

# Allow Lovable (browser) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; later restrict to your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    title: str
    youtube_id: str
    chunk_count: int


class BriefRequest(BaseModel):
    video_hint: str  # YouTube URL or ID


class BriefResponse(BaseModel):
    brief_markdown: str


class BriefPdfRequest(BaseModel):
    brief_markdown: str


# ---------- Endpoints ----------


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    General chat endpoint using the Evrika agent.
    """
    reply = agent_respond(req.message)
    return ChatResponse(reply=reply)


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Ingest a YouTube video into Supabase (if not already present).
    """
    meta = ingest_youtube(req.url)
    return IngestResponse(
        title=meta.get("title", ""),
        youtube_id=meta.get("youtube_id", ""),
        chunk_count=meta.get("chunk_count", 0),
    )


@app.post("/brief", response_model=BriefResponse)
def create_brief(req: BriefRequest) -> BriefResponse:
    """
    Step 1: Generate a Markdown Evrika Brief from a YouTube URL/ID.

    The frontend should show this text and allow the user to edit it
    before turning it into a PDF.
    """
    brief_md = generate_brief_text(req.video_hint)
    return BriefResponse(brief_markdown=brief_md)


@app.post("/brief/pdf")
def create_brief_pdf(req: BriefPdfRequest):
    """
    Step 2: Take the (possibly edited) Markdown brief from the user
    and return a generated PDF.
    """
    filename = "evrika_brief.pdf"
    save_brief_as_pdf(req.brief_markdown, filename)
    return FileResponse(
        filename,
        media_type="application/pdf",
        filename=filename,
    )
