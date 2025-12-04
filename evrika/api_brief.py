# api_brief.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from evrika.rag_pipeline import generate_brief_text, save_brief_as_pdf


app = FastAPI(
    title="Evrika Briefs API (Brief + PDF only)",
    version="0.1.0",
)

# Allow browser (Lovable) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BriefRequest(BaseModel):
    video_hint: str  # YouTube URL or ID


class BriefResponse(BaseModel):
    brief_markdown: str


class BriefPdfRequest(BaseModel):
    brief_markdown: str


@app.post("/brief", response_model=BriefResponse)
def create_brief(req: BriefRequest) -> BriefResponse:
    """
    Step 1: Generate a Markdown Evrika Brief from a YouTube URL/ID.
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
