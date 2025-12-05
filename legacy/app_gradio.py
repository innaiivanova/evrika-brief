# app_gradio.py

import gradio as gr

from evrika.rag_pipeline import (
    ingest_youtube,
    video_chat,
    generate_brief,
    save_brief_as_pdf,
)


# 1) Ingest video -------------------------------------------------------------

def gr_ingest(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "Please paste a YouTube URL."

    try:
        meta = ingest_youtube(url)
        return (
            "✅ Ingestion complete:\n"
            f"- Title: {meta.get('title')}\n"
            f"- YouTube ID: {meta.get('youtube_id')}\n"
            f"- Chunks stored: {meta.get('chunk_count')}"
        )
    except Exception as e:
        return f"❌ Error ingesting video: {e}"


# 2) Q&A ----------------------------------------------------------------------

def gr_qa(message: str, history):
    """
    Q&A handler for Chatbot.
    Uses the RAG function video_chat (Supabase + LLM).

    history is a list of [user, bot] pairs (default Chatbot format).
    """
    history = history or []

    question = (message or "").strip()
    if not question:
        answer = "Please ask a question about your ingested videos."
    else:
        # video_chat is a LangChain Tool; call it with a dict of arguments
        try:
            answer = video_chat.invoke({"question": question})
        except Exception as e:
            answer = f"❌ Error during Q&A: {e}"

    history = history + [[question, answer]]
    return "", history


# 3) Brief generation + PDF ---------------------------------------------------

def gr_generate_brief(hint: str) -> str:
    """
    Generate a 1-page brief for a given video/topic hint,
    using stored transcript chunks.
    """
    hint = (hint or "").strip()
    if not hint:
        return "Please provide a hint (e.g., video title or main topic)."

    try:
        brief_text = generate_brief.invoke({"video_hint": hint})
        return brief_text
    except Exception as e:
        return f"❌ Error generating brief: {e}"


def gr_export_pdf(brief_text: str):
    """
    Save the given brief_text as a PDF and return the path so Gradio
    can serve it as a downloadable file.
    """
    if not brief_text or not brief_text.strip():
        return None  # nothing to export

    filename = "evrika_brief.pdf"
    try:
        # save_brief_as_pdf is also a Tool; call with dict args
        save_brief_as_pdf.invoke(
            {
                "brief_text": brief_text,
                "filename": filename,
            }
        )
        return filename
    except Exception as e:
        print("Error saving PDF:", e)
        return None


# 4) Gradio UI ----------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Evrika Briefs – Video Q&A + 1-Pager")

    # --- Tab 1: Ingest -------------------------------------------------------
    with gr.Tab("1. Ingest Video"):
        gr.Markdown(
            "Paste a YouTube URL, then click **Ingest**. "
            "The transcript will be stored in Supabase with embeddings."
        )
        url_in = gr.Textbox(label="YouTube URL")
        ingest_btn = gr.Button("Ingest")
        ingest_out = gr.Textbox(label="Ingestion log", lines=6)

        ingest_btn.click(gr_ingest, inputs=url_in, outputs=ingest_out)

    # --- Tab 2: Ask Questions (RAG) -----------------------------------------
    with gr.Tab("2. Ask Questions (RAG)"):
        gr.Markdown(
            "Ask questions about any ingested videos. "
            "Answers come from the stored transcripts (Supabase + embeddings)."
        )
        chatbot = gr.Chatbot(
            label="Ask about your videos",
            height=400,
        )
        qa_box = gr.Textbox(
            label="Your question",
            placeholder="Example: What are the main ideas of the last video?",
        )
        qa_btn = gr.Button("Ask")

        qa_box.submit(gr_qa, [qa_box, chatbot], [qa_box, chatbot])
        qa_btn.click(gr_qa, [qa_box, chatbot], [qa_box, chatbot])

    # --- Tab 3: Brief + PDF --------------------------------------------------
    with gr.Tab("3. Generate Brief & PDF"):
        gr.Markdown(
            "Generate a structured 1-page Evrika brief from the stored content, "
            "then save it as a PDF."
        )
        brief_hint = gr.Textbox(
            label="Video / topic hint",
            placeholder="Example: the title of the video or a key phrase",
        )
        brief_btn = gr.Button("Generate brief")

        brief_text = gr.Textbox(
            label="Generated Evrika Brief",
            lines=25,
        )

        pdf_btn = gr.Button("Save brief as PDF")
        pdf_file = gr.File(label="Download brief PDF")

        brief_btn.click(gr_generate_brief, inputs=brief_hint, outputs=brief_text)
        pdf_btn.click(gr_export_pdf, inputs=brief_text, outputs=pdf_file)


if __name__ == "__main__":
    demo.launch()
