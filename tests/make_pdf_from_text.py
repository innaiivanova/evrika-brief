# make_pdf_from_text.py
from pathlib import Path

from evrika.rag_pipeline import save_brief_as_pdf


def main():
    # 1. Read the text from brief.md
    brief_path = Path("brief.md")
    if not brief_path.exists():
        print("brief.md not found. Create a file named 'brief.md' next to this script.")
        return

    brief_text = brief_path.read_text(encoding="utf-8")

    # 2. Generate the PDF
    output_name = "evrika_brief.pdf"
    save_brief_as_pdf(brief_text, output_name)

    print(f"✅ PDF generated: {output_name}")


if __name__ == "__main__":
    main()
