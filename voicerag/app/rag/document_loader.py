from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)