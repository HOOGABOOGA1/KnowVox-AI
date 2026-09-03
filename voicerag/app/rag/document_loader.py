from pathlib import Path
from typing import List, Dict, Any
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


def load_document_with_metadata(file_path: str) -> List[Dict[str, Any]]:
    """Extract text and metadata from PDF, TXT, MD, CSV, JSON files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    docs = []
    if ext == ".pdf":
        reader = PdfReader(path)
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                docs.append({
                    "text": page_text,
                    "metadata": {"source": path.name, "page": idx + 1}
                })
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        docs.append({
            "text": content,
            "metadata": {"source": path.name, "page": 1}
        })
    return docs