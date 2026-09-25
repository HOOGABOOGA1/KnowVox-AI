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


def extract_text_from_image(file_path: str) -> str:
    """Extract text from PNG, JPG, JPEG, WEBP, BMP images via Windows native hardware OCR."""
    p = Path(file_path)
    try:
        from PIL import Image
        import winocr
        import asyncio
        import concurrent.futures

        img = Image.open(p)
        if img.mode not in ('RGB', 'RGBA', 'L'):
            img = img.convert('RGB')

        async def _do_ocr(image):
            res = await winocr.recognize_pil(image, 'en')
            return res.text.strip() if res else ''

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                text = pool.submit(lambda: asyncio.run(_do_ocr(img))).result()
        else:
            text = asyncio.run(_do_ocr(img))

        return text or f"[Image uploaded: {p.name} - Visual diagram or illustration with no embedded text]"
    except Exception as e:
        return f"[Image uploaded: {p.name}]"


def load_document_with_metadata(file_path: str) -> List[Dict[str, Any]]:
    """Extract text and metadata from PDF, TXT, MD, CSV, JSON, and Image files."""
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
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        img_text = extract_text_from_image(str(path))
        docs.append({
            "text": img_text,
            "metadata": {"source": path.name, "page": 1, "type": "image"}
        })
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        docs.append({
            "text": content,
            "metadata": {"source": path.name, "page": 1}
        })
    return docs