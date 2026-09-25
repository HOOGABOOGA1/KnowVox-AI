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


_blip_processor = None
_blip_model = None


def get_offline_image_caption(file_path: Any) -> str:
    """Generate an on-device local neural caption for photos/scenes using local BLIP weights."""
    global _blip_processor, _blip_model
    try:
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model_id = "Salesforce/blip-image-captioning-base"
        if _blip_processor is None or _blip_model is None:
            _blip_processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
            _blip_model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True)

        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        inputs = _blip_processor(img, return_tensors='pt')
        out = _blip_model.generate(**inputs, max_new_tokens=60)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True).strip()
        return caption
    except Exception:
        return ""


def extract_text_from_image(file_path: str) -> str:
    """Extract text from PNG, JPG, JPEG, WEBP, BMP images via Windows native hardware OCR & Local Neural Vision."""
    p = Path(file_path)
    ocr_text = ""
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
                ocr_text = pool.submit(lambda: asyncio.run(_do_ocr(img))).result()
        else:
            ocr_text = asyncio.run(_do_ocr(img))
    except Exception:
        ocr_text = ""

    caption = get_offline_image_caption(p)

    parts = []
    if caption:
        parts.append(f"[Visual Scene Analysis: {caption}. This photograph visually depicts {caption}.]")
    if ocr_text:
        parts.append(f"[Embedded Text / OCR Content:\n{ocr_text}]")
    elif not caption:
        parts.append(f"[Photo uploaded: {p.name} - Photograph or visual scene with no readable text]")

    return "\n\n".join(parts)


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