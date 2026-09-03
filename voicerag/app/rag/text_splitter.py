from typing import List, Dict, Any


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 100, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = ""
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = [s for s in text.split(separator) if s] if separator else list(text)
        current_doc = []
        total_len = 0

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_doc else 0
            if total_len + piece_len + sep_len > self.chunk_size:
                if total_len > 0:
                    doc_text = separator.join(current_doc)
                    if doc_text.strip():
                        final_chunks.append(doc_text.strip())
                    while current_doc and total_len > self.chunk_overlap:
                        total_len -= len(current_doc.pop(0)) + (len(separator) if current_doc else 0)
                if piece_len > self.chunk_size and new_separators:
                    sub_chunks = self._split(piece, new_separators)
                    final_chunks.extend(sub_chunks)
                    current_doc = []
                    total_len = 0
                else:
                    current_doc.append(piece)
                    total_len += piece_len
            else:
                current_doc.append(piece)
                total_len += piece_len + sep_len

        if current_doc:
            doc_text = separator.join(current_doc)
            if doc_text.strip():
                final_chunks.append(doc_text.strip())

        return final_chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 900,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """Split documents into overlapping chunks with source & page metadata."""
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for i, text in enumerate(split_texts):
            chunks.append({
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": i
                }
            })
    return chunks