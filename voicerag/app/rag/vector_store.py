import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings


class VectorStore:
    def __init__(self, persist_dir: str = None):
        persist_path = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_path, exist_ok=True)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_path)

        # ChromaDB ONNX-based MiniLM embeddings (fast, lightweight, runs locally without external dependencies)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Get or create collection for knowledge base with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name="voice_rag_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], conversation_id: str = "default") -> int:
        """Store document chunks in ChromaDB with conversation-scoped metadata and unique IDs in batches."""
        if not chunks:
            return 0

        batch_size = 64
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            documents = [c["text"] for c in batch]
            metadatas = []
            ids = []

            for idx, c in enumerate(batch):
                meta = dict(c["metadata"])
                meta["conversation_id"] = conversation_id
                metadatas.append(meta)

                src = meta.get("source", "document")
                page = meta.get("page", 1)
                global_idx = i + idx
                ids.append(f"{conversation_id}_{src}_p{page}_c{global_idx}")

            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            total_indexed += len(batch)

        return total_indexed

    def _is_summary_query(self, query_text: str) -> bool:
        """Detect generic overview/summary queries that ask about the entire document."""
        if not query_text:
            return False
        q = query_text.lower().strip()
        summary_triggers = [
            "what is in this", "what is in the", "what is this document", "what is this pdf",
            "what's in this", "what's in the", "summarize", "summary", "overview",
            "about this document", "about the document", "about this file", "about the file",
            "about this pdf", "about the pdf", "tell me about this document", "tell me about this file",
            "explain this document", "explain the document", "what does this document",
            "what does this file", "contents of", "what is the document about", "what are the documents about",
            "main points", "key takeaways", "give me an overview"
        ]
        return any(trigger in q for trigger in summary_triggers)

    def get_overview_chunks(
        self,
        conversation_id: Optional[str] = None,
        max_chunks: int = 4
    ) -> List[Dict[str, Any]]:
        """Fetch representative/introductory chunks for high-level document overviews."""
        try:
            data = self.collection.get()
            if not data or not data.get("documents"):
                return []

            matched = []
            for doc, meta, id_ in zip(data["documents"], data["metadatas"], data["ids"]):
                if conversation_id and meta.get("conversation_id") != conversation_id:
                    continue
                matched.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": 0.1,
                    "chunk_id": meta.get("chunk_id", 0)
                })

            if not matched:
                return []

            # Sort by chunk_id so beginning/introductory chunks come first
            matched.sort(key=lambda x: x.get("chunk_id", 0))
            return matched[:max_chunks]
        except Exception:
            return []

    def _find_exact_identifier_chunks(
        self,
        query_text: str,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find exact chapter/law/section/rule chunks (e.g. 'law 17', 'law17', 'chapter 3')."""
        import re
        q = query_text.lower()
        m = re.search(r'\b(law|rule|chapter|section|part|article|clause|module|unit|step|lesson|act|policy)\s*#?\s*(\d+|[ivxlcdm]+)\b', q)
        if not m:
            return []

        term, num = m.group(1), m.group(2)
        try:
            data = self.collection.get()
            if not data or not data.get("documents"):
                return []

            matches = []
            for doc, meta, id_ in zip(data["documents"], data["metadatas"], data["ids"]):
                if conversation_id and meta.get("conversation_id") != conversation_id:
                    continue
                d_clean = doc.strip()
                d_lower = d_clean.lower()
                score = 0
                if d_clean.startswith(f"{num}\n") or d_lower.startswith(f"{term} {num}\n") or d_lower.startswith(f"{term} {num}:") or d_lower.startswith(f"{term} {num} "):
                    score = 100
                elif f"\n{num}\n" in doc or f"\n{term} {num}\n" in d_lower or f"\n{term} {num}:" in d_lower:
                    score = 80
                elif f"{term} {num}" in d_lower:
                    score = 40

                if score > 0:
                    matches.append((score, {
                        "text": doc,
                        "metadata": meta,
                        "distance": 0.05,
                        "chunk_id": meta.get("chunk_id", 0)
                    }))

            matches.sort(key=lambda x: x[0], reverse=True)
            # If we found an actual chapter start (score 100), return ONLY score 100 chapter chunks and exclude index lists
            chapter_chunks = [m[1] for m in matches if m[0] == 100]
            if chapter_chunks:
                return chapter_chunks[:4]

            high_scores = [m[1] for m in matches if m[0] >= 80]
            if high_scores:
                return high_scores[:4]
            return [m[1] for m in matches[:4]]
        except Exception:
            return []

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        max_distance: float = 0.85,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a user query.
        Uses Hybrid Exact-Identifier Matching + Semantic Vector Search.
        """
        count = self.collection.count()
        if count == 0:
            return []

        import re
        # Normalize concatenated phrases like 'law17' -> 'law 17', 'module2' -> 'module 2'
        clean_query = re.sub(r'\b(law|rule|chapter|section|part|article|clause|module|unit|step|lesson|act|policy)(\d+|[ivxlcdm]+)\b', r'\1 \2', query_text, flags=re.IGNORECASE)

        # 1. Check for exact identifier matches (e.g. Law 17, Chapter 3)
        exact_chunks = self._find_exact_identifier_chunks(clean_query, conversation_id)
        if not exact_chunks and conversation_id:
            exact_chunks = self._find_exact_identifier_chunks(clean_query, None)

        # 2. If it is an overview/summary query and no specific law/chapter was asked, fetch overview
        if not exact_chunks and self._is_summary_query(clean_query):
            overview = self.get_overview_chunks(conversation_id, max_chunks=top_k)
            if overview:
                return overview

        # 3. Semantic search with robust matching
        where_filter = None
        if conversation_id:
            where_filter = {"conversation_id": conversation_id}

        actual_k = min(top_k, count)
        results = None
        try:
            results = self.collection.query(
                query_texts=[clean_query],
                n_results=actual_k,
                where=where_filter
            )
        except Exception:
            pass

        # If scoped search yielded 0 docs, try without where_filter
        if not results or not results.get("documents") or not results["documents"] or not results["documents"][0]:
            try:
                results = self.collection.query(
                    query_texts=[clean_query],
                    n_results=actual_k
                )
            except Exception:
                results = None

        retrieved = []
        all_matches = []
        if results and results.get("documents") and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                item = {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                }
                all_matches.append(item)
                if dist is None or dist <= 1.25:
                    retrieved.append(item)

        if not retrieved and all_matches:
            retrieved = all_matches[:top_k]

        # Filter out Table of Contents / Index chunks unless user explicitly asked for contents
        user_wants_toc = any(w in clean_query.lower() for w in ["contents", "table of contents", "index", "chapters list"])
        if not user_wants_toc and retrieved:
            clean_content_chunks = [c for c in retrieved if not self._is_table_of_contents(c.get("text", ""))]
            if clean_content_chunks:
                retrieved = clean_content_chunks

        # If exact chapter/law/section matches are found, prioritize them exclusively
        if exact_chunks:
            return exact_chunks[:top_k]

        # 4. If standard search found nothing, fallback to overview chunks
        if not retrieved:
            overview = self.get_overview_chunks(conversation_id, max_chunks=top_k)
            if overview:
                return overview

        return retrieved[:top_k]

    def _is_table_of_contents(self, text: str) -> bool:
        """Detect whether a chunk is merely a structural Table of Contents or Index list."""
        if not text:
            return False
        import re
        t = text.strip()
        if re.search(r'^\s*(?:table\s+of\s+)?contents\b', t, re.IGNORECASE):
            return True
        if re.search(r'^\s*(?:index|subject\s+index|author\s+index)\b', t, re.IGNORECASE):
            return True
        if len(re.findall(r'chapter\s*[-—–:]?\s*\d+', t, re.IGNORECASE)) >= 3:
            return True
        if len(re.findall(r'[\.·\-_]{3,}\s*\d+', t)) >= 3:
            return True
        return False

    def delete_document(self, source_name: str, conversation_id: Optional[str] = None) -> int:
        """Delete all chunks for a specific document from ChromaDB (optionally scoped to a conversation)."""
        try:
            data = self.collection.get()
            if not data or not data["ids"]:
                return 0

            matching_ids = []
            for id_, meta in zip(data["ids"], data["metadatas"]):
                if meta.get("source") == source_name:
                    if conversation_id is None or meta.get("conversation_id") == conversation_id:
                        matching_ids.append(id_)

            if matching_ids:
                self.collection.delete(ids=matching_ids)
                return len(matching_ids)
            return 0
        except Exception:
            return 0

    def delete_conversation_vectors(self, conversation_id: str) -> int:
        """Delete all vectors belonging to a conversation."""
        try:
            data = self.collection.get()
            if not data or not data["ids"]:
                return 0

            matching_ids = [
                id_ for id_, meta in zip(data["ids"], data["metadatas"])
                if meta.get("conversation_id") == conversation_id
            ]
            if matching_ids:
                self.collection.delete(ids=matching_ids)
                return len(matching_ids)
            return 0
        except Exception:
            return 0

    def clear_all(self) -> bool:
        """Clear all indexed documents and reset collection."""
        try:
            self.client.delete_collection("voice_rag_knowledge")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="voice_rag_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        return True

    def get_indexed_documents(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of distinct documents indexed in the vector store (optionally scoped to a conversation)."""
        try:
            data = self.collection.get()
            if not data or not data["metadatas"]:
                return []

            source_counts = {}
            for meta in data["metadatas"]:
                if conversation_id and meta.get("conversation_id") != conversation_id:
                    continue
                src = meta.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            return [
                {"name": src, "chunks": f"{cnt} Chunks"}
                for src, cnt in sorted(source_counts.items())
            ]
        except Exception:
            return []