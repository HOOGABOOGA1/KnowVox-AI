import os
import sqlite3
import json
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import base_dir


class ConversationManager:
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = base_dir / "data" / "conversations.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    audio_base64 TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_documents (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    chunks_count INTEGER DEFAULT 0,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def create_conversation(self, title: str = "New Chat") -> Dict[str, Any]:
        conv_id = f"conv_{uuid.uuid4().hex[:10]}"
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now)
            )
            conn.commit()
        return {
            "id": conv_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "doc_count": 0,
            "message_count": 0
        }

    def get_or_create_default(self) -> Dict[str, Any]:
        convs = self.list_conversations()
        if convs:
            return convs[0]
        return self.create_conversation("General Discussion")

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    c.id, 
                    c.title, 
                    c.created_at, 
                    c.updated_at,
                    (SELECT COUNT(*) FROM conversation_documents d WHERE d.conversation_id = c.id) as doc_count,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
                FROM conversations c
                ORDER BY c.updated_at DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title.strip(), now, conv_id)
            )
            conn.commit()
        return True

    def delete_conversation(self, conv_id: str) -> bool:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversation_documents WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
        return True

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        audio_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        now = datetime.datetime.utcnow().isoformat()
        sources_json = json.dumps(sources or [])

        with self._get_connection() as conn:
            # Ensure conversation exists
            cur = conn.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,))
            conv = cur.fetchone()
            if not conv:
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "New Chat", now, now)
                )
            else:
                # Auto-title conversation on first user query if still generic
                if role == "user" and conv["title"] in ["New Chat", "New Conversation", "General Discussion"]:
                    words = content.strip().split()
                    if words:
                        smart_title = " ".join(words[:5])
                        if len(smart_title) > 30:
                            smart_title = smart_title[:30] + "..."
                        conn.execute(
                            "UPDATE conversations SET title = ? WHERE id = ?",
                            (smart_title, conv_id)
                        )

                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conv_id)
                )

            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, sources, audio_base64, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conv_id, role, content, sources_json, audio_base64, now)
            )
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "audio_base64": audio_base64,
            "created_at": now
        }

    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conv_id,)
            )
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                try:
                    item["sources"] = json.loads(item.get("sources") or "[]")
                except Exception:
                    item["sources"] = []
                result.append(item)
            return result

    def add_document(
        self,
        conv_id: str,
        filename: str,
        original_name: str,
        chunks_count: int
    ) -> Dict[str, Any]:
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            # Check if document with same filename already in this conversation
            conn.execute(
                "DELETE FROM conversation_documents WHERE conversation_id = ? AND filename = ?",
                (conv_id, filename)
            )
            conn.execute(
                """
                INSERT INTO conversation_documents (id, conversation_id, filename, original_name, chunks_count, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, conv_id, filename, original_name, chunks_count, now)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id)
            )
            conn.commit()

        return {
            "id": doc_id,
            "conversation_id": conv_id,
            "filename": filename,
            "original_name": original_name,
            "chunks_count": chunks_count,
            "uploaded_at": now
        }

    def get_documents(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM conversation_documents WHERE conversation_id = ? ORDER BY uploaded_at ASC",
                (conv_id,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, conv_id: str, filename: str) -> bool:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM conversation_documents WHERE conversation_id = ? AND (filename = ? OR original_name = ?)",
                (conv_id, filename, filename)
            )
            conn.commit()
        return True

    def clear_documents(self, conv_id: str) -> bool:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM conversation_documents WHERE conversation_id = ?",
                (conv_id,)
            )
            conn.commit()
        return True
