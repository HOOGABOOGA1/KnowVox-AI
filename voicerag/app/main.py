import os
import shutil
import uuid
import base64
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings, base_dir
from app.core.conversation_manager import ConversationManager
from app.rag.document_loader import load_document_with_metadata
from app.rag.text_splitter import chunk_documents
from app.rag.vector_store import VectorStore
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.rag_service import RAGService
from app.models.schemas import (
    Message,
    QueryRequest,
    QueryResponse,
    IngestResponse,
    CreateConversationRequest,
    RenameConversationRequest,
    UploadDocRequest,
    OpenDocRequest,
    SettingsRequest,
)

app = FastAPI(
    title="VoiceRAG Assistant",
    description="Voice-Enabled Grounded Knowledge Retrieval System with Scoped Conversations",
    version="1.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
DATA_DIR = base_dir / "data"
AUDIO_DIR = DATA_DIR / "audio"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Mount audio directory to serve voice responses
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Initialize services
conversation_manager = ConversationManager()
vector_store = VectorStore(persist_dir=str(CHROMA_DIR))
stt_service = STTService()
tts_service = TTSService()
rag_service = RAGService()


# -------------------------------------------------------------
# ROOT & HEALTH CHECK
# -------------------------------------------------------------

@app.get("/")
def home(request: Request):
    """Serves the web UI."""
    ui_index = UI_DIR / "index.html"
    if ui_index.exists():
        return FileResponse(str(ui_index), media_type="text/html")
    return {"message": "VoiceRAG Assistant API is online", "status": "ready"}

@app.get("/logo.jpeg")
def get_logo():
    return FileResponse(str(UI_DIR / "logo.jpeg"))

@app.get("/logo.ico")
def get_ico():
    return FileResponse(str(UI_DIR / "logo.ico"))

@app.get("/knowvox_logo.jpeg")
def get_knowvox_logo():
    return FileResponse(str(UI_DIR / "knowvox_logo.jpeg"))


@app.get("/health")
def health():
    return {"status": "healthy"}


# -------------------------------------------------------------
# CONVERSATION MANAGEMENT
# -------------------------------------------------------------

@app.get("/api/conversations")
def list_conversations():
    """Returns all conversations ordered by recency."""
    convs = conversation_manager.list_conversations()
    if not convs:
        new_conv = conversation_manager.create_conversation("New Chat")
        return [new_conv]
    return convs


@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest = CreateConversationRequest()):
    """Creates a new conversation session."""
    return conversation_manager.create_conversation(req.title or "New Chat")


@app.get("/api/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str):
    """Returns messages and attached documents for a specific conversation."""
    try:
        messages = conversation_manager.get_messages(conversation_id)
        documents = conversation_manager.get_documents(conversation_id)
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "messages": messages,
            "documents": documents
        }
    except Exception as e:
        return {
            "status": "error",
            "conversation_id": conversation_id,
            "messages": [],
            "documents": [],
            "error": str(e)
        }


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deletes a conversation, its messages, and its scoped vectors from ChromaDB."""
    try:
        vector_store.delete_conversation_vectors(conversation_id)
        conversation_manager.delete_conversation(conversation_id)
        return {"status": "success", "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@app.post("/api/conversations/{conversation_id}/rename")
@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameConversationRequest):
    """Renames a conversation."""
    try:
        conversation_manager.rename_conversation(conversation_id, req.title)
        return {"status": "success", "conversation_id": conversation_id, "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename conversation: {str(e)}")


# -------------------------------------------------------------
# DOCUMENT MANAGEMENT & SCOPED INGESTION
# -------------------------------------------------------------

@app.get("/api/documents")
def get_documents(conversation_id: Optional[str] = Query(default="default")):
    """Returns documents attached to a conversation."""
    conv_id = conversation_id or "default"
    docs = conversation_manager.get_documents(conv_id)
    return [
        {
            "name": d.get("original_name") or d.get("filename"),
            "chunks": f"{d.get('chunks_count', 0)} Chunks"
        }
        for d in docs
    ]


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(default="default")
):
    """Upload a document, chunk it, and index it into ChromaDB scoped to a conversation."""
    conv_id = conversation_id or "default"
    clean_filename = os.path.basename(file.filename)
    saved_filename = f"{conv_id}_{clean_filename}"
    file_path = str(UPLOADS_DIR / saved_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        documents = load_document_with_metadata(file_path)
        if not documents:
            raise HTTPException(status_code=400, detail=f"No text could be extracted from {clean_filename}.")

        for d in documents:
            d["metadata"]["source"] = clean_filename

        chunks = chunk_documents(documents)
        indexed_count = vector_store.add_chunks(chunks, conversation_id=conv_id)
        conversation_manager.add_document(conv_id, saved_filename, clean_filename, indexed_count)

        return IngestResponse(
            status="success",
            filename=clean_filename,
            chunks_indexed=indexed_count,
            message=f"Successfully indexed {indexed_count} chunks from {clean_filename}."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/documents/upload")
async def upload_document_base64(req: UploadDocRequest):
    """Uploads and indexes a document via Base64 payload scoped to a conversation."""
    try:
        conv_id = req.conversation_id or "default"
        clean_filename = os.path.basename(req.filename)
        saved_filename = f"{conv_id}_{clean_filename}"
        target_path = UPLOADS_DIR / saved_filename

        file_bytes = base64.b64decode(req.base64_data)
        with open(target_path, "wb") as f:
            f.write(file_bytes)

        docs = load_document_with_metadata(str(target_path))
        if not docs:
            return {
                "status": "error",
                "message": f"No text could be extracted from {clean_filename}",
                "chunks_indexed": 0
            }

        for d in docs:
            d["metadata"]["source"] = clean_filename

        chunks = chunk_documents(docs)
        indexed_count = vector_store.add_chunks(chunks, conversation_id=conv_id)
        conversation_manager.add_document(conv_id, saved_filename, clean_filename, indexed_count)

        return {
            "status": "success",
            "filename": clean_filename,
            "chunks_indexed": indexed_count,
            "conversation_id": conv_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "chunks_indexed": 0}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str, conversation_id: Optional[str] = Query(default="default")):
    """Deletes a document and its vectors from a conversation."""
    try:
        conv_id = conversation_id or "default"
        clean_filename = os.path.basename(filename)

        # 1. Delete vectors
        vector_store.delete_document(clean_filename, conversation_id=conv_id)
        # 2. Delete from database
        conversation_manager.delete_document(conv_id, clean_filename)
        # 3. Delete file on disk
        candidates = [UPLOADS_DIR / f"{conv_id}_{clean_filename}", UPLOADS_DIR / clean_filename]
        for fpath in candidates:
            if fpath.exists():
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        return {"status": "success", "message": f"Deleted {clean_filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/documents")
def clear_all_documents(conversation_id: Optional[str] = Query(default="default")):
    """Clears all documents for a specific conversation."""
    try:
        conv_id = conversation_id or "default"
        vector_store.delete_conversation_vectors(conv_id)
        conversation_manager.clear_documents(conv_id)
        return {"status": "success", "message": "Documents cleared for this conversation."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/documents/open")
def open_document_page(req: OpenDocRequest):
    """Opens the source document directly at a specific page on the host machine."""
    try:
        clean_filename = os.path.basename(req.filename).strip()
        conv_id = req.conversation_id or "default"

        candidate_paths = [
            UPLOADS_DIR / f"{conv_id}_{clean_filename}",
            UPLOADS_DIR / clean_filename,
            DATA_DIR / clean_filename,
        ]

        target_path = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                target_path = p
                break

        if not target_path and UPLOADS_DIR.exists():
            for p in UPLOADS_DIR.glob(f"*{clean_filename}*"):
                if p.is_file():
                    target_path = p
                    break

        if not target_path or not target_path.exists():
            return {"status": "error", "message": f"Document '{clean_filename}' not found."}

        page_num = max(1, int(req.page or 1))
        resolved_uri = target_path.resolve().as_uri() + f"#page={page_num}"

        opened = False
        try:
            subprocess.Popen(f'start msedge "{resolved_uri}"', shell=True)
            opened = True
        except Exception:
            pass

        if not opened:
            try:
                webbrowser.open(resolved_uri)
                opened = True
            except Exception:
                pass

        if not opened:
            try:
                os.startfile(str(target_path))
            except Exception:
                pass

        return {"status": "success", "file": str(target_path), "page": page_num}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------
# QUERY & RAG GENERATION (TEXT & VOICE)
# -------------------------------------------------------------

@app.post("/api/query-text", response_model=QueryResponse)
async def query_text(request: QueryRequest):
    """Text-in RAG query with scoped retrieval and audio synthesis."""
    if not request.query or not request.query.strip():
        return QueryResponse(
            query="",
            answer="Please enter a question.",
            sources=[],
            audio_url=None,
            audio_base64=None,
            conversation_id=request.conversation_id
        )

    conv_id = request.conversation_id or "default"

    # 1. Save user query into conversation history
    try:
        conversation_manager.add_message(conv_id, "user", request.query.strip())
    except Exception:
        pass

    # 2. Retrieve scoped chunks from ChromaDB
    chunks = []
    try:
        attached_docs = conversation_manager.get_documents(conv_id)
        if attached_docs:
            chunks = vector_store.query(request.query, top_k=request.top_k or 4, conversation_id=conv_id)
    except Exception:
        chunks = []

    # 3. Grounded Generation with Career OS Profile
    history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    try:
        profile_res = get_career_profile()
        user_profile = profile_res.get("profile") if profile_res.get("status") == "success" else None
        result = rag_service.generate_answer(
            query=request.query,
            context_chunks=chunks,
            conversation_history=history_dicts,
            user_profile=user_profile
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
    except Exception as e:
        answer = f"Could not generate answer: {str(e)}"
        sources = []

    # 4. Synthesize voice audio
    audio_filename = None
    audio_base64 = None
    try:
        if request.voice:
            tts_service.voice = request.voice
        audio_path = await tts_service.text_to_speech(answer, output_dir=str(AUDIO_DIR))
        if audio_path and os.path.exists(audio_path):
            audio_filename = Path(audio_path).name
            with open(audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"TTS voice error: {e}")

    # 5. Save assistant response into database
    try:
        conversation_manager.add_message(conv_id, "assistant", answer, sources, audio_base64)
    except Exception:
        pass

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        audio_url=f"/audio/{audio_filename}" if audio_filename else None,
        audio_base64=audio_base64,
        conversation_id=conv_id
    )


@app.post("/api/query-voice", response_model=QueryResponse)
async def query_voice(
    audio_file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(default="default"),
    voice: Optional[str] = Form(default="en-US-JennyNeural")
):
    """Voice-in, Voice-out RAG query (STT -> Grounded RAG -> TTS)."""
    conv_id = conversation_id or "default"
    temp_audio_path = str(AUDIO_DIR / f"input_{uuid.uuid4().hex}_{audio_file.filename}")
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # 1. Speech to Text
        transcribed_text = stt_service.transcribe(temp_audio_path)
        if not transcribed_text or "error" in transcribed_text.lower() or transcribed_text.strip() == "":
            return QueryResponse(
                query="",
                answer="I could not hear any speech clearly. Please speak again.",
                sources=[],
                audio_url=None,
                audio_base64=None,
                conversation_id=conv_id
            )

        # 2. Save user message
        try:
            conversation_manager.add_message(conv_id, "user", transcribed_text.strip())
        except Exception:
            pass

        # 3. Retrieve scoped context & generate answer
        chunks = []
        try:
            attached_docs = conversation_manager.get_documents(conv_id)
            if attached_docs:
                chunks = vector_store.query(transcribed_text, top_k=4, conversation_id=conv_id)
        except Exception:
            chunks = []

        profile_res = get_career_profile()
        user_profile = profile_res.get("profile") if profile_res.get("status") == "success" else None
        result = rag_service.generate_answer(query=transcribed_text, context_chunks=chunks, user_profile=user_profile)
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # 4. Text to Speech
        audio_filename = None
        audio_base64 = None
        try:
            if voice:
                tts_service.voice = voice
            audio_path = await tts_service.text_to_speech(answer, output_dir=str(AUDIO_DIR))
            if audio_path and os.path.exists(audio_path):
                audio_filename = Path(audio_path).name
                with open(audio_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"TTS error in query_voice: {e}")

        # 5. Save assistant message
        try:
            conversation_manager.add_message(conv_id, "assistant", answer, sources, audio_base64)
        except Exception:
            pass

        return QueryResponse(
            query=transcribed_text,
            answer=answer,
            sources=sources,
            audio_url=f"/audio/{audio_filename}" if audio_filename else None,
            audio_base64=audio_base64,
            conversation_id=conv_id
        )
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


@app.post("/api/synthesize")
async def synthesize_speech_api(req: Dict[str, Any]):
    """Dynamic voice synthesis on demand with custom voice accent."""
    text = req.get("text", "")
    voice = req.get("voice", "en-US-JennyNeural")
    try:
        audio_path = await tts_service.text_to_speech(text, output_dir=str(AUDIO_DIR), voice=voice)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                return {
                    "status": "success",
                    "audio_base64": base64.b64encode(f.read()).decode("utf-8"),
                    "voice": voice
                }
    except Exception as e:
        return {"status": "error", "message": str(e), "audio_base64": None}
    return {"status": "error", "message": "Synthesis failed", "audio_base64": None}


# -------------------------------------------------------------
# SETTINGS & CONFIGURATION
# -------------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    """Returns current configuration status."""
    key = settings.GEMINI_API_KEY
    masked_key = (key[:6] + "..." + key[-4:]) if len(key) > 10 else ("Configured" if key else "")
    return {
        "has_key": bool(key),
        "masked_key": masked_key,
        "model": settings.DEFAULT_MODEL,
        "models_list": [
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.6-flash",
            "gemini-3.7-flash"
        ]
    }


@app.post("/api/settings")
def save_settings(req: SettingsRequest):
    """Save new Gemini API key and model selection."""
    try:
        if req.api_key and req.api_key.strip():
            clean_key = req.api_key.strip()
            settings.update_api_key(clean_key)
            rag_service.api_key = clean_key
            stt_service.api_key = clean_key
        if req.model and req.model.strip():
            clean_model = req.model.strip()
            settings.update_model(clean_model)
            rag_service.model = clean_model

        return {"status": "success", "message": "Settings updated successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------
# CAREER OS PROFILE ENDPOINTS
# -------------------------------------------------------------

@app.get("/api/career-profile")
def get_career_profile():
    """Fetches the active user profile from career_os.db."""
    import sqlite3
    candidate_paths = [
        base_dir.parent / "career_os.db",
        base_dir / "career_os.db",
        DATA_DIR / "career_os.db"
    ]
    db_path = None
    for p in candidate_paths:
        if p.exists():
            db_path = p
            break

    if not db_path:
        return {"status": "empty", "profile": None}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            return {"status": "success", "profile": dict(row)}
        return {"status": "empty", "profile": None}
    except Exception as e:
        return {"status": "error", "message": str(e), "profile": None}


@app.post("/api/career-profile")
def save_career_profile(profile_data: Dict[str, Any]):
    """Saves or updates user profile in career_os.db."""
    import sqlite3
    candidate_paths = [
        base_dir.parent / "career_os.db",
        base_dir / "career_os.db",
        DATA_DIR / "career_os.db"
    ]
    db_path = candidate_paths[0]
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                education TEXT,
                skills TEXT,
                interests TEXT,
                career_goal TEXT,
                experience TEXT,
                location TEXT
            )
        """)
        conn.execute("""
            INSERT INTO users (name, education, skills, interests, career_goal, experience, location)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            profile_data.get("name", "User"),
            profile_data.get("education", ""),
            profile_data.get("skills", ""),
            profile_data.get("interests", ""),
            profile_data.get("career_goal", ""),
            profile_data.get("experience", ""),
            profile_data.get("location", "")
        ))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Career OS Profile updated successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}