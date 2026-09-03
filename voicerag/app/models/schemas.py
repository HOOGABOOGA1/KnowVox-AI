from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = "default"
    conversation_history: Optional[List[Dict[str, str]]] = []
    voice: Optional[str] = "en-IN-NeerjaNeural"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    audio_base64: Optional[str] = None
    conversation_id: Optional[str] = "default"

class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    conversation_id: Optional[str] = "default"

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"

class RenameConversationRequest(BaseModel):
    title: str

class UploadDocRequest(BaseModel):
    filename: str
    base64_data: str
    conversation_id: Optional[str] = "default"

class OpenDocRequest(BaseModel):
    filename: str
    page: Optional[int] = 1
    conversation_id: Optional[str] = "default"

class SettingsRequest(BaseModel):
    api_key: Optional[str] = ""
    model: Optional[str] = "gemini-3.1-flash-lite"
