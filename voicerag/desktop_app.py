import asyncio
import base64
import os
import sys
import wave
from pathlib import Path
import webview
import sounddevice as sd
import numpy as np

# Add voicerag to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings, base_dir
from app.core.conversation_manager import ConversationManager
from app.rag.document_loader import load_document_with_metadata
from app.rag.text_splitter import chunk_documents
from app.rag.vector_store import VectorStore
from app.services.rag_service import RAGService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService


class DesktopBridgeAPI:
    def __init__(self):
        self.base_dir = base_dir
        self.data_dir = self.base_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.audio_dir = self.data_dir / "audio"
        self.chroma_dir = self.data_dir / "chroma_db"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.cm = ConversationManager()
        self.vs = VectorStore(persist_dir=str(self.chroma_dir))
        self.rag = RAGService()
        self.stt = STTService()
        self.tts = TTSService()

    # ==========================================
    # Conversation Management
    # ==========================================
    def list_conversations(self):
        """Returns all chat conversations ordered by recency."""
        try:
            convs = self.cm.list_conversations()
            if not convs:
                new_conv = self.cm.create_conversation("New Chat")
                return [new_conv]
            return convs
        except Exception as e:
            return [{"id": "default", "title": "Main Chat", "doc_count": 0, "message_count": 0}]

    def create_conversation(self, title: str = "New Chat"):
        """Creates a new empty chat session."""
        try:
            return self.cm.create_conversation(title or "New Chat")
        except Exception as e:
            return {"id": "conv_error", "title": "New Chat", "doc_count": 0, "message_count": 0}

    def get_conversation_history(self, conversation_id: str):
        """Returns full chat history and attached documents for a conversation."""
        try:
            messages = self.cm.get_messages(conversation_id)
            documents = self.cm.get_documents(conversation_id)
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

    def delete_conversation(self, conversation_id: str):
        """Deletes a conversation, its messages, and its scoped vectors from ChromaDB."""
        try:
            # 1. Delete vectors from ChromaDB
            self.vs.delete_conversation_vectors(conversation_id)
            # 2. Delete from SQLite
            self.cm.delete_conversation(conversation_id)
            return {"status": "success", "conversation_id": conversation_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def rename_conversation(self, conversation_id: str, new_title: str):
        """Renames a conversation."""
        try:
            self.cm.rename_conversation(conversation_id, new_title)
            return {"status": "success", "conversation_id": conversation_id, "title": new_title}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ==========================================
    # Query & RAG (Scoped per Conversation)
    # ==========================================
    def query_text(
        self,
        query: str,
        conversation_id: str = None,
        history: list = None,
        voice: str = "en-US-JennyNeural"
    ):
        """Process a text query scoped strictly to the active conversation."""
        if not query or not query.strip():
            return {"answer": "Please enter a question.", "sources": [], "audio_base64": None}

        conv_id = conversation_id or "default"

        # 1. Save user query into conversation history
        try:
            self.cm.add_message(conv_id, "user", query.strip())
        except Exception:
            pass

        # 2. Semantic search scoped strictly to this conversation's documents
        chunks = []
        q_lower = query.lower().strip()
        pure_profile_kw = [
            "who am i", "my name", "my profile", "profile", "my goal", "my skill", "my skills",
            "about me", "my career", "career info", "career goal", "my experience", "my education",
            "what is my goal", "what are my skills", "what should i do", "what work should i",
            "my roadmap", "guide me", "my resume", "my path", "my background", "my interest", "my interests"
        ]
        is_profile_focused = any(pk in q_lower for pk in pure_profile_kw) and not any(dk in q_lower for dk in ["pdf", "document", "file", "doc", "notes", "text", "paper", "attached", "uploaded"])

        try:
            attached_docs = self.cm.get_documents(conv_id)
            if attached_docs and not is_profile_focused:
                chunks = self.vs.query(query, top_k=4, conversation_id=conv_id)
        except Exception:
            chunks = []

        # 3. Grounded generation with Career OS Profile
        try:
            profile_res = self.get_career_profile()
            user_profile = profile_res.get("profile") if profile_res.get("status") == "success" else None
            result = self.rag.generate_answer(query, chunks, history or [], user_profile=user_profile)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"Could not generate answer: {str(e)}"
            sources = []

        # 4. Text to Speech
        audio_base64 = None
        try:
            target_voice = voice or "en-US-JennyNeural"
            self.tts.voice = target_voice
            audio_path = asyncio.run(self.tts.text_to_speech(answer, output_dir=str(self.audio_dir), voice=target_voice))
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"TTS synthesis handled: {e}")

        # 5. Save assistant response into database
        try:
            self.cm.add_message(conv_id, "assistant", answer, sources, audio_base64)
        except Exception:
            pass

        return {
            "conversation_id": conv_id,
            "answer": answer,
            "sources": sources,
            "audio_base64": audio_base64
        }

    def record_and_answer(
        self,
        duration: int = 5,
        conversation_id: str = None,
        history: list = None,
        voice: str = "en-US-JennyNeural"
    ):
        """Records from microphone, transcribes, runs RAG on conversation docs, and synthesizes audio."""
        mic_audio_path = str(self.audio_dir / "desktop_mic_input.wav")
        sample_rate = 16000
        dur = max(2, min(int(duration), 15))
        conv_id = conversation_id or "default"

        try:
            # 1. Record live microphone with AGC
            raw_rec = sd.rec(int(dur * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()

            rec_float = raw_rec.astype(np.float32)
            max_val = np.max(np.abs(rec_float))
            if max_val > 15.0:
                target_peak = 22000.0
                gain = min(5.0, target_peak / max_val)
                audio_data = np.clip(rec_float * gain, -32767, 32767).astype(np.int16)
            else:
                audio_data = raw_rec

            with wave.open(mic_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            # 2. Transcribe voice
            transcription = self.stt.transcribe(mic_audio_path)
            if not transcription or "could not transcribe" in transcription.lower() or "error" in transcription.lower() or transcription.strip() == "":
                return {
                    "conversation_id": conv_id,
                    "transcription": "",
                    "answer": "I could not hear any speech clearly. Please check your microphone and speak again.",
                    "sources": [],
                    "audio_base64": None
                }

            # 3. Save user message
            try:
                self.cm.add_message(conv_id, "user", transcription.strip())
            except Exception:
                pass

            # 4. Retrieve scoped chunks & generate answer
            chunks = []
            try:
                attached_docs = self.cm.get_documents(conv_id)
                if attached_docs:
                    chunks = self.vs.query(transcription, top_k=4, conversation_id=conv_id)
            except Exception:
                chunks = []

            profile_res = self.get_career_profile()
            user_profile = profile_res.get("profile") if profile_res.get("status") == "success" else None
            result = self.rag.generate_answer(transcription, chunks, history or [], user_profile=user_profile)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # 5. Synthesize voice
            audio_base64 = None
            try:
                target_voice = voice or "en-US-JennyNeural"
                self.tts.voice = target_voice
                audio_path = asyncio.run(self.tts.text_to_speech(answer, output_dir=str(self.audio_dir), voice=target_voice))
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"TTS voice error in record_and_answer (handled): {e}")

            # 6. Save assistant message
            try:
                self.cm.add_message(conv_id, "assistant", answer, sources, audio_base64)
            except Exception:
                pass

            return {
                "conversation_id": conv_id,
                "transcription": transcription,
                "answer": answer,
                "sources": sources,
                "audio_base64": audio_base64
            }
        except Exception as e:
            return {
                "conversation_id": conv_id,
                "transcription": "",
                "answer": f"Audio processing error: {str(e)}",
                "sources": [],
                "audio_base64": None
            }

    def synthesize_speech(self, text: str, voice: str = "en-US-JennyNeural"):
        """Synthesizes text on demand with the requested voice accent and returns Base64 audio."""
        try:
            target_voice = voice or "en-US-JennyNeural"
            audio_path = asyncio.run(self.tts.text_to_speech(text, output_dir=str(self.audio_dir), voice=target_voice))
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    return {
                        "status": "success",
                        "audio_base64": base64.b64encode(f.read()).decode("utf-8"),
                        "voice": target_voice
                    }
        except Exception as e:
            return {"status": "error", "message": str(e), "audio_base64": None}
        return {"status": "error", "message": "Failed to generate audio", "audio_base64": None}

    # ==========================================
    # Live Voice-to-Voice Hands-Free Engine
    # ==========================================
    def record_live_speech(self):
        """
        Phase 1 of Live Call:
        Listens with sensitive adaptive VAD, captures speech with a pre-buffer,
        and saves to live_turn_input.wav.
        Returns immediately once user finishes speaking so UI can switch to 'thinking'.
        """
        import time
        time.sleep(0.15)

        sample_rate = 16000
        block_duration = 0.08  # 80ms blocks
        block_samples = int(sample_rate * block_duration)
        max_duration = 16.0
        max_silence_blocks = int(1.1 / block_duration)  # ~1.1s silence ends turn
        min_speech_blocks = 3   # at least 240ms of active speech

        pre_buffer = []  # Ring buffer to catch first consonant
        recorded_chunks = []
        speech_detected = False
        silence_count = 0
        total_blocks = 0
        max_blocks = int(max_duration / block_duration)

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                # 100ms calibration
                calib, _ = stream.read(int(sample_rate * 0.10))
                calib_rms = np.sqrt(np.mean(calib.astype(np.float64)**2)) if len(calib) > 0 else 15.0
                # Highly sensitive threshold so normal & quiet speech is caught reliably
                threshold = max(28.0, min(180.0, float(calib_rms * 1.25)))

                while total_blocks < max_blocks:
                    block, overflowed = stream.read(block_samples)
                    total_blocks += 1
                    rms = np.sqrt(np.mean(block.astype(np.float64)**2))

                    if not speech_detected:
                        pre_buffer.append(block)
                        if len(pre_buffer) > 4:
                            pre_buffer.pop(0)

                        if rms > threshold:
                            speech_detected = True
                            recorded_chunks.extend(pre_buffer)
                            recorded_chunks.append(block)
                            silence_count = 0
                        elif total_blocks > int(4.5 / block_duration):
                            # Silence timeout (user said nothing)
                            return {"status": "timeout"}
                    else:
                        recorded_chunks.append(block)
                        if rms > threshold * 0.85:
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= max_silence_blocks:
                                break
        except Exception as e:
            return {"status": "error", "message": f"Mic error: {e}"}

        if not speech_detected or len(recorded_chunks) < min_speech_blocks:
            return {"status": "silence"}

        # Save to WAV with Automatic Gain Control (AGC) so quiet speech is normalized
        raw_audio = np.concatenate(recorded_chunks, axis=0).astype(np.float32)
        max_val = np.max(np.abs(raw_audio))
        if max_val > 15.0:
            target_peak = 22000.0  # ~68% max int16
            gain = min(5.0, target_peak / max_val)
            audio_data = np.clip(raw_audio * gain, -32767, 32767).astype(np.int16)
        else:
            audio_data = raw_audio.astype(np.int16)

        mic_audio_path = str(self.audio_dir / "live_turn_input.wav")
        with wave.open(mic_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return {"status": "speech_detected", "wav_path": mic_audio_path}

    def process_live_speech(
        self,
        conversation_id: str = None,
        history: list = None,
        voice: str = "en-US-JennyNeural"
    ):
        """
        Phase 2 of Live Call:
        Transcribes recorded WAV via on-device Faster-Whisper / STT, runs RAG, and synthesizes audio.
        """
        conv_id = conversation_id or "default"
        mic_audio_path = str(self.audio_dir / "live_turn_input.wav")
        if not os.path.exists(mic_audio_path):
            return {"status": "error", "message": "No audio file recorded"}

        # 1. Transcribe speech
        transcription = self.stt.transcribe(mic_audio_path)
        if not transcription or "could not transcribe" in transcription.lower() or "error" in transcription.lower() or transcription.strip() == "":
            return {
                "status": "unclear",
                "conversation_id": conv_id,
                "transcription": "",
                "answer": "",
                "sources": [],
                "audio_base64": None
            }

        # 2. Process query & generate voice
        res = self.query_text(
            query=transcription.strip(),
            conversation_id=conv_id,
            history=history,
            voice=voice
        )
        res["status"] = "success"
        res["transcription"] = transcription.strip()
        return res

    def live_voice_turn(
        self,
        conversation_id: str = None,
        history: list = None,
        voice: str = "en-US-JennyNeural"
    ):
        """Combined turn for backward compatibility."""
        rec = self.record_live_speech()
        if rec.get("status") == "speech_detected":
            return self.process_live_speech(conversation_id, history, voice)
        return rec
    # ==========================================
    def upload_document(self, filename: str, base64_data: str, conversation_id: str = None):
        """Uploads and indexes a document strictly into the given conversation."""
        try:
            conv_id = conversation_id or "default"
            file_bytes = base64.b64decode(base64_data)
            
            # Save file on disk
            clean_filename = os.path.basename(filename)
            saved_filename = f"{conv_id}_{clean_filename}"
            target_path = self.uploads_dir / saved_filename
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            # Ingest & index into ChromaDB with conversation scope
            docs = load_document_with_metadata(str(target_path))
            if not docs:
                return {
                    "status": "error",
                    "message": f"No text could be extracted from {clean_filename}",
                    "chunks_indexed": 0
                }

            # Stamp original source filename in metadata
            for d in docs:
                d["metadata"]["source"] = clean_filename

            chunks = chunk_documents(docs)
            indexed_count = self.vs.add_chunks(chunks, conversation_id=conv_id)

            # Save in database
            self.cm.add_document(conv_id, saved_filename, clean_filename, indexed_count)

            return {
                "status": "success",
                "filename": clean_filename,
                "chunks_indexed": indexed_count,
                "conversation_id": conv_id
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "chunks_indexed": 0}

    def delete_document(self, filename: str, conversation_id: str = None):
        """Delete a document from a specific conversation and disk."""
        try:
            conv_id = conversation_id or "default"
            clean_filename = os.path.basename(filename)
            
            # 1. Delete vectors
            self.vs.delete_document(clean_filename, conversation_id=conv_id)
            
            # 2. Delete from database
            self.cm.delete_document(conv_id, clean_filename)

            # 3. Delete file from disk
            candidate_names = [f"{conv_id}_{clean_filename}", clean_filename]
            for cname in candidate_names:
                fpath = self.uploads_dir / cname
                if fpath.exists():
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

            return {"status": "success", "message": f"Deleted {clean_filename}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def clear_all_documents(self, conversation_id: str = None):
        """Clear all documents strictly from the given conversation."""
        try:
            conv_id = conversation_id or "default"
            self.vs.delete_conversation_vectors(conv_id)
            self.cm.clear_documents(conv_id)
            return {"status": "success", "message": "Documents cleared for this conversation."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_documents(self, conversation_id: str = None):
        """Returns documents attached to this conversation."""
        try:
            conv_id = conversation_id or "default"
            docs = self.cm.get_documents(conv_id)
            return [
                {
                    "name": d.get("original_name") or d.get("filename"),
                    "chunks": f"{d.get('chunks_count', 0)} Chunks"
                }
                for d in docs
            ]
        except Exception:
            return []

    def open_document_page(self, filename: str, page: int = 1, conversation_id: str = None):
        """Opens the source document directly at the specific page number."""
        try:
            import subprocess
            import webbrowser
            clean_filename = os.path.basename(filename).strip()
            conv_id = conversation_id or "default"

            # Check potential locations for the file
            candidate_paths = [
                self.uploads_dir / f"{conv_id}_{clean_filename}",
                self.uploads_dir / clean_filename,
                self.data_dir / clean_filename,
                self.base_dir / "data" / clean_filename,
            ]

            target_path = None
            for p in candidate_paths:
                if p.exists() and p.is_file():
                    target_path = p
                    break

            if not target_path and self.uploads_dir.exists():
                for p in self.uploads_dir.glob(f"*{clean_filename}*"):
                    if p.is_file():
                        target_path = p
                        break

            if not target_path or not target_path.exists():
                return {"status": "error", "message": f"Document '{clean_filename}' not found."}

            page_num = max(1, int(page or 1))
            resolved_uri = target_path.resolve().as_uri() + f"#page={page_num}"

            # Microsoft Edge natively jumps directly to #page=N on Windows
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

    # ==========================================
    # Configuration & Settings
    # ==========================================
    def get_settings(self):
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

    def save_settings(self, api_key: str, model: str):
        """Save new Gemini API key and model selection."""
        try:
            if api_key and api_key.strip():
                clean_key = api_key.strip()
                settings.update_api_key(clean_key)
                self.rag.api_key = clean_key
                self.stt.api_key = clean_key
            if model and model.strip():
                clean_model = model.strip()
                settings.update_model(clean_model)
                self.rag.model = clean_model

            return {"status": "success", "message": "Settings updated successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ==========================================
    # Career OS Profile Integration
    # ==========================================
    def get_career_profile(self):
        """Fetches the latest Career OS user profile from career_os.db."""
        import sqlite3
        candidate_paths = [
            self.base_dir.parent / "career_os.db",
            self.base_dir / "career_os.db",
            self.data_dir / "career_os.db"
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

    def save_career_profile(self, name: str, education: str, skills: str, interests: str, career_goal: str, experience: str, location: str):
        """Saves/Updates Career OS user profile in career_os.db."""
        import sqlite3
        candidate_paths = [
            self.base_dir.parent / "career_os.db",
            self.base_dir / "career_os.db",
            self.data_dir / "career_os.db"
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
            """, (name, education, skills, interests, career_goal, experience, location))
            conn.commit()
            conn.close()
            return {"status": "success", "message": "Career OS Profile updated successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def start_app():
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('knowvox.voicerag.assistant.1.0')
    except Exception:
        pass

    ui_html_path = Path(get_resource_path("ui/index.html")).resolve()
    api = DesktopBridgeAPI()

    window = webview.create_window(
        title="KnowVox - Voice RAG Assistant",
        url=ui_html_path.as_uri(),
        js_api=api,
        width=1240,
        height=860,
        min_size=(940, 640),
        background_color="#121316"
    )

    webview.start(debug=False)


if __name__ == "__main__":
    start_app()
