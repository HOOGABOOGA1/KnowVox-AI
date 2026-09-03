import os
import uuid
import re
from pathlib import Path
import edge_tts
from app.core.config import base_dir


class TTSService:
    def __init__(self, voice: str = "en-US-JennyNeural"):
        """
        voice options:
        - "en-US-JennyNeural" (Female, natural American English)
        - "en-US-GuyNeural" (Male, natural American English)
        - "en-IN-NeerjaNeural" (Female, Indian English)
        - "en-IN-PrabhatNeural" (Male, Indian English)
        - "en-GB-SoniaNeural" (Female, British English)
        """
        self.voice = voice

    def _clean_text_for_speech(self, text: str) -> str:
        """Remove document citations, markdown tags, filenames, dashes, and formatting so speech flows naturally."""
        if not text:
            return "I have no response to speak."

        cleaned = text

        # 1. Remove unicode emojis and symbols
        cleaned = re.sub(r"[\U00010000-\U0010ffff]", " ", cleaned)

        # 2. Remove offline headers, banner texts, notes
        cleaned = re.sub(r"(?i)\*?\*?offline knowledge match[^\n]*\*?\*?", " ", cleaned)
        cleaned = re.sub(r"(?i)\*?\*?local (?:ai|knowledge)[^\n]*\*?\*?", " ", cleaned)
        cleaned = re.sub(r"(?i)\(retrieved directly from[^\)]*\)", " ", cleaned)
        cleaned = re.sub(r"(?i)\(retrieved from[^\)]*\)", " ", cleaned)
        cleaned = re.sub(r"(?i)\(note:[^\)]*\)", " ", cleaned)
        cleaned = re.sub(r"(?i)Key Findings from Document", " ", cleaned)
        cleaned = re.sub(r"(?i)Detailed Document Analysis & Findings", " ", cleaned)
        cleaned = re.sub(r"(?i)Say 'explain more'[^\n]*", " ", cleaned)
        cleaned = re.sub(r"(?i)Tip: You can ask[^\n]*", " ", cleaned)

        # 3. Remove all parenthetical source citations like (Source: The+48+Laws..., Page 146):
        cleaned = re.sub(r"(?i)\(source:[^\)]*\):?", " ", cleaned)
        cleaned = re.sub(r"(?i)\((?:from\s+)?page\s*\d+[^\)]*\):?", " ", cleaned)
        cleaned = re.sub(r"(?i)\[source:[^\]]*\]:?", " ", cleaned)
        cleaned = re.sub(r"(?i)\[page\s*\d+[^\]]*\]:?", " ", cleaned)
        cleaned = re.sub(r"(?i)\bPoint\s*\d+\b:?", " ", cleaned)

        # 4. Remove any stray filenames or extensions
        cleaned = re.sub(r"[0-9a-zA-Z_\+\-]+\.(?:pdf|docx?|txt|md|xlsx?|csv)", " ", cleaned, flags=re.IGNORECASE)

        # 5. Fix hyphenated word breaks across lines (e.g. "vic- tim" -> "victim", "un- lucky" -> "unlucky")
        cleaned = re.sub(r'(\b\w+)[-—–]\s+(\w+\b)', r'\1\2', cleaned)

        # 6. Replace multiple dashes or isolated dashes with commas (so TTS pauses naturally instead of saying 'dash dash')
        cleaned = re.sub(r"[-—–]{2,}", ", ", cleaned)
        cleaned = re.sub(r"\s+[-—–]\s+", ", ", cleaned)
        cleaned = re.sub(r"[-—–]", " ", cleaned)

        # 7. Remove OCR artifacts
        cleaned = re.sub(r'[_\[\]]+UDGMENT', 'JUDGMENT:', cleaned, flags=re.IGNORECASE)

        # 8. Remove markdown formatting
        cleaned = re.sub(r"#{1,6}\s*", " ", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

        # 9. Remove bullet markers
        cleaned = re.sub(r"^\s*[-*•]\s*", " ", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*\d+\.\s*", " ", cleaned, flags=re.MULTILINE)

        # 10. Clean up special characters that TTS might pronounce literally
        for ch in ["_", "~", "<", ">", "|", "•", "\u2022", "*", "#", "[", "]", "{", "}", "\\", "/", "«", "»", "^", "+"]:
            cleaned = cleaned.replace(ch, " ")

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    async def text_to_speech(self, text: str, output_dir: str = None, voice: str = None) -> str:
        """Synthesize text into an MP3 file with the selected voice accent and return the file path."""
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = base_dir / "data" / "audio"

        out_path.mkdir(parents=True, exist_ok=True)

        chosen_voice = voice or self.voice or "en-US-JennyNeural"
        spoken_text = self._clean_text_for_speech(text)
        if not spoken_text:
            spoken_text = "I have no response to speak."

        filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
        output_file = str(out_path / filename)

        import socket
        def _is_online(timeout=0.6) -> bool:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(("8.8.8.8", 53))
                s.close()
                return True
            except Exception:
                return False

        online = _is_online(timeout=0.5)

        if online:
            try:
                communicate = edge_tts.Communicate(spoken_text, chosen_voice)
                await communicate.save(output_file)
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    return output_file
            except Exception:
                # Fallback to JennyNeural if chosen voice had a minor network glitch
                if chosen_voice != "en-US-JennyNeural":
                    try:
                        communicate = edge_tts.Communicate(spoken_text, "en-US-JennyNeural")
                        await communicate.save(output_file)
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            return output_file
                    except Exception:
                        pass

        # 100% Offline fallback using pyttsx3 (SAPI5 native Windows speech)
        try:
            offline_filename = f"response_{uuid.uuid4().hex[:8]}.wav"
            offline_file = str(out_path / offline_filename)
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.save_to_file(spoken_text, offline_file)
            engine.runAndWait()
            if os.path.exists(offline_file):
                return offline_file
        except Exception as e:
            print(f"Offline TTS fallback error: {e}")

        return output_file