import os
import json
import wave
import base64
import requests
from typing import Optional
from app.core.config import settings

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import vosk
    vosk.SetLogLevel(-1)  # Silence verbose C++ logs
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

import socket


def check_internet_connection(timeout: float = 0.8) -> bool:
    """Quick check if internet is reachable to avoid long DNS timeouts when offline."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except Exception:
        return False


class STTService:
    def __init__(self, api_key: Optional[str] = None, model_size: str = "tiny.en"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if self.api_key == "your_gemini_api_key_here":
            self.api_key = ""
        self.recognizer = sr.Recognizer() if HAS_SR else None
        self.model_size = model_size
        self._whisper_model = None
        self._vosk_model = None

    def _get_api_key(self) -> str:
        return self.api_key or settings.GEMINI_API_KEY

    def _get_vosk_model(self):
        """Lazily initialize on-device Vosk model for 100% offline transcription."""
        if self._vosk_model is None and HAS_VOSK:
            try:
                self._vosk_model = vosk.Model(lang="en-us")
            except Exception as e:
                print(f"Vosk model init notice: {e}")
        return self._vosk_model

    def _get_whisper_model(self):
        """Lazily initialize on-device faster-whisper for offline transcription."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except Exception:
                try:
                    from faster_whisper import WhisperModel
                    self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                except Exception:
                    pass
        return self._whisper_model

    def _transcribe_vosk(self, audio_file_path: str) -> str:
        """Transcribes WAV audio using on-device Vosk model with zero internet."""
        try:
            model = self._get_vosk_model()
            if not model:
                return ""

            wf = wave.open(audio_file_path, "rb")
            sample_rate = wf.getframerate()
            rec = vosk.KaldiRecognizer(model, sample_rate)
            rec.SetWords(False)

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"):
                        results.append(res["text"])

            final_res = json.loads(rec.FinalResult())
            if final_res.get("text"):
                results.append(final_res["text"])
            wf.close()

            return " ".join(results).strip()
        except Exception as e:
            print(f"Vosk offline transcription error: {e}")
            return ""

    def _words_to_num(self, w: str) -> str:
        units = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19
        }
        tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        tokens = w.strip().lower().split()
        if not tokens:
            return w
        val = 0
        for tok in tokens:
            if tok in units:
                val += units[tok]
            elif tok in tens:
                val += tens[tok]
            else:
                return w
        return str(val) if val > 0 else w

    def _post_process_stt(self, text: str) -> str:
        if not text:
            return ""
        import re
        t = text.lower().strip()

        # 1. Phonetic fixes for Indian names & common misrecognitions in en-us Vosk
        phonetic_map = [
            (r'\bchina\s*(?:king|key|kia|can|kya|q)\b', 'chanakya'),
            (r'\bshining\s*(?:car|key)\b', 'chanakya'),
            (r'\bturn\s*a\s*key\b', 'chanakya'),
            (r'\bchinook\s*(?:yeah|ya)\b', 'chanakya'),
            (r'\bchanak\s*ya\b', 'chanakya'),
            (r'\b(?:neety|neati|niti|neat\s*tea)\b', 'neeti'),
            (r'\bneed\s*to\b(?=\s+sutra|\s+by|\s+book|\s+advice|\s+chanakya|\s+management)', 'neeti'),
            (r'\bnight\b(?=\s+sutra|\s+chanakya|\s+management)', 'neeti'),
            (r'\bshooter\b(?=\s+chanakya|\s+management)', 'sutra'),
            (r'\bsootra\b', 'sutra'),
            (r'\ba\s*chariot\b', 'acharya'),
            (r'\barch\s*area\b', 'acharya'),
            (r'\bcaught\s*ill\b', 'kautilya'),
            (r'\bcoat\s*till\s*ya\b', 'kautilya'),
            (r'\bdrama\b(?=\s+and\s+karma|\s+sutra|\s+chanakya)', 'dharma'),
            (r'\brobert\s*green\b', 'robert greene'),
            (r'\bis\s*hunt\b', 'ishant'),
            (r'\bis\s*shunt\b', 'ishant'),
            (r'\b(?:low|lot|raw|lo|la)\s+(?=\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)', 'law '),
        ]

        for pattern, repl in phonetic_map:
            t = re.sub(pattern, repl, t)

        # 2. Convert spoken number words to digits for laws, chapters, rules
        num_pattern = r'\b(law|rule|chapter|section|module|unit|page|part|article)\s+((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))\b'
        def _repl_num(m):
            prefix = m.group(1)
            num = self._words_to_num(m.group(2))
            return f'{prefix} {num}'

        t = re.sub(num_pattern, _repl_num, t, flags=re.IGNORECASE)

        return t

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file to text with phonetic and numerical normalization.
        """
        raw_text = self._transcribe_raw(audio_file_path)
        return self._post_process_stt(raw_text)

    def _transcribe_raw(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            return ""

        online = check_internet_connection(timeout=0.6)

        # 1. If online, try SpeechRecognition (Fast online Google STT)
        if online and self.recognizer and audio_file_path.lower().endswith(".wav"):
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    if text and text.strip():
                        return text.strip()
            except Exception:
                pass  # Fall back

        # 2. If online and API key exists, try Gemini Multimodal Audio
        api_key = self._get_api_key()
        if online and api_key:
            try:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()

                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                ext = os.path.splitext(audio_file_path)[1].lower()
                mime_types = {
                    ".mp3": "audio/mp3",
                    ".wav": "audio/wav",
                    ".ogg": "audio/ogg",
                    ".m4a": "audio/mp4",
                    ".webm": "audio/webm"
                }
                mime_type = mime_types.get(ext, "audio/mp3")

                for model in ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash"]:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": "Transcribe the spoken words in this audio exactly as uttered."},
                                {"inline_data": {"mime_type": mime_type, "data": base64_audio}}
                            ]
                        }]
                    }
                    try:
                        response = requests.post(url, json=payload, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
                            transcription = "".join([p.get("text", "") for p in parts]).strip()
                            if transcription and "error" not in transcription.lower():
                                return transcription
                    except Exception:
                        continue
            except Exception:
                pass

        # 3. 100% Offline On-Device Vosk Model (Runs locally on CPU with zero internet)
        vosk_text = self._transcribe_vosk(audio_file_path)
        if vosk_text and vosk_text.strip():
            return vosk_text.strip()

        # 4. 100% Offline Whisper Fallback
        try:
            whisper = self._get_whisper_model()
            if whisper:
                segments, info = whisper.transcribe(audio_file_path, beam_size=1)
                whisper_text = " ".join([s.text for s in segments]).strip()
                if whisper_text:
                    return whisper_text
        except Exception:
            pass

        return ""

