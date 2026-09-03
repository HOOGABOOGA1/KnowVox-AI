import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def get_base_dir() -> Path:
    """Get the persistent base directory whether running from source or PyInstaller .exe"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent

base_dir = get_base_dir()

# Look for .env in all candidate locations
env_candidates = [
    base_dir / ".env",
    base_dir.parent / ".env",
    base_dir / "voicerag" / ".env",
    Path.cwd() / ".env",
    Path.home() / ".knowvox" / ".env",
    Path.home() / ".voicerag" / ".env"
]
if hasattr(sys, '_MEIPASS'):
    env_candidates.append(Path(sys._MEIPASS) / ".env")

for candidate in env_candidates:
    if candidate.exists():
        load_dotenv(dotenv_path=candidate, override=False)

load_dotenv(override=False)

class Settings:
    PROJECT_NAME: str = "KnowVox"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = str(base_dir / "data" / "chroma_db")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    @classmethod
    def update_api_key(cls, new_key: str):
        cls.GEMINI_API_KEY = new_key
        os.environ["GEMINI_API_KEY"] = new_key
        # Save to base_dir .env
        env_file = base_dir / ".env"
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={new_key}\nGEMINI_MODEL={cls.DEFAULT_MODEL}\n")
        except Exception:
            pass

    @classmethod
    def update_model(cls, new_model: str):
        cls.DEFAULT_MODEL = new_model
        os.environ["GEMINI_MODEL"] = new_model

settings = Settings()