from fastapi import FastAPI

app = FastAPI(
    title="VoiceRAG",
    description="SIH012 Voice-Enabled Knowledge Retrieval System",
    version="0.1.0"
)


@app.get("/")
def home():
    return {
        "message": "VoiceRAG is online",
        "status": "ready"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }