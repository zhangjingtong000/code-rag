import os
import sys

# Ensure project root is on path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.health import router as health_router
from backend.routes.query import router as query_router
from backend.routes.audio import router as audio_router
from backend.config import settings


class NoCacheStaticFiles(StaticFiles):
    """Static files with no-cache headers for development."""
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app = FastAPI(
    title="RAG Code Q&A System",
    description="代码知识库 RAG 问答系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/rag")
app.include_router(query_router, prefix="/api/rag")
app.include_router(audio_router, prefix="/api/rag")

# Mount frontend static files (no-cache for development)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", NoCacheStaticFiles(directory=frontend_path, html=True), name="frontend")


@app.on_event("startup")
async def startup_event():
    print(f"[RAG System] Starting on {settings.host}:{settings.port}")
    print(f"[RAG System] LLM: {settings.deepseek_model}")
    print(f"[RAG System] Embedding: {settings.ollama_embed_model} @ {settings.ollama_base_url}")
    print(f"[RAG System] Frontend: http://{settings.host}:{settings.port}/frontend/index.html")
