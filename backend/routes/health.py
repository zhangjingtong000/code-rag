"""Health check and index status endpoints."""

import os
import json
from fastapi import APIRouter
from backend.schemas.models import HealthResponse, IndexStatusResponse, RebuildRequest, RebuildResponse
from backend.config import settings

router = APIRouter()


def _read_metadata():
    path = os.path.join(settings.data_dir, "chunk_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@router.get("/health", response_model=HealthResponse)
async def health():
    meta = _read_metadata()
    return HealthResponse(
        status="healthy",
        llm_model=settings.deepseek_model,
        embed_model=settings.ollama_embed_model,
        index_status="ready" if meta else "not_built",
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    meta = _read_metadata()
    if meta is None:
        return IndexStatusResponse(
            chunks_count=0,
            files_count=0,
            status="not_built",
        )

    # Check if currently building
    lock_path = os.path.join(settings.data_dir, ".building")
    if os.path.exists(lock_path):
        with open(lock_path, 'r') as f:
            progress = f.read().strip()
        return IndexStatusResponse(
            chunks_count=meta.get('chunks_count', 0),
            files_count=meta.get('files_count', 0),
            last_built=meta.get('last_built'),
            status="building",
            progress=progress,
        )

    return IndexStatusResponse(
        chunks_count=meta.get('chunks_count', 0),
        files_count=meta.get('files_count', 0),
        last_built=meta.get('last_built'),
        status="ready",
    )


@router.post("/index/rebuild", response_model=RebuildResponse)
async def rebuild_index(req: RebuildRequest):
    lock_path = os.path.join(settings.data_dir, ".building")
    if os.path.exists(lock_path):
        return RebuildResponse(status="already_building", message="索引正在构建中，请稍后再试")

    # Import here to avoid circular imports
    from backend.indexing.indexer import Indexer
    indexer = Indexer()

    # Run in background (for simplicity we run synchronously in a thread)
    import threading

    def _rebuild():
        try:
            with open(lock_path, 'w') as f:
                f.write("starting...")
            indexer.build_index(mode=req.mode)
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

    thread = threading.Thread(target=_rebuild, daemon=True)
    thread.start()

    return RebuildResponse(
        status="started",
        message=f"索引重建已启动（模式: {req.mode}），通过 /api/rag/index/status 查看进度"
    )
