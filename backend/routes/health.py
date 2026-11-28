"""Health check and index status endpoints."""

import os
import json
from fastapi import APIRouter, HTTPException
from backend.schemas.models import HealthResponse, IndexStatusResponse, RebuildRequest, RebuildResponse
from backend.config import settings

router = APIRouter()

# 后台重建的终态。构建线程里抛异常时（例如语料扫到 0 个文件），光打日志不够：
# .building 锁文件删掉之后 /index/status 会退回 ready，前端永远看不出这次重建
# 是失败的。状态存在内存里，因为覆盖写索引的只有本进程这一个线程。
_rebuild_state = {'status': 'idle', 'message': None}


def _read_metadata(path: str | None = None):
    """Read chunk metadata, tolerating a missing or corrupt file.

    A half-written / corrupt ``chunk_metadata.json`` used to raise straight out
    of the endpoint (HTTP 500) and the UI showed "服务未连接". Missing or
    unreadable metadata is a recoverable state: report it as "not built".
    """
    path = path or os.path.join(settings.data_dir, "chunk_metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[Index API] chunk_metadata.json 读取失败（按未构建处理）: {exc}", flush=True)
        return None
    return data if isinstance(data, dict) else None


def _empty_index_detail() -> str:
    return ("索引为空（0 chunk）：语料目录里没有可索引的源文件，索引文件不覆盖写。"
            "请检查 data/source-* 或 RAG_SOURCE_DIRS 指向的目录是否存在源文件，"
            "然后重新构建索引。")


@router.get("/health", response_model=HealthResponse)
async def health():
    meta = _read_metadata()
    if _rebuild_state['status'] == 'error':
        index_state = "error"
    elif not meta:
        index_state = "not_built"
    elif meta.get('chunks_count'):
        index_state = "ready"
    else:
        # 0 chunk 的索引不算就绪，和 /index/status 保持一致
        index_state = "error"

    return HealthResponse(
        status="healthy",
        llm_model=settings.deepseek_model,
        embed_model=settings.ollama_embed_model,
        index_status=index_state,
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    meta = _read_metadata() or {}
    lock_path = os.path.join(settings.data_dir, ".building")

    # Check if currently building
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

    if _rebuild_state['status'] == 'error':
        return IndexStatusResponse(
            chunks_count=meta.get('chunks_count', 0),
            files_count=meta.get('files_count', 0),
            last_built=meta.get('last_built'),
            status="error",
            progress=_rebuild_state['message'],
        )

    if not meta:
        return IndexStatusResponse(
            chunks_count=0,
            files_count=0,
            status="not_built",
        )

    # 0 chunk 的索引不算就绪：允许提问只会得到「源码中未找到」的空回答
    if not meta.get('chunks_count'):
        return IndexStatusResponse(
            chunks_count=0,
            files_count=meta.get('files_count', 0),
            last_built=meta.get('last_built'),
            status="error",
            progress=_empty_index_detail(),
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

    # 先扫一遍语料再决定要不要起线程。以前不管有没有文件都回 started，建完写出
    # chunks_count=0 的 metadata，状态反而变 ready。扫描是纯文件名遍历，相对
    # 嵌入开销可以忽略。
    from backend.indexing.scanner import CodeScanner
    if not CodeScanner(settings).scan():
        raise HTTPException(status_code=409, detail=_empty_index_detail())

    # Import here to avoid circular imports
    from backend.indexing.indexer import Indexer
    indexer = Indexer()

    # Run in background (for simplicity we run synchronously in a thread)
    import threading

    def _rebuild():
        _rebuild_state['status'] = 'building'
        _rebuild_state['message'] = None
        try:
            with open(lock_path, 'w') as f:
                f.write("starting...")
            indexer.build_index(mode=req.mode)
            _rebuild_state['status'] = 'ok'
        except Exception as e:  # 线程里的异常没人接，必须自己记下来
            _rebuild_state['status'] = 'error'
            _rebuild_state['message'] = str(e)
            print(f"[Index API] rebuild failed: {e}", flush=True)
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

    thread = threading.Thread(target=_rebuild, daemon=True)
    thread.start()

    return RebuildResponse(
        status="started",
        message=f"索引重建已启动（模式: {req.mode}），通过 /api/rag/index/status 查看进度"
    )
