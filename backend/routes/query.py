"""Query endpoint with SSE streaming — sync, no async/thread complexity."""

import json
import os
import uuid
import time
import sys
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas.models import QueryRequest, QueryResponse
from backend.config import settings

router = APIRouter()

# In-memory query store (for demo / single user)
_query_store: dict = {}


def _log(msg: str):
    """Write log to stderr (unbuffered) + to a debug file."""
    sys.stderr.write(f"[RAG] {msg}\n")
    sys.stderr.flush()
    try:
        with open("data/rag_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _sse_event(event: str, data: dict) -> str:
    """Format an SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query", response_model=QueryResponse)
def create_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 空索引上允许提问只会拿到「源码中未找到相关实现」的空回答，还会白烧一次
    # LLM 调用。索引没建、或建出来是 0 chunk，都在这里直接拦掉并说清怎么修。
    meta_path = settings.metadata_path
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = None

    if not meta:
        raise HTTPException(
            status_code=409,
            detail="索引尚未构建（data/chunk_metadata.json 缺失），先运行 python build_index.py full",
        )
    if not meta.get('chunks_count'):
        raise HTTPException(
            status_code=409,
            detail=("索引为空（0 chunk）：语料目录里没有可索引的源文件，检索不会有结果。"
                    "请检查 data/source-* 或 RAG_SOURCE_DIRS，然后重建索引"),
        )

    query_id = uuid.uuid4().hex[:12]
    _query_store[query_id] = {
        "question": req.question,
        "chat_history": [m.model_dump() for m in req.chat_history],
        "status": "pending",
    }

    return QueryResponse(
        query_id=query_id,
        stream_url=f"/api/rag/query/stream/{query_id}",
    )


@router.get("/query/stream/{query_id}")
def stream_query(query_id: str):
    """Sync generator — FastAPI runs it in a thread pool automatically."""
    if query_id not in _query_store:
        raise HTTPException(status_code=404, detail="查询不存在或已过期")

    qdata = _query_store[query_id]
    qdata["status"] = "streaming"

    def generate():
        last_time = time.time()
        try:
            from backend.retrieval.query_expansion import expand_query
            from backend.indexing.embedding import EmbeddingService
            from backend.retrieval.vector_store import VectorStore
            from backend.retrieval.bm25_store import BM25Store
            from backend.retrieval.fusion import hybrid_search
            from backend.retrieval.reranker import rerank
            from backend.generation.context_builder import ContextBuilder
            from backend.generation.context_builder import estimate_tokens
            from backend.generation.prompt_manager import PromptManager
            from backend.generation.llm_service import LLMService

            question = qdata["question"]
            chat_history = qdata.get("chat_history", [])

            _log(f"Query: {question[:80]}...")

            # 1. Expand query
            expanded_query = expand_query(question)

            # 2. Embed query
            embed_svc = EmbeddingService()
            query_embedding = embed_svc.embed_query(expanded_query)

            # 3. Vector search
            vs = VectorStore()
            vector_results = vs.search(query_embedding, k=settings.vector_k)

            # 4. BM25 search
            bm25 = BM25Store()
            bm25.load(settings.bm25_path)
            bm25_results = bm25.search(expanded_query, k=settings.bm25_k)

            # 5. Hybrid fusion
            fusion_results = hybrid_search(vector_results, bm25_results, question=question)

            # Metadata
            source_files = []
            for r in fusion_results[:10]:
                meta = r.get('metadata', {})
                fpath = meta.get('file_path', '')
                if fpath:
                    source_files.append(f"{meta.get('source', '')}:{fpath}")
            source_files = list(set(source_files))
            _log(f"Top files: {source_files[:5]}")
            yield _sse_event("metadata", {
                "files_count": len(source_files),
                "files": source_files,
            })

            # 6. Cross-encoder rerank (universal quality boost)
            if len(fusion_results) > 15:
                fusion_results = rerank(question, fusion_results, top_n=20)
                _log(f"Reranked to {len(fusion_results)} candidates")

            # 7. Build context
            ctx_builder = ContextBuilder()
            context = ctx_builder.build(fusion_results)

            # 7. Build messages
            prompt_mgr = PromptManager()
            messages = prompt_mgr.build_messages(
                question=question,
                context=context,
                chat_history=chat_history,
            )

            # 8. LLM streaming
            input_size = estimate_tokens(context) + 500
            dyn_max = max(4096, min(16384, 64000 - input_size))
            _log(f"context ~{input_size} tokens, output budget ~{dyn_max} tokens")

            llm = LLMService()
            full_text = ""

            for token in llm.generate_stream(messages, max_tokens=dyn_max):
                full_text += token
                yield _sse_event("token", {"text": token})

                # Heartbeat check: if no token for 15s, send heartbeat comment
                now = time.time()
                if now - last_time > 15:
                    yield ": heartbeat\n\n"
                    last_time = now

            _log(f"Done. Total output: {len(full_text)} chars, {len(full_text.split())} words")
            yield _sse_event("done", {"full_text": full_text})

        except Exception as e:
            import traceback
            _log(f"ERROR: {e}\n{traceback.format_exc()}")
            yield _sse_event("error", {"error": str(e)})
        finally:
            qdata["status"] = "completed"
            _query_store.pop(query_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
