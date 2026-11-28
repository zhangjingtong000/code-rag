"""Query endpoint with SSE streaming — sync, no async/thread complexity."""

import json
import os
import queue
import threading
import uuid
import time
import sys
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas.models import QueryRequest, QueryResponse
from backend.config import settings

router = APIRouter()

# How long the LLM may stay silent before an SSE comment is sent.
HEARTBEAT_SECONDS = 15.0

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


def _iter_tokens_with_heartbeat(tokens, heartbeat_seconds: float = HEARTBEAT_SECONDS):
    """Relay LLM tokens while yielding ('heartbeat', None) during long silences.

    A sync generator cannot yield while blocked inside ``iter_lines()``, so the
    old in-loop check only ever ran *after* a token arrived — precisely the case
    where a heartbeat is pointless. The silent case (first token, or a long
    pause between bursts) never produced one. The reader therefore runs in a
    worker thread and the generator waits on a queue with a timeout.

    Yields ('token', text) or ('heartbeat', None); re-raises reader exceptions
    so the caller's existing error handling still applies.
    """
    events: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def _pump():
        try:
            for token in tokens:
                events.put(('token', token))
        except BaseException as exc:  # forwarded to the consuming generator
            events.put(('error', exc))
        finally:
            events.put(('end', None))

    threading.Thread(target=_pump, daemon=True).start()

    while True:
        try:
            kind, payload = events.get(timeout=heartbeat_seconds)
        except queue.Empty:
            yield ('heartbeat', None)
            continue
        if kind == 'end':
            return
        if kind == 'error':
            raise payload
        yield (kind, payload)


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
        try:
            from backend.retrieval.query_expansion import expand_query
            from backend.indexing.embedding import EmbeddingService
            from backend.retrieval.vector_store import VectorStore
            from backend.retrieval.fusion import hybrid_search
            from backend.retrieval.reranker import rerank
            from backend.retrieval.bm25_store import load_cached
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

            # 4. BM25 search — cached per process, rebuilt only when the pkl changes
            bm25 = load_cached(settings.bm25_path)
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

            # 8. Build messages
            prompt_mgr = PromptManager()
            messages = prompt_mgr.build_messages(
                question=question,
                context=context,
                chat_history=chat_history,
            )

            # 9. LLM streaming. The input budget must cover everything that is
            # actually sent — System prompt, chat history and the context — not
            # just the context string. SYSTEM_PROMPT alone is ~183 tokens and
            # 4 turns of history can be several thousand.
            input_size = sum(estimate_tokens(m.get('content', '')) for m in messages)
            dyn_max = max(4096, min(16384, 64000 - input_size))
            _log(f"input ~{input_size} tokens, output budget ~{dyn_max} tokens")

            llm = LLMService()
            full_text = ""

            for kind, payload in _iter_tokens_with_heartbeat(
                    llm.generate_stream(messages, max_tokens=dyn_max)):
                if kind == 'heartbeat':
                    yield ": heartbeat\n\n"
                    continue
                full_text += payload
                yield _sse_event("token", {"text": payload})

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
