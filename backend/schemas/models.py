from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []


class QueryResponse(BaseModel):
    query_id: str
    stream_url: str


class IndexStatusResponse(BaseModel):
    chunks_count: int
    files_count: int
    last_built: Optional[str] = None
    status: str  # 'ready' | 'building' | 'error' | 'not_built'
    progress: Optional[str] = None  # e.g. "45/350 chunks"
    eta_seconds: Optional[int] = None


class RebuildRequest(BaseModel):
    mode: str = 'incremental'  # 'full' or 'incremental'


class RebuildResponse(BaseModel):
    status: str  # 'started' | 'already_building'
    message: str


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embed_model: str
    index_status: str
