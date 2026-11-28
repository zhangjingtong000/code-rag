from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class QueryRequest(BaseModel):
    question: str
    # 可变默认值必须每个实例一份：list 字面量会在所有请求之间共享同一个对象。
    chat_history: list[ChatMessage] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query_id: str
    stream_url: str


class IndexStatusResponse(BaseModel):
    chunks_count: int
    files_count: int
    last_built: Optional[str] = None
    status: str  # 'ready' | 'building' | 'error' | 'not_built'
    progress: Optional[str] = None  # e.g. "45/350 chunks"


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
