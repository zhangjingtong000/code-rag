"""Query expansion — Chinese term mapping + optional LLM expansion."""

from backend.config import settings

# Domain-specific Chinese → English term mapping
# Organized by subsystem for maintainability
TERM_MAP = {
    # ── Project Domain (WeChat Mini-Program) ──
    '登录': ['login', 'auth', 'authentication', 'signin'],
    '注册': ['register', 'signup'],
    '鉴权': ['auth', 'authorization', 'authenticate', 'appkey'],
    '识别': ['recognize', 'recognition', 'detect', 'classification'],
    '图像识别': ['image recognition', 'recognize', 'scene recognition'],
    '场景': ['scene', 'scenery', 'scenario'],
    '标签': ['tag', 'label', 'classification'],
    '分类': ['category', 'classification', 'classify'],
    '图片': ['image', 'picture', 'photo'],
    '上传': ['upload', 'file upload', 'chooseMedia'],
    '压缩': ['compress', 'resize', 'compression', 'compressImage'],
    '文字': ['text', 'ocr', 'character'],
    '提取': ['extract', 'extraction'],
    '问答': ['chat', 'qa', 'question answering'],
    '对话': ['chat', 'conversation', 'dialogue'],
    '缓存': ['cache', 'caching', 'wx.setStorageSync'],
    '存储': ['store', 'storage', 'persist', 'cloud storage'],
    '记录': ['record', 'history', 'log'],
    '收藏': ['favorite', 'bookmark', 'star'],
    '分享': ['share', 'forward'],
    '云函数': ['cloud function', 'cloudfunction', 'faas', 'wx.cloud.callFunction'],
    '数据库': ['database', 'db', 'cloud database', 'wx.cloud.database'],
    '页面': ['page', 'view', 'screen', 'navigateTo'],
    '组件': ['component', 'widget'],
    '导航': ['navigation', 'navigate', 'router', 'tabBar'],
    '跳转': ['navigate', 'redirect', 'route', 'navigateTo'],
    '配置': ['config', 'configuration', 'settings', 'app.json'],
    '接口': ['api', 'interface', 'endpoint', 'API_BASE'],
    '路由': ['router', 'route', 'routing', 'FastAPI'],
    '服务': ['service', 'server', 'FastAPI'],
    '中间件': ['middleware', 'CORS', 'rate limit'],
    '限流': ['rate limit', 'throttle', 'middleware'],
    '安全': ['security', 'moderate', 'imgSecCheck'],
    '审核': ['moderate', 'moderation', 'security check'],
    '分支': ['branch', 'main', 'release'],
    '主分支': ['main branch', 'main'],
    '发布分支': ['release branch', 'release'],
    '模型': ['model', 'LLM', 'deepseek', 'qwen'],

    # ── RAG: Indexing Pipeline ──
    '扫描': ['scanner', 'scan', 'CodeScanner', 'file scan'],
    '扫描器': ['scanner', 'CodeScanner', 'file walker'],
    '包含': ['include', 'include_extensions', 'whitelist'],
    '排除': ['exclude', 'exclude_patterns', 'blacklist', 'ignore'],
    '文件扩展名': ['file extension', 'include_extensions', 'file suffix'],
    '分块': ['chunk', 'chunking', 'split', 'chunker', 'CodeChunker'],
    '块类型': ['chunk_type', 'chunk type', 'function', 'class', 'page'],
    '函数块': ['function chunk', 'function', 'method chunk'],
    '类块': ['class chunk', 'class'],
    '页面块': ['page chunk', 'Page block', 'Page({})'],
    '模板块': ['template chunk', 'template', 'WXML template'],
    '样式块': ['style chunk', 'style', 'WXSS style'],
    '配置块': ['config chunk', 'config', 'JSON chunk'],
    '模块级': ['module_level', 'module level', 'top-level'],
    '边界': ['boundary', 'split', 'chunk boundary', 'def boundary'],
    '子块': ['subchunk', 'sub-chunk', 'sub_split', 'inner split'],
    '装饰器': ['decorator', '@decorator'],
    '重叠': ['overlap', 'chunk_overlap', 'overlap_lines'],
    '头部': ['header', 'module header', 'imports block'],
    '语言检测': ['language detection', 'file language'],

    # ── RAG: Embedding ──
    '嵌入': ['embedding', 'embed', 'vector', 'embeddings'],
    '向量': ['vector', 'embedding', 'chromadb', 'ChromaDB'],
    '向量检索': ['vector search', 'vector retrieval', 'embedding search'],
    'Ollama': ['Ollama', 'ollama', 'ollama embed', 'local embed'],
    '批量大小': ['batch size', 'embed_batch_size', 'batch embed'],
    '回退': ['fallback', 'one-by-one', 'single embed'],
    '查询嵌入': ['query embedding', 'embed_query'],
    '向量维度': ['dimension', 'embedding dim', '768 dim'],
    '零向量': ['zero vector', 'fallback zeros'],
    '余弦': ['cosine', 'cosine distance', 'cosine similarity'],
    '相似度': ['similarity', 'cosine', 'distance'],

    # ── RAG: Retrieval ──
    '检索': ['retrieval', 'search', 'retrieve', 'hybrid search'],
    '关键词': ['keyword', 'BM25', 'bm25', 'token'],
    '混合': ['hybrid', 'fusion', 'RRF', 'reciprocal rank'],
    '排序': ['rank', 'ranking', 'RRF', 'rerank', 'score'],
    '融合': ['fusion', 'RRF', 'merge', 'hybrid'],
    '去重': ['dedup', 'deduplicate', 'deduplication'],
    '每文件限制': ['per-file limit', 'max_chunks_per_file', 'max per file'],
    '前K个': ['top-k', 'top_k', 'top K', 'k results'],
    '语义': ['semantic', 'embedding', 'similarity', 'cosine'],

    # ── RAG: Indexing Operations ──
    '索引': ['index', 'indexing', 'indexer', 'build index'],
    '全量': ['full rebuild', 'full index', 'mode full'],
    '增量': ['incremental', 'incremental index', 'mode incremental'],
    '重建': ['rebuild', 'reindex', 'build_index', 'build index'],
    '构建锁': ['lock file', 'building lock', '.building'],
    '进度': ['progress', 'build progress', 'indexing progress'],
    '哈希': ['hash', 'md5', 'file hash', 'filehash'],
    '变更检测': ['change detection', 'hash diff', 'md5 compare'],
    '孤立块': ['orphan chunk', 'stale chunk', 'orphan cleanup'],
    '元数据': ['metadata', 'chunk metadata', 'chunk_metadata'],
    '来源': ['source', 'source label', 'branch source'],
    '文件计数': ['files count', 'file count'],
    '块计数': ['chunks count', 'chunk count'],

    # ── RAG: BM25 ──
    '分词': ['tokenize', 'tokenizer', 'split', 'BM25'],
    '语料库': ['corpus', 'BM25 corpus', 'document corpus'],
    '文档': ['document', 'indexed document', 'bm25 document'],
    '持久化': ['persist', 'save', 'load', 'pickle'],
    'BM25分数': ['BM25 score', 'keyword score', 'bm25 score'],

    # ── RAG: LLM & Generation ──
    '生成': ['generation', 'generate', 'completion', 'LLM'],
    '流式': ['stream', 'SSE', 'streaming', 'EventSource'],
    '流式生成': ['streaming generation', 'generate_stream', 'SSE stream'],
    '非流式': ['non-streaming', 'generate', 'sync generation'],
    '最大令牌': ['max tokens', 'max_tokens', 'token limit'],
    '温度': ['temperature', 'temp', 'sampling temperature'],
    '完成原因': ['finish reason', 'finish_reason', 'stop length'],
    '增量内容': ['delta', 'token delta', 'content delta'],
    'DeepSeek': ['DeepSeek', 'deepseek', 'deepseek-chat', 'deepseek API'],
    '令牌估算': ['estimate tokens', 'estimate_tokens', 'token count'],
    '令牌预算': ['token budget', 'context budget', 'token limit'],
    '对话历史': ['chat history', 'chat_history', 'conversation history'],

    # ── RAG: Context Building ──
    '上下文': ['context', 'contextual', 'context window'],
    '上下文窗口': ['context window', 'context length', 'context size'],
    '截断': ['truncate', 'truncation', 'context truncation'],
    '命中范围': ['hit range', 'hit_ranges', 'hit lines'],
    '展开行': ['expand lines', 'context expand', 'expand_lines'],
    '完整文件': ['full file', 'full source file', 'complete file'],
    '项目背景': ['project background', 'project context', 'project info'],

    # ── RAG: Prompt ──
    '提示词': ['prompt', 'system prompt', 'System Prompt'],
    '系统提示': ['system prompt', 'system message', 'SYSTEM_PROMPT'],
    '硬性要求': ['hard constraint', 'hard rule', 'mandatory rule'],
    '角色': ['role', 'message role', 'user system assistant'],

    # ── RAG: Query & SSE ──
    '查询ID': ['query id', 'query_id', 'stream query'],
    '查询扩展': ['query expansion', 'expand_query', 'query expansion L1'],
    'LLM扩展': ['LLM expansion', 'expand_query_llm', 'L2 expansion'],
    '心跳': ['heartbeat', 'SSE heartbeat', 'keep alive'],
    '查询存储': ['query store', 'in-memory store', '_query_store'],
    '调试日志': ['debug log', 'rag_debug.log', 'log file'],

    # ── RAG: Server & Health ──
    '健康检查': ['health check', 'health', 'health endpoint'],
    '状态': ['status', 'index status', 'server status'],
    '就绪': ['ready', 'index ready'],
    '未构建': ['not built', 'not_built', 'index missing'],
    '构建中': ['building', 'index building', 'currently building'],
    '模式': ['mode', 'index mode', 'full incremental'],
    '主机': ['host', 'server host', '0.0.0.0'],
    '端口': ['port', 'server port', 'uvicorn port'],
    '数据目录': ['data directory', 'data_dir', 'data folder'],
    '前端': ['frontend', 'static files', 'index.html'],
    '启动': ['startup', 'server startup', 'on_event startup'],
    '项目根目录': ['project root', 'project_root', 'root dir'],
    'Pydantic': ['pydantic', 'schema', 'data model', 'request model'],

    # ── RAG: Fusion ──
    'RRF': ['RRF', 'reciprocal rank fusion', 'rank fusion'],

    # ── Models & APIs ──
    '千问': ['qwen', 'dashscope', 'tongyi'],
    '火山': ['volcano', 'volcengine', 'doubao'],  # 已废弃，保留兼容
    '腾讯云': ['tencent cloud', 'cloudbase', 'tencent'],
    '知识库': ['knowledge base', 'index', 'vector store', 'ChromaDB'],
    'ChromaDB': ['ChromaDB', 'chromadb', 'vector database', 'persistent client'],
    '流水线': ['pipeline', 'indexing pipeline', 'processing pipeline'],
    'FastAPI': ['FastAPI', 'fastapi', 'CORS', 'middleware'],
    'Uvicorn': ['uvicorn', 'server', 'ASGI', 'reload'],
    'SSE': ['SSE', 'server-sent events', 'event stream', 'StreamingResponse'],
    'CORS': ['CORS', 'cross-origin', 'allow origins'],

    # ── RAG: ASR / Voice Input ──
    '语音': ['voice', 'speech', 'audio', 'ASR', 'speech-to-text', 'transcribe'],
    '语音输入': ['voice input', 'speech input', 'microphone', 'MediaRecorder', 'getUserMedia', 'audio recording'],
    '录音': ['recording', 'audio capture', 'MediaRecorder', 'microphone'],
    '麦克风': ['microphone', 'mic', 'audio input', 'getUserMedia'],
    '语音识别': ['ASR', 'speech recognition', 'Qwen3-ASR', 'speech-to-text', 'STT', 'transcription'],
    '转写': ['transcribe', 'transcription', 'speech-to-text', 'ASR'],
    '音频': ['audio', 'WAV', 'PCM', 'waveform', 'sound'],
    'Qwen3': ['Qwen3-ASR', 'Qwen', 'speech recognition model', 'Alibaba'],
    'ASR': ['automatic speech recognition', 'speech-to-text', 'STT', 'Qwen3-ASR', 'transcriber'],
    'transcriber': ['transcribe', 'transcription', 'ASR', 'speech recognition', 'Qwen3-ASR'],
    'MediaRecorder': ['MediaRecorder API', 'getUserMedia', 'audio recording', 'webm', 'browser recording'],
}


def expand_query(question: str) -> str:
    """Expand a Chinese question with English term equivalents.

    Returns the expanded query string for both vector and BM25 search.
    Automatically disambiguates between RAG system and mini-program context.
    """
    expansion_terms = []

    # ── Context Detection ──
    is_rag_context = bool(
        'rag' in question.lower() or
        'RAG' in question or
        '问答系统' in question
    )

    # ── L1: Static mapping ──
    for zh_term, en_terms in TERM_MAP.items():
        if zh_term in question:
            expansion_terms.extend(en_terms[:3])  # take top 3 to avoid explosion

    # ── RAG context disambiguation ──
    # When user asks about "RAG的前端" / "RAG的检索", bias towards rag source files
    if is_rag_context:
        expansion_terms.extend([
            'source-rag', 'rag-system', 'RAG backend', 'rag frontend',
            'web page', 'web frontend', 'HTML CSS JS',
            'index.html', 'style.css', 'app.js', 'markdown.js', 'streaming.js',
            'ChatApp', 'StreamHandler', 'NoCacheStaticFiles',
            'frontend layout', 'frontend design', 'frontend/css', 'frontend/js',
            # Voice / ASR files
            'audio.py', 'transcriber', 'Qwen3-ASR', 'ASR', 'voice input',
            'MediaRecorder', '_webmToWav', '_initMic', 'mic-btn',
            'backend/asr', 'backend/routes/audio',
        ])

    if expansion_terms:
        # Deduplicate
        seen = set()
        unique_terms = []
        for t in expansion_terms:
            if t.lower() not in seen:
                unique_terms.append(t)
                seen.add(t.lower())
        expansion_part = ' '.join(unique_terms)
        return f"{question} {expansion_part}"

    return question


def expand_query_llm(question: str) -> str:
    """L2: Use LLM to generate English keywords (cached, toggleable)."""
    if not settings.llm_expansion_enabled:
        return expand_query(question)

    # Try static first as base
    base = expand_query(question)

    try:
        from backend.generation.llm_service import LLMService
        llm = LLMService()
        prompt = f"""将以下中文问题中的技术术语扩展为3-5个英文关键词，以空格分隔。只返回关键词，不返回其他内容。

问题：{question}

关键词："""
        keywords = llm.generate([{"role": "user", "content": prompt}])
        if keywords and len(keywords) < 200:
            return f"{base} {keywords}"
    except Exception:
        pass

    return base
