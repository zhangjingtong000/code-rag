"""Query expansion — Chinese term mapping + optional LLM expansion."""

from backend.config import settings

# Domain-specific Chinese → English term mapping
TERM_MAP = {
    '登录': ['login', 'auth', 'authentication', 'signin'],
    '注册': ['register', 'signup'],
    '鉴权': ['auth', 'authorization', 'authenticate', 'appkey'],
    '识别': ['recognize', 'recognition', 'detect', 'classification'],
    '场景': ['scene', 'scenery', 'scenario', 'scenery recognition'],
    '标签': ['tag', 'label', 'classification', 'label detection'],
    '分类': ['category', 'classification', 'classify'],
    '图片': ['image', 'picture', 'photo', 'image recognition'],
    '上传': ['upload', 'file upload', 'chooseMedia'],
    '压缩': ['compress', 'resize', 'compression', 'compressImage'],
    '文字': ['text', 'ocr', 'character', 'text recognition'],
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
    '流式': ['stream', 'SSE', 'streaming', 'EventSource'],
    '分块': ['chunk', 'chunking', 'split'],
    '嵌入': ['embedding', 'embed', 'vector'],
    '检索': ['retrieval', 'search', 'retrieve'],
    '融合': ['fusion', 'RRF', 'merge'],
    '生成': ['generation', 'generate', 'completion'],
    '上下文': ['context', 'contextual'],
    '提示词': ['prompt', 'system prompt'],
    '分支': ['branch', 'main', 'release'],
    '火山': ['volcano', 'volcengine', 'doubao'],
    '腾讯云': ['tencent cloud', 'cloudbase', 'tencent'],
    '模型': ['model', 'LLM', 'deepseek', 'doubao'],
    '安全': ['security', 'moderate', 'imgSecCheck'],
    '审核': ['moderate', 'moderation', 'security check'],
}


def expand_query(question: str) -> str:
    """Expand a Chinese question with English term equivalents.

    Returns the expanded query string for both vector and BM25 search.
    """
    expansion_terms = []

    # L1: Static mapping
    for zh_term, en_terms in TERM_MAP.items():
        if zh_term in question:
            expansion_terms.extend(en_terms[:3])  # take top 3 to avoid explosion

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
