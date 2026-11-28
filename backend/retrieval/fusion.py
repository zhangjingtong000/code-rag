"""RRF fusion + source-aware boosting + dedup + same-file limit + full-file context fill."""

import re

from backend.config import settings


# ── Source / language boost multipliers ──

# Default: no boost
DEFAULT_BOOST = 1.0

# `'rag' in question` matched "average latency?", "drag handler" and
# "fragment cache" and doubled every rag-source chunk. The lookarounds keep
# "rag" as a standalone word while still allowing Chinese neighbours
# ("这个项目的rag在哪"), which `\b` does not (Chinese chars are word chars).
_RAG_WORD = re.compile(r'(?<![a-z0-9_])rag(?![a-z0-9_])')

# When query mentions specific source, boost chunks from that source
SOURCE_BOOST = {
    'rag': 2.0,      # "rag" in query → rag chunks boosted 2x in RRF
    'main': 1.5,     # "main分支" → main chunks boosted
    'release': 1.5,  # "release分支" → release chunks boosted
}

# When query asks about frontend/UI, boost HTML/CSS/JS files
LANG_BOOST = {
    'html': 1.8,
    'css': 1.8,
    'javascript': 1.5,
    'wxml': 1.3,
    'wxss': 1.3,
}


def detect_query_context(question: str) -> dict:
    """Analyze the question to determine boost preferences.

    Returns a dict with:
      - source_boost: {source_label: multiplier} or None
      - lang_boost: {lang: multiplier} or None
      - is_frontend_q: bool
    """
    q_lower = question.lower()
    result = {
        'source_boost': {},
        'lang_boost': {},
        'is_frontend_q': False,
    }

    # ── Source detection ──
    if _RAG_WORD.search(q_lower) or any(w in q_lower for w in ('问答系统', '检索系统')):
        result['source_boost']['rag'] = SOURCE_BOOST['rag']
    if any(w in q_lower for w in ('main分支', 'main 分支', '主分支')):
        result['source_boost']['main'] = SOURCE_BOOST['main']
    if any(w in q_lower for w in ('release分支', 'release 分支', '发布分支')):
        result['source_boost']['release'] = SOURCE_BOOST['release']

    # ── Frontend detection ──
    if any(w in question for w in ('前端', '界面', 'UI', '页面布局', '设计风格', '样式', 'CSS', 'HTML')):
        result['is_frontend_q'] = True
        result['lang_boost']['html'] = LANG_BOOST['html']
        result['lang_boost']['css'] = LANG_BOOST['css']
        result['lang_boost']['javascript'] = LANG_BOOST['javascript']
        result['lang_boost']['wxml'] = LANG_BOOST['wxml']
        result['lang_boost']['wxss'] = LANG_BOOST['wxss']

    # ── Backend/API detection ──
    if any(w in question for w in ('API', '端点', '路由', '后端', 'FastAPI', '接口', '中间件')):
        result['lang_boost']['python'] = 1.5

    return result


def boost_results(results: list[dict], context: dict) -> list[dict]:
    """Apply source and language boosts to result scores.

    The boost is applied to a synthetic 'boost_score' field that RRF uses
    as a multiplier on the reciprocal rank contribution.
    """
    for r in results:
        meta = r.get('metadata', {})
        boost = DEFAULT_BOOST

        # Source boost
        src = meta.get('source', '')
        src_boost = context.get('source_boost', {})
        if src in src_boost:
            boost *= src_boost[src]

        # Language boost
        lang = meta.get('language', '')
        lang_boost = context.get('lang_boost', {})
        if lang in lang_boost:
            boost *= lang_boost[lang]

        r['_boost'] = boost
    return results


def rrf_fusion(vector_results: list[dict], bm25_results: list[dict],
               k: int = 60, context: dict = None) -> list[dict]:
    """RRF with optional per-chunk boost multipliers.

    boosted_rrf = Σ (boost / (k + rank))
    """
    scores = {}

    for rank, r in enumerate(vector_results):
        key = r['chunk_id']
        b = r.get('_boost', DEFAULT_BOOST)
        scores[key] = scores.get(key, 0) + b / (k + rank + 1)

    for rank, r in enumerate(bm25_results):
        key = r['chunk_id']
        b = r.get('_boost', DEFAULT_BOOST)
        scores[key] = scores.get(key, 0) + b / (k + rank + 1)

    # Merge metadata
    all_results = {}
    for rlist in [vector_results, bm25_results]:
        for r in rlist:
            key = r['chunk_id']
            if key not in all_results:
                all_results[key] = r

    # Sort by boosted RRF score descending
    merged = []
    for key, score in scores.items():
        if key in all_results:
            entry = dict(all_results[key])
            entry['rrf_score'] = score
            merged.append(entry)

    merged.sort(key=lambda x: x['rrf_score'], reverse=True)
    return merged


def dedup_and_limit(results: list[dict], max_per_file: int = 3) -> list[dict]:
    """Remove duplicates and enforce per-file chunk limit."""
    seen_chunks = set()
    file_counts = {}
    output = []

    for r in results:
        cid = r['chunk_id']
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)

        meta = r.get('metadata', {})
        fpath = meta.get('file_path', '')
        if file_counts.get(fpath, 0) >= max_per_file:
            continue

        file_counts[fpath] = file_counts.get(fpath, 0) + 1
        output.append(r)

    return output


def hybrid_search(vector_results: list[dict], bm25_results: list[dict],
                  question: str = '') -> list[dict]:
    """Full hybrid search pipeline with source-aware boosting.

    1. Detect query context (source/language preferences)
    2. Apply boosts to both result sets
    3. RRF fusion with boosted scores
    4. Dedup + per-file limit

    Returns list of result dicts with metadata and content.
    """
    context = detect_query_context(question)

    vector_results = boost_results(vector_results, context)
    bm25_results = boost_results(bm25_results, context)

    merged = rrf_fusion(vector_results, bm25_results,
                        k=settings.rrf_k, context=context)
    filtered = dedup_and_limit(merged, max_per_file=settings.max_chunks_per_file)
    return filtered
