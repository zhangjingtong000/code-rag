"""Cross-encoder re-ranker using DeepSeek — universal quality boost for all query types."""

import requests
from backend.config import settings


def rerank(query: str, candidates: list[dict], top_n: int = 15) -> list[dict]:
    """Re-rank candidates using DeepSeek as a cross-encoder.

    Takes top-K candidates from RRF fusion, asks DeepSeek to rank them
    by relevance to the query, returns re-ordered top-N.

    Works for ALL question types without domain-specific rules.
    """
    if len(candidates) <= top_n:
        return candidates

    # Build candidate list with IDs and content summaries
    items = []
    for i, c in enumerate(candidates):
        meta = c.get('metadata', {})
        content = (c.get('content', '') or '')[:300]  # first 300 chars for context
        items.append({
            'id': i,
            'source': meta.get('source', ''),
            'file': meta.get('file_path', ''),
            'preview': content.replace('\n', ' ')[:200],
        })

    # Build prompt
    lines = [f"{it['id']}. [{it['source']}:{it['file']}] {it['preview']}" for it in items]
    candidates_text = '\n'.join(lines)

    prompt = f"""你是一个代码检索排序助手。根据用户问题，对下列代码片段按相关性从高到低排序。

## 用户问题
{query}

## 候选代码片段
{candidates_text}

## 指令
选出与问题最相关的片段，只返回它们的序号，用逗号分隔（如: 3, 7, 1, 12, 5）。返回 {top_n} 个序号。只返回序号，不要其他内容。"""

    try:
        session = requests.Session()
        session.trust_env = False

        resp = session.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.deepseek_api_key}",
            },
            json={
                "model": settings.deepseek_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()['choices'][0]['message']['content'].strip()

        # Parse ranked IDs
        ranked_ids = []
        for part in result.split(','):
            part = part.strip().strip('.')
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates) and idx not in ranked_ids:
                    ranked_ids.append(idx)

        if len(ranked_ids) < 3:
            return candidates[:top_n]  # fallback to original order

        # Reorder candidates by ranked IDs, append unranked at end
        reranked = [candidates[i] for i in ranked_ids if i < len(candidates)]
        seen = set(ranked_ids)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)

        reranked = reranked[:top_n]

        # ContextBuilder 只在文件组内取 max(rrf_score) 再排序，它看不到这里的
        # 新顺序。不把这层顺序写回分数，重排就只起到「截到 top_n」的作用，对
        # 最终上下文没有任何影响。按新顺序重写成严格递减的分数（与 RRF 同量级）。
        for rank, candidate in enumerate(reranked):
            candidate['rrf_score'] = 1.0 / (rank + 1)

        return reranked

    except Exception as e:
        print(f"[Reranker] failed: {e}, returning original order", flush=True)
        return candidates[:top_n]
