"""RRF fusion + dedup + same-file limit + full-file context fill."""

from backend.config import settings


def rrf_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — merge heterogeneous search results."""
    scores = {}

    for rank, r in enumerate(vector_results):
        key = r['chunk_id']
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, r in enumerate(bm25_results):
        key = r['chunk_id']
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    # Merge metadata from both sources
    all_results = {}
    for rlist in [vector_results, bm25_results]:
        for r in rlist:
            key = r['chunk_id']
            if key not in all_results:
                all_results[key] = r

    # Sort by RRF score descending
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


def hybrid_search(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    """Full hybrid search pipeline: RRF → dedup → limit.

    Returns list of result dicts with metadata and content.
    After this, context_builder reads full file content for top-K files.
    """
    merged = rrf_fusion(vector_results, bm25_results, k=settings.rrf_k)
    filtered = dedup_and_limit(merged, max_per_file=settings.max_chunks_per_file)
    return filtered
