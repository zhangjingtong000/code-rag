"""BM25 keyword search store with mixed Chinese/English tokenizer."""

import os
import pickle
from rank_bm25 import BM25Okapi

from backend.indexing.chunker import chunk_key


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English code text.

    Splits on whitespace/punctuation but keeps Chinese chars as individual tokens,
    English identifiers intact, and handles code identifiers like recognizeByUrl.
    """
    if not text:
        return []

    # Normalize
    text = text.lower()

    tokens = []
    current_token = ''

    for ch in text:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            # Chinese character: flush current, add as separate token
            if current_token:
                tokens.append(current_token)
                current_token = ''
            tokens.append(ch)
        elif ch.isalnum() or ch == '_':
            current_token += ch
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            # Skip punctuation/whitespace unless meaningful
            if ch in '.-></:':
                continue

    if current_token:
        tokens.append(current_token)

    return tokens


def _chunk_metadata(chunk) -> dict:
    """Metadata carried by every BM25 hit.

    Without it the context builder cannot tell which file / line range a keyword
    hit points at, and silently drops the hit.
    """
    return {
        'source': chunk.source,
        'file_path': chunk.file_path,
        'language': chunk.language,
        'start_line': chunk.start_line,
        'end_line': chunk.end_line,
        'chunk_type': chunk.chunk_type,
    }


class BM25Store:

    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.documents = []
        self.metadatas = []

    def build(self, chunks):
        """Build BM25 index from chunks."""
        kept = [ch for ch in chunks if ch.content.strip()]
        self.documents = [ch.content for ch in kept]
        self.chunk_ids = [self._chunk_key(ch) for ch in kept]
        self.metadatas = [_chunk_metadata(ch) for ch in kept]

        if not self.documents:
            self.bm25 = None
            return

        corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

    def _chunk_key(self, chunk) -> str:
        # Same id as VectorStore._chunk_id — see chunker.chunk_key
        return chunk_key(chunk)

    def search(self, query: str, k: int = 15) -> list[dict]:
        """Search BM25 for query. Returns list of result dicts."""
        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'chunk_id': self.chunk_ids[idx],
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                    'score': float(scores[idx]),
                })
        return results

    def save(self, path: str):
        """Persist BM25 index to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'documents': self.documents,
            'chunk_ids': self.chunk_ids,
            'metadatas': self.metadatas,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load BM25 index from disk."""
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.chunk_ids = data['chunk_ids']
        # Indexes built before metadata was persisted degrade to empty metadata.
        self.metadatas = data.get('metadatas') or [{} for _ in self.documents]

        if self.documents:
            corpus = [tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(corpus)
        return True
