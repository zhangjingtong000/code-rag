"""BM25 keyword search store with mixed Chinese/English tokenizer."""

import os
import re
import pickle
from rank_bm25 import BM25Okapi


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


class BM25Store:

    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.documents = []

    def build(self, chunks):
        """Build BM25 index from chunks."""
        self.documents = [ch.content for ch in chunks if ch.content.strip()]
        self.chunk_ids = [self._chunk_key(ch) for ch in chunks if ch.content.strip()]

        if not self.documents:
            self.bm25 = None
            return

        corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

    def _chunk_key(self, chunk) -> str:
        return f"{chunk.source}:{chunk.file_path}:{chunk.start_line}"

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
                    'score': float(scores[idx]),
                })
        return results

    def save(self, path: str):
        """Persist BM25 index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'documents': self.documents,
            'chunk_ids': self.chunk_ids,
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

        if self.documents:
            corpus = [tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(corpus)
        return True
