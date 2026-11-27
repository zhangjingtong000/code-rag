"""Offline regression tests for the retrieval plumbing.

Run:  python -m unittest discover -s tests -v

No network, no Ollama, no LLM and no real index are involved.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.indexing.chunker import CodeChunk, chunk_key
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.fusion import dedup_and_limit, rrf_fusion
from backend.retrieval.vector_store import VectorStore


def make_chunk(source, path, start, end, content, chunk_type='function'):
    return CodeChunk(
        source=source,
        file_path=path,
        language='python',
        content=content,
        start_line=start,
        end_line=end,
        chunk_type=chunk_type,
    )


class ChunkIdentityTest(unittest.TestCase):
    """Both stores must derive the same id for the same chunk."""

    def test_both_stores_use_the_same_id(self):
        chunk = make_chunk('main', 'app/main.py', 3, 20, 'def handler():\n    pass')

        self.assertEqual(VectorStore._chunk_id(None, chunk), chunk_key(chunk))
        self.assertEqual(BM25Store()._chunk_key(chunk), chunk_key(chunk))

    def test_same_chunk_is_fused_once(self):
        chunk = make_chunk('main', 'app/main.py', 3, 20, 'def handler():\n    pass')
        metadata = {
            'source': 'main', 'file_path': 'app/main.py', 'language': 'python',
            'start_line': 3, 'end_line': 20, 'chunk_type': 'function',
        }
        cid = chunk_key(chunk)
        vector = [{'chunk_id': cid, 'content': chunk.content,
                   'metadata': metadata, 'score': 0.81}]
        bm25 = [{'chunk_id': cid, 'content': chunk.content,
                 'metadata': metadata, 'score': 6.4}]

        merged = rrf_fusion(vector, bm25, k=60)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['chunk_id'], cid)


class BM25MetadataTest(unittest.TestCase):
    """Keyword hits must carry metadata, otherwise the context builder drops them."""

    def _store(self):
        store = BM25Store()
        store.build([
            make_chunk('main', 'app/recognize.py', 1, 12,
                       'def recognize_by_url(url):\n    return detect(url)'),
            make_chunk('release', 'app/tencent.py', 1, 12,
                       'def detect_label(img):\n    return cv(img)'),
            make_chunk('main', 'frontend/app.js', 1, 30,
                       'Page({ onLoad: function () { this.loadHistory(); } })'),
            make_chunk('main', 'frontend/utils/api.js', 1, 40,
                       'function uploadImage(file) { return wx.uploadFile(file); }'),
            make_chunk('rag', 'backend/config.py', 1, 20,
                       'class Settings:\n    port = 8001'),
        ])
        return store

    def test_hits_expose_file_and_line_range(self):
        results = self._store().search('recognize_by_url', k=5)

        self.assertTrue(results)
        self.assertEqual(results[0]['metadata']['file_path'], 'app/recognize.py')
        self.assertEqual(results[0]['metadata']['source'], 'main')
        self.assertEqual(results[0]['metadata']['start_line'], 1)
        self.assertEqual(results[0]['metadata']['end_line'], 12)

    def test_metadata_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bm25.pkl')
            self._store().save(path)

            reloaded = BM25Store()
            self.assertTrue(reloaded.load(path))
            hit = reloaded.search('recognize_by_url', k=5)[0]

        self.assertEqual(hit['metadata']['file_path'], 'app/recognize.py')


class DedupLimitTest(unittest.TestCase):

    def _result(self, path, start, score):
        return {
            'chunk_id': f'main:{path}:{start}:{start + 5}',
            'content': '',
            'metadata': {'source': 'main', 'file_path': path,
                         'start_line': start, 'end_line': start + 5},
            'rrf_score': score,
        }

    def test_same_file_is_capped(self):
        results = [self._result('app/main.py', i * 10, 1.0 - i * 0.1) for i in range(5)]

        self.assertEqual(len(dedup_and_limit(results, max_per_file=3)), 3)

    def test_duplicate_chunk_kept_once(self):
        first = self._result('app/main.py', 10, 1.0)
        duplicate = dict(first, rrf_score=0.5)

        self.assertEqual(len(dedup_and_limit([first, duplicate], max_per_file=3)), 1)


if __name__ == '__main__':
    unittest.main()
