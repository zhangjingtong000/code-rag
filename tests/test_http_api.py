"""HTTP 层回归测试：`/api/rag/query` 与 SSE 流。

这两个端点以前只有函数级测试（heartbeat / reranker / context 组装），HTTP 层
没人碰：状态码、metadata 事件、事件格式（`id:` 行）都是空白区。这里用
FastAPI TestClient + mock 掉的检索/生成层跑通「建查询 → 取流 → 收事件」。

离线：向量、BM25、LLM 全部替换成假实现，不联网、不需要 Key。
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.routes import query as query_mod  # noqa: E402


def _write_metadata(path: str, chunks_count: int = 12) -> None:
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'files_count': 3, 'chunks_count': chunks_count,
                   'last_built': 'x', 'sources': {}}, f)


class _FakeLLM:
    instances = []

    def __init__(self, *args, **kwargs):
        _FakeLLM.instances.append(self)
        self.messages = None

    def generate_stream(self, messages, max_tokens=0):
        self.messages = messages
        yield '你好'
        yield '，世界'


class _FakeEmbeddingService:
    def embed_query(self, query):
        return [0.0] * 8


class _FakeVectorStore:
    def search(self, embedding, k=25):
        return []


class _FakeBM25:
    def search(self, query, k=25):
        return []


class QueryHttpTest(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        query_mod._query_store.clear()

    def _settings(self, metadata_path: str):
        return types.SimpleNamespace(
            metadata_path=metadata_path,
            vector_k=5,
            bm25_k=5,
            bm25_path=metadata_path,
        )

    def test_missing_index_is_409(self):
        with tempfile.TemporaryDirectory(prefix='rag-http-') as tmp:
            missing = os.path.join(tmp, 'chunk_metadata.json')
            with mock.patch.object(query_mod, 'settings', self._settings(missing)):
                resp = self.client.post('/api/rag/query', json={'question': '怎么实现'})

        self.assertEqual(resp.status_code, 409, resp.text)

    def test_empty_index_is_409(self):
        with tempfile.TemporaryDirectory(prefix='rag-http-') as tmp:
            meta = os.path.join(tmp, 'chunk_metadata.json')
            _write_metadata(meta, chunks_count=0)
            with mock.patch.object(query_mod, 'settings', self._settings(meta)):
                resp = self.client.post('/api/rag/query', json={'question': '怎么实现'})

        self.assertEqual(resp.status_code, 409, resp.text)

    def test_blank_question_is_400(self):
        with tempfile.TemporaryDirectory(prefix='rag-http-') as tmp:
            meta = os.path.join(tmp, 'chunk_metadata.json')
            _write_metadata(meta)
            with mock.patch.object(query_mod, 'settings', self._settings(meta)):
                resp = self.client.post('/api/rag/query', json={'question': '   '})

        self.assertEqual(resp.status_code, 400, resp.text)

    def test_stream_of_unknown_query_is_404(self):
        resp = self.client.get('/api/rag/query/stream/does-not-exist')
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_create_and_stream_emits_events_with_ids(self):
        _FakeLLM.instances.clear()
        with tempfile.TemporaryDirectory(prefix='rag-http-') as tmp:
            meta = os.path.join(tmp, 'chunk_metadata.json')
            _write_metadata(meta)
            settings = self._settings(meta)

            with mock.patch.object(query_mod, 'settings', settings), \
                    mock.patch.object(query_mod, '_log', lambda *a, **k: None), \
                    mock.patch('backend.retrieval.query_expansion.expand_query_llm',
                               lambda q: q), \
                    mock.patch('backend.indexing.embedding.EmbeddingService',
                               _FakeEmbeddingService), \
                    mock.patch('backend.retrieval.vector_store.VectorStore',
                               _FakeVectorStore), \
                    mock.patch('backend.retrieval.bm25_store.load_cached',
                               lambda path: _FakeBM25()), \
                    mock.patch('backend.retrieval.fusion.hybrid_search',
                               lambda *a, **k: []), \
                    mock.patch('backend.generation.llm_service.LLMService', _FakeLLM):
                created = self.client.post(
                    '/api/rag/query', json={'question': '融合怎么做的'})
                self.assertEqual(created.status_code, 200, created.text)
                payload = created.json()
                self.assertIn('query_id', payload)
                self.assertEqual(payload['stream_url'],
                                 f"/api/rag/query/stream/{payload['query_id']}")

                streamed = self.client.get(payload['stream_url'])

        self.assertEqual(streamed.status_code, 200, streamed.text)
        self.assertEqual(streamed.headers['content-type'].split(';')[0],
                         'text/event-stream')
        body = streamed.text
        self.assertIn('event: metadata', body)
        self.assertIn('event: token', body)
        self.assertIn('event: done', body)
        self.assertIn('id: 1', body, 'SSE 必须发 id:（前端据此记录进度）')
        self.assertIn('你好', body)
        # 流结束后查询从进程内表里清掉
        self.assertNotIn(payload['query_id'], query_mod._query_store)

    def test_stream_reports_model_errors_as_sse_error_event(self):
        with tempfile.TemporaryDirectory(prefix='rag-http-') as tmp:
            meta = os.path.join(tmp, 'chunk_metadata.json')
            _write_metadata(meta)

            boom = mock.Mock(side_effect=RuntimeError('ollama down'))
            with mock.patch.object(query_mod, 'settings', self._settings(meta)), \
                    mock.patch.object(query_mod, '_log', lambda *a, **k: None), \
                    mock.patch('backend.retrieval.query_expansion.expand_query_llm',
                               lambda q: q), \
                    mock.patch('backend.indexing.embedding.EmbeddingService', boom):
                created = self.client.post(
                    '/api/rag/query', json={'question': '融合怎么做的'})
                streamed = self.client.get(created.json()['stream_url'])

        self.assertEqual(streamed.status_code, 200)
        self.assertIn('event: error', streamed.text)
        self.assertIn('ollama down', streamed.text)


if __name__ == '__main__':
    unittest.main()
