"""Offline regression tests for the retrieval plumbing.

Run:  python -m unittest discover -s tests -v

No network, no Ollama, no LLM and no real index are involved.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.indexing.chunker import CodeChunk, chunk_key
from backend.retrieval.bm25_store import BM25Store, load_cached
from backend.retrieval.fusion import dedup_and_limit, detect_query_context, rrf_fusion
from backend.retrieval.query_expansion import expand_query
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


class RagWordBoundaryTest(unittest.TestCase):
    """'rag' 必须是独立单词，不能靠子串命中。"""

    def _rag_boosted(self, question: str) -> bool:
        return 'rag' in detect_query_context(question)['source_boost']

    def test_english_words_containing_rag_do_not_boost(self):
        for question in ('average latency?', 'drag handler', 'fragment cache',
                         'storage size', 'paragraph 3'):
            self.assertFalse(self._rag_boosted(question), question)

    def test_standalone_rag_still_boosts(self):
        for question in ('rag 的融合怎么做', '这个项目的rag在哪', 'RAG pipeline 是什么'):
            self.assertTrue(self._rag_boosted(question), question)


class ExpandQueryWordBoundaryTest(unittest.TestCase):
    """expand_query 是同一链路的另一半：它也不能靠 'rag' 子串判断。

    修复前 fusion 改成了词边界，query_expansion 仍是 `'rag' in ...`，
    于是 "average latency of the search" 会被扩出 source-rag / index.html /
    style.css / app.js 等一串写死的本仓库文件名。这里与 RagWordBoundaryTest
    对称：不命中时不注入，命中时仍然注入，且两条路径必须给出同一结论。
    """

    REPO_TERMS = ('source-rag', 'rag-system', 'index.html', 'app.js', 'transcriber')

    def test_english_words_containing_rag_do_not_inject_repo_files(self):
        for question in ('average latency?', 'drag handler', 'fragment cache',
                         'storage size', 'paragraph 3'):
            expanded = expand_query(question)
            for term in self.REPO_TERMS:
                self.assertNotIn(term, expanded, f'{question!r} 被注入了 {term}')

    def test_standalone_rag_still_injects_repo_files(self):
        for question in ('rag 的前端怎么组织', '这个项目的rag在哪', 'RAG pipeline 是什么'):
            self.assertIn('source-rag', expand_query(question), question)

    def test_question_answering_system_still_injects(self):
        self.assertIn('source-rag', expand_query('问答系统怎么做的'))

    def test_both_halves_agree_on_rag_context(self):
        for question in ('average latency?', 'drag handler', 'fragment cache',
                         'storage size', 'paragraph 3', 'rag 的融合怎么做',
                         '这个项目的rag在哪', '问答系统怎么做的'):
            boosted = 'rag' in detect_query_context(question)['source_boost']
            injected = 'source-rag' in expand_query(question)
            self.assertEqual(boosted, injected, f'{question!r} 两条路径结论不一致')


class ExpandQueryLlmTest(unittest.TestCase):
    """L2 扩展真的接在链路上：默认关闭，开启后按问题缓存、只调一次 LLM。"""

    def test_disabled_is_exactly_static_expansion(self):
        from backend.retrieval import query_expansion as qe

        with mock.patch.object(qe.settings, 'llm_expansion_enabled', False):
            self.assertEqual(
                qe.expand_query_llm('rag 的融合'),
                qe.expand_query('rag 的融合'),
            )

    def test_enabled_calls_llm_once_and_caches(self):
        from backend.retrieval import query_expansion as qe

        calls = []

        class FakeLLM:
            def generate(self, messages):
                calls.append(messages)
                return 'alpha beta'

        with mock.patch.object(qe.settings, 'llm_expansion_enabled', True), \
                mock.patch('backend.generation.llm_service.LLMService', FakeLLM):
            qe._LLM_CACHE.clear()
            first = qe.expand_query_llm('缓存测试问题')
            second = qe.expand_query_llm('缓存测试问题')
            qe._LLM_CACHE.clear()

        self.assertEqual(first, second)
        self.assertIn('alpha beta', first)
        self.assertEqual(len(calls), 1, '同一问题不应重复调用 LLM')


class BM25CacheTest(unittest.TestCase):
    """BM25 索引按 (mtime, size) 在进程内复用，重建后自动失效。"""

    def _write_store(self, path):
        store = BM25Store()
        store.build([make_chunk('rag', 'a.py', 1, 5, 'def alpha():\n    pass')])
        store.save(path)

    def test_reuses_the_loaded_store_until_the_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bm25.pkl')
            self._write_store(path)

            first = load_cached(path)
            self.assertIs(load_cached(path), first, '同一个索引文件被重复加载')

            # build_index 覆盖同一个 pkl：mtime 变了就必须丢掉缓存
            os.utime(path, (time.time() + 30, time.time() + 30))
            self.assertIsNot(load_cached(path), first)


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
