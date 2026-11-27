"""End-to-end smoke test: 建索引 → 检索 → 组装上下文。

语料就是本仓库自身的源码；嵌入换成确定性假向量（不连 Ollama），生成层
（LLM）不参与。因此这条链路不需要模型、不需要 API Key、不联网。

Run:  pytest -q        （也可以 python -m unittest discover -s tests -v）
"""

import os
import sys
import gc
import shutil
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402
from backend.generation import context_builder as context_builder_mod  # noqa: E402
from backend.generation.context_builder import ContextBuilder  # noqa: E402
from backend.indexing import indexer as indexer_mod  # noqa: E402
from backend.indexing.indexer import Indexer  # noqa: E402
from backend.retrieval import fusion as fusion_mod  # noqa: E402
from backend.retrieval import vector_store as vector_store_mod  # noqa: E402
from backend.retrieval.bm25_store import BM25Store, tokenize  # noqa: E402
from backend.retrieval.fusion import hybrid_search  # noqa: E402
from backend.retrieval.query_expansion import expand_query  # noqa: E402
from backend.retrieval.vector_store import VectorStore  # noqa: E402

EMBED_DIM = 768  # must match the production embedding model


def fake_embed(texts, dim=EMBED_DIM) -> np.ndarray:
    """Deterministic token-hash embedding — a stand-in for the Ollama model.

    Same input always gives the same vector, and texts sharing tokens get
    non-zero cosine similarity, which is enough to exercise the vector path.
    """
    if isinstance(texts, str):
        texts = [texts]
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            vecs[row, zlib.crc32(token.encode('utf-8')) % dim] += 1.0
        norm = np.linalg.norm(vecs[row])
        if norm:
            vecs[row] /= norm
    return vecs


class _FakeEmbedder:
    """Duck-typed replacement for EmbeddingService — no Ollama involved."""

    def embed(self, texts):
        return fake_embed(texts)

    def embed_query(self, query):
        return fake_embed([query])[0]


class _TempSettings(Settings):
    """Real settings with the data dir and the corpus list pointed elsewhere."""

    def __init__(self, data_dir: str, source_dirs: list):
        super().__init__()
        self._data_dir = data_dir
        self._source_dirs = source_dirs
        # The corpus is the repository itself; skip build artifacts under data/.
        self.exclude_patterns = list(self.exclude_patterns) + ['data']

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def source_dirs(self) -> list:
        return self._source_dirs


class EndToEndSmokeTest(unittest.TestCase):
    """Build a real index out of this repo, then retrieve and build context."""

    def _cleanup(self, path):
        """ChromADB keeps the temp files open on Windows — close before removing."""
        gc.collect()
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)

    def test_index_search_and_context(self):
        tmp = tempfile.mkdtemp(prefix='rag-e2e-')
        cfg = _TempSettings(tmp, [{'path': ROOT, 'label': 'repo'}])

        try:
            with mock.patch.object(indexer_mod, 'settings', cfg), \
                    mock.patch.object(vector_store_mod, 'settings', cfg), \
                    mock.patch.object(fusion_mod, 'settings', cfg), \
                    mock.patch.object(context_builder_mod, 'settings', cfg):

                # ── 1. 建索引（扫描 → 分块 → 假嵌入 → 落盘）──
                builder = Indexer()
                builder.embedder = _FakeEmbedder()
                builder.build_index(mode='full')

                self.assertGreater(builder.vs.count(), 0, '向量库为空')
                self.assertTrue(os.path.exists(cfg.bm25_path), 'BM25 索引未落盘')

                # ── 2. 检索（查询扩展 → 向量 + BM25 → RRF 融合）──
                query = 'chunk_key 分块器 边界'
                expanded = expand_query(query)

                vs = VectorStore()
                vector_hits = vs.search(fake_embed([expanded])[0], k=cfg.vector_k)

                bm25 = BM25Store()
                self.assertTrue(bm25.load(cfg.bm25_path))
                keyword_hits = bm25.search(expanded, k=cfg.bm25_k)

                self.assertTrue(vector_hits, '向量检索没有结果')
                self.assertTrue(keyword_hits, 'BM25 检索没有结果')

                fused = hybrid_search(vector_hits, keyword_hits, question=query)

                # ── 3. 断言至少召回一条，且指向真实文件 ──
                self.assertGreaterEqual(len(fused), 1, '融合后没有任何结果')
                for hit in fused:
                    self.assertTrue(hit['metadata'].get('file_path'), hit)

                top_paths = [h['metadata']['file_path'] for h in fused[:10]]
                self.assertTrue(
                    any(p.endswith('chunker.py') for p in top_paths),
                    f'chunker.py 未进入前 10：{top_paths}',
                )
                self.assertTrue(any(p.endswith('.py') for p in top_paths), top_paths)

                # ── 4. 上下文组装（整文件读取 + 预算控制）──
                context = ContextBuilder().build(fused)
                self.assertIn('项目背景', context)
                self.assertIn('chunk_key', context)
        finally:
            del builder, vs
            self._cleanup(tmp)


class SourceDirsOverrideTest(unittest.TestCase):
    """RAG_SOURCE_DIRS 必须能覆盖默认语料路径，默认值保持原样。"""

    def test_default_source_dirs_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('RAG_SOURCE_DIRS', None)
            dirs = Settings().source_dirs

        self.assertEqual([d['label'] for d in dirs], ['main', 'release', 'rag'])
        for entry in dirs:
            self.assertTrue(
                entry['path'].replace('\\', '/').endswith('data/source-' + entry['label']),
                entry,
            )

    def test_env_override_with_labels(self):
        with mock.patch.dict(os.environ, {'RAG_SOURCE_DIRS': f'corpus={ROOT}'}):
            dirs = Settings().source_dirs

        self.assertEqual(dirs, [{'path': os.path.abspath(ROOT), 'label': 'corpus'}])

    def test_env_override_bare_path_uses_basename(self):
        with mock.patch.dict(os.environ, {'RAG_SOURCE_DIRS': ROOT}):
            dirs = Settings().source_dirs

        self.assertEqual(len(dirs), 1)
        self.assertEqual(dirs[0]['label'], os.path.basename(ROOT.rstrip('\\/')))


if __name__ == '__main__':
    unittest.main()
