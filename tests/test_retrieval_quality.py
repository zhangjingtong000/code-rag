"""检索质量指标：recall@5 与 MRR，语料就是本仓库源码。

README 之前只有「57 个单测」，断言的是「不会崩」，没有任何命中率/质量指标。
这里给一条可复现的下界：只用 BM25（不走向量、不调 LLM），对 10 个中文问题
检查期望文件是否进前 5，并算 MRR。纯离线，几秒跑完。

阈值取实测值留出余量（实测 recall@5 = 0.90、MRR = 0.708）。真实链路是
BM25 + 向量 + RRF，只会比这个下界更好；这个数字掉下来就说明切块、语义标签
或查询扩展被改坏了。
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402
from backend.indexing.chunker import CodeChunker  # noqa: E402
from backend.indexing.indexer import _file_semantic_tag  # noqa: E402
from backend.indexing.scanner import CodeScanner  # noqa: E402
from backend.retrieval.bm25_store import BM25Store  # noqa: E402
from backend.retrieval.query_expansion import expand_query  # noqa: E402

# (中文问题, 期望命中的文件)。问题刻意用用户会说的话，而不是文件里的标识符。
QUESTIONS = [
    ('BM25 和向量怎么融合', 'backend/retrieval/fusion.py'),
    ('上下文超出预算时怎么截断', 'backend/generation/context_builder.py'),
    ('索引增量构建只嵌入改过的文件', 'backend/indexing/indexer.py'),
    ('SSE 心跳怎么发', 'backend/routes/query.py'),
    ('查询扩展术语映射表', 'backend/retrieval/query_expansion.py'),
    ('语音转写调用子进程', 'backend/asr/transcriber.py'),
    ('chunk_key 怎么保证两套索引一致', 'backend/indexing/chunker.py'),
    ('向量库封装 search', 'backend/retrieval/vector_store.py'),
    ('全量重建时清空向量库', 'backend/indexing/indexer.py'),
    ('健康检查返回索引状态', 'backend/routes/health.py'),
]

MIN_RECALL_AT_5 = 0.8
MIN_MRR = 0.6


def _build_repo_bm25() -> BM25Store:
    """本仓库源码 → 切块（带语义标签）→ BM25。与 Indexer 的内容一致。"""

    class _RepoSettings(Settings):
        @property
        def source_dirs(self) -> list:
            # 语料固定为本仓库自身，不受运行环境里的 RAG_SOURCE_DIRS 影响
            return [{'path': ROOT, 'label': 'repo'}]

    cfg = _RepoSettings()
    cfg.exclude_patterns = list(cfg.exclude_patterns) + ['data']

    chunker = CodeChunker(cfg)
    chunks = []
    for fi in CodeScanner(cfg).scan():
        tag = _file_semantic_tag(fi)
        for ch in chunker.chunk_file(fi):
            if not ch.content.strip():
                continue
            if ch.chunk_type != 'config':
                ch.content = tag + '\n' + ch.content
            chunks.append(ch)

    store = BM25Store()
    store.build(chunks)
    return store


class RetrievalQualityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store = _build_repo_bm25()
        cls.ranks = {}
        for question, expected in QUESTIONS:
            hits = cls.store.search(expand_query(question), k=10)
            paths = [h['metadata'].get('file_path', '') for h in hits]
            cls.ranks[question] = next(
                (i + 1 for i, p in enumerate(paths) if p.endswith(expected)), None)

    def test_recall_at_5(self):
        hits = sum(1 for r in self.ranks.values() if r and r <= 5)
        recall = hits / len(QUESTIONS)
        self.assertGreaterEqual(
            recall, MIN_RECALL_AT_5,
            f'recall@5={recall:.2f} < {MIN_RECALL_AT_5}；明细：{self.ranks}')

    def test_mrr(self):
        mrr = sum(1.0 / r if r else 0.0 for r in self.ranks.values()) / len(QUESTIONS)
        self.assertGreaterEqual(
            mrr, MIN_MRR, f'MRR={mrr:.3f} < {MIN_MRR}；明细：{self.ranks}')

    def test_every_question_has_a_hit_beyond_the_tail(self):
        """至少 9/10 的问题在 top-10 里找到期望文件（回归时能定位到具体问题）。"""
        found = sum(1 for r in self.ranks.values() if r)
        self.assertGreaterEqual(found, 9, self.ranks)


if __name__ == '__main__':
    unittest.main()
