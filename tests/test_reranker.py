"""精排顺序必须真的改变最终上下文的文件顺序（P0 回归）。

修复前：``rerank()`` 只返回了新顺序，``ContextBuilder.build()`` 仍然按
``rrf_score`` 重排文件，于是这次 DeepSeek 重排对最终上下文完全无效——实测
传 ['low.py', 'high.py'] 和 ['high.py', 'low.py'] 得到的上下文都是
['low.py', 'high.py']。这里锁住三件事：新顺序写回分数、ContextBuilder 会用
这个顺序、调用失败时退回原顺序。

不联网：DeepSeek 的 HTTP 会话整个被 mock 掉。
"""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.generation import context_builder as context_builder_mod  # noqa: E402
from backend.generation.context_builder import ContextBuilder  # noqa: E402
from backend.retrieval import reranker  # noqa: E402


def _candidate(file_path: str, start: int, rrf: float) -> dict:
    return {
        'chunk_id': f'rag:{file_path}:{start}:{start + 5}',
        'content': f'# {file_path} chunk {start}\ndef handler():\n    pass',
        'rrf_score': rrf,
        'metadata': {
            'source': 'rag',
            'file_path': file_path,
            'start_line': start,
            'end_line': start + 5,
        },
    }


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {'choices': [{'message': {'content': self._content}}]}


class RerankOrderTest(unittest.TestCase):

    def _candidates(self) -> list[dict]:
        # a.py 有 11 块（RRF 更高），b.py 有 10 块；总 21 条才会真的调模型
        cands = [_candidate('a.py', i * 10, 1.0 - i * 0.01) for i in range(11)]
        cands += [_candidate('b.py', i * 10, 0.5 - i * 0.01) for i in range(10)]
        return cands

    def _patch_llm(self, content=None, error=None):
        cls = mock.patch.object(reranker.requests, 'Session')
        session_cls = cls.start()
        self.addCleanup(cls.stop)
        if error is not None:
            session_cls.return_value.post.side_effect = error
        else:
            session_cls.return_value.post.return_value = _FakeResponse(content)
        return session_cls

    def test_new_order_is_written_into_rrf_score_and_context_follows_it(self):
        cands = self._candidates()
        # 把原本第 11 名（b.py）排到第一，a.py 的块排在其后
        order = '11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 10'
        self._patch_llm(order)

        ranked = reranker.rerank('这个问题怎么实现', cands, top_n=20)

        self.assertEqual(ranked[0]['metadata']['file_path'], 'b.py')
        scores = [c['rrf_score'] for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         '重排后的 rrf_score 不是递减的，ContextBuilder 会把它排乱')
        self.assertEqual(len(scores), len(set(scores)))

        # ContextBuilder 只认 rrf_score：修复前这里第一个文件会是 a.py
        with mock.patch.object(
                context_builder_mod.ContextBuilder, '_read_source_file',
                lambda self, source, rel, root: f'# {rel}\nVALUE = 1\n'):
            context = ContextBuilder().build(ranked)

        self.assertLess(context.index('文件: b.py'), context.index('文件: a.py'),
                        '精排的新顺序没有影响最终上下文的文件顺序')

    def test_model_failure_keeps_the_rrf_order(self):
        cands = self._candidates()
        self._patch_llm(error=RuntimeError('boom'))

        ranked = reranker.rerank('问题', cands, top_n=20)

        self.assertEqual(ranked[0]['metadata']['file_path'], 'a.py')
        self.assertEqual([c['rrf_score'] for c in ranked],
                         [c['rrf_score'] for c in cands[:20]])

    def test_short_candidate_list_never_calls_the_api(self):
        session_cls = self._patch_llm('0, 1, 2')

        ranked = reranker.rerank('问题', self._candidates()[:5], top_n=20)

        self.assertEqual(len(ranked), 5)
        session_cls.assert_not_called()


if __name__ == '__main__':
    unittest.main()
