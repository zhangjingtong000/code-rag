"""上下文组装的回归测试：超预算时要按预期截断，而不是把整文件硬塞进去。

锁住三件事：32K 预算不能超、单个文件按命中行截断后要留「已截断」标注、
预算不够时后面的文件直接不收录（不是悄悄截掉一半塞进去）。切分/组装逻辑
被改坏时（比如预算检查写错、截断不留痕），这几条会立刻失败。
"""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.generation import context_builder as context_builder_mod  # noqa: E402
from backend.generation.context_builder import (  # noqa: E402
    ContextBuilder,
    estimate_tokens,
)
from backend.semantic_tags import load_config  # noqa: E402


def _padding_file(prefix: str, n_lines: int, sentinel: str) -> str:
    """生成一个「每行都很像」的大文件，末行放哨兵用于检测尾部泄漏。"""
    lines = [f'{prefix}_LINE_{i:03d} = {i}' for i in range(n_lines - 1)]
    lines.append(sentinel)
    return '\n'.join(lines) + '\n'


SMALL_FILE = _padding_file('a', 12, 'A_END_SENTINEL = True') + '\n'
BIG_FILE = _padding_file('b', 400, 'B_TAIL_SENTINEL = True')
FILES = {'small.py': SMALL_FILE, 'big.py': BIG_FILE}


def _fake_read_source_file(self, source, rel_path, project_root):
    return FILES.get(rel_path)


def _hit(rel_path: str, start: int, end: int, rrf: float) -> dict:
    return {
        'content': '',
        'rrf_score': rrf,
        'metadata': {
            'source': 'rag',
            'file_path': rel_path,
            'start_line': start,
            'end_line': end,
        },
    }


class ContextBudgetTest(unittest.TestCase):
    """超预算时按命中行截断并标注；预算真的不够时直接不收录后续文件。"""

    def _builder(self, max_tokens: int) -> ContextBuilder:
        builder = ContextBuilder()
        # 收敛截断半径，让测试里的数字可算且不依赖默认 300 行
        builder.expand_lines = 5
        builder.max_tokens = max_tokens
        return builder

    def _base_tokens(self) -> tuple[int, int]:
        header = load_config()['project_background'].rstrip() + '\n'
        return estimate_tokens(header), estimate_tokens(SMALL_FILE)

    def test_truncates_big_file_and_drops_file_when_budget_is_gone(self):
        header_tokens, small_tokens = self._base_tokens()
        results = [
            _hit('small.py', 1, 12, 0.9),   # 小文件：完整进上下文
            _hit('big.py', 100, 104, 0.8),  # 大文件：命中中段，尾部应被截掉
        ]

        # ── 场景一：预算只够装下小文件 → 大文件必须整份不收录 ──
        tight = self._builder(header_tokens + small_tokens + 100)
        with mock.patch.object(
                context_builder_mod.ContextBuilder, '_read_source_file',
                _fake_read_source_file):
            out = tight.build(results)

        self.assertIn('## 项目背景', out)
        self.assertIn('A_END_SENTINEL', out, '小文件应完整收录')
        self.assertNotIn('big.py', out, '预算不够时不应收录后续文件')
        self.assertNotIn('B_TAIL_SENTINEL', out)
        self.assertLessEqual(
            estimate_tokens(out), tight.max_tokens,
            '上下文超出 token 预算')

        # ── 场景二：预算够装截断后的大文件 → 只保留命中行附近并标注 ──
        loose = self._builder(header_tokens + small_tokens + 1000)
        with mock.patch.object(
                context_builder_mod.ContextBuilder, '_read_source_file',
                _fake_read_source_file):
            out = loose.build(results)

        self.assertIn('### [来源: rag | 文件: big.py]', out, '大文件应进上下文')
        self.assertIn('b_LINE_100', out, '命中行必须在保留范围内')
        self.assertIn('已截断', out, '截断处必须显式标注')
        self.assertNotIn('B_TAIL_SENTINEL', out, '大文件尾部应被截掉')
        self.assertLessEqual(
            estimate_tokens(out), loose.max_tokens,
            '截断后仍然超出 token 预算')


if __name__ == '__main__':
    unittest.main()
