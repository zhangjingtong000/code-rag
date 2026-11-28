"""整文件上下文 + 行号的回归测试。

锁住两个「文档说 A、实际做 B」的问题：

1. 快速开始只把 ``--source-dirs "rag=."`` 传给 build_index.py，随后启动的
   服务进程里没有 ``RAG_SOURCE_DIRS``。旧实现回落到默认 ``data/source-rag``，
   目录不存在 → 返回 None → 静默退化成 chunk 拼接，README 承诺的「整份文件」
   并没有兑现。现在构建索引时把语料路径写进 ``chunk_metadata.json``，服务端
   优先读它；真的读不到时打 warning 并在上下文里显式标注。
2. System Prompt 要求每段代码标注行号，但 ContextBuilder 输出的是裸代码块，
   行号只能由模型猜。现在每行左侧带真实行号，截断后仍然对得上。

离线：只读临时目录里的文件，不连模型、不连数据库。
"""

import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402
from backend.generation import context_builder as context_builder_mod  # noqa: E402
from backend.generation.context_builder import ContextBuilder  # noqa: E402

CORPUS_FILE = '\n'.join([
    '"""module docstring."""',
] + [f'LINE_{i} = {i}' for i in range(3, 59)] + [
    'def search(self, query):',
    '    return self._store.search(query)',
] + [f'TAIL_{i} = {i}' for i in range(59, 62)]) + '\n'


def _hit(rel_path: str, start: int, end: int, rrf: float = 0.9, source: str = 'rag') -> dict:
    return {
        'content': f'[文件说明: chunk of {rel_path}]',
        'rrf_score': rrf,
        'metadata': {
            'source': source,
            'file_path': rel_path,
            'start_line': start,
            'end_line': end,
        },
    }


class _TempSettings(Settings):
    """Default corpus dirs (which do not exist) + a temp data dir."""

    def __init__(self, data_dir: str):
        super().__init__()
        self._data_dir = data_dir

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def source_dirs(self) -> list:
        # 默认行为：data/source-<label>，测试里这些目录并不存在，
        # 模拟「启动服务时没有 RAG_SOURCE_DIRS」。
        return [
            {'path': os.path.join(self._data_dir, 'data', 'source-main'), 'label': 'main'},
            {'path': os.path.join(self._data_dir, 'data', 'source-release'), 'label': 'release'},
            {'path': os.path.join(self._data_dir, 'data', 'source-rag'), 'label': 'rag'},
        ]


def _write_metadata(data_dir: str, source_dirs: list) -> None:
    import json
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'chunk_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'chunks_count': 1, 'files_count': 1, 'source_dirs': source_dirs}, f)


class IndexedCorpusPathTest(unittest.TestCase):
    """chunk_metadata.json 里记录的语料路径优先于默认 data/source-<label>。"""

    def test_full_file_is_read_from_index_metadata_without_env(self):
        with tempfile.TemporaryDirectory(prefix='rag-corpus-') as tmp:
            corpus = os.path.join(tmp, 'corpus')
            os.makedirs(corpus)
            with open(os.path.join(corpus, 'mod.py'), 'w', encoding='utf-8') as f:
                f.write(CORPUS_FILE)

            data_dir = os.path.join(tmp, 'data')
            _write_metadata(data_dir, [{'label': 'rag', 'path': corpus}])

            cfg = _TempSettings(data_dir)
            with mock.patch.dict(os.environ, {}, clear=False), \
                    mock.patch.object(context_builder_mod, 'settings', cfg):
                os.environ.pop('RAG_SOURCE_DIRS', None)
                context = ContextBuilder().build([_hit('mod.py', 1, 58)])

        self.assertIn('TAIL_60 = 60', context, '没有读到整文件，退化成了 chunk 拼接')
        self.assertNotIn('chunk of mod.py', context, '上下文里出现了 chunk 拼接内容')
        self.assertNotIn('整文件不可用', context)

    def test_missing_corpus_warns_and_is_marked_in_context(self):
        with tempfile.TemporaryDirectory(prefix='rag-corpus-') as tmp:
            data_dir = os.path.join(tmp, 'data')
            # 元数据里的语料目录早已不存在
            _write_metadata(data_dir, [{'label': 'rag', 'path': os.path.join(tmp, 'gone')}])

            cfg = _TempSettings(data_dir)
            with mock.patch.object(context_builder_mod, 'settings', cfg), \
                    self.assertLogs(context_builder_mod.logger, level=logging.WARNING) as logs:
                context = ContextBuilder().build([_hit('mod.py', 1, 58)])

        self.assertTrue(
            any('chunk' in line and 'mod.py' in line for line in logs.output),
            logs.output,
        )
        self.assertIn('整文件不可用', context, 'fallback 必须显式标注，不能静默')
        self.assertIn('chunk of mod.py', context)


class LineNumberTest(unittest.TestCase):
    """上下文每行左侧必须是源文件的真实行号。"""

    def _build(self, rel_path: str, content: str, hit: dict, **overrides) -> str:
        with tempfile.TemporaryDirectory(prefix='rag-lines-') as tmp:
            # 默认语料目录 data/source-rag 存在时不需要任何元数据
            corpus = os.path.join(tmp, 'data', 'source-rag')
            os.makedirs(corpus)
            with open(os.path.join(corpus, rel_path), 'w', encoding='utf-8') as f:
                f.write(content)

            cfg = _TempSettings(tmp)
            with mock.patch.object(context_builder_mod, 'settings', cfg):
                builder = ContextBuilder()
                for key, value in overrides.items():
                    setattr(builder, key, value)
                return builder.build([hit])

    def test_every_line_carries_its_real_line_number(self):
        context = self._build('mod.py', CORPUS_FILE, _hit('mod.py', 58, 59))

        self.assertRegex(context, r'(?m)^\s*1\s{2}"""module docstring\."""$')
        self.assertRegex(context, r'(?m)^\s*58\s{2}def search\(self, query\):$')
        self.assertRegex(context, r'(?m)^\s*59\s{2}    return self\._store\.search\(query\)$')
        self.assertRegex(context, r'(?m)^\s*60\s{2}TAIL_59 = 59$')

    def test_line_numbers_stay_correct_after_truncation(self):
        content = '\n'.join(f'L{i} = {i}' for i in range(1, 401)) + '\n'
        context = self._build(
            'big.py', content, _hit('big.py', 200, 202),
            expand_lines=5, max_tokens=1500, single_file_truncate=2000)

        self.assertRegex(context, r'(?m)^\s*200\s{2}L200 = 200$')
        self.assertNotRegex(context, r'(?m)^\s*1\s{2}L1 = 1$',
                            '截断后不应保留文件开头')

    def test_truncation_stays_within_budget(self):
        content = '\n'.join(f'L{i} = {i}' for i in range(1, 401)) + '\n'
        context = self._build(
            'big.py', content, _hit('big.py', 200, 202),
            expand_lines=5, max_tokens=1500, single_file_truncate=2000)

        from backend.generation.context_builder import estimate_tokens
        self.assertLessEqual(estimate_tokens(context), 1500)


if __name__ == '__main__':
    unittest.main()
