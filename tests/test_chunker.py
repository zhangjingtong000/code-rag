"""结构感知分块的回归测试：Python 文件的切分边界必须落在 def/class 之间。

这是「按代码结构切、不按固定字符数切」的最小证据。把切分退化成定长（比如
每 512 字符一刀）时，下面的断言会直接失败：函数被劈成两块、装饰器与它修饰的
函数被分开、首个 def 之前的模块级代码整段丢失、行号与源码对不上。
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402
from backend.indexing.chunker import CodeChunker, chunk_key  # noqa: E402
from backend.indexing.scanner import FileInfo  # noqa: E402

SOURCE = '''"""示例模块：只用于分块边界回归测试。"""

import os

MAX_BATCH = 8
_BASE = os.sep


def alpha(value):
    """函数 A。"""
    return value + 1


@decorator
def gamma(value):
    return value * 2


class Beta:
    """类 B。"""

    def run(self):
        return MAX_BATCH
'''


def _file_info(content=SOURCE, rel_path='demo/sample.py'):
    return FileInfo(
        source='rag',
        path=os.path.join(ROOT, 'demo', 'sample.py'),
        relative_path=rel_path,
        language='python',
        content=content,
        size=len(content),
        md5='0' * 32,
    )


def _wxml_source(blocks=40):
    """超长 WXML：40 个 <view class=...> 块，必然触发按 view 切分。"""
    lines = ['<template name="page">']
    for i in range(blocks):
        lines.append(f'<view class="card card-{i}">')
        lines.append(f'  <text class="title">标题 {i}</text>')
        lines.append(f'  <text class="body">第 {i} 块内容，用来验证声明的行号能切出同样的文本。</text>')
        lines.append('</view>')
    lines.append('</template>')
    return '\n'.join(lines) + '\n'


class WxmlLineNumberTest(unittest.TestCase):
    """WXML 切块的行号必须指向内容本身，不能整体偏移一行。

    旧实现用 `(?:^|\\n)` 取匹配位置，match.start() 落在换行符上，导致
    start_line 少算一行；40 个块里 39 个与内容错位。Python 用例覆盖不到这条
    路径（所以之前全绿也发现不了），这里单独锁住。
    """

    def setUp(self):
        self.source = _wxml_source()
        self.lines = self.source.split('\n')
        self.chunker = CodeChunker(Settings())
        self.chunks = self.chunker.chunk_file(FileInfo(
            source='rag',
            path=os.path.join(ROOT, 'demo', 'page.wxml'),
            relative_path='demo/page.wxml',
            language='wxml',
            content=self.source,
            size=len(self.source),
            md5='1' * 32,
        ))

    def test_page_is_split_into_many_chunks(self):
        self.assertGreater(len(self.chunks), 30, '超长 WXML 没有被切分')

    def test_declared_line_range_matches_chunk_content(self):
        for chunk in self.chunks:
            sliced = '\n'.join(self.lines[chunk.start_line - 1:chunk.end_line])
            self.assertEqual(
                sliced.strip(), chunk.content.strip(),
                f'行号与内容错位: {chunk.start_line}-{chunk.end_line}')

    def test_start_line_points_at_the_first_content_line(self):
        for chunk in self.chunks:
            first_line = chunk.content.split('\n')[0].strip()
            self.assertEqual(
                self.lines[chunk.start_line - 1].strip(), first_line,
                f'start_line={chunk.start_line} 不是块首行')

    def test_line_ranges_are_inside_the_file(self):
        for chunk in self.chunks:
            self.assertGreaterEqual(chunk.start_line, 1)
            self.assertLessEqual(chunk.end_line, len(self.lines))


class PythonBoundaryTest(unittest.TestCase):
    """Python 按顶层 def/class 切块，块边界与源码行号自洽。"""

    def setUp(self):
        self.chunker = CodeChunker(Settings())
        self.lines = SOURCE.split('\n')
        self.chunks = self.chunker.chunk_file(_file_info())

    def test_boundaries_follow_def_and_class(self):
        contents = [c.content for c in self.chunks]

        # 1) 三种块类型都出现：模块级、函数、类
        types = {c.chunk_type for c in self.chunks}
        self.assertIn('function', types)
        self.assertIn('class', types)
        self.assertIn('module_level', types)

        # 2) 函数不会被劈成两块：含 alpha 的块只有一个，且不含后面 def 的内容
        alpha_chunks = [c for c in self.chunks if 'def alpha' in c.content]
        self.assertEqual(len(alpha_chunks), 1, '同一个函数被切成了多块')
        self.assertIn('return value + 1', alpha_chunks[0].content)
        self.assertNotIn('def gamma', alpha_chunks[0].content)
        self.assertNotIn('class Beta', alpha_chunks[0].content)

        # 3) 装饰器与它修饰的函数留在同一块里
        gamma_chunks = [c for c in self.chunks if 'def gamma' in c.content]
        self.assertEqual(len(gamma_chunks), 1)
        self.assertTrue(
            gamma_chunks[0].content.lstrip().startswith('@decorator'),
            '装饰器被切到了函数外面',
        )

        # 4) 类块整体保留，方法体不被单独拆走
        class_chunks = [c for c in self.chunks if c.chunk_type == 'class']
        self.assertEqual(len(class_chunks), 1)
        self.assertTrue(class_chunks[0].content.lstrip().startswith('class Beta'))
        self.assertIn('def run', class_chunks[0].content)

        # 5) 首个 def 之前的模块级代码必须进索引，不能只留下 import 头
        module_text = '\n'.join(
            c.content for c in self.chunks if c.chunk_type == 'module_level')
        self.assertIn('MAX_BATCH = 8', module_text)
        self.assertIn('_BASE = os.sep', module_text)

        # 6) 每块声明的行号区间切出来的源码，必须与块内容完全一致
        for chunk in self.chunks:
            sliced = '\n'.join(self.lines[chunk.start_line - 1:chunk.end_line])
            self.assertEqual(sliced, chunk.content,
                             f'块行号与源码错位: {chunk.start_line}-{chunk.end_line}')
            self.assertGreaterEqual(chunk.start_line, 1)
            self.assertLessEqual(chunk.end_line, len(self.lines))

        # 7) 同一文件里块 id 互不相同（向量库与 BM25 共用这个 id）
        keys = [chunk_key(c) for c in self.chunks]
        self.assertEqual(len(keys), len(set(keys)))

        # 8) 拼起来不丢内容：所有函数/类定义都在某一块里出现且只出现一次
        for marker in ('def alpha', '@decorator', 'def gamma', 'class Beta',
                       'def run'):
            hits = sum(1 for text in contents if marker in text)
            self.assertEqual(hits, 1, f'{marker} 出现 {hits} 次（应为 1）')


if __name__ == '__main__':
    unittest.main()
