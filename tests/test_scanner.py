"""文件筛选的回归测试：排除规则不能误伤正常文件。

修复前用「子串包含」判断：``'test_'`` 会把 ``contest_utils.py`` 一起排掉，
``'images'`` 会排掉任何路径里含这个词的文件。现在按路径段精确匹配，只有
以 ``test_`` 开头的**文件名**才排除。
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402
from backend.indexing.scanner import CodeScanner  # noqa: E402


class ExcludeRuleTest(unittest.TestCase):

    def setUp(self):
        self.scanner = CodeScanner(Settings())

    def test_test_prefix_does_not_reject_contest_utils(self):
        self.assertTrue(self.scanner._should_include('src/contest_utils.py'),
                        'contest_utils.py 被 test_ 规则误伤')
        self.assertFalse(self.scanner._should_include('tests/test_chunker.py'),
                         'test_chunker.py 应被排除')

    def test_images_matches_a_directory_not_a_substring(self):
        self.assertFalse(self.scanner._should_include('assets/images/logo.py'))
        self.assertTrue(self.scanner._should_include('src/images_utils.py'))

    def test_exact_pattern_matches_a_segment(self):
        self.assertFalse(self.scanner._should_include('node_modules/pkg/index.js'))

        cfg = Settings()
        cfg.exclude_patterns = list(cfg.exclude_patterns) + ['data']
        scanner = CodeScanner(cfg)

        self.assertFalse(scanner._should_include('data/corpus.py'))
        self.assertTrue(scanner._should_include('backend/metadata.py'),
                        'metadata.py 被 data 规则误伤')

    def test_other_extensions_still_apply(self):
        self.assertFalse(self.scanner._should_include('assets/logo.png'))


if __name__ == '__main__':
    unittest.main()
