"""语义标签 / 项目背景外置配置的回归测试。

标签表与背景文案从 ``backend/semantic_tags.json`` 读；文件缺失或损坏必须回退
到内置默认值（索引链路不能因此跑不起来），为此这里同时锁住「读到文件」和
「回退可用」两条路径。

Run:  pytest -q        （也可以 python -m unittest discover -s tests -v）
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import semantic_tags  # noqa: E402
from backend.indexing.indexer import _file_semantic_tag  # noqa: E402


class _FileInfo:
    """Minimal stand-in for scanner.FileInfo (only what the tagger reads)."""

    def __init__(self, relative_path, source='repo', language='python'):
        self.relative_path = relative_path
        self.source = source
        self.language = language


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)


class SemanticConfigTest(unittest.TestCase):
    """默认配置生效、缺失回退、坏文件回退、环境变量覆盖。"""

    def test_default_file_is_used(self):
        cfg = semantic_tags.load_config()

        self.assertEqual(cfg['source'], 'file', f'默认配置未生效: {cfg["path"]}')
        self.assertGreater(len(cfg['file_semantic_tags']), 10, '标签表看起来是空的')
        self.assertIn('## 项目背景', cfg['project_background'])

    def test_default_file_describes_this_repo(self):
        tags = semantic_tags.load_config()['file_semantic_tags']

        self.assertIn('backend/indexing/indexer.py', tags)
        self.assertIn('backend/generation/context_builder.py', tags)

    def test_missing_file_falls_back_to_defaults(self):
        missing = os.path.join(tempfile.mkdtemp(prefix='rag-tags-'), 'nope.json')
        cfg = semantic_tags.load_config(missing)

        self.assertEqual(cfg['source'], 'default')
        self.assertTrue(cfg['file_semantic_tags'], '回退后标签表不能为空')
        self.assertIn('## 项目背景', cfg['project_background'])

    def test_broken_json_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory(prefix='rag-tags-') as tmp:
            path = os.path.join(tmp, 'broken.json')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('{ this is not json')
            cfg = semantic_tags.load_config(path)

        self.assertEqual(cfg['source'], 'default')
        self.assertIn('## 项目背景', cfg['project_background'])

    def test_env_override_points_at_another_project(self):
        with tempfile.TemporaryDirectory(prefix='rag-tags-') as tmp:
            path = os.path.join(tmp, 'other_project.json')
            _write_json(path, {
                'project_background': '## 项目背景\n另一个项目。\n---',
                'file_semantic_tags': {'src/app.py': '另一个项目的入口'},
            })
            with mock.patch.dict(os.environ, {semantic_tags.CONFIG_ENV: path}):
                cfg = semantic_tags.load_config()

        self.assertEqual(cfg['source'], 'file')
        self.assertEqual(cfg['file_semantic_tags'], {'src/app.py': '另一个项目的入口'})
        self.assertIn('另一个项目', cfg['project_background'])


class IndexerTagTest(unittest.TestCase):
    """indexer 的标签取自配置，而不是写死在 Python 里。"""

    def test_indexer_reads_tags_from_config(self):
        with tempfile.TemporaryDirectory(prefix='rag-tags-') as tmp:
            path = os.path.join(tmp, 'tags.json')
            _write_json(path, {'file_semantic_tags': {'src/app.py': '自定义文件说明'}})
            with mock.patch.dict(os.environ, {semantic_tags.CONFIG_ENV: path}):
                hit = _file_semantic_tag(_FileInfo('src/app.py'))
                generic = _file_semantic_tag(_FileInfo('src/unknown.py'))

        self.assertEqual(hit, '[文件说明: 自定义文件说明]')
        self.assertIn('Python后端代码', generic)


if __name__ == '__main__':
    unittest.main()
