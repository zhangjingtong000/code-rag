"""chunk_metadata.json 的读取容错。

文件被写坏（构建中途断电、手工编辑错误）时，``json.load`` 会直接抛异常，
/health 和 /index/status 变成 HTTP 500，前端只看到「服务未连接」。这两个
端点应当在元数据不可读时按「未构建 / 空索引」处理，而不是把 500 抛给用户。
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.routes import health as health_mod  # noqa: E402


class MetadataReadTest(unittest.TestCase):

    def _write(self, tmp, text):
        path = os.path.join(tmp, 'chunk_metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory(prefix='rag-health-') as tmp:
            self.assertIsNone(
                health_mod._read_metadata(os.path.join(tmp, 'chunk_metadata.json')))

    def test_corrupt_json_returns_none(self):
        with tempfile.TemporaryDirectory(prefix='rag-health-') as tmp:
            path = self._write(tmp, '{ this is not json')
            self.assertIsNone(health_mod._read_metadata(path))

    def test_non_object_json_returns_none(self):
        with tempfile.TemporaryDirectory(prefix='rag-health-') as tmp:
            path = self._write(tmp, '[1, 2, 3]')
            self.assertIsNone(health_mod._read_metadata(path))

    def test_valid_json_is_returned(self):
        with tempfile.TemporaryDirectory(prefix='rag-health-') as tmp:
            path = self._write(
                tmp, '{"chunks_count": 177, "files_count": 35, "last_built": "x"}')
            meta = health_mod._read_metadata(path)

        self.assertEqual(meta['chunks_count'], 177)
        self.assertEqual(meta['files_count'], 35)


if __name__ == '__main__':
    unittest.main()
