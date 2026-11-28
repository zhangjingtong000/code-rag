"""服务默认值的安全回归：不显式设置时只监听本机。

全链路没有鉴权，绑定 ``0.0.0.0`` 等于把索引里的源码、DeepSeek 额度和音频
上传接口暴露给整个网段。默认值必须是 ``127.0.0.1``，要对外服务得显式设置
``RAG_HOST``。
"""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config import Settings  # noqa: E402


class ServerDefaultsTest(unittest.TestCase):

    def test_default_host_is_loopback(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('RAG_HOST', None)
            self.assertEqual(Settings().host, '127.0.0.1')

    def test_rag_host_env_override(self):
        with mock.patch.dict(os.environ, {'RAG_HOST': '0.0.0.0'}):
            self.assertEqual(Settings().host, '0.0.0.0')

    def test_default_port_is_8001(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('RAG_PORT', None)
            self.assertEqual(Settings().port, 8001)

    def test_rag_port_env_override(self):
        with mock.patch.dict(os.environ, {'RAG_PORT': '9000'}):
            self.assertEqual(Settings().port, 9000)


if __name__ == '__main__':
    unittest.main()
