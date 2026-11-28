"""路由装配层的冒烟测试：``from backend.app import app`` 必须能跑通。

其余测试都是按模块直接调函数，从不 import ``backend.app``，所以「路由装配
阶段失败」这类问题一路漏过去：``backend/routes/audio.py`` 用了
``UploadFile = File(...)``，一旦 ``python-multipart`` 没装，FastAPI 在
``include_router`` 时就抛 "Form data requires python-multipart to be
installed."，应用直接起不来，而 18 条单测仍然全绿。

这条用例锁住整条装配链：应用对象能 import 出来、``app.routes`` 非空、health /
query / audio / index 四条路由真的挂上了、audio 路由按 multipart 解析上传
（就是上面那个依赖点），并且 OpenAPI schema 能生成（会走一遍所有路由依赖）。
不启动服务、不连模型、不需要 API Key。

Run:  pytest -q        （也可以 python -m unittest discover -s tests -v）
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 断言的正是这一行：装配阶段任何依赖缺失都会在这里炸掉
from backend.app import app  # noqa: E402


class AppAssemblyTest(unittest.TestCase):

    def test_app_imports_and_routes_assembled(self):
        self.assertTrue(app.routes, 'app.routes 为空：路由一个都没装上')

        paths = {getattr(r, 'path', '') for r in app.routes}
        for expected in (
            '/api/rag/health',
            '/api/rag/query',
            '/api/rag/index/status',
            '/api/rag/audio/transcribe',
        ):
            self.assertIn(expected, paths, f'{expected} 没有装配上')

        # audio 路由的 File(...) 就是 python-multipart 的隐式依赖
        audio = [
            r for r in app.routes
            if getattr(r, 'path', '') == '/api/rag/audio/transcribe'
        ]
        self.assertEqual(len(audio), 1)
        body_field = audio[0].body_field
        self.assertIsNotNone(body_field, 'audio 路由没有解析出文件上传字段')
        self.assertEqual(
            getattr(body_field.field_info, 'media_type', ''),
            'multipart/form-data',
            'audio 路由没有按 multipart 解析上传（File(...) 未生效）',
        )

        # OpenAPI 生成会遍历全部路由依赖，能建出来说明装配完整
        schema = app.openapi()
        self.assertEqual(schema['info']['title'], 'RAG Code Q&A System')
        self.assertIn('/api/rag/audio/transcribe', schema['paths'])


if __name__ == '__main__':
    unittest.main()
