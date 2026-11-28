"""路由装配层的冒烟测试：``from backend.app import app`` 必须能跑通。

其余测试都是按模块直接调函数，从不 import ``backend.app``，所以「路由装配
阶段失败」这类问题一路漏过去：``backend/routes/audio.py`` 用了
``UploadFile = File(...)``，一旦 ``python-multipart`` 没装，FastAPI 在
``include_router`` 时就抛 "Form data requires python-multipart to be
installed."，应用直接起不来，而 18 条单测仍然全绿。

断言只看稳定接口：``app.routes`` 非空、OpenAPI schema 里四条路由都在、
audio 路由的请求体是 ``multipart/form-data``（就是上面那个依赖点）。
不直接翻 ``app.routes`` 里的 APIRoute——FastAPI 0.165 起 ``include_router``
改成用 ``_IncludedRouter`` 包一层，子路由不再摊平进 ``app.routes``。

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

        schema = app.openapi()
        self.assertEqual(schema['info']['title'], 'RAG Code Q&A System')

        paths = schema['paths']
        for expected in (
            '/api/rag/health',
            '/api/rag/index/status',
            '/api/rag/index/rebuild',
            '/api/rag/query',
            '/api/rag/audio/transcribe',
        ):
            self.assertIn(expected, paths, f'{expected} 没有装配上')

        # audio 路由的 File(...) 就是 python-multipart 的隐式依赖：
        # 解析成 multipart/form-data 才说明 File(...) 真的生效了
        audio_post = paths['/api/rag/audio/transcribe']['post']
        content = audio_post.get('requestBody', {}).get('content', {})
        self.assertIn(
            'multipart/form-data', content,
            'audio 路由没有按 multipart 解析上传（File(...) 未生效）',
        )


if __name__ == '__main__':
    unittest.main()
