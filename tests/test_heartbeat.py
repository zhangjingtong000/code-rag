"""SSE 心跳的回归测试。

修复前心跳检查写在 ``for token in llm.generate_stream(...)`` 循环体内：同步
生成器阻塞在 ``iter_lines()`` 时根本走不到那行，所以「一直没有 token」这一
最需要心跳的场景永远发不出心跳。现在由后台线程读流、主生成器带超时等待。

不联网、不起服务：只测 ``_iter_tokens_with_heartbeat`` 这个纯生成器。
"""

import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.routes.query import _iter_tokens_with_heartbeat  # noqa: E402


class HeartbeatTest(unittest.TestCase):

    def test_heartbeat_is_emitted_while_the_model_is_silent(self):
        def slow_source():
            time.sleep(0.3)
            yield 'hello'

        events = list(_iter_tokens_with_heartbeat(slow_source(), heartbeat_seconds=0.05))
        kinds = [kind for kind, _ in events]

        self.assertIn('heartbeat', kinds, '模型沉默期间没有发出任何心跳')
        self.assertEqual(kinds[-1], 'token')
        self.assertIn(('token', 'hello'), events)

    def test_tokens_pass_through_unchanged(self):
        events = list(_iter_tokens_with_heartbeat(iter(['a', 'b']), heartbeat_seconds=5))

        self.assertEqual(events, [('token', 'a'), ('token', 'b')])

    def test_reader_exception_propagates(self):
        def broken_source():
            yield 'a'
            raise ValueError('llm broke')

        with self.assertRaises(ValueError):
            list(_iter_tokens_with_heartbeat(broken_source(), heartbeat_seconds=5))


if __name__ == '__main__':
    unittest.main()
