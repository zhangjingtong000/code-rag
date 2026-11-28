"""ASR 错误分层：装了 qwen-asr 但没下模型时必须落到 503，不是 500。

旧实现把 ``<project>/models/qwen3-asr-0.6b`` 这个 Windows 绝对路径直接传给
``Qwen3ASRModel.from_pretrained``，HF 认为它不是合法 repo id 而抛 ValueError；
``audio.py`` 只 catch RuntimeError，于是返回 500，README 承诺的 503 落空。
注释里写的「fall back to HF model name」也不存在。

这里锁住三件事：本地目录不存在时退回 HF repo id、加载失败包装成 RuntimeError、
``/audio/transcribe`` 把 RuntimeError → 503、ValueError → 422。

离线：qwen_asr / torch 都是假的。
"""

import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.asr import transcriber  # noqa: E402
from backend.routes import audio as audio_mod  # noqa: E402


def _fake_qwen_asr(record: dict, fail: Exception | None = None):
    module = types.ModuleType('qwen_asr')

    class Qwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            record['model_path'] = model_path
            if fail is not None:
                raise fail
            return cls()

        def transcribe(self, audio):
            return []

    module.Qwen3ASRModel = Qwen3ASRModel
    return module


def _reset_model_cache():
    if hasattr(transcriber._transcribe_python, '_model'):
        del transcriber._transcribe_python._model


class PythonBackendModelPathTest(unittest.TestCase):

    def setUp(self):
        _reset_model_cache()
        self.addCleanup(_reset_model_cache)

    def test_missing_local_model_dir_falls_back_to_hf_repo_id(self):
        record = {}
        fake_torch = types.SimpleNamespace(bfloat16='bfloat16')
        missing_dir = os.path.join(ROOT, 'models', 'definitely-not-downloaded')

        with mock.patch.dict(sys.modules, {'qwen_asr': _fake_qwen_asr(record),
                                          'torch': fake_torch}), \
                mock.patch.object(transcriber.settings, 'asr_model_dir', missing_dir):
            result = transcriber._transcribe_python('audio.wav')

        self.assertEqual(result, '')
        self.assertEqual(record['model_path'], transcriber.settings.asr_model_name,
                         '本地模型不存在时必须退回 HF repo id，不能把 Windows 绝对路径当 repo id')

    def test_existing_local_model_dir_is_used(self):
        record = {}
        fake_torch = types.SimpleNamespace(bfloat16='bfloat16')

        with mock.patch.dict(sys.modules, {'qwen_asr': _fake_qwen_asr(record),
                                          'torch': fake_torch}), \
                mock.patch.object(transcriber.settings, 'asr_model_dir', ROOT), \
                mock.patch.object(transcriber.settings, 'asr_model_name', 'Qwen/whatever'):
            transcriber._transcribe_python('audio.wav')

        self.assertEqual(os.path.normpath(record['model_path']), os.path.normpath(ROOT))

    def test_model_load_failure_is_runtime_error(self):
        record = {}
        fake_torch = types.SimpleNamespace(bfloat16='bfloat16')
        missing_dir = os.path.join(ROOT, 'models', 'nope')

        with mock.patch.dict(
                sys.modules,
                {'qwen_asr': _fake_qwen_asr(record, fail=ValueError("not a valid repo id")),
                 'torch': fake_torch}), \
                mock.patch.object(transcriber.settings, 'asr_model_dir', missing_dir):
            with self.assertRaises(RuntimeError) as ctx:
                transcriber._transcribe_python('audio.wav')

        self.assertIn('not a valid repo id', str(ctx.exception))


class AudioRouteStatusTest(unittest.TestCase):
    """错误分层：后端不可用 503、音频解不开 422、其余 500。"""

    def setUp(self):
        self.client = TestClient(app)

    def _post(self):
        return self.client.post(
            '/api/rag/audio/transcribe',
            files={'file': ('clip.wav', b'RIFF' + b'\x00' * 800, 'audio/wav')},
        )

    def test_unavailable_backend_is_503(self):
        with mock.patch.object(audio_mod, 'transcribe',
                               side_effect=RuntimeError('No ASR backend available')):
            resp = self._post()

        self.assertEqual(resp.status_code, 503, resp.text)
        self.assertIn('ASR', resp.json()['detail'])

    def test_undecodable_audio_is_422_not_500(self):
        with mock.patch.object(audio_mod, 'transcribe',
                               side_effect=ValueError('bad wav header')):
            resp = self._post()

        self.assertEqual(resp.status_code, 422, resp.text)

    def test_unexpected_failure_is_still_500(self):
        with mock.patch.object(audio_mod, 'transcribe',
                               side_effect=OSError('disk gone')):
            resp = self._post()

        self.assertEqual(resp.status_code, 500, resp.text)


if __name__ == '__main__':
    unittest.main()
