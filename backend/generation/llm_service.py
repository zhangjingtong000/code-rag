"""LLM service — DeepSeek API with streaming SSE support."""

import json
import requests
from backend.config import settings


class LLMService:

    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model

    def generate(self, messages: list[dict]) -> str:
        """Non-streaming generation (for summaries, etc.)."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']

    def generate_stream(self, messages: list[dict], max_tokens: int = 16384):
        """Streaming generation — yields tokens one at a time."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
            "stop": None,
        }

        session = requests.Session()
        # Bypass proxy for DeepSeek API to avoid connection drops
        session.trust_env = False

        response = session.post(
            url, headers=headers, json=payload,
            stream=True, timeout=(30, 300),
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    choice = data['choices'][0]
                    delta = choice.get('delta', {}).get('content', '')
                    if delta:
                        yield delta
                    # Log finish reason if present
                    finish = choice.get('finish_reason')
                    if finish:
                        print(f"[LLM] finish_reason: {finish}", flush=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
