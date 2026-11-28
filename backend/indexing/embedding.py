"""Embedding service — Ollama API wrapper with batch support."""

import json
import urllib.request
import numpy as np

# Build a no-proxy opener for localhost (bypasses Windows system proxy)
_proxy_handler = urllib.request.ProxyHandler({})
_no_proxy_opener = urllib.request.build_opener(_proxy_handler)


class EmbeddingService:

    def __init__(self, config=None):
        if config is None:
            from backend.config import settings as config
        self.base_url = config.ollama_base_url
        self.model = config.ollama_embed_model
        self.batch_size = config.embed_batch_size

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        payload = json.dumps({"model": self.model, "input": texts}).encode('utf-8')
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method='POST',
        )
        with _no_proxy_opener.open(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get("embeddings", [])

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts in batches. Returns (n, dim) ndarray."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                embeddings = self._embed_batch(batch)
                if embeddings and len(embeddings) == len(batch):
                    all_embeddings.extend(embeddings)
                    continue
            except Exception:
                pass

            # Fallback: one by one
            for j, text in enumerate(batch):
                try:
                    emb = self._embed_batch([text])
                    if emb:
                        all_embeddings.extend(emb)
                    else:
                        raise Exception("empty result")
                except Exception:
                    print(f"[Embedding] chunk {i + j} failed, using zeros", flush=True)
                    all_embeddings.append([0.0] * 768)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed([query])[0]
