"""ChromaDB vector store wrapper."""

import os
import hashlib
import chromadb
from backend.config import settings
from backend.indexing.chunker import CodeChunk


class VectorStore:

    def __init__(self):
        os.makedirs(settings.chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_id(self, chunk: CodeChunk) -> str:
        raw = f"{chunk.source}:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def add_chunks(self, chunks: list[CodeChunk], embeddings, batch_size: int = 5000):
        """Batch insert chunks with embeddings (ChromaDB has max batch size limit)."""
        ids = [self._chunk_id(ch) for ch in chunks]
        documents = [ch.content for ch in chunks]
        metadatas = [
            {
                "source": ch.source,
                "file_path": ch.file_path,
                "language": ch.language,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "chunk_type": ch.chunk_type,
            }
            for ch in chunks
        ]
        emb_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = ids[start:end]

            # Upsert: remove existing then add
            existing = self.collection.get(ids=batch_ids)
            if existing and existing.get('ids'):
                self.collection.delete(ids=[id for id in batch_ids if id in existing['ids']])

            self.collection.add(
                ids=batch_ids,
                documents=documents[start:end],
                embeddings=emb_list[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  [VS] Batch {start//batch_size + 1}: {start}-{end}/{total}")

    def search(self, query_embedding, k: int = 15) -> list[dict]:
        """Search by vector similarity. Returns list of result dicts."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results.get('ids') and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                output.append({
                    'chunk_id': chunk_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1.0 - results['distances'][0][i],  # cosine → similarity
                })
        return output

    def delete_by_source_key(self, source_key: str):
        """Delete all chunks matching source:relpath pattern."""
        # source_key format: "main:2/pages/index/index.js"
        source, _, relpath = source_key.partition(':')
        results = self.collection.get(where={"$and": [
            {"source": source},
            {"file_path": relpath},
        ]})
        if results and results.get('ids'):
            self.collection.delete(ids=results['ids'])

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete the collection and recreate."""
        try:
            self.client.delete_collection("code_chunks")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"},
        )
