"""Indexer orchestrator — scan → chunk → embed → store."""

import os
import sys
import json
import pickle
import hashlib
from datetime import datetime

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.indexing.scanner import CodeScanner
from backend.indexing.chunker import CodeChunker
from backend.indexing.embedding import EmbeddingService
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_store import BM25Store


class Indexer:

    def __init__(self):
        self.scanner = CodeScanner(settings)
        self.chunker = CodeChunker(settings)
        self.embedder = EmbeddingService(settings)
        self.vs = VectorStore()
        self.bm25 = BM25Store()

    def build_index(self, mode: str = 'full'):
        """Build or rebuild the entire index."""
        data_dir = settings.data_dir

        # Load old hashes for incremental detection
        hash_path = settings.filehash_path
        old_hashes = {}
        if mode == 'incremental' and os.path.exists(hash_path):
            with open(hash_path, 'r', encoding='utf-8') as f:
                old_hashes = json.load(f)

        # Scan
        print("[Indexer] Scanning source files...")
        files = self.scanner.scan()

        # Hash tracking
        new_hashes = {}
        changed_files = []
        all_chunks = []
        current = 0

        for fi in files:
            key = f"{fi.source}:{fi.relative_path}"
            new_hashes[key] = fi.md5

            if mode == 'incremental' and key in old_hashes and old_hashes[key] == fi.md5:
                continue  # unchanged

            changed_files.append(fi)

        # Process changed files
        if mode == 'full':
            print(f"[Indexer] Full rebuild: processing all {len(files)} files")
            # Clear existing data
            self.vs.reset()
        else:
            print(f"[Indexer] Incremental: {len(changed_files)}/{len(files)} files changed")

        # Chunk
        for fi in (changed_files if mode == 'incremental' else files):
            chunks = self.chunker.chunk_file(fi)

            # Low-info config files skip vector embedding
            for ch in chunks:
                if ch.chunk_type == 'config':
                    ch.content = ''  # mark for BM25-only

            all_chunks.extend(chunks)

        if not all_chunks and mode == 'incremental':
            # Clean up orphans (deleted source files)
            all_source_keys = set(new_hashes.keys())
            old_source_keys = set(old_hashes.keys())
            orphan_keys = old_source_keys - all_source_keys
            if orphan_keys:
                print(f"[Indexer] Cleaning {len(orphan_keys)} orphan chunks")
                for key in orphan_keys:
                    self.vs.delete_by_source_key(key)

            # Still rebuild BM25 with ALL files (current state)
            all_chunks_for_bm25 = []
            for fi in files:
                all_chunks_for_bm25.extend(self.chunker.chunk_file(fi))
            self._rebuild_bm25(all_chunks_for_bm25)
            self._save_metadata(files, all_chunks_for_bm25)
            self._save_hashes(new_hashes)
            print("[Indexer] Incremental rebuild complete (no new chunks)")
            return

        if not all_chunks:
            print("[Indexer] No chunks to index")
            return

        print(f"[Indexer] {len(all_chunks)} chunks to embed")

        # Embed chunks that need vectors (not config-only)
        embed_chunks = [(i, ch) for i, ch in enumerate(all_chunks) if ch.content]
        embed_texts = [ch.content for _, ch in embed_chunks]
        embed_indices = [i for i, _ in embed_chunks]

        if embed_texts:
            lock_path = os.path.join(data_dir, ".building")
            self._write_progress(lock_path, f"0/{len(embed_texts)}")

            embeddings = None
            total_texts = len(embed_texts)

            # Batch embed with progress
            batch_size = settings.embed_batch_size
            all_embs = []
            for i in range(0, total_texts, batch_size):
                batch = embed_texts[i:i + batch_size]
                try:
                    batch_embs = self.embedder.embed(batch)
                    all_embs.append(batch_embs)
                except Exception as e:
                    print(f"[Indexer] Embed batch failed at {i}: {e}")
                    # Fallback one-by-one
                    for text in batch:
                        try:
                            all_embs.append(self.embedder.embed([text]))
                        except Exception:
                            all_embs.append(self.embedder.embed(["error fallback"]))

                progress = f"{min(i + batch_size, total_texts)}/{total_texts} chunks"
                self._write_progress(lock_path, progress)

            embeddings = np.vstack(all_embs)

            # Store in vector DB
            store_chunks = [all_chunks[i] for i in embed_indices]
            self.vs.add_chunks(store_chunks, embeddings)

        # Rebuild BM25 (with ALL chunks for keyword search, including config-only)
        self._rebuild_bm25(all_chunks)

        # Save metadata
        self._save_metadata(files, all_chunks)
        self._save_hashes(new_hashes)

        lock_path = os.path.join(data_dir, ".building")
        if os.path.exists(lock_path):
            os.remove(lock_path)

        print(f"[Indexer] Index build complete: {len(files)} files, {len(all_chunks)} chunks")

    def _rebuild_bm25(self, all_chunks):
        print("[Indexer] Rebuilding BM25 index...")
        self.bm25.build(all_chunks)
        self.bm25.save(settings.bm25_path)

    def _save_metadata(self, files, chunks):
        meta = {
            'files_count': len(files),
            'chunks_count': len(chunks),
            'last_built': datetime.now().isoformat(),
            'sources': {},
        }
        for ch in chunks:
            src = ch.source
            if src not in meta['sources']:
                meta['sources'][src] = {'files': set(), 'chunks': 0}
            meta['sources'][src]['files'].add(ch.file_path)
            meta['sources'][src]['chunks'] += 1

        # Convert sets to counts
        for src in meta['sources']:
            meta['sources'][src]['files'] = len(meta['sources'][src]['files'])

        with open(settings.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_hashes(self, hashes):
        with open(settings.filehash_path, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)

    def _write_progress(self, lock_path, msg):
        try:
            with open(lock_path, 'w') as f:
                f.write(msg)
        except Exception:
            pass


# Need numpy for vstack
import numpy as np
