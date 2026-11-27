"""Indexer orchestrator — scan → chunk → embed → store."""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.indexing.scanner import CodeScanner
from backend.indexing.chunker import CodeChunker, CodeChunk, chunk_key
from backend.indexing.embedding import EmbeddingService
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_store import BM25Store
from backend.semantic_tags import load_config as _load_semantic_config


class Indexer:

    def __init__(self):
        self.scanner = CodeScanner(settings)
        self.chunker = CodeChunker(settings)
        self.embedder = EmbeddingService(settings)
        self.vs = VectorStore()
        self.bm25 = BM25Store()

    def build_index(self, mode: str = 'full'):
        """Build or rebuild the whole index.

        Chunking is cheap, embedding is not. Every run therefore re-chunks the
        entire corpus (the BM25 corpus must cover every file, otherwise keyword
        search silently shrinks to whatever changed), while only new/changed
        files are embedded and written to the vector store.
        """
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
        new_hashes = {f"{fi.source}:{fi.relative_path}": fi.md5 for fi in files}

        if mode == 'full':
            print(f"[Indexer] Full rebuild: processing all {len(files)} files")
            self.vs.reset()
            changed_keys = set(new_hashes)
        else:
            changed_keys = {k for k, md5 in new_hashes.items() if old_hashes.get(k) != md5}
            print(f"[Indexer] Incremental: {len(changed_keys)}/{len(files)} files changed")

        # Chunk every file: all_chunks feeds BM25, chunks_to_embed feeds vectors
        all_chunks = []
        chunks_to_embed = []

        for fi in files:
            key = f"{fi.source}:{fi.relative_path}"
            chunks = [ch for ch in self.chunker.chunk_file(fi) if ch.content.strip()]

            # Generate semantic tag for this file
            tag_text = _file_semantic_tag(fi)

            for ch in chunks:
                if ch.chunk_type != 'config':
                    # Prepend tag to code content (for BM25 + LLM context).
                    # Config chunks keep their raw text: keyword-only, never embedded.
                    ch.content = tag_text + '\n' + ch.content
            all_chunks.extend(chunks)

            # Create standalone semantic tag chunk (NOT diluted by code)
            # This gives the tag undiluted embedding signal for retrieval
            tag_chunk = CodeChunk(
                source=fi.source,
                file_path=fi.relative_path,
                language='summary',
                content=tag_text,
                start_line=0,
                end_line=0,
                chunk_type='semantic_tag',
                file_md5=fi.md5,
            )
            all_chunks.append(tag_chunk)

            if key in changed_keys:
                chunks_to_embed.extend(chunks)
                chunks_to_embed.append(tag_chunk)

        if not files:
            print("[Indexer] No source files found")

        # Drop vector entries of changed or deleted files before re-adding them,
        # so line ranges that no longer exist stop being retrievable.
        if mode == 'incremental':
            stale_keys = changed_keys | (set(old_hashes) - set(new_hashes))
            if stale_keys:
                print(f"[Indexer] Dropping stale chunks of {len(stale_keys)} files")
            for key in stale_keys:
                self.vs.delete_by_source_key(key)

        # Embed only new/changed chunks (config chunks are keyword-only)
        embed_items = [ch for ch in chunks_to_embed if ch.chunk_type != 'config']
        if embed_items:
            print(f"[Indexer] {len(embed_items)} chunks to embed")

            lock_path = os.path.join(data_dir, ".building")
            self._write_progress(lock_path, f"0/{len(embed_items)}")

            pairs = []
            total = len(embed_items)
            batch_size = settings.embed_batch_size

            for i in range(0, total, batch_size):
                batch = embed_items[i:i + batch_size]
                try:
                    batch_embs = self.embedder.embed([ch.content for ch in batch])
                    if len(batch_embs) != len(batch):
                        raise ValueError(
                            f"embedding count mismatch: {len(batch_embs)} != {len(batch)}"
                        )
                    pairs.extend(zip(batch, batch_embs))
                except Exception as e:
                    print(f"[Indexer] Embed batch failed at {i}: {e}")
                    # Fallback one-by-one; a still-failing chunk gets a zero vector
                    for ch in batch:
                        try:
                            emb = self.embedder.embed([ch.content])[0]
                        except Exception:
                            emb = np.zeros(768, dtype=np.float32)
                        pairs.append((ch, emb))

                self._write_progress(lock_path, f"{min(i + batch_size, total)}/{total} chunks")

            # Deduplicate chunks sharing an id (e.g. semantic_tag vs. a
            # module_level chunk at line 0)
            seen_ids = set()
            unique_pairs = []
            for ch, emb in pairs:
                cid = chunk_key(ch)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                unique_pairs.append((ch, emb))
            if len(unique_pairs) < len(pairs):
                print(f"[Indexer] Dedup: {len(pairs)} -> {len(unique_pairs)} unique chunks")

            embeddings = np.vstack([emb for _, emb in unique_pairs])
            self.vs.add_chunks([ch for ch, _ in unique_pairs], embeddings)

        # Rebuild BM25 from the whole corpus — keyword search must not depend on
        # which files happened to change
        self._rebuild_bm25(all_chunks)

        # Save metadata
        self._save_metadata(files, all_chunks)
        self._save_hashes(new_hashes)

        lock_path = os.path.join(data_dir, ".building")
        if os.path.exists(lock_path):
            os.remove(lock_path)

        print(f"[Indexer] Index build complete: {len(files)} files, "
              f"{len(all_chunks)} chunks ({len(embed_items)} embedded)")

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


# ── File-level semantic tags ──
# The tag table lives in backend/semantic_tags.json: change that file to describe
# a different corpus, no Python edit needed. Missing/broken file falls back to the
# built-in defaults in backend.semantic_tags, so indexing still works.


def _file_semantic_tag(file_info, chunk=None) -> str:
    """Generate a natural-language tag describing a file's purpose.

    Returns Chinese text that gets embedded as a standalone semantic chunk
    AND prepended to code chunks (for BM25 + LLM readability).
    """
    rel = file_info.relative_path.replace('\\', '/')
    source = file_info.source
    tags = _load_semantic_config()['file_semantic_tags']

    # Try source-specific tag first (e.g. 'release::app/config.py'), then generic
    source_key = f'{source}::{rel}'
    if source_key in tags:
        return f'[文件说明: {tags[source_key]}]'
    if rel in tags:
        return f'[文件说明: {tags[rel]}]'
    lang = file_info.language

    # Generic by language
    if lang == 'python':
        return f'[文件说明: Python后端代码 来源:{source} 路径:{rel}]'
    if lang == 'javascript':
        return f'[文件说明: JavaScript前端逻辑代码 来源:{source} 路径:{rel}]'
    if lang in ('html', 'wxml'):
        return f'[文件说明: 前端页面模板 来源:{source} 路径:{rel}]'
    if lang in ('css', 'wxss'):
        return f'[文件说明: 前端样式表 来源:{source} 路径:{rel}]'
    if lang == 'json':
        return f'[文件说明: JSON配置文件 来源:{source} 路径:{rel}]'

    return f'[文件说明: 源代码文件 来源:{source} 路径:{rel}]'
