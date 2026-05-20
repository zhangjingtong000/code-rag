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
        semantic_chunks = []  # standalone tag-only chunks for high-precision retrieval
        for fi in (changed_files if mode == 'incremental' else files):
            chunks = self.chunker.chunk_file(fi)

            # Generate semantic tag for this file
            tag_text = _file_semantic_tag(fi)

            for ch in chunks:
                if ch.chunk_type == 'config':
                    ch.content = ''  # mark for BM25-only
                else:
                    # Prepend tag to code content (for BM25 + LLM context)
                    ch.content = tag_text + '\n' + ch.content

            # Create standalone semantic tag chunk (NOT diluted by code)
            # This gives the tag undiluted embedding signal for retrieval
            from backend.indexing.chunker import CodeChunk
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
            semantic_chunks.append(tag_chunk)

            all_chunks.extend(chunks)

        # Add semantic chunks AFTER regular chunks so they get embedded too
        all_chunks.extend(semantic_chunks)

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


# ── File-level semantic tags ──
# These provide natural-language context so the embedding model understands
# WHAT a file is, not just what code it contains.

_FILE_SEMANTIC_TAGS = {
    # RAG system frontend
    'frontend/index.html': 'RAG问答系统前端主页面 聊天界面布局 左侧问答记录栏 中间对话区 右侧快捷提问面板 三栏布局',
    'frontend/css/style.css': 'RAG系统前端样式表 浅蓝白配色设计 CSS变量体系 响应式布局 代码块样式 消息气泡 三栏布局',
    'frontend/js/app.js': 'RAG系统前端核心聊天逻辑 ChatApp类 消息管理 发送消息 流式响应处理 侧栏问题追踪 点击跳转',
    'frontend/js/streaming.js': 'RAG系统SSE流式处理器 StreamHandler 打字机缓冲效果 16ms定时器平滑渲染 流中断重试',
    'frontend/js/markdown.js': 'RAG系统Markdown渲染模块 代码语法高亮 代码折叠收起 一键复制 来源引用标注',
    # RAG system backend
    'backend/app.py': 'RAG系统FastAPI应用入口 CORS中间件 静态文件挂载 NoCacheStaticFiles 启动事件',
    'backend/config.py': 'RAG系统全局配置 Settings数据类 模型参数 分块参数 检索参数 上下文预算 路径配置',
    'backend/indexing/scanner.py': 'RAG系统文件扫描器 CodeScanner 双目录扫描 文件过滤 语言检测 MD5哈希',
    'backend/indexing/chunker.py': 'RAG系统代码分块器 CodeChunker 结构感知分块 Python/JS/WXML/WXSS 函数类边界',
    'backend/indexing/embedding.py': 'RAG系统嵌入服务 EmbeddingService Ollama Jina v2模型 批量嵌入 代理绕过',
    'backend/indexing/indexer.py': 'RAG系统索引构建器 Indexer 扫描分块嵌入存储流水线 增量索引 孤儿清理',
    'backend/retrieval/vector_store.py': 'RAG系统向量存储 ChromaDB封装 余弦距离 元数据过滤 add_chunks search',
    'backend/retrieval/bm25_store.py': 'RAG系统BM25关键词检索引擎 中英文分词 语料库构建 持久化pickle',
    'backend/retrieval/fusion.py': 'RAG系统混合检索融合 RRF倒数排序融合 去重 同文件限制 来源感知加权',
    'backend/retrieval/query_expansion.py': 'RAG系统查询扩展 中英文术语映射 150+领域词表 RAG上下文消歧',
    'backend/generation/llm_service.py': 'RAG系统LLM服务 DeepSeek Chat API 流式生成 SSE 代理绕过',
    'backend/generation/prompt_manager.py': 'RAG系统Prompt管理 短System硬护栏 重User上下文 消息构建',
    'backend/generation/context_builder.py': 'RAG系统上下文构建器 全文件检索 令牌预算控制 截断策略 命中范围扩展',
    'backend/routes/health.py': 'RAG系统健康检查API 索引状态 重建端点',
    'backend/routes/query.py': 'RAG系统查询API 同步生成器 SSE流式响应 心跳保活 调试日志',
    'backend/schemas/models.py': 'RAG系统Pydantic数据模型 请求响应Schema',
    'run.py': 'RAG系统启动入口 uvicorn服务器 端口8001 热重载',
    'build_index.py': 'RAG系统索引构建脚本 命令行独立运行',
}


def _file_semantic_tag(file_info, chunk=None) -> str:
    """Generate a natural-language tag describing a file's purpose.

    Returns Chinese text that gets embedded as a standalone semantic chunk
    AND prepended to code chunks (for BM25 + LLM readability).
    """
    rel = file_info.relative_path.replace('\\', '/')

    # Exact match for key architectural files
    if rel in _FILE_SEMANTIC_TAGS:
        return f'[文件说明: {_FILE_SEMANTIC_TAGS[rel]}]'

    source = file_info.source
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
