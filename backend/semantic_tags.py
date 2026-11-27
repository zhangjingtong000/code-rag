"""Semantic tags & project background — externalised configuration.

标签表与项目背景文案原本写死在 ``indexer.py`` / ``context_builder.py`` 里，是为
示例项目（示例小程序项目）准备的。现在两者都从
``backend/semantic_tags.json`` 读：换项目时改这个 JSON 即可，不用动 Python 源码。

配置文件缺失或格式不对时回退到内置默认值，索引与上下文仍能构建。
"""

import json
import os

CONFIG_FILENAME = 'semantic_tags.json'
# 可选：指向另一份配置（例如每个项目一份），默认用与 backend/ 同级目录下的文件
CONFIG_ENV = 'RAG_SEMANTIC_TAGS'

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)

# 内置默认值：只在配置文件不可用时兜底，覆盖主要路径 + 按语言的通用标签
_DEFAULT_FILE_SEMANTIC_TAGS = {
    'backend/indexing/indexer.py': '索引构建器 扫描分块嵌入存储流水线 增量索引 孤儿清理',
    'backend/indexing/chunker.py': '代码分块器 结构感知分块 函数类边界',
    'backend/retrieval/vector_store.py': '向量存储 余弦距离 元数据过滤 add_chunks search',
    'backend/retrieval/bm25_store.py': 'BM25关键词检索引擎 中英文分词 语料库构建 持久化',
    'backend/generation/context_builder.py': '上下文构建器 全文件检索 令牌预算控制 截断策略',
    'run.py': '启动入口 uvicorn服务器 热重载',
}

_DEFAULT_PROJECT_BACKGROUND = """## 项目背景
未找到语义标签配置文件 backend/semantic_tags.json，当前使用内置默认背景文案。
检索到的代码标注了来源。
---"""


def config_path(path: str | None = None) -> str:
    """解析配置文件路径：显式参数 > RAG_SEMANTIC_TAGS > backend/semantic_tags.json。"""
    if path:
        return path
    override = os.environ.get(CONFIG_ENV, '').strip()
    return override or _DEFAULT_PATH


def load_config(path: str | None = None) -> dict:
    """读配置并返回 ``{'file_semantic_tags', 'project_background', 'source', 'path'}``。

    文件不存在、不是合法 JSON 或字段缺失时逐项回退到内置默认值，不抛异常——
    索引链路不能因为一份配置缺失就跑不起来。
    """
    resolved = config_path(path)
    try:
        with open(resolved, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, ValueError):
        raw = None

    if not isinstance(raw, dict):
        return {
            'file_semantic_tags': dict(_DEFAULT_FILE_SEMANTIC_TAGS),
            'project_background': _DEFAULT_PROJECT_BACKGROUND,
            'source': 'default',
            'path': resolved,
        }

    tags = raw.get('file_semantic_tags')
    background = raw.get('project_background')
    return {
        'file_semantic_tags': tags if isinstance(tags, dict) and tags
        else dict(_DEFAULT_FILE_SEMANTIC_TAGS),
        'project_background': background if isinstance(background, str) and background.strip()
        else _DEFAULT_PROJECT_BACKGROUND,
        'source': 'file',
        'path': resolved,
    }
