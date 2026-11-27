import os
from dataclasses import dataclass, field


def get_env(key: str, default: str = '') -> str:
    value = os.environ.get(key)
    if value:
        return value
    return default


def parse_source_dirs(spec: str) -> list:
    """Parse a corpus description into scanner entries.

    ``spec`` is split on ``os.pathsep`` (';' on Windows, ':' elsewhere). Each
    entry is either a bare directory path or ``label=path``; a bare path uses its
    last path component as the label. Empty entries are ignored.
    """
    entries = []
    for raw in spec.split(os.pathsep):
        item = raw.strip()
        if not item:
            continue
        if '=' in item:
            label, _, path = item.partition('=')
            label, path = label.strip(), path.strip()
        else:
            label, path = '', item
        path = os.path.abspath(os.path.expanduser(path))
        if not label:
            label = os.path.basename(path.rstrip('\\/')) or 'source'
        entries.append({'path': path, 'label': label})
    return entries


@dataclass
class Settings:
    # DeepSeek API
    deepseek_api_key: str = get_env('DEEPSEEK_API_KEY')
    deepseek_base_url: str = 'https://api.deepseek.com/v1'
    deepseek_model: str = 'deepseek-chat'

    # Ollama Embedding
    ollama_base_url: str = 'http://127.0.0.1:11434'
    ollama_embed_model: str = 'unclemusclez/jina-embeddings-v2-base-code:q5'

    # Source directories to index (paths resolved at access time).
    # Override with RAG_SOURCE_DIRS, e.g. ``label=C:\code\repo;other=D:\code\x``
    # (separator = os.pathsep; a bare path takes its basename as label).
    @property
    def source_dirs(self) -> list:
        import os as _os
        override = get_env('RAG_SOURCE_DIRS')
        if override:
            return parse_source_dirs(override)
        root = self.project_root
        return [
            {'path': _os.path.join(root, 'data', 'source-main'), 'label': 'main'},
            {'path': _os.path.join(root, 'data', 'source-release'), 'label': 'release'},
            {'path': _os.path.join(root, 'data', 'source-rag'), 'label': 'rag'},
        ]

    # File extensions to include
    include_extensions: tuple = ('.py', '.js', '.wxml', '.wxss', '.json', '.html', '.css')

    # Exclude patterns (directory or filename)
    # site-packages / venv: vendored third-party code is noise for code QA —
    # in one snapshot it accounted for ~86% of all indexed files.
    exclude_patterns: list = field(default_factory=lambda: [
        '__pycache__', '.git', 'node_modules', 'site-packages', 'venv', '.venv',
        'images', 'package-lock.json', 'project.private.config.json', 'sitemap.json',
        '.gitignore', 'figma', 'test_',
    ])

    # Exclude file extensions (binary / media)
    exclude_extensions: tuple = ('.png', '.svg', '.jpg', '.jpeg', '.gif', '.ico')

    # Chunking
    chunk_min_chars: int = 2000
    chunk_max_chars: int = 4000
    chunk_page_max_chars: int = 5000  # force re-split if Page({}) exceeds this
    chunk_wxml_max_chars: int = 3000  # force re-split if WXML exceeds this
    chunk_overlap_lines: int = 5

    # Embedding batch size
    embed_batch_size: int = 30

    # Retrieval
    vector_k: int = 25
    bm25_k: int = 25
    rrf_k: int = 60
    max_chunks_per_file: int = 3
    full_file_top_k: int = 4  # how many top files to include as full content

    # Context budget (tokens)
    context_max_tokens: int = 32000  # ~32K input, leaves ~32K for output
    single_file_truncate_tokens: int = 20000  # truncate oversized single file
    context_expand_lines: int = 300  # ±lines around hit chunk for truncation

    # ASR (speech-to-text)
    asr_backend: str = 'auto'  # auto | qwen3-asr-cli | qwen-asr-py | antirez-c
    asr_model_name: str = 'Qwen/Qwen3-ASR-0.6B'
    asr_model_dir: str = 'models/qwen3-asr-0.6b'   # local model from ModelScope
    asr_binary_path: str = ''  # path to antirez/qwen_asr binary
    asr_device: str = 'cuda:0'
    asr_timeout: int = 60  # seconds

    # Streaming
    typewriter_frame_ms: int = 16
    typewriter_chars_min: int = 8
    typewriter_chars_max: int = 15

    # Query expansion
    llm_expansion_enabled: bool = False  # toggle L2 LLM expansion

    # Server
    host: str = '0.0.0.0'
    port: int = 8001

    # Paths (absolute, resolved against project root)
    @property
    def project_root(self) -> str:
        # Config is at <project>/backend/config.py
        import os as _os
        return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    @property
    def data_dir(self) -> str:
        import os as _os
        return _os.path.join(self.project_root, 'data')

    @property
    def chroma_dir(self) -> str:
        import os as _os
        return _os.path.join(self.data_dir, 'chroma_db')

    @property
    def bm25_path(self) -> str:
        import os as _os
        return _os.path.join(self.data_dir, 'bm25_index.pkl')

    @property
    def metadata_path(self) -> str:
        import os as _os
        return _os.path.join(self.data_dir, 'chunk_metadata.json')

    @property
    def filehash_path(self) -> str:
        import os as _os
        return _os.path.join(self.data_dir, 'file_hashes.json')

    @property
    def summaries_path(self) -> str:
        import os as _os
        return _os.path.join(self.data_dir, 'file_summaries.json')

    @property
    def asr_temp_dir(self) -> str:
        import os as _os
        path = _os.path.join(self.data_dir, 'temp_audio')
        _os.makedirs(path, exist_ok=True)
        return path


settings = Settings()
