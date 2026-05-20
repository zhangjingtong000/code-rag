import os
from dataclasses import dataclass, field


def get_env(key: str, default: str = '') -> str:
    value = os.environ.get(key)
    if value:
        return value
    return default


@dataclass
class Settings:
    # DeepSeek API
    deepseek_api_key: str = get_env('DEEPSEEK_API_KEY')
    deepseek_base_url: str = 'https://api.deepseek.com/v1'
    deepseek_model: str = 'deepseek-chat'

    # Ollama Embedding
    ollama_base_url: str = 'http://127.0.0.1:11434'
    ollama_embed_model: str = 'unclemusclez/jina-embeddings-v2-base-code:q5'

    # Source directories to index
    source_dirs: list = field(default_factory=lambda: [
        {'path': 'data/source-main', 'label': 'main'},
        {'path': 'data/source-release', 'label': 'release'},
        {'path': 'data/source-rag', 'label': 'rag'},
    ])

    # File extensions to include
    include_extensions: tuple = ('.py', '.js', '.wxml', '.wxss', '.json')

    # Exclude patterns (directory or filename)
    exclude_patterns: list = field(default_factory=lambda: [
        '__pycache__', '.git', 'node_modules', 'images',
        'package-lock.json', 'project.private.config.json', 'sitemap.json',
        '.gitignore', 'figma',
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
    vector_k: int = 15
    bm25_k: int = 15
    rrf_k: int = 60
    max_chunks_per_file: int = 3
    full_file_top_k: int = 4  # how many top files to include as full content

    # Context budget (tokens)
    context_max_tokens: int = 32000  # ~32K input, leaves ~32K for output
    single_file_truncate_tokens: int = 20000  # truncate oversized single file
    context_expand_lines: int = 300  # ±lines around hit chunk for truncation

    # Streaming
    typewriter_frame_ms: int = 16
    typewriter_chars_min: int = 8
    typewriter_chars_max: int = 15

    # Query expansion
    llm_expansion_enabled: bool = False  # toggle L2 LLM expansion

    # Server
    host: str = '0.0.0.0'
    port: int = 8001

    # Paths (relative to project root)
    @property
    def data_dir(self) -> str:
        return 'data'

    @property
    def chroma_dir(self) -> str:
        return 'data/chroma_db'

    @property
    def bm25_path(self) -> str:
        return 'data/bm25_index.pkl'

    @property
    def metadata_path(self) -> str:
        return 'data/chunk_metadata.json'

    @property
    def filehash_path(self) -> str:
        return 'data/file_hashes.json'

    @property
    def summaries_path(self) -> str:
        return 'data/file_summaries.json'


settings = Settings()
