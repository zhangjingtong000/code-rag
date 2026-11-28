"""Context builder — full-file retrieval with token budget control."""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.semantic_tags import load_config as _load_semantic_config

logger = logging.getLogger(__name__)

# 每个文件段还有 ### 标题 + 代码围栏，不能算进正文预算
_BUDGET_OVERHEAD_TOKENS = 200
# 已经有文件进上下文后，剩余预算低于这个就不值得再收一个残段
_MIN_SECTION_TOKENS = 500
_TRUNCATION_MARKER = '\n// ... [已截断，完整文件见源码]\n'

# 读不到整文件时的显式标注：以前这里静默退化成 chunk 拼接，上下文开头是
# 「[文件说明: …]」，README 承诺的「整份文件」其实没有兑现。
_FALLBACK_NOTE = (
    '// [整文件不可用：以下内容由检索命中的 chunk 拼接而成，不是完整文件，'
    '行号不可引用。请确认构建索引时的语料目录仍然存在。]\n'
)

# 语料路径缓存：key = chunk_metadata.json 路径，value = (mtime, {label: path})
_SOURCE_DIRS_CACHE: dict = {}


# Approximate token counter. Deliberately conservative: over-estimating only
# costs a little context, under-estimating blows the model window.
# DeepSeek counts a Chinese character as ~1 token and ~4 ASCII chars as one
# token. The old formula (`chinese // 2 + other // 3`) halved Chinese and
# floored each class separately, so 38 Chinese characters measured 18 tokens —
# the budget was silently exceeded on every Chinese-heavy context.
def estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed Chinese/English text."""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars + 3) // 4


def indexed_source_dirs() -> dict:
    """Read the corpus dirs recorded by build_index.py into the index metadata.

    快速开始只把 `--source-dirs "rag=."` 传给 build_index.py，随后启动的
    服务进程里没有 RAG_SOURCE_DIRS；旧实现回落到默认 ``data/source-<label>``，
    目录不存在便返回 None，于是整文件上下文静默降级成 chunk 拼接。
    索引构建时已经把真实语料路径写进 ``chunk_metadata.json``，服务端优先用它。

    Returns {label: absolute_path}; missing or unreadable metadata yields {}.
    """
    path = settings.metadata_path
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return {}

    cached = _SOURCE_DIRS_CACHE.get(path)
    if cached and cached[0] == stamp:
        return cached[1]

    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(meta, dict):
        return {}

    dirs = {}
    for entry in meta.get('source_dirs') or []:
        if isinstance(entry, dict) and entry.get('label') and entry.get('path'):
            dirs[entry['label']] = entry['path']

    _SOURCE_DIRS_CACHE[path] = (stamp, dirs)
    return dirs


class ContextBuilder:

    def __init__(self):
        self.max_tokens = settings.context_max_tokens
        self.single_file_truncate = settings.single_file_truncate_tokens
        self.expand_lines = settings.context_expand_lines
        self.top_k = settings.full_file_top_k

    def build(self, fusion_results: list[dict]) -> str:
        """Build formatted context from fusion results.

        Strategy:
        1. Group by unique file (source + path)
        2. For each file, track which line ranges were hit (for truncation)
        3. Sort files by best RRF score
        4. Fill full file contents until token budget is hit
        5. Truncate the file that exceeds budget (using hit ranges)
        """
        # Group results by (source, file_path), track best score and hit ranges
        file_groups = {}
        for r in fusion_results:
            meta = r.get('metadata', {})
            key = (meta.get('source', ''), meta.get('file_path', ''))
            if not key[1]:
                continue

            if key not in file_groups:
                file_groups[key] = {
                    'best_score': 0,
                    'hit_ranges': [],  # (start_line, end_line) tuples
                }

            file_groups[key]['best_score'] = max(
                file_groups[key]['best_score'],
                r.get('rrf_score', 0)
            )
            file_groups[key]['hit_ranges'].append(
                (meta.get('start_line', 1), meta.get('end_line', meta.get('start_line', 1) + 100))
            )

        # Sort by best score descending, then keep only the configured number
        # of full files (`full_file_top_k`). It used to be read into `top_k` and
        # never applied, so the budget alone decided how many files went in.
        sorted_files = sorted(
            file_groups.items(),
            key=lambda x: x[1]['best_score'],
            reverse=True
        )[:self.top_k]

        # Build context sections
        sections = []
        current_tokens = 0

        # Background header — text lives in backend/semantic_tags.json so a
        # different corpus only needs a config edit, not a code change.
        header = _load_semantic_config()['project_background'].rstrip() + '\n'
        current_tokens = estimate_tokens(header)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for (source, rel_path), file_info in sorted_files:
            # Read full file content
            full_content = self._read_source_file(source, rel_path, project_root)
            if full_content is None:
                # Fallback: use chunk contents. This is a real degradation (no
                # full file, no trustworthy line numbers), so it is logged and
                # marked in the context instead of passing for full-file content.
                logger.warning(
                    "整文件读取失败，退化为 chunk 拼接：%s:%s", source, rel_path)
                chunk_texts = []
                for r in fusion_results:
                    meta = r.get('metadata', {})
                    if meta.get('source') == source and meta.get('file_path') == rel_path:
                        chunk_texts.append(r.get('content', ''))
                full_content = _FALLBACK_NOTE + '\n...\n'.join(chunk_texts)
            else:
                # 每行左侧是这一行在源文件里的真实行号：System Prompt 要求
                # 回答标注 `[来源 | 文件 | 行号: X-Y]`，而模型只能看到上下文。
                # 上下文是裸代码块时，行号全靠模型编（截图里的 78-95 与真实
                # 的 58-75 差 20 行就是这么来的）。
                full_content = self._number_lines(full_content)

            file_tokens = estimate_tokens(full_content)

            # Check if we should truncate this file
            if file_tokens > self.single_file_truncate:
                # Oversized single file: truncate to hit range ± expand
                hit_ranges = file_info['hit_ranges']
                full_content = self._truncate_to_relevant(full_content, hit_ranges)
                file_tokens = estimate_tokens(full_content)

            # Check if adding this file would exceed budget
            if current_tokens + file_tokens > self.max_tokens:
                remaining = self.max_tokens - current_tokens - _BUDGET_OVERHEAD_TOKENS
                if sections:
                    # Truncate this file to fit remaining budget
                    if remaining >= _MIN_SECTION_TOKENS:
                        sections.append(self._format_file_section(
                            source, rel_path,
                            self._fit_to_budget(
                                full_content, file_info['hit_ranges'], remaining)))
                    break  # stop adding more files

                # 第一个文件就超预算时，原实现直接 break，返回的上下文只有项目
                # 背景 header、一个文件都没有，也不报错——模型只能凭空回答。
                # 这里至少收一个截断到剩余预算的文件；连一段都放不下时写一条
                # 显式提示，让「没检索到内容」可见，而不是静默。
                if remaining > 0:
                    sections.append(self._format_file_section(
                        source, rel_path,
                        self._fit_to_budget(
                            full_content, file_info['hit_ranges'], remaining)))
                else:
                    sections.append(
                        '> [上下文预算不足：项目背景已占满 '
                        f'{self.max_tokens} token 预算，未能收录任何源文件。'
                        '请调大 context_max_tokens 或缩短项目背景。]')
                break

            sections.append(
                self._format_file_section(source, rel_path, full_content)
            )
            current_tokens += file_tokens

        body = '\n\n'.join(sections)
        return header + body

    @staticmethod
    def _number_lines(text: str, start_line: int = 1) -> str:
        """Prefix each line with its real 1-based source line number."""
        lines = text.split('\n')
        width = max(3, len(str(start_line + len(lines) - 1)))
        return '\n'.join(
            f'{start_line + i:>{width}}  {line}' for i, line in enumerate(lines)
        )

    def _fit_to_budget(self, content: str, hit_ranges: list[tuple],
                       budget_tokens: int) -> str:
        """Truncate ``content`` to the hit ranges and then to a token budget.

        Never returns an empty string when there is any budget: a context that
        silently drops the only matching file is worse than a short excerpt.
        The expand radius shrinks before the raw character cut, so the hit
        lines themselves survive a tight budget.
        """
        for radius in (self.expand_lines, self.expand_lines // 4, 20, 5, 1):
            truncated = self._truncate_to_relevant(content, hit_ranges, radius)
            if estimate_tokens(truncated) <= budget_tokens:
                return truncated

        truncated = self._truncate_to_relevant(content, hit_ranges, 1)
        marker = '' if '已截断' in truncated else _TRUNCATION_MARKER
        allowance = budget_tokens - estimate_tokens(marker) - 1
        if allowance <= 0:
            return marker.strip()

        # 2 chars/token is the safe lower bound (Chinese is 1 char/token);
        # then shrink until the estimate actually fits.
        cut = truncated[:max(0, allowance * 2)]
        while cut and estimate_tokens(cut) > allowance:
            cut = cut[:int(len(cut) * 0.9)]
        return cut + marker

    def _truncate_to_relevant(self, content: str, hit_ranges: list[tuple],
                              expand_lines: int | None = None) -> str:
        """Truncate file content to the most relevant sections (hit ranges ± expand_lines).

        ``content`` is expected to be line-numbered already (see
        ``_number_lines``): slicing keeps the real line numbers attached to the
        lines, so a truncated snippet still cites the right range.
        """
        lines = content.split('\n')
        if expand_lines is None:
            expand_lines = self.expand_lines

        if not hit_ranges:
            # No hit info: return first portion
            return '\n'.join(lines[:expand_lines * 2])

        # Merge overlapping hit ranges
        merged = []
        for start, end in sorted(hit_ranges):
            if merged and start <= merged[-1][1] + expand_lines:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Expand ranges
        expanded = []
        for start, end in merged:
            exp_start = max(1, start - expand_lines)
            exp_end = min(len(lines), end + expand_lines)
            expanded.append((exp_start, exp_end))

        # Build truncated content
        parts = []
        prev_end = 0
        for exp_start, exp_end in expanded:
            if exp_start > prev_end + 10:
                parts.append(f'// ... [已截断 {prev_end + 1}-{exp_start - 1} 行，完整文件见源码] ...')
            parts.append('\n'.join(lines[exp_start - 1:exp_end]))
            prev_end = exp_end

        if prev_end < len(lines):
            parts.append(f'// ... [已截断 {prev_end + 1}-{len(lines)} 行，完整文件见源码] ...')

        return '\n'.join(parts)

    def _format_file_section(self, source: str, rel_path: str, content: str) -> str:
        """Format a single file's content for the LLM context.

        ``content`` must already carry line numbers: this function only adds the
        heading and the code fence.
        """
        lang_map = {
            'python': 'python', 'javascript': 'javascript',
            'wxml': 'html', 'wxss': 'css', 'json': 'json',
            'html': 'html', 'css': 'css',
        }
        # Determine language from extension
        ext = os.path.splitext(rel_path)[1].lower()
        lang = lang_map.get({
            '.py': 'python', '.js': 'javascript',
            '.wxml': 'wxml', '.wxss': 'wxss', '.json': 'json',
            '.html': 'html', '.css': 'css',
        }.get(ext, ext.lstrip('.')), '')

        return f"""### [来源: {source} | 文件: {rel_path}]

```{lang}
{content}
```"""

    def _candidate_dirs(self, source: str, project_root: str) -> list[str]:
        """Corpus dirs to try for ``source``, most authoritative first.

        1. the path recorded in the index metadata (what actually was indexed)
        2. ``settings.source_dirs`` (RAG_SOURCE_DIRS / defaults)
        3. legacy ``data/source-<label>``
        """
        candidates = []

        indexed = indexed_source_dirs().get(source)
        if indexed:
            candidates.append(indexed)

        for entry in settings.source_dirs:
            if entry.get('label') == source and entry.get('path'):
                candidates.append(entry['path'])

        candidates.append(f'data/source-{source}')

        resolved = []
        for path in candidates:
            if not path:
                continue
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            path = os.path.normpath(path)
            if path not in resolved:
                resolved.append(path)
        return resolved

    def _read_source_file(self, source: str, rel_path: str, project_root: str) -> str | None:
        """Read full source file, resolved against the configured corpus dirs."""
        for dir_name in self._candidate_dirs(source, project_root):
            full_path = os.path.join(dir_name, rel_path)
            if not os.path.exists(full_path):
                continue
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(full_path, 'r', encoding='utf-8-sig') as f:
                        return f.read()
                except Exception:
                    continue
            except Exception:
                continue
        return None
