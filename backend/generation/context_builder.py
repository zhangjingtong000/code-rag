"""Context builder — full-file retrieval with token budget control."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings


# Approximate token counter (char-based heuristic: ~4 chars per token for Chinese, ~3 for English)
def estimate_tokens(text: str) -> int:
    """Rough token estimate. ~3-4 chars per token on average for mixed Chinese/English."""
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 3  # Chinese ~2 chars/token, other ~3-4


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

        # Sort by best score descending
        sorted_files = sorted(
            file_groups.items(),
            key=lambda x: x[1]['best_score'],
            reverse=True
        )

        # Build context sections
        sections = []
        current_tokens = 0

        # Background header
        header = f"""## 项目背景
示例小程序项目是一个 AI 图像识别工具，包含图像标签识别、OCR 文字提取、图片对比等功能。
项目源码分为三部分：
- **main 分支**：完整 AI 功能，使用火山方舟 LLM/MLLM + DeepSeek 进行场景识别
- **release 分支**：合规精简版，使用腾讯云 CV API（DetectLabelPro + GeneralBasicOCR）
- **rag 来源**：本 RAG 问答系统自身的源码（向量检索 + 代码分块 + LLM 生成管线）

main 和 release 是两个独立分支，同一功能可能实现不同。检索到的代码标注了来源。
---
"""
        current_tokens = estimate_tokens(header)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for (source, rel_path), file_info in sorted_files:
            # Read full file content
            full_content = self._read_source_file(source, rel_path, project_root)
            if full_content is None:
                # Fallback: use chunk contents
                chunk_texts = []
                for r in fusion_results:
                    meta = r.get('metadata', {})
                    if meta.get('source') == source and meta.get('file_path') == rel_path:
                        chunk_texts.append(r.get('content', ''))
                full_content = '\n...\n'.join(chunk_texts)

            file_tokens = estimate_tokens(full_content)

            # Check if we should truncate this file
            if file_tokens > self.single_file_truncate:
                # Oversized single file: truncate to hit range ± expand
                hit_ranges = file_info['hit_ranges']
                full_content = self._truncate_to_relevant(full_content, hit_ranges)
                file_tokens = estimate_tokens(full_content)

            # Check if adding this file would exceed budget
            if current_tokens + file_tokens > self.max_tokens:
                if sections:
                    # Truncate this file to fit remaining budget
                    remaining = self.max_tokens - current_tokens - 200  # 200 for overhead
                    if remaining > 500:  # only if we can fit something meaningful
                        hit_ranges = file_info['hit_ranges']
                        truncated = self._truncate_to_relevant(full_content, hit_ranges)
                        # Further truncate if still too big
                        if estimate_tokens(truncated) > remaining:
                            truncated = truncated[:remaining * 3]  # approximate char cutoff
                            truncated += '\n// ... [已截断，完整文件见源码]\n'
                        sections.append(
                            self._format_file_section(source, rel_path, truncated)
                        )
                break  # stop adding more files

            sections.append(
                self._format_file_section(source, rel_path, full_content)
            )
            current_tokens += file_tokens

        body = '\n\n'.join(sections)
        return header + body

    def _truncate_to_relevant(self, content: str, hit_ranges: list[tuple]) -> str:
        """Truncate file content to the most relevant sections (hit ranges ± expand_lines)."""
        lines = content.split('\n')

        if not hit_ranges:
            # No hit info: return first portion
            return '\n'.join(lines[:self.expand_lines * 2])

        # Merge overlapping hit ranges
        merged = []
        for start, end in sorted(hit_ranges):
            if merged and start <= merged[-1][1] + self.expand_lines:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Expand ranges
        expanded = []
        for start, end in merged:
            exp_start = max(1, start - self.expand_lines)
            exp_end = min(len(lines), end + self.expand_lines)
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
        """Format a single file's content for the LLM context."""
        lang_map = {
            'python': 'python', 'javascript': 'javascript',
            'wxml': 'html', 'wxss': 'css', 'json': 'json',
        }
        # Determine language from extension
        ext = os.path.splitext(rel_path)[1].lower()
        lang = lang_map.get({
            '.py': 'python', '.js': 'javascript',
            '.wxml': 'wxml', '.wxss': 'wxss', '.json': 'json',
        }.get(ext, ext.lstrip('.')), '')

        return f"""### [来源: {source} | 文件: {rel_path}]

```{lang}
{content}
```"""

    def _read_source_file(self, source: str, rel_path: str, project_root: str) -> str | None:
        """Read full source file from data/source-{label}/ directory."""
        source_dir_map = {
            'main': 'data/source-main',
            'release': 'data/source-release',
            'rag': 'data/source-rag',
        }
        dir_name = source_dir_map.get(source, f'data/source-{source}')
        full_path = os.path.join(project_root, dir_name, rel_path)

        if not os.path.exists(full_path):
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='utf-8-sig') as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None
