"""Structure-aware code chunker — splits files at function/class/component boundaries."""

import re
from dataclasses import dataclass, field
from backend.indexing.scanner import FileInfo


@dataclass
class CodeChunk:
    source: str
    file_path: str         # relative path
    language: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str        # 'function' | 'class' | 'page' | 'template' | 'style' | 'config' | 'module_level'
    file_md5: str = ''


class CodeChunker:

    def __init__(self, config):
        self.min_chars = config.chunk_min_chars
        self.max_chars = config.chunk_max_chars
        self.page_max_chars = config.chunk_page_max_chars
        self.wxml_max_chars = config.chunk_wxml_max_chars
        self.overlap_lines = config.chunk_overlap_lines

    def chunk_file(self, file_info: FileInfo) -> list[CodeChunk]:
        lang = file_info.language
        if lang == 'python':
            return self._chunk_python(file_info)
        if lang == 'javascript':
            return self._chunk_javascript(file_info)
        if lang == 'wxml':
            return self._chunk_wxml(file_info)
        if lang == 'wxss':
            return self._chunk_style(file_info, 'wxss')
        if lang == 'css':
            return self._chunk_style(file_info, 'css')
        if lang == 'html':
            return self._chunk_wxml(file_info)  # same strategy as WXML
        if lang == 'json':
            return self._chunk_config(file_info)
        return []

    # ─── Python ──────────────────────────────────────────

    def _chunk_python(self, fi: FileInfo) -> list[CodeChunk]:
        lines = fi.content.split('\n')
        chunks = []
        header_end = 0

        # Gather import/docstring header
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                header_end = i + 1
            elif stripped.startswith('import ') or stripped.startswith('from '):
                header_end = i + 1
            elif stripped == '':
                header_end = i + 1
            else:
                break

        if header_end > 0:
            header_text = '\n'.join(lines[:header_end])
            if len(header_text.strip()) > 0:
                chunks.append(CodeChunk(
                    source=fi.source, file_path=fi.relative_path, language=fi.language,
                    content=header_text, start_line=1, end_line=header_end,
                    chunk_type='module_level', file_md5=fi.md5,
                ))

        # Find top-level def/class boundaries
        boundaries = []
        in_decorator = False
        decorator_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('@'):
                if not in_decorator:
                    decorator_start = i
                    in_decorator = True
                continue
            is_boundary = (
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('async def ')
            )
            if is_boundary and re.match(r'^\S', line):  # column 0 only
                start = decorator_start if in_decorator else i
                boundaries.append((start, stripped))
                in_decorator = False
            elif in_decorator and not is_boundary:
                in_decorator = False  # decorator not followed by def/class

        if not boundaries:
            # No clear boundaries — single chunk
            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=fi.content, start_line=1, end_line=len(lines),
                chunk_type='module_level', file_md5=fi.md5,
            ))
            return chunks

        # Build chunks between boundaries
        for idx, (start, _decl) in enumerate(boundaries):
            end_line = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
            chunk_lines = lines[start:end_line]
            chunk_text = '\n'.join(chunk_lines)

            chunk_type = 'class' if _decl.startswith('class ') else 'function'

            # Sub-chunk large classes at method boundaries
            if chunk_type == 'class' and len(chunk_text) > self.page_max_chars:
                sub_chunks = self._subchunk_methods(lines, start, end_line, fi, chunk_type)
                chunks.extend(sub_chunks)
                continue

            # If still too large, try to sub-split at inner def boundaries
            if len(chunk_text) > self.max_chars and chunk_type in ('function', 'class'):
                sub = self._subchunk_inner(lines, start, end_line, fi)
                if sub:
                    chunks.extend(sub)
                    continue

            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=chunk_text, start_line=start + 1, end_line=end_line,
                chunk_type=chunk_type, file_md5=fi.md5,
            ))

        return chunks

    def _subchunk_methods(self, lines, start, end, fi, parent_type):
        """Split a large class body at method boundaries."""
        chunks = []
        # Class header stays with first chunk
        class_header_end = start
        for i in range(start, min(start + 10, end)):
            if lines[i].strip().startswith('class '):
                class_header_end = i + 1
                break

        current_start = start
        current_method_start = class_header_end
        for i in range(class_header_end, end):
            stripped = lines[i].strip()
            if (stripped.startswith('def ') or stripped.startswith('async def ')) and re.match(r'^\s{4}', lines[i]):
                if current_method_start < i:
                    chunk_text = '\n'.join(lines[current_start:i])
                    if len(chunk_text.strip()) > 0:
                        chunks.append(CodeChunk(
                            source=fi.source, file_path=fi.relative_path, language=fi.language,
                            content=chunk_text, start_line=current_start + 1, end_line=i,
                            chunk_type=parent_type, file_md5=fi.md5,
                        ))
                current_start = current_method_start = i

        # Last piece
        if current_start < end:
            chunk_text = '\n'.join(lines[current_start:end])
            if len(chunk_text.strip()) > 0:
                chunks.append(CodeChunk(
                    source=fi.source, file_path=fi.relative_path, language=fi.language,
                    content=chunk_text, start_line=current_start + 1, end_line=end,
                    chunk_type=parent_type, file_md5=fi.md5,
                ))

        return chunks

    def _subchunk_inner(self, lines, start, end, fi):
        """Split oversized function/class block at inner def boundaries."""
        indent = '    '
        boundaries = [start]
        for i in range(start + 1, end):
            stripped = lines[i].strip()
            if (stripped.startswith('def ') or stripped.startswith('async def ')) and lines[i].startswith(indent):
                boundaries.append(i)

        if len(boundaries) < 2:
            return None  # can't sub-split

        chunks = []
        for idx in range(len(boundaries)):
            b_start = boundaries[idx]
            b_end = boundaries[idx + 1] if idx + 1 < len(boundaries) else end
            chunk_text = '\n'.join(lines[b_start:b_end])
            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=chunk_text, start_line=b_start + 1, end_line=b_end,
                chunk_type='function', file_md5=fi.md5,
            ))
        return chunks

    # ─── JavaScript ───────────────────────────────────────

    def _chunk_javascript(self, fi: FileInfo) -> list[CodeChunk]:
        lines = fi.content.split('\n')
        chunks = []

        # Find boundary lines: function, class, Page({, App({, Component({
        boundaries = []
        in_page_block = False
        page_start = 0
        brace_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect Page({ or App({ or Component({
            if re.match(r'^(Page|App|Component)\s*\(\s*\{', stripped) and not in_page_block:
                boundaries.append(i)
                in_page_block = True
                page_start = i
                brace_depth = 1
                continue

            if in_page_block:
                brace_depth += stripped.count('{') - stripped.count('}')
                if brace_depth <= 0:
                    in_page_block = False
                    # Check if page block too large
                    page_text = '\n'.join(lines[page_start:i + 1])
                    if len(page_text) > self.page_max_chars:
                        # Re-split at methodName(){ boundaries
                        sub = self._split_page_methods(lines, page_start, i + 1, fi)
                        chunks.extend(sub)
                    else:
                        chunks.append(CodeChunk(
                            source=fi.source, file_path=fi.relative_path, language=fi.language,
                            content=page_text, start_line=page_start + 1, end_line=i + 1,
                            chunk_type='page', file_md5=fi.md5,
                        ))
                    continue

            if in_page_block:
                continue

            # Top-level function or class
            if re.match(r'^(function |async function |class |const \w+\s*=\s*(async\s*)?\(|module\.exports)', stripped):
                if stripped.startswith('module.exports'):
                    continue
                boundaries.append(i)

        if not boundaries:
            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=fi.content, start_line=1, end_line=len(lines),
                chunk_type='module_level', file_md5=fi.md5,
            ))
            return chunks

        # Build chunks between boundaries
        for idx in range(len(boundaries)):
            start = boundaries[idx]
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
            chunk_text = '\n'.join(lines[start:end])

            stype = 'page' if re.match(r'(Page|App|Component)\s*\(', lines[start].strip()) else 'function'

            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=chunk_text, start_line=start + 1, end_line=end,
                chunk_type=stype, file_md5=fi.md5,
            ))

        return chunks

    def _split_page_methods(self, lines, start, end, fi):
        """Split oversized Page block at methodName(){ boundaries."""
        chunks = []
        # Page header as first chunk (Page({ ... first few lines)
        header_end = start + 3
        for i in range(start, min(start + 8, end)):
            if '{' in lines[i]:
                header_end = i + 1
                break

        method_starts = [start]
        for i in range(header_end, end - 1):
            stripped = lines[i].strip()
            if re.match(r'^\w+\s*\(', stripped) and not stripped.startswith(('if', 'for', 'while', 'return', 'let', 'const', 'var', 'this.')):
                method_starts.append(i)

        for idx in range(len(method_starts)):
            m_start = method_starts[idx]
            m_end = method_starts[idx + 1] if idx + 1 < len(method_starts) else end
            chunk_text = '\n'.join(lines[m_start:m_end])
            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=chunk_text, start_line=m_start + 1, end_line=m_end,
                chunk_type='function', file_md5=fi.md5,
            ))
        return chunks

    # ─── WXML ─────────────────────────────────────────────

    def _chunk_wxml(self, fi: FileInfo) -> list[CodeChunk]:
        content = fi.content
        lines = content.split('\n')

        if len(content) < self.wxml_max_chars:
            return [CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=content, start_line=1, end_line=len(lines),
                chunk_type='template', file_md5=fi.md5,
            )]

        # Split at <template> or major <view class="xxx"> boundaries
        chunks = []
        splits = list(re.finditer(r'(?:^|\n)(?=\s*(?:<template|<scroll-view|<view\s+class=))', content))
        split_positions = [m.start() for m in splits]

        if len(split_positions) < 2:
            return [CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=content, start_line=1, end_line=len(lines),
                chunk_type='template', file_md5=fi.md5,
            )]

        for idx in range(len(split_positions)):
            start_pos = split_positions[idx]
            end_pos = split_positions[idx + 1] if idx + 1 < len(split_positions) else len(content)
            chunk_text = content[start_pos:end_pos].strip()
            if not chunk_text:
                continue
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            chunks.append(CodeChunk(
                source=fi.source, file_path=fi.relative_path, language=fi.language,
                content=chunk_text, start_line=start_line, end_line=end_line,
                chunk_type='template', file_md5=fi.md5,
            ))
        return chunks

    # ─── WXSS ─────────────────────────────────────────────

    def _chunk_style(self, fi: FileInfo, _lang: str) -> list[CodeChunk]:
        lines = fi.content.split('\n')
        return [CodeChunk(
            source=fi.source, file_path=fi.relative_path, language=fi.language,
            content=fi.content, start_line=1, end_line=len(lines),
            chunk_type='style', file_md5=fi.md5,
        )]

    # ─── JSON config ──────────────────────────────────────

    def _chunk_config(self, fi: FileInfo) -> list[CodeChunk]:
        """Minimal config files — included for BM25 but may skip vector embedding."""
        lines = fi.content.split('\n')
        return [CodeChunk(
            source=fi.source, file_path=fi.relative_path, language=fi.language,
            content=fi.content, start_line=1, end_line=len(lines),
            chunk_type='config', file_md5=fi.md5,
        )]
