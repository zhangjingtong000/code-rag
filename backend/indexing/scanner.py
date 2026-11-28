"""File scanner — walks source directories with include/exclude rules."""

import os
import hashlib
from dataclasses import dataclass


@dataclass
class FileInfo:
    source: str           # 'main' | 'release' | 'rag'
    path: str             # absolute path
    relative_path: str    # e.g. '1/app/main.py'
    language: str         # 'python' | 'javascript' | 'wxml' | 'wxss' | 'json'
    content: str
    size: int
    md5: str = ''


class CodeScanner:

    def __init__(self, config):
        self.source_dirs = config.source_dirs
        self.include_exts = config.include_extensions
        # Normalised once; matching is per path segment, never a substring
        self.exclude_patterns = [p.lower() for p in config.exclude_patterns]
        self.exclude_exts = config.exclude_extensions

    def _language(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.py':
            return 'python'
        if ext == '.js':
            return 'javascript'
        if ext == '.wxml':
            return 'wxml'
        if ext == '.wxss':
            return 'wxss'
        if ext == '.json':
            return 'json'
        if ext == '.html':
            return 'html'
        if ext == '.css':
            return 'css'
        return 'unknown'

    def _excluded(self, name: str, is_dir: bool = False) -> bool:
        """Match a single path segment against the exclude patterns.

        The old rule was `pattern in whole_path`, so `'test_'` rejected
        `contest_utils.py` and `'images'` rejected any path containing that
        word. A pattern now matches a whole directory/file name; a trailing
        underscore additionally means "filename prefix", which keeps `test_`
        catching `test_chunker.py` without touching `contest_utils.py`.
        """
        name = name.lower()
        for pattern in self.exclude_patterns:
            if name == pattern:
                return True
            if not is_dir and pattern.endswith('_') and name.startswith(pattern):
                return True
        return False

    def _should_include(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.include_exts:
            return False
        if ext in self.exclude_exts:
            return False

        parts = [p for p in filepath.replace('\\', '/').split('/') if p]
        if not parts:
            return False
        if any(self._excluded(part, is_dir=True) for part in parts[:-1]):
            return False
        return not self._excluded(parts[-1])

    def scan(self) -> list[FileInfo]:
        files = []

        for src_cfg in self.source_dirs:
            src_dir = src_cfg['path']
            src_label = src_cfg['label']

            if not os.path.exists(src_dir):
                print(f"[Scanner] Source directory not found: {src_dir}")
                continue

            for root, dirs, _filenames in os.walk(src_dir):
                # Filter directories in-place
                dirs[:] = [d for d in dirs
                           if not d.startswith('.') and not self._excluded(d, is_dir=True)]

                for filename in _filenames:
                    full_path = os.path.join(root, filename)
                    if not self._should_include(full_path):
                        continue

                    rel_path = os.path.relpath(full_path, src_dir).replace('\\', '/')

                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(full_path, 'r', encoding='utf-8-sig') as f:
                                content = f.read()
                        except Exception:
                            continue
                    except Exception:
                        continue

                    md5 = hashlib.md5(content.encode('utf-8')).hexdigest()

                    files.append(FileInfo(
                        source=src_label,
                        path=full_path,
                        relative_path=rel_path,
                        language=self._language(full_path),
                        content=content,
                        size=len(content),
                        md5=md5,
                    ))

        print(f"[Scanner] Found {len(files)} files across {len(self.source_dirs)} sources")
        return files
