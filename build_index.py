"""Standalone index builder script.

Usage:
    python build_index.py [full|incremental] [--source-dirs "label=path;label=path"]

--source-dirs overrides the corpus directories for this run (same format as the
RAG_SOURCE_DIRS environment variable, which is set from the flag).
"""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from backend.config import parse_source_dirs, settings  # noqa: E402
from backend.indexing.indexer import Indexer  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Build the code index.')
    parser.add_argument('mode', nargs='?', default='incremental',
                        choices=['full', 'incremental'],
                        help='full rebuild (default: incremental)')
    parser.add_argument('--source-dirs', default=None,
                        help="corpus dirs, os.pathsep-separated, each 'label=path' "
                             "or a bare path (overrides RAG_SOURCE_DIRS)")
    args = parser.parse_args(argv)

    if args.source_dirs:
        os.environ['RAG_SOURCE_DIRS'] = args.source_dirs

    dirs = settings.source_dirs
    if not dirs:
        print("[build_index] No source directories configured.")
        return 2
    for entry in dirs:
        exists = os.path.isdir(entry['path'])
        print(f"[build_index] source {entry['label']}: {entry['path']}"
              f"{'' if exists else '  <- 目录不存在'}")

    print(f"Starting index build (mode={args.mode})...")
    idx = Indexer()
    idx.build_index(mode=args.mode)
    print("=== INDEX BUILD COMPLETE ===")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
