"""Standalone index builder script. Usage: python build_index.py [full|incremental]"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from backend.indexing.indexer import Indexer

mode = sys.argv[1] if len(sys.argv) > 1 else 'incremental'
print(f"Starting index build (mode={mode})...")
idx = Indexer()
idx.build_index(mode=mode)
print("=== INDEX BUILD COMPLETE ===")
