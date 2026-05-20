"""Standalone index builder script."""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from backend.indexing.indexer import Indexer

print("Starting index build...")
idx = Indexer()
idx.build_index(mode='full')
print("=== INDEX BUILD COMPLETE ===")
