#!/usr/bin/env python3
"""RAG Code Q&A System — Entry Point"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    import uvicorn
    from backend.config import settings

    print("=== RAG Code Q&A System ===")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Frontend: http://{settings.host}:{settings.port}/frontend/index.html")
    print(f"API Docs: http://{settings.host}:{settings.port}/docs")
    print()

    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        # Code reload is a dev convenience: it restarts the process on every
        # file change, which drops in-process state and is wrong for a demo
        # someone just launched. Opt in with RAG_DEV=1.
        reload=os.environ.get("RAG_DEV") == "1",
        timeout_keep_alive=300,  # Keep SSE connections alive for 5 minutes
    )
