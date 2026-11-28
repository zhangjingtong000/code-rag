# 代码库问答 RAG

[![CI](https://github.com/zhangjingtong000/code-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/zhangjingtong000/code-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

把一份源码目录灌进索引，然后用中文提问：这个接口怎么实现、BM25 和向量怎么融合。检索按函数/类边界定位，回答时喂给模型的是整份文件，并且文件每一行都带真实行号。

<p align="center">
  <img src="docs/screenshots/01-home.png" width="32%" alt="首页：三栏布局，索引状态就绪">
  <img src="docs/screenshots/02-answer.png" width="32%" alt="带代码高亮和行号来源的回答">
  <img src="docs/screenshots/03-voice.png" width="32%" alt="按住麦克风语音输入">
</p>

<sub>截图取自同一会话，语料为本仓库自身源码（自举索引 35 个文件、约 180 个块）。</sub>

## 它做什么

建索引时按函数、类、组件边界切块，每块附一句语义标签；提问后同时跑向量检索（jina-embeddings-v2-base-code，768 维）和 BM25 关键词检索，两路排名用 RRF 融合；排在前面的文件整份读进上下文，超出 32K token 预算时按命中行 ±300 行截断；DeepSeek 流式生成，前端边收边渲染。

```text
POST /api/rag/query              → query_id
GET  /api/rag/query/stream/{id}  → SSE：metadata → token… → done
  查询扩展  157 组中英术语映射（LLM 扩展默认关）
  双路召回  向量 k=25 + BM25 k=25
  融合      1/(60+rank) + 来源/语言加权 → 去重 → 同文件 ≤3 块
  上下文    按 RRF 排名补全整文件，32K 预算，命中行 ±300 行截断
  生成      DeepSeek 流式输出，前端打字机渲染
```

## 快速开始

先装好 [Ollama](https://ollama.com/) 并拉取嵌入模型，再按平台二选一。

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull unclemusclez/jina-embeddings-v2-base-code:q5
export DEEPSEEK_API_KEY=sk-...
python build_index.py full --source-dirs "rag=."   # 用本仓库自身源码建索引
python run.py                                      # http://127.0.0.1:8001/frontend/index.html
```

Windows（cmd）：

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull unclemusclez/jina-embeddings-v2-base-code:q5
set DEEPSEEK_API_KEY=sk-...
python build_index.py full --source-dirs "rag=."
python run.py
```

PowerShell 里环境变量换成 `$env:DEEPSEEK_API_KEY="sk-..."`。

`--source-dirs` 只对构建脚本生效，它会把语料路径写进 `data/chunk_metadata.json`；`run.py` 启动时从这份元数据读回整文件，所以**不需要**再为服务进程设一遍 `RAG_SOURCE_DIRS`。

默认语料是 `data/source-main`、`data/source-release`、`data/source-rag` 三个目录；语料在别处时用 `--source-dirs` 或 `RAG_SOURCE_DIRS` 指过去。多个目录用 `os.pathsep` 分隔——Windows 是分号，macOS/Linux 是冒号：

```bat
set RAG_SOURCE_DIRS=main=D:\code\app-main;release=D:\code\app-release
```

```bash
export RAG_SOURCE_DIRS=main=/code/app-main:release=/code/app-release
```

索引是本地生成物，仓库不附带 `data/`，首次使用要自己构建。

不开浏览器也能问：

```bash
curl -X POST http://127.0.0.1:8001/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"BM25 和向量怎么融合"}'
# 返回 query_id 与 stream_url，再 GET stream_url 取 SSE 事件流
```

## 功能

| 能力 | 说明 |
| --- | --- |
| 问答界面 | 三栏布局，Markdown 渲染 + 代码高亮，代码块可折叠、可复制，生成中可中断 |
| 双路检索 | 向量与 BM25 同时召回，RRF 融合；候选超过 15 条时用 DeepSeek 重排 |
| 行号来源 | 上下文每行左侧标注源文件真实行号，系统提示只允许照抄，不允许模型推算 |
| 索引管理 | 全量 / 增量构建，增量只重新嵌入改过的文件，BM25 语料每次全量重建；页面顶栏显示构建状态 |
| 多轮对话 | 携带最近 10 条问答历史，历史长度与输入预算一起计算 |
| 语音输入 | 按住麦克风说话，松开后文字填进输入框（需另装 ASR 后端） |
| HTTP 接口 | `POST /api/rag/query` 取 query_id，`GET /api/rag/query/stream/{id}` 走 SSE，事件带 `id:`，静默 15 秒发一次心跳 |

## 设计要点

**结构感知切块**：Python 按 `def`/`class` 断，类里再按方法细分；JS 认 `function`/`class` 以及 `var X = class` 这类表达式；HTML/WXML 在 `<template>`、`<view class>` 边界处断，小文件整份一块；CSS/WXSS 和 JSON 整文件一块（JSON 只进 BM25）。同一切块在两套索引里共用一个 `chunk_key`，RRF 不会把同一块当成两条结果。

**整文件上下文 + 预算截断**：检索只负责挑出相关文件，进上下文的仍是完整文件，最多 4 个。32K 预算按 RRF 排名依次装文件，单个文件超过 20K token 时保留命中行 ±300 行并留下截断标记。第一个文件就超预算时也会收一段截断内容，不返回空上下文。语料路径随索引一起落在 `chunk_metadata.json`，服务端按它读文件；实在读不到时打 warning，并在上下文开头显式标注「整文件不可用，以下为 chunk 拼接」。

**行号可引用**：`ContextBuilder` 给每行加真实行号，截断切的是同一条行数组，所以片段里的行号仍然对得上源文件。System Prompt 第 3 条要求模型照抄左侧数字。

**RRF 与来源加权**：两路排名按 `1/(60+rank)` 相加；问题里出现 rag、main/release 分支、前端或接口关键词时，对相应来源和语言乘 1.3–2.0。同一文件最多保留 3 块，把位置让给别的文件。

**可选语音**：浏览器录 16kHz 单声道 WAV，`POST /api/rag/audio/transcribe`，后端起子进程调用本机 Qwen3-ASR，支持 qwen3-asr-cli、qwen-asr 包、antirez 的 C 二进制三种后端并自动探测，录音上限 30 秒。后端不可用返回 503，音频解不开返回 422。

## 评测

`tests/test_retrieval_quality.py` 用本仓库源码建索引，对 10 个中文问题算 recall@5 与 MRR，纯离线、几秒跑完：

| 指标 | 当前下界 |
| --- | --- |
| recall@5 | 0.90（9/10） |
| MRR | 0.71 |

这条评测只走 BM25，不含向量与 RRF；线上混合链路不会比它更差。数值掉下来说明切块、语义标签或查询扩展被改坏了。

## 已知边界

- 单机 demo：查询表和 BM25 缓存都在进程内存里，按单用户、中小语料设计。
- 没有鉴权，默认只监听 `127.0.0.1`；要对外服务得显式设 `RAG_HOST=0.0.0.0`，并自行补鉴权、CORS 白名单和限流。
- 检索依赖 Ollama 里的 `jina-embeddings-v2-base-code:q5`，问答依赖 DeepSeek API Key，少一样就只跑得通半条链路。
- 语音输入要另装 ASR 后端，没装、或装了包但本地模型没下载时 `/audio/transcribe` 返回 503。
- 157 组中英术语映射、加权倍数、项目背景（`backend/semantic_tags.json`）都是给本仓库配的，换技术栈要改 Python 和这份 JSON。
- 自举建索引时 `exclude_patterns` 里的 `test_` 是「文件名前缀」规则，所以 `tests/` 下的 `test_*.py` 不进索引（截图里的 35 个文件不含单测）。
- SSE 不重放已经发出的 token，事件里的 `id:` 只用于记录进度，不做断点续传。

换项目时用 `RAG_SOURCE_DIRS`（或 `--source-dirs`）指向新语料，再改 `backend/semantic_tags.json` 与 `backend/retrieval/query_expansion.py` 里的术语表。

## 技术栈

Python 3.10+ · FastAPI + uvicorn（SSE）· Chroma + rank-bm25 · jina-embeddings-v2-base-code（Ollama）· DeepSeek Chat · 原生 JS + marked / DOMPurify / highlight.js。

## 目录结构

```text
backend/
  indexing/    扫描、切块、嵌入、索引编排
  retrieval/   向量库、BM25、RRF 融合、重排、查询扩展
  generation/  上下文组装（整文件 + 行号）、提示词、DeepSeek 流式调用
  routes/      query / health / audio 三组接口
  asr/         Qwen3-ASR 子进程封装
frontend/      三栏界面、SSE 接收、Markdown 渲染
tests/         离线单测与检索评测，不依赖模型和 Key
docs/          DESIGN.md（设计报告）、CHANGELOG.md（逐轮修复）、截图
build_index.py 索引构建入口
run.py         服务入口
```

## 测试

```bash
pip install -r requirements-dev.txt   # 运行依赖 + pytest + ruff + httpx
pytest -q                             # 78 passed，不需要模型和 API Key
```

离线覆盖分块边界、检索融合、上下文预算、行号、整文件回退、SSE 心跳、HTTP 状态码、ASR 错误分层和路由装配；`tests/test_e2e_smoke.py` 用本仓库源码加假向量跑通「建索引 → 双路检索 → RRF → 组装上下文」。设计细节见 [docs/DESIGN.md](docs/DESIGN.md)，逐轮修复见 [docs/CHANGELOG.md](docs/CHANGELOG.md)。

## License

MIT © 2026 zhangjingtong000
