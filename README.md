# 代码库问答系统（RAG for source code）

把一份源码仓库灌进索引，然后用中文提问：「图像识别怎么实现的」「上传做了哪些校验」。检索按函数/类边界定位，回答时喂给模型的是**完整文件**，不是切碎的片段。

**一句话差异**：切块按代码结构切，不按固定字符数切；检索同时跑语义和精确两条路；上下文以整文件为单元，而不是 top-k 碎片。

## 不是又一个 RAG demo

拿文档问答那套（定长切块 + 向量召回 + 塞 top-k）直接套代码，会在这三处翻车：

**1. 切分粒度。** 文档按段落切没问题，代码按 512 字符切会把函数劈成两半。本项目按语言各自的结构切：Python 在 `def`/`class`/装饰器边界切，超大类再按方法切；JS 认 `Page({...})`/`App({...})` 并单独处理超大 Page 块；WXML/HTML 按 `<template>`/`<view class>` 切；WXSS/CSS 整文件一块；JSON 单块，只进关键词索引、不占向量。宁可块大小不均，也不在函数中间下刀。

**2. 两条检索路径。** 「登录」要能匹配到 `login`（语义），`getUserInfo` 必须精确命中（关键词）。只有向量找不到标识符，只有 BM25 理解不了同义词。所以两路都跑：Chroma 向量 + rank-bm25，RRF 融合，再按来源/语言加权。查询扩展维护 160 组中英术语映射（`识别→recognize`、`分块→chunk`）。

**3. 调用链依赖。** 代码答案的价值在于完整：调用方、被调用方、import、配置。检索只负责定位文件；命中文件以**全文**进上下文（32K token 预算），单个超大文件按命中行 ±300 行截断，截断处显式标注。分块管「找得准」，不管「答得全」。

## 架构与检索流程

```
POST /api/rag/query → query_id → GET /api/rag/query/stream/{id}（SSE）
        │
   查询扩展（160 组术语映射 + 可选 LLM 扩展）
        ├── 向量检索  Chroma + jina-embeddings-v2-base-code（Ollama，768 维）
        ├── 关键词检索 rank-bm25（中英混排分词，标识符不拆）
        └── RRF 融合  1/(60+rank) + 来源/语言加权 → 去重 → 同文件最多 3 块
        │
   上下文组装：按 RRF 排名补全整文件，32K 预算，超出按命中行 ±300 截断
        │
   DeepSeek 流式生成 → SSE → 前端打字机渲染（Markdown + 高亮 + 折叠/复制）
```

索引侧：`scanner` 扫 `data/source-*` → `chunker` 按结构切块 → 每块前置一段文件语义标签（这个文件是干什么的）→ 向量入库 + BM25 落盘。增量构建只对变更文件重新嵌入，关键词语料每次全量重建。语义标签表与项目背景文案是外置配置（`backend/semantic_tags.json`），换项目改那个文件即可，不用动 Python 源码。

```
backend/        FastAPI：indexing / retrieval / generation / routes / asr
backend/semantic_tags.json  文件语义标签表 + 项目背景文案（换项目改这里）
frontend/       原生 JS 单页（三栏：问答记录 | 对话 | 快捷提问）
docs/DESIGN.md  设计报告：选型、踩坑、调参速查
tests/          离线回归测试 + 端到端 smoke test（pytest -q）
run.py          启动入口（uvicorn，8001）
build_index.py  索引构建脚本（full / incremental，--source-dirs 覆盖语料）
```

## 快速开始

需要：Python 3.10+、本地 Ollama（嵌入模型）、DeepSeek API Key。

```bash
# 1. 依赖
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. 嵌入模型（768 维）
ollama pull unclemusclez/jina-embeddings-v2-base-code:q5
curl http://127.0.0.1:11434/api/embed -d "{\"model\":\"unclemusclez/jina-embeddings-v2-base-code:q5\",\"input\":[\"test\"]}"

# 3. 放入待问答的源码，默认目录 data/source-main/
#    只扫 .py/.js/.wxml/.wxss/.json/.html/.css；可再建 source-release/、source-rag/ 做多分支对比

# 4. 建索引（语料不在默认位置时，用 RAG_SOURCE_DIRS 或 --source-dirs 指定）
#    格式：“标签=路径”，多个用系统路径分隔符（Windows 为 ;）连接
$env:RAG_SOURCE_DIRS="main=D:\code\my-app;release=D:\code\my-app-release"
python build_index.py full --source-dirs "main=D:\code\my-app"
# 增量：python build_index.py incremental

# 5. 启动
set DEEPSEEK_API_KEY=sk-...       # PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
python run.py
# 打开 http://127.0.0.1:8001/frontend/index.html
```

`RAG_SOURCE_DIRS` 不设时行为不变（默认仍是 `data/source-{main,release,rag}`）。标签只写了路径没写标签时，取目录名当标签。

换项目时不用改 Python：`backend/semantic_tags.json` 里 `file_semantic_tags` 的键是相对路径（可加 source 前缀，如 `release::app/config.py`）、值是这个文件做什么，`project_background` 是拼进 LLM 上下文的项目背景。也可以用 `RAG_SEMANTIC_TAGS` 指向另一份配置；文件缺失或格式不对时回退到内置默认值，索引照常可建。

不装模型、不配 key 也能验证检索链路：`pytest -q`（`python -m unittest discover -s tests -v` 等价）。其中 `tests/test_e2e_smoke.py` 是端到端 smoke test：拿本仓库自己的源码当语料，跑「建索引 → 向量 + BM25 检索 → RRF 融合 → 组装上下文」，嵌入用确定性假向量，不连 Ollama、不调 LLM。

## 已知问题

原始快照的 3 个缺陷已在本仓库修复，并补了离线回归测试：

1. 向量 id 用 md5、BM25 id 用 `source:path:start_line`，同一块在 RRF 里算两条 → 统一由 `chunker.chunk_key()` 生成，两库共用。
2. BM25 结果没有 metadata，被 `context_builder` 整条丢弃 → metadata 随 pkl 持久化，关键词命中现在能进上下文。
3. 增量构建 BM25 只用变更文件重建，未变更文件从此搜不到 → 关键词语料每次全量重建，只有嵌入走增量。
4. 附带修复：JSON 配置块的 content 原被清空（既没进向量也没进 BM25）；切块器会丢掉首个 `def`/`class` 之前的模块级代码、漏掉 `var X = class {}` 形式、并把 Page 块切两遍；默认排除规则补上 `site-packages`/`venv`。

本轮补上的两处（原先列在「仍然存在」里）：

- **已补端到端测试**：`tests/test_e2e_smoke.py` 覆盖「建索引 → 检索 → 返回结果」全链路并断言至少召回一条。真实的 Ollama 嵌入与 DeepSeek 生成仍不在自动化范围内（要模型 + Key），smoke test 用假嵌入跑通链路，替代了原先那些未入库的一次性集成脚本。
- **已补语料路径可覆盖**：`RAG_SOURCE_DIRS`（环境变量）与 `build_index.py --source-dirs`（命令行）可指定语料目录，默认值不变；`context_builder` 读整文件时也跟随同一份配置。
- **已把语义标签与项目背景外置成配置**：文件语义标签表（原 `indexer._FILE_SEMANTIC_TAGS`，69 条）与上下文里的项目背景文案都搬进了 `backend/semantic_tags.json`，加载器是 `backend/semantic_tags.py`；换项目只改 JSON（或用 `RAG_SEMANTIC_TAGS` 指向另一份），不用改 Python 源码。文件缺失/损坏时回退到内置默认值，索引与上下文仍能构建。回归测试见 `tests/test_semantic_config.py`。

仍然存在（如实列出）：

- **索引不兼容（说明，不是缺陷）**：chunk id 方案变过，旧索引必须删掉 `data/` 重建。
- **定位是单机 demo，非生产服务**：查询表在进程内存、无鉴权、无并发保护，CORS 全开；ASR 依赖外部工具，异常未兜底。要上线得自己补鉴权、限流、多进程共享状态和 CORS 白名单。
- **本仓库不附带索引**：`data/`（chroma + BM25 pkl + 元数据）和模型权重都不入库，需按上面的步骤自行构建。仓库也从未附带第三方库语料——早期快照混入过 `site-packages`（1,287 个文件里 1,111 个是第三方库），现已清掉并补了默认排除规则。

## 技术栈

| 层 | 选型 |
|---|---|
| 服务 | FastAPI + uvicorn，SSE 流式（同步生成器 + 心跳保活） |
| 向量库 | ChromaDB（持久化，cosine） |
| 关键词 | rank-bm25（自定义中英文混排分词） |
| 嵌入 | jina-embeddings-v2-base-code（Ollama，768 维） |
| 生成 | DeepSeek Chat（64K 窗口，动态输出预算） |
| 精排 | 候选 > 15 时用 DeepSeek 重排（可失败降级） |
| 前端 | 原生 HTML/CSS/JS + marked@4.3.0 + highlight.js 11.9.0（版本锁定） |
| 语音 | 可选 Qwen3-ASR（子进程调用，未安装则不可用） |

## License

MIT © 2026 zhangjingtong000
