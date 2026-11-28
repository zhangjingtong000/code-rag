# 代码库问答系统（RAG for source code）

把一份源码仓库灌进索引，然后用中文提问：「这个项目怎么建索引」「BM25 和向量怎么融合」。检索按函数/类边界定位，回答时喂给模型的是**完整文件**，不是切碎的片段。

**一句话差异**：代码类文件按结构边界切块，不按固定字符数切；检索同时跑语义和精确两条路；上下文以整文件为单元，而不是 top-k 碎片。

## 定位

单机代码问答 demo：一份源码目录、一个本地嵌入模型、一个 API Key，起一个进程回答问题。

查询表、索引句柄都放在进程内存里，没有鉴权，CORS 全开，也没有多进程共享状态；ASR 依赖外部工具子进程。适合本机、单用户、中小体量语料。要对外提供服务，需要自行补上鉴权、限流、CORS 白名单、多进程状态共享，以及 ASR 的异常兜底。

> **安全边界（起服务前先读）**：默认 `host = "0.0.0.0"`（`backend/config.py`）、CORS `allow_origins=["*"]`、全链路零鉴权。这意味着同一网段任何人都能直接调 `/api/rag/query`，读到你的源码片段、消耗你的 DeepSeek 额度，也能 POST `/api/rag/audio/transcribe` 往磁盘写临时音频文件。**只在可信内网或本机使用**；本机自用建议把 `host` 改成 `127.0.0.1`，对外必须自己加鉴权 + CORS 白名单 + 反向代理限流。

## 不是又一个 RAG demo

拿文档问答那套（定长切块 + 向量召回 + 塞 top-k）直接套代码，会在这三处翻车：

**1. 切分粒度。** 文档按段落切没问题，代码按 512 字符切会把函数劈成两半。本项目按文件类型分五类处理：Python 在 `def`/`class`/装饰器边界切，超大类再按方法切；JS 认 `Page({...})`/`App({...})` 并单独处理超大 Page 块；WXML/HTML 仅在超过 3000 字符时才按 `<template>`/`<view class>` 切；WXSS/CSS 整文件一块；JSON 整文件单块，只进关键词索引、不占向量。所以真正按结构切的是 Python 与 JS 两套，WXML 是「超长才切」的条件策略，WXSS/JSON 是整文件单块——不是五套并列的结构策略。宁可块大小不均，也不在函数中间下刀。

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

索引侧：`scanner` 扫 `data/source-*` → `chunker` 按结构切块 → 每块前置一段文件语义标签（这个文件是干什么的）→ 向量入库 + BM25 落盘。增量构建只对变更文件重新嵌入，关键词语料每次全量重建。语义标签表与项目背景文案是外置配置（`backend/semantic_tags.json`），换项目改那个文件即可；但检索侧还有术语表、加权倍数、查询扩展分支写死在 Python 里，见「换项目时要改什么」。

```
backend/        FastAPI：indexing / retrieval / generation / routes / asr
backend/semantic_tags.json  文件语义标签表 + 项目背景文案（换项目改这里）
frontend/       原生 JS 单页（三栏：问答记录 | 对话 | 快捷提问）
docs/DESIGN.md  设计报告：选型、踩坑、调参速查
tests/          离线回归测试 + 端到端 smoke test（pytest -q）
run.py          启动入口（uvicorn，8001）
build_index.py  索引构建脚本（full / incremental，--source-dirs 覆盖语料）
```

## 前置要求

| 项 | 说明 |
|---|---|
| Python | 3.10+ |
| 嵌入模型 | 本地 Ollama，并拉取 `unclemusclez/jina-embeddings-v2-base-code:q5`（768 维） |
| LLM | DeepSeek API Key，通过环境变量 `DEEPSEEK_API_KEY` 提供 |
| 索引 | 仓库不附带 `data/`（Chroma 库、BM25 pkl、chunk 元数据）与模型权重，首次使用需自行构建，命令见「快速开始」 |
| 语料 | 待问答的源码放进 `data/source-main/`（可选 `source-release/`、`source-rag/` 做多分支对比），只扫 `.py/.js/.wxml/.wxss/.json/.html/.css` |

## 快速开始

以下步骤假定「前置要求」已满足。

```bash
# 1. 依赖
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. 嵌入模型（768 维）
ollama pull unclemusclez/jina-embeddings-v2-base-code:q5
curl http://127.0.0.1:11434/api/embed -d "{\"model\":\"unclemusclez/jina-embeddings-v2-base-code:q5\",\"input\":[\"test\"]}"

# 3. 放入待问答的源码，默认目录 data/source-main/
#    可再建 source-release/、source-rag/ 做多分支对比

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

## 换项目时要改什么

换项目时**只有一部分能靠配置表迁走**，另一部分写死在 Python 里：

- **可配置**：`backend/semantic_tags.json` 里 `file_semantic_tags` 的键是相对路径（可加 source 前缀，如 `release::app/config.py`）、值是这个文件做什么，`project_background` 是拼进 LLM 上下文的项目背景。**这份配置的默认内容描述的就是本仓库自身的源码**，换语料时整体替换（本仓库的标签表只收录仓库里真实存在的路径）。也可以用 `RAG_SEMANTIC_TAGS` 指向另一份配置；文件缺失或格式不对时回退到内置默认值，索引照常可建。
- **硬编码，换项目要改 Python**：`backend/retrieval/query_expansion.py` 的 `TERM_MAP`（160 组中英术语映射，其中混着 `云函数`/`wx.cloud.database`/`imgSecCheck`/`chooseMedia` 这类小程序词，以及本仓库自己的文件名、类名），`expand_query()` 里 `is_rag_context` 的判断（只认 `rag` / `问答系统`）和它追加的那串写死的文件名；`backend/retrieval/fusion.py` 的 `SOURCE_BOOST`（`rag` 2.0、`main`/`release` 1.5）、`LANG_BOOST`（`wxml`/`wxss`/`html`/`css`/`javascript` 各自倍数）与 `detect_query_context()` 里的关键词分支。这些是「本项目自己的知识」，不读配置；换到别的语料要想检索质量不塌，得逐条改。

## 测试

```bash
pytest -q
```

不装模型、不配 key 也能验证检索链路（`python -m unittest discover -s tests -v` 等价）。其中 `tests/test_e2e_smoke.py` 是端到端 smoke test：拿本仓库自己的源码当语料，跑「建索引 → 向量 + BM25 检索 → RRF 融合 → 组装上下文」，嵌入用确定性假向量，不连 Ollama、不调 LLM。

`tests/test_chunker.py` 锁分块边界（Python 函数/类不被劈开、装饰器跟着它修饰的函数、首个 `def` 前的模块级代码不丢、块行号与源码一致）；`tests/test_context_builder.py` 锁上下文预算（超预算时按命中行截断并留「已截断」标注、预算不够时后续文件不收录、总 token 不越界）。这两块过去没有直接覆盖。

`tests/test_app_import.py` 是路由装配层的冒烟测试：`from backend.app import app` 必须成立、`app.routes` 非空、health/query/index/audio 四条路由都挂上、audio 路由按 multipart 解析上传、OpenAPI schema 能生成。其余测试都是直接调模块函数，从不 import `backend.app`，所以「路由装配阶段就崩」的问题（例如缺 `python-multipart`）能带着 18 条全绿溜过去——这条用例专门堵这个洞。

当前 **19 passed**（12 条离线单测 + 6 条语义配置用例 + 1 条应用装配冒烟）。

## 实现要点

快照里几处会让链路静默失效的地方已经修掉，并补了离线回归：

- **同一块两套 id**：向量侧用 md5、BM25 侧用 `source:path:start_line`，同一块在 RRF 里被算成两条 → 统一由 `chunker.chunk_key()` 生成，两库共用。
- **BM25 命中被丢弃**：BM25 结果不带 metadata，`context_builder` 整条跳过 → metadata 随 pkl 持久化，关键词命中能进上下文。
- **增量构建后旧文件搜不到**：BM25 只按变更文件重建 → 关键词语料每次全量重建，只有嵌入走增量。
- **切块漏内容**：JSON 配置块 content 原被清空；首个 `def`/`class` 之前的模块级代码、`var X = class {}` 形式、Page 块重复切 → 均已修正，默认排除规则补上 `site-packages`/`venv`。
- **链路可回归**：`tests/test_e2e_smoke.py` 覆盖「建索引 → 双路检索 → RRF → 组装上下文」并断言至少召回一条。真实嵌入与生成需要模型和 Key，不在自动化范围内。
- **语料与文案外置**：`RAG_SOURCE_DIRS` / `build_index.py --source-dirs` 覆盖语料目录，默认值不变；文件语义标签与项目背景移到 `backend/semantic_tags.json`（可用 `RAG_SEMANTIC_TAGS` 指向另一份）。检索侧的 `TERM_MAP`、boost 倍数、`expand_query()` 分支仍是硬编码，换项目要改 Python。
- **前端不再直接吃 marked 的输出**：`renderMarkdown()` 先过 DOMPurify 再进 `innerHTML`；`showError()` 改用 `textContent`。
- **多轮对话不再重复注入当前问题**：`_getChatHistory()` 排除末尾「当前问题 + 空助手占位」两条，之前 `slice(-11, -1)` 只排掉占位，同一句问题会同时出现在 `chat_history` 和 `question` 里。

## 使用注意

chunk id 方案变过（现由 `chunker.chunk_key()` 统一生成），旧版本建出的索引与新代码不兼容。升级后先删掉旧索引再重建：

```bash
rm -rf data/chroma_db data/bm25_index.pkl data/chunk_metadata.json data/file_hashes.json
python build_index.py full
```

直接删整个 `data/` 也可以，但那会连带删掉 `data/source-*` 下的语料；语料只放在那里时先备份，或用 `--source-dirs` 指向仓库外的目录。

**前端依赖外网 CDN。** `frontend/index.html` 从 jsdelivr 取 marked 4.3.0、从 cdnjs 取 DOMPurify 3.0.8 和 highlight.js 11.9.0（含 python / javascript 语言包），共 5 个外部资源。国内网络访问这两个 CDN 经常很慢或被拒；任一脚本加载失败，页面就是白屏或 Markdown 不渲染（脚本在 `<head>` 里同步阻塞）。离线或内网环境需自行把这几个文件放到 `frontend/vendor/`，再改 `index.html` 的引用。

**LLM 输出不作为可信内容。** 模型复述的是被索引的源码，源码本身不可控，所以 `renderMarkdown()` 在 `innerHTML` 之前先过 DOMPurify（只放行渲染器自己产生的 `onclick="toggleFold('id')"` / `onclick="copyCode(this,'id')"`）；DOMPurify 没加载成功时退化为标签/属性黑名单过滤，安全性下降但仍强于裸 `innerHTML`。要接不可信语料，请把 CDN 换成自托管并收紧白名单。

**索引为空时直接报错，不再假装就绪。** 语料目录扫不到文件时 `Indexer.build_index()` 抛异常且**不改动旧索引**；`POST /api/rag/index/rebuild` 返回 `409`（不再回 `started`）；`chunk_metadata.json` 里 `chunks_count == 0` 时 `/api/rag/index/status` 报 `error`，`POST /api/rag/query` 也回 `409` 并说明原因，不会白烧一次 LLM 调用。

## 技术栈

| 层 | 选型 |
|---|---|
| 服务 | FastAPI + uvicorn，SSE 流式（同步生成器 + 心跳保活） |
| 向量库 | ChromaDB（持久化，cosine） |
| 关键词 | rank-bm25（自定义中英文混排分词） |
| 嵌入 | jina-embeddings-v2-base-code（Ollama，768 维） |
| 生成 | DeepSeek Chat（64K 窗口，动态输出预算） |
| 精排 | 候选 > 20 条时调 DeepSeek 重排并截到 20 条（16–20 条不调模型，直接按 RRF 顺序取前 20；调模型失败则降级为原序） |
| 前端 | 原生 HTML/CSS/JS + marked@4.3.0 + DOMPurify@3.0.8 + highlight.js 11.9.0（版本锁定，全部走外网 CDN，见「使用注意」） |
| 语音 | 可选 Qwen3-ASR（子进程调用，未安装则不可用） |

## License

MIT © 2026 zhangjingtong000
