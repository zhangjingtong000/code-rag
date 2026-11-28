# 变更记录

记录每轮「静默失效」修复：现象、根因、现在的行为，以及锁住它的离线回归。
README 只留定位与用法，细节都放这里。

## 2026-11-28（第二轮）：两处「文档说 A、实际做 B」

### 快速开始承诺整文件，实际静默退化成 chunk 拼接

- **现象**：README 写「回答时喂给模型的是整份文件」，但快速开始只执行
  `python build_index.py full --source-dirs "rag=."`，紧接着的 `python run.py`
  没有 `RAG_SOURCE_DIRS`。`ContextBuilder._read_source_file()` 按
  `settings.source_dirs` 找文件，启动时是默认的 `data/source-rag`（不存在），
  返回 `None` 后静默拼 chunk。
- **实测**（同一文件 `backend/retrieval/vector_store.py`）：照 README 原样，
  上下文开头是 `[文件说明: RAG系统向量存储…]`；把语料路径带到服务端后，开头是
  `"""ChromaDB vector store wrapper."""`，`def search` 落在第 58 行（源文件确为
  58）。前者没有任何日志或警告。
- **现在**：`Indexer._save_metadata()` 把构建时的语料路径写进
  `data/chunk_metadata.json` 的 `source_dirs`；`ContextBuilder` 优先按它解析文件
  （其次 `RAG_SOURCE_DIRS`，最后旧的 `data/source-<label>`）。真的读不到时打
  `logger.warning`，并在上下文里显式标注「整文件不可用，以下为 chunk 拼接，
  行号不可引用」。
- **锁住**：`tests/test_full_file_context.py::IndexedCorpusPathTest`（不设环境
  变量也能读到整文件；语料消失时 warning + 标注）。

### 「每段代码标注行号」，但行号是模型编的

- **现象**：System Prompt 硬性要求第 3 条要模型标 `[来源 | 文件 | 行号: X-Y]`，
  而 `ContextBuilder.build()` 输出的是裸代码块，一个行号都没有。它自己截图里
  `02-answer.png` 标 `vector_store.py | 行号: 78-95`，实际 `def search` 在
  58–75 行，整体偏 20 行。
- **现在**：`_number_lines()` 给每行加真实行号（` 58      def search(...)`），
  截断切的是同一条行数组，片段里的行号仍然对得上；System Prompt 改成「行号只能
  取自左侧数字，不得自行推算」。
- **锁住**：`tests/test_full_file_context.py::LineNumberTest`（完整文件逐行、
  截断后仍命中第 200 行、截断后仍在预算内）。

### 其他

- `expand_query()` 不再往查询里塞 20 多个写死的本仓库文件名（`index.html`、
  `style.css`、`app.js`、`ChatApp`、`_initMic`…），只注入 `source-rag`、
  `RAG backend` 这类与语料无关的方向词；`TERM_MAP` 删掉上一个项目的微信小程序
  词条（`wx.cloud`、`chooseMedia`、`navigateTo`、`tabBar`、`imgSecCheck`、
  `腾讯云`、`火山`、`云函数`），157 组。
- 新增检索评测：`tests/test_retrieval_quality.py` 用本仓库源码建 BM25，对 10
  个中文问题算 recall@5 与 MRR（当前 0.90 / 0.71），README 里公开了这两个数字。
- 新增 HTTP 层测试：`tests/test_http_api.py` 覆盖 `/api/rag/query` 的 400/409、
  SSE 的 `metadata/token/done/id`、未知 query_id 的 404、异常转 `event: error`。
- ASR 错误分层：本地模型目录不存在时退回 HF repo id（注释承诺过、代码没实现的
  分支），模型加载失败包装成 `RuntimeError` → 503；音频解不开 → 422。
  `tests/test_asr_backend.py` 锁住。
- 重排阈值与 README 对齐：`rerank(..., top_n=15)`，16–20 条候选不再被
  `len(candidates) <= top_n` 提前返回。
- SSE 事件补 `id:`；前端删掉实际不生效的 `Last-Event-ID` 续传逻辑。
- `requirements.txt` 全部加上界；`requirements-dev.txt` 补 `httpx`（TestClient
  需要）。
- `config.py` 六个 property 里重复的 `import os as _os` 去掉，统一用顶部 import。
- README：Windows / macOS 快速开始分开写，`RAG_SOURCE_DIRS` 分隔符说明
  （`;` vs `:`），加 CI / License / Python badge，说明 `tests/` 不进自举索引、
  SSE 不做断点续传。

## 2026-11-28：把「只修一半」的问题补齐

### `rag` 词边界只修了链路的一半

- **现象**：`expand_query("average latency of the search")` 会把 `source-rag`、
  `rag-system`、`index.html`、`style.css`、`app.js` 等 20 多个写死的本仓库
  文件名塞进查询；`average` / `drag` / `fragment` / `paragraph` 都会触发。
- **根因**：`fusion.py` 已改成 `_RAG_WORD` 词边界，但同一条链路上的
  `query_expansion.py` 仍是 `'rag' in question.lower()`。
- **现在**：两处共用 `fusion._RAG_WORD`（`(?<![a-z0-9_])rag(?![a-z0-9_])`），
  中文紧邻（"这个项目的rag在哪"）仍算独立单词。
  `tests/test_retrieval.py::ExpandQueryWordBoundaryTest` 与
  `RagWordBoundaryTest` 对称，并断言两条路径结论一致。

### 默认监听地址从 `0.0.0.0` 改为 `127.0.0.1`

- **现象**：默认配置 + CORS `*` + 全链路无鉴权，同网段任何人都能读取索引里的
  源码片段、消耗 DeepSeek 额度、向磁盘写临时音频。
- **现在**：`Settings.host` 默认 `127.0.0.1`；需要对外时显式设置
  `RAG_HOST=0.0.0.0`（`RAG_PORT` 同理）。`tests/test_server_config.py` 锁默认值。

### 首个文件超预算时不再返回空上下文

- **现象**：`max_tokens=1000` + 单文件 3000 行时，输出里只有项目背景 header，
  `contains huge.py: False`，也不报错——模型只能凭空回答「源码中未找到」。
- **根因**：`context_builder.py` 在 `if sections:` 为空时直接 `break`。
- **现在**：首个文件超预算也会收录一段按命中行截断的内容；命中行优先保留
  （先收窄 ±行半径，再按字符截断）；连一段都放不下时写入显式提示
  `[上下文预算不足…]`，不再静默。
  `tests/test_context_builder.py` 新增三个场景。

### 存在但不生效的配置

- `expand_query_llm()` 已接入 `/api/rag/query`：`settings.llm_expansion_enabled`
  为真才调 LLM（默认关闭时与 `expand_query()` 完全一致），结果按问题进程内缓存，
  与 DESIGN 5.2 的「首次调用缓存」一致。
- `full_file_top_k` 真正限制进上下文的文件数（原来只读进 `__init__` 就没人用）。
- 删掉从未使用的 `Settings.summaries_path`、`IndexStatusResponse.eta_seconds`，
  以及 `typewriter_frame_ms` / `typewriter_chars_{min,max}`（前端打字机参数其实
  硬编码在 `streaming.js`，后端这三个字段没人读）。
- `app.js` 的 `maxRecordMs` 原来只是赋值，录音判断写死 `elapsed >= 30`；现在真的
  用它判断 30 秒上限。

### 前端长回答不再每帧全量重排

- **现象**：打字机每 16ms 一帧，每帧对全文 `marked.parse` + DOMPurify +
  `innerHTML` 全量替换，几千字的代码回答持续全量重排。
- **现在**：渲染节流到 100ms；正文超过 4000 字符后走 `textContent` 快路径
  （不解析 Markdown），完成后整体渲染一次。`appendStreamText` 只累积文本。

### 其他

- `health.py::_read_metadata()` 捕获坏 JSON / 非对象 JSON，按「未构建」处理，
  `/health`、`/index/status` 不再 500。`tests/test_health_metadata.py` 锁住。
- `fusion.rrf_fusion()` 删掉从未使用的 `context` 参数（加权已由 `_boost` 承载）。
- `query._log()` 的调试日志改为跟随 `settings.data_dir`，不再依赖进程 CWD。
- `QueryRequest.chat_history` 改用 `Field(default_factory=list)`，不再共享可变默认值。
- 新增 `pyproject.toml`（ruff + pytest 配置）；CI 跑 3.10 / 3.11 / 3.12 并执行
  `ruff check`，与 README 的「Python 3.10+」一致。

## 更早的修复（原 README「实现要点」）

- **同一块两套 id**：向量侧用 md5、BM25 侧用 `source:path:start_line`，同一块在
  RRF 里被算成两条 → 统一由 `chunker.chunk_key()` 生成，两库共用。
- **BM25 命中被丢弃**：BM25 结果不带 metadata，`context_builder` 整条跳过 →
  metadata 随 pkl 持久化，关键词命中能进上下文。
- **增量构建后旧文件搜不到**：BM25 只按变更文件重建 → 关键词语料每次全量重建，
  只有嵌入走增量。
- **切块漏内容**：JSON 配置块 content 原被清空；首个 `def`/`class` 之前的模块级
  代码、`var X = class {}` 形式、Page 块重复切 → 均已修正，默认排除规则补上
  `site-packages`/`venv`。
- **链路可回归**：`tests/test_e2e_smoke.py` 覆盖「建索引 → 双路检索 → RRF →
  组装上下文」并断言至少召回一条。真实嵌入与生成需要模型和 Key，不在自动化范围。
- **语料与文案外置**：`RAG_SOURCE_DIRS` / `build_index.py --source-dirs` 覆盖语料
  目录，默认值不变；文件语义标签与项目背景移到 `backend/semantic_tags.json`
  （可用 `RAG_SEMANTIC_TAGS` 指向另一份）。检索侧的 `TERM_MAP`、boost 倍数、
  `expand_query()` 分支仍是硬编码，换项目要改 Python。
- **前端不再直接吃 marked 的输出**：`renderMarkdown()` 先过 DOMPurify 再进
  `innerHTML`；`showError()` 改用 `textContent`。
- **多轮对话不再重复注入当前问题**：`_getChatHistory()` 排除末尾「当前问题 +
  空助手占位」两条，之前 `slice(-11, -1)` 只排掉占位。
- **精排顺序写回分数**：`rerank()` 只返回新顺序时 `ContextBuilder` 仍按
  `rrf_score` 排文件，重排对最终上下文没有影响。现在按新顺序重写 `rrf_score`，
  `tests/test_reranker.py` 锁住「新顺序决定上下文文件顺序」。
- **WXML 行号不再偏一行**：`_chunk_wxml` 的正则曾把换行符算进匹配起点，
  `start_line` 整体晚一行（40 块里 39 块错位）。改成行首零宽匹配。
- **token 预算与估算**：输入预算从「context + 500」改为对 system prompt +
  对话历史 + context 求和；`estimate_tokens` 不再把中文折半（38 汉字算 38 tokens，
  旧算法是 18），截断字符数也按 2 chars/token 保守取。
- **SSE 心跳真的会发**：心跳检查原先写在 `for token in ...` 循环体内，模型沉默时
  永远走不到；现在后台线程读流、生成器带超时等待，沉默 15 秒就发一次注释。
- **BM25 进程内缓存**：查询路径不再每次 `BM25Store().load()` 全语料重建
  `BM25Okapi`，按 pkl 的 (mtime, size) 复用，重建索引后自动失效。
- **排除规则按路径段匹配**：`test_`、`images` 不再做子串匹配，
  `contest_utils.py`、`metadata.py` 这类文件名不会被误伤。
- **默认不热重载**：`run.py` 的 `reload` 改为 `RAG_DEV=1` 才开；
  `@app.on_event("startup")` 换成 lifespan；`allow_origins=["*"]` +
  `allow_credentials=True` 这个浏览器不认的组合改成 `allow_credentials=False`；
  `convert_to_wav()` 与 `_postProcessCodeBlocks()` 两个死函数删掉。
