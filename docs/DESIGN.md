# 面向项目代码库的 RAG 问答系统设计指南

> 适用场景：为任意项目的源码仓库搭建 AI 问答助手，让用户用自然语言询问代码细节、功能实现、项目结构等问题。
> 本指南基于真实项目中从零搭建到稳定运行的完整经验，涵盖设计思路、常见陷阱、调试方法和可迁移的解决方案。

> **本文的经验来自另一个项目，不是本仓库的说明书。** 下面这套流程最早是在一个
> 「微信小程序 + 图像识别」项目（未开源，main / release 双分支）上打磨出来的，
> 本仓库是它的自举改造版：用同一套流程问答本仓库自己的源码。因此文中提到的
> 小程序文件结构——`Page({...})`、WXML/WXSS、`app.json`、`sitemap.json`、
> main/release 分支对比、`def recognize(image)` 这类示例——在本仓库里并不存在。
> 4.3 / 4.4 的分块规则、8.4 的示例问题、8.5 的代码块示例都属于那个项目；本仓库
> 对应位置的实际情况是：WXSS/JSON 整文件单块、WXML/HTML 仅在超过 3000 字符时
> 才按 `<view class>` 切分、`fusion.SOURCE_BOOST` 里的 main/release 倍数是按那对
> 分支标的。读的时候把「小程序」当成「你手上的语料」即可。

---

## 一、RAG 用于代码问答的特殊性

代码问答和普通文档问答有本质区别：

| 维度 | 文档问答 RAG | 代码问答 RAG |
|------|-------------|-------------|
| 语义单位 | 段落、句子 | 函数、类、组件 |
| 相关性 | 语义相似即可 | 语义 + 精确匹配（函数名、类名） |
| 上下文依赖 | 低（段落相对独立） | 高（函数调用链、import、继承） |
| 用户提问 | 自然语言 | 自然语言 + 技术术语混用 |
| 答案要求 | 概括或摘录 | 完整代码 + 解释，不能省略 |

---

## 二、核心架构

```
用户提问
  → 查询扩展（术语映射：中文→英文、缩写→全称）
  → 混合检索
      ├─ 向量检索（语义匹配）
      ├─ 关键词检索（精确匹配：BM25）
      └─ RRF 排名融合
  → 全文件补齐（高相关文件传完整内容，碎片看不懂代码）
  → 上下文组装（System 极短硬护栏 + User 承载完整上下文）
  → LLM 流式生成（同步生成器，非异步线程）
  → 前端渲染（Markdown + 代码高亮 + 打字机缓冲 + 智能滚动）
```

核心思想：**Chunk 只用于检索精准定位，最终传给 LLM 的是完整文件或大段上下文**。

---

## 三、模型选择

### 3.1 生成模型（LLM）

选择标准按优先级：

1. **上下文窗口足够大**（≥ 64K tokens，推荐）— 代码文件动辄几百行，需要一次传入多个完整文件
2. **代码理解能力强** — 能读懂多语言代码，能解释逻辑
3. **指令跟随好** — 能遵守 "贴完整代码"、"标注来源"、"禁止省略" 等约束
4. **支持流式** — 改善用户体验
5. **成本可控**

**推荐模型**：

| 模型 | 上下文 | 优势 | 适用场景 |
|------|--------|------|----------|
| **DeepSeek Chat** | 64K | 代码理解好、指令跟随强、流式稳定、中文优秀、性价比高 | 首选推荐，实测效果好 |
| GPT-4o / GPT-4o-mini | 128K | 综合能力强、生态好 | 预算充足时 |
| Qwen3.5 / Qwen-Coder | 32-128K | 代码专精、开源可本地部署 | 数据敏感需本地运行时 |
| Claude Sonnet 4 | 200K | 超长上下文、代码分析深度好 | 大项目多文件对比 |

> **实践建议**：用同一套问题集（覆盖定位/实现/解释/列举/排查各 2-3 题）对比候选模型，重点关注完整性和准确性。实测 DeepSeek Chat 在 64K 窗口 + 动态 token 预算下输出稳定，性价比突出。

**API 调用注意事项**：

- 走代理时用 `session.trust_env = False` 绕过，避免代理超时断开长连接
- 超时设 `timeout=(30, 300)` — 30 秒连接超时，300 秒读取超时（流式需要）
- `max_tokens` 动态计算而非写死：`min(16384, 总窗口 - 输入token数 - 缓冲)`

### 3.2 嵌入模型（Embedding）

选择标准：

| 指标 | 说明 |
|------|------|
| 代码语义敏感 | 通用模型对代码的理解远不如代码专用模型 |
| 上下文窗口 | 至少 512 tokens，推荐 8192 tokens（允许更大的 chunk） |
| 批量接口 | 知识库构建必须用批量请求，逐条调用本地嵌入服务的时间成本不可接受 |
| 部署方式 | 本地部署（数据不出本机，无 API 费用）vs 云端 API（省运维但花钱） |

**推荐模型**：

| 模型 | 维度 | 上下文 | 部署 | 适用场景 |
|------|------|--------|------|----------|
| **Jina Embeddings v2 Base Code** | 768 | 8192 | Ollama 本地 / API | **首选推荐**，代码专用，8192 窗口允许大 chunk |
| BGE-M3 | 1024 | 8192 | Ollama / 本地 | 多语言 + 代码混合，中文好 |
| BGE-small-zh-v1.5 | 512 | 512 | sentence-transformers | 轻量中文场景（33MB） |
| text-embedding-3-small | 1536 | 8191 | OpenAI API | 不想本地部署时 |

**Jina v2 量化版使用方式**（推荐）：

```bash
# 1. 拉取量化模型（~100MB）
ollama pull unclemusclez/jina-embeddings-v2-base-code:q5

# 2. 验证 API
curl http://localhost:11434/api/embed \
  -d '{"model":"unclemusclez/jina-embeddings-v2-base-code:q5","input":["test code"]}'

# 3. Python 调用
resp = requests.post("http://localhost:11434/api/embed",
    json={"model": "unclemusclez/jina-embeddings-v2-base-code:q5", "input": texts},
    timeout=120)
embeddings = resp.json()["embeddings"]
```

> **陷阱**：Ollama 中模型名称必须包含完整路径（`user/model:tag` 而非 `model:tag`），否则返回 `404 {"error":"model not found"}`。**先用 curl 验证模型名称和 API 端点完全正确**，再写代码。Windows 系统代理可能拦截 localhost 请求导致 404，排查时注意区分"模型名称错误"和"代理干扰"两种 404。

### 3.3 两种检索方式的角色分工

| | 向量检索 | 关键词检索 (BM25) |
|------|----------|----------------------|
| 擅长 | "进度条" 匹配到 `progressBar` | `getUserInfo` 精确查找 |
| 擅长 | 相近语义的发现 | 函数名、类名、变量名的精确匹配 |
| 弱点 | 对精确标识符不敏感 | 不理解同义词 |
| 结论 | 两者互补，缺一不可 | |

---

## 四、分块策略

### 4.1 核心矛盾

- **分块太小**（500-1000 字符）：检索精准但 LLM 只看到代码碎片 → 答非所问
- **分块太大**（5000+ 字符）：LLM 能看到完整逻辑，但检索变模糊
- **不分块（全文检索）**：项目小了可以，大了就超出上下文窗口

### 4.2 推荐方案：三层粒度

```
粒1: 文件级说明
     - 每个文件一条，描述职责和关键结构
     - 本仓库实现为人工维护的语义标签表（indexer._FILE_SEMANTIC_TAGS），
       未接 LLM 自动摘要
     - 用于"哪个文件有我要的东西"

粒2: 结构感知分块（code chunk）
     - 在函数/类/组件边界处断开，绝不在中间切断
     - chunk_size = 2000-4000 字符（根据语言调整）
     - 用于检索和初筛

粒3: 全文件上下文
     - 对高相关文件，传完整内容给 LLM
     - 用于最终生成答案
```

**流程**：用粒1+粒2定位相关文件 → 将命中的完整文件（粒3）传给 LLM。

### 4.3 结构感知分块规则

> 注：下表是按前述小程序项目（WXML/WXSS/JSON + Page 组件）整理的。本仓库没有
> 那些文件；这里的 Python/JS 规则与本仓库一致，WXML 一栏是「超长才切」的条件策略。

| 语言 | 分块边界 | 特殊处理 |
|------|----------|----------|
| Python | `class` `def` `async def` 定义 | 装饰器归属前一 block；类内方法过多时在 method 边界再切 |
| JS/TS | `function` `class` `Page({` `App({` `Component({` | `Page({...})` 整体保留（生命周期完整性）；超大 Page 在 `methodName(){` 边界切 |
| WXML/HTML | `<template>` `<scroll-view>` `<view class="xxx">` | 超大模板在 view 层级强制切分 |
| CSS/WXSS | 通常单文件一块 | 文件不大，不必细分 |
| JSON/YAML 配置 | 单文件一块 | **低信息密度降权**：配置类文件跳过向量嵌入，仅纳入 BM25 关键词索引 |

**核心原则**：绝不在函数/方法/类的中间切断。宁可 chunk 大小不均，也要保证语义完整。

### 4.4 兜底策略

- 超大 `Page({})` 或单函数超 5000 字符 → 强制在内部 `methodName(){` 边界再切
- 超大 WXML（>3000 字符）→ 在 `<view class="...">` 层级强制切分
- 低信息密度文件（`app.json` `package.json` `sitemap.json`）→ 跳过向量嵌入，仅 BM25

### 4.5 分块大小建议

| 项目类型 | 推荐 chunk_size | 理由 |
|------|---------------|------|
| 前端项目（JS/TS + HTML + CSS） | 2000-4000 字符 | Page/Component 结构较大 |
| Python 后端 | 1500-3000 字符 | 函数较短，逻辑密集 |
| Java/C# 企业项目 | 2000-4000 字符 | 方法多、类结构复杂 |
| 配置文件密集 | 整文件 / 1000-1500 字符 | 配置项短小 |

---

## 五、检索策略

### 5.1 混合检索 = 向量 + 关键词

不能只有一种。向量擅长语义，关键词擅长精确。用 **RRF** 融合：

```
RRF_score(doc) = Σ 1/(k + rank_in_list)
```

k 值建议 60。RRF 只依赖排名位置，不依赖绝对分数，天然适合融合异构检索结果。

### 5.2 查询扩展（两层，可开关）

用户的自然语言和代码标识符之间存在语义鸿沟：

```
用户说 "登录" → 扩展为 [login, auth, authentication, signin]
用户说 "进度条" → 扩展为 [progress, progressBar]
```

- **L1 静态映射**（默认开启，零延迟）：维护项目中英文术语映射表，随着使用不断补充
- **L2 LLM 扩展**（可配置开关）：用 LLM 将中文问题扩展为 3-5 个英文关键词，首次调用缓存

### 5.3 检索结果后处理

1. **去重**：同一 chunk 只保留一个
2. **同文件限制**：同一文件最多 3 个 chunk（防单一文件垄断）
3. **全文件补齐**：对 RRF top-K 命中的文件，读取完整内容传给 LLM
4. **上下文扩展**：截断时以命中 chunk 的行号范围为中心，前后各扩 300 行；同文件多 chunk 命中取并集

### 5.4 上下文溢出控制（关键）

设总 token 上限（如 ~32-35K，给 LLM 64K 窗口留足输出空间），全文件按 RRF 排名依次放入。截断策略：

1. 按 RRF 排名依次放入完整文件
2. 当下一个文件会导致总额超出上限时，对该文件（无论其本身大小）只保留「最相关区间」
3. 单个文件本身超过阈值（如 20K tokens）时提前截断
4. 「最相关区间」= 该文件在检索阶段被命中 chunk 的行号范围 ±300 行
5. 截断处插入注释标记（`// ... [已截断，完整文件见源码]`），让 LLM 知道此处不完整
6. 截断后仍超出 → 停止添加更多文件
7. 进上下文的文件数另有上限 `full_file_top_k`（默认 4）；若第一个文件就超预算，也保底收录一段按命中行截断的内容——连一段都放不下时写入显式「预算不足」提示，绝不返回一个只有背景 header 的空上下文

**核心公式**：`输出预算 = min(16384, 总窗口 - 输入token数 - 缓冲)`。动态计算避免上下文挤占输出空间。

---

## 六、Prompt 设计

### 6.1 核心理念：短 System + 重 User

经过多轮调试确认的最有效策略：

- **System message**：极短，只设硬护栏（3 条红线，违反作废）
- **User message**：承载项目背景、分支说明、完整源码上下文、回答格式模板
- LLM 对 user 角色的指令遵循度显著高于 system 角色

### 6.2 System Prompt 模板

```
你是专业的代码分析助手。

硬性要求（违反任何一条回答作废）：
1. 禁止省略代码 — 不可写 "..."、"同上"、"类似"、"其余相同"
2. 禁止编造 — 只基于上下文中的源码回答，没有的就说没有
3. 强制标注来源 — 每段代码必须标注 [来源: xxx | 文件: xxx | 行号: X-Y]

除此之外，你可以自由组织回答：解释粒度、行文风格、代码引用范围均由你判断。
目标是让提问者完全理解代码的实现逻辑。
```

### 6.3 User Message 模板

```
## 项目背景
[项目简介，1-2 句]

## 分支说明（如适用）
[如果项目有多个分支，说明各自的定位和差异]

## 相关源代码（完整文件）
[来源: main | 文件: path/to/file.py]
```language
...完整文件内容...
```

## 用户问题
{用户原始问题}

## 回答要求
1. 先概述（1-3 句说清核心逻辑）
2. 贴完整代码（标注来源、文件、行号）
3. 逐段解释实现细节
4. 如果涉及多来源/多分支，分别说明各自实现
```

### 6.4 上下文窗口分配

如果模型支持 64K 上下文：

- ~50% 给检索到的源码（完整文件优先，片段补充）
- ~5% 给 System Prompt + 对话历史
- ~45% 给生成的回答

**不要**让上下文挤占输出空间。输入 token 越多，LLM 可用于回答的 token 越少。

---

## 七、流式输出

### 7.1 架构选择

**推荐：同步生成器 + FastAPI 线程池**。

异步 + 线程 + 队列的复杂架构容易引入难以调试的竞态和死锁。对于 demo 和单用户场景，同步生成器完全够用：

```python
def generate():  # sync generator
    for token in llm.generate_stream(messages):
        yield f"data: {json.dumps({'text': token})}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(generate(), media_type="text/event-stream")
```

FastAPI 自动将同步生成器放入线程池，不阻塞事件循环。

### 7.2 SSE 连接保活

长连接可能被代理、路由器、uvicorn 关闭。措施：

1. **心跳**：15 秒无 token 时发送 SSE 注释 `: heartbeat\n\n`
2. **uvicorn**：`timeout_keep_alive=300`（5 分钟）
3. **HTTP 头**：`Cache-Control: no-cache` + `Connection: keep-alive` + `X-Accel-Buffering: no`

### 7.3 打字机缓冲

LLM 的流式推送通常是 Burst 模式——一次推一批 token，停几秒再推下一批。直接渲染就是"一大段 → 卡住 → 一大段"。

方案：前端维护字符缓冲区和独立定时器：

- `setInterval` 16ms → 从缓冲区取 8-15 个字符追加到显示区
- SSE 接收的 token 追加到缓冲区尾部
- 流结束时一次性 flush 剩余缓冲

### 7.4 自动滚动不打扰用户

- 检测用户是否在底部（容差 ~120px）
- 仅当用户在底部时才自动滚动
- 用户向上滚动查阅历史时完全停止自动滚动
- **不要**用 `isStreaming` 状态覆盖用户的滚动位置

---

## 八、前端设计

### 8.1 整体设计风格

推荐采用**浅色清爽风格 + 单一品牌色**，而非纯白色或暗色。理由：

- 代码块保留深色背景更易读（亮色代码块刺眼）
- 品牌色统一贯穿（按钮、链接、高亮、气泡），避免杂乱配色
- CSS 变量管理全局色彩体系，方便整体换肤

**推荐 CSS 变量体系**：

```css
:root {
  --bg: #f1f5f9;              /* 页面底色（浅灰蓝） */
  --bg-card: #ffffff;          /* 卡片/气泡底色 */
  --primary: #3b82f6;          /* 品牌色 */
  --primary-light: #eff6ff;    /* 品牌色浅底 */
  --text-primary: #0f172a;     /* 主文字 */
  --text-secondary: #475569;   /* 次文字 */
  --code-bg: #1e293b;          /* 代码块深色底 */
  --border: #e2e8f0;           /* 边框 */
  --radius: 8px;               /* 圆角 */
  --radius-lg: 14px;           /* 大圆角 */
}
```

**字体选择**：
- 标题：衬线体（如 Georgia / Noto Serif SC），增加学术感
- 正文：系统默认无衬线栈（`-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`）
- 代码：等宽字体（JetBrains Mono / Cascadia Code / Fira Code）
- 避免 Inter/Roboto/Arial（过于通用，缺乏辨识度）

### 8.2 三栏布局

推荐经典的 **记录 | 对话 | 快捷提问** 三栏结构：

```
┌──────────┬────────────────────┬──────────┐
│ 问答记录  │                    │ 快捷提问  │
│ (250px)  │   对话区 + 输入框    │ (260px)  │
│          │   (flex: 1)        │          │
│ ·问题1   │                    │ 快速入门  │
│ ·问题2   │   [助手回答...]     │ ·问题A   │
│ ·问题3   │                    │ ·问题B   │
│          │   [用户提问...]     │ 功能实现  │
│          │                    │ ·问题C   │
│          │   [输入框_____🚀]  │ ·问题D   │
└──────────┴────────────────────┴──────────┘
```

- 窄屏（≤900px）时自动隐藏右栏；这是 `style.css` 里唯一一条断点，再窄（含手机竖屏）也仍是两栏，左栏只是收窄，未改成顶部横条
- `max-width` 限制总宽度避免超宽屏内容被拉长

### 8.3 左侧栏：问答记录

**行为**：

- 每次用户提问后自动追加一项，显示问题文本
- 最新问题自动出现在列表底部
- 点击历史问题 → 对话区平滑滚动到对应位置 → 目标消息短暂高亮（`box-shadow` 光环 1.8 秒后消失）
- 当前查看的问题高亮（左侧蓝色竖条 + 浅蓝底色）
- 首次打开时显示占位文字："还没有提问，输入问题开始"
- **仅本次会话有效**，刷新页面即清空

**实现要点**：

```javascript
// 每个消息 DOM 元素赋予唯一 id，sidebar 按钮存储 data-msg-id
div.id = 'msg-' + (++counter);

// 点击侧栏按钮 → scrollIntoView + 高亮动画
msgEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
msgEl.style.boxShadow = '0 0 0 3px rgba(59,130,246,0.2)';
setTimeout(() => msgEl.style.boxShadow = '', 1800);

// 更新 active 样式
oldActive.classList.remove('active');
sidebarBtn.classList.add('active');
```

### 8.4 右侧栏：快捷提问

**设计原则**：

- **始终可见**，不受对话进度影响
- 按难度或主题**分组**（如：快速入门 / 功能实现 / 架构与安全）
- 每组带小标题，问题以按钮形式纵向排列
- 点击后填入输入框并自动发送，无需二次操作
- 问题措辞从易到难、从通用到深入

**问题设计技巧**：

> 注：下表的示例问题来自前述小程序项目，本仓库没有页面、OCR、main/release
> 分支对比这类对象；换语料时按自己的文件结构重写这一栏。

| 分组 | 示例问题 | 目的 |
|------|---------|------|
| 快速入门 | "项目有哪些页面？"、"app.json 里配置了什么？" | 熟悉项目结构 |
| 功能实现 | "图像识别怎么实现的？"、"OCR 的数据流是怎样的？" | 理解核心逻辑 |
| 架构与安全 | "图片上传做了哪些安全校验？"、"main 和 release 分支有什么区别？" | 深入架构对比 |

### 8.5 代码块

> 注：下面 `def recognize(image)` 只是示意代码块长什么样，来自前述小程序项目，
> 本仓库没有这个函数。

每个代码块包含完整的交互元素：

```
┌─────────────────────────────────────┐
│ Python  │  [收起] [复制]            │ ← 代码块头部
├─────────────────────────────────────┤
│ def recognize(image):              │
│     result = model.predict(image)  │ ← 语法高亮代码
│     return result                  │
└─────────────────────────────────────┘
```

- **语法高亮**：使用 highlight.js，按语言自动着色（深色主题如 github-dark）
- **语言标签**：头部左侧显示语言名，全大写，等宽字体
- **一键复制**：`navigator.clipboard.writeText()` + "已复制" 反馈（1.8 秒后恢复）；降级方案用 `document.execCommand('copy')`
- **折叠/展开**：

```javascript
// 折叠时限制 max-height + 降低透明度
codeBody.classList.toggle('folded');    // max-height: 60px
codeBody.style.opacity = '0.5';        // 视觉提示已折叠
btn.textContent = isFolded ? '展开' : '收起';
```

折叠状态默认为展开（用户第一眼需要看到完整代码），折叠后显示代码前几行作为预览。

**代码块渲染 — Markdown 解析器自定义**：

使用 marked.js 的自定义 Renderer 注入代码块头部：

```javascript
renderer.code = function(code, lang) {
  var blockId = 'code-' + randomId();
  var highlighted = hljs.highlight(code, { language: lang || 'text' }).value;

  return '<div class="code-block">' +
    '<div class="code-header">' +
      '<span>' + (lang || 'text') + '</span>' +
      '<button onclick="toggleFold(\'' + blockId + '\')">收起</button>' +
      '<button onclick="copyCode(\'' + blockId + '\')">复制</button>' +
    '</div>' +
    '<div class="code-body" id="' + blockId + '">' +
      '<pre><code>' + highlighted + '</code></pre>' +
    '</div>' +
    '</div>';
};
```

**复制按钮的 clipboard API 降级**：

```javascript
if (navigator.clipboard) {
  navigator.clipboard.writeText(code).then(...);
} else {
  // Fallback: 创建隐藏 textarea → select → execCommand('copy')
  var ta = document.createElement('textarea');
  ta.value = code;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  document.execCommand('copy');
  document.body.removeChild(ta);
}
```

### 8.6 Markdown 渲染

**关键教训**：CDN 引用的库必须固定版本号。

```html
<!-- ❌ 危险：API 可能随时变化 -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<!-- ✅ 安全：版本锁定 -->
<script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
```

marked.js v4 → v15 的 API 完全变化（`setOptions` 移除、`Renderer` 签名变更、`parser.parseInline` 移除）。不固定版本号会导致页面静默崩溃，排查困难。

另外注意不同版本 Renderer 的方法签名不同：
- marked v4：`renderer.code = function(code, lang)` — 位置参数
- marked v5+：`renderer.code = function({ text, lang })` — 对象解构

渲染函数加 try-catch 兜底，解析失败时直接显示原文（转义 HTML）。

### 8.7 来源引用渲染

LLM 输出的来源标注（如 `[来源: main | 文件: xxx | 行号: Y-Z]`）可以用自定义 renderer 识别并包裹为特殊样式：

```javascript
renderer.paragraph = function(text) {
  if (/^\[来源:/.test(text.trim())) {
    return '<div class="source-ref">' + text + '</div>';
  }
  return '<p>' + text + '</p>';
};
```

渲染效果：等宽字体、小字号、灰色底卡片，区别于正文段落。

### 8.8 前端缓存控制

开发阶段给静态文件加 `Cache-Control: no-cache` 响应头，避免浏览器缓存旧版 JS/CSS，导致修改不生效但误以为代码有问题。

```python
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
```

### 8.9 响应式适配

| 屏幕宽度 | 布局 |
|---------|------|
| > 900px | 三栏完整展示（问答记录 / 对话 / 快捷提问） |
| ≤ 900px | 隐藏右栏，两栏（问答记录 / 对话）；侧栏条目、头部与消息区字号同时收小 |

实现上只有 `@media (max-width: 900px)` 这一个断点，没有 768px 那一档：比 900px 更窄的屏幕（含手机竖屏）仍然保持两栏，左栏按固定宽度收窄，不会变成顶部横条，快捷提问 chips 也不再有单独的缩小规则。要单栏布局得自己加断点。

---

## 九、知识库构建

### 9.1 文件筛选

- **包含**：源码文件（按项目语言添加后缀白名单）、配置文件
- **排除**：node_modules、site-packages、venv、.git、构建产物、图片/视频/字体等二进制文件、lock 文件、IDE 私有配置
- **特殊**：HTML/WXML 是前端项目的核心，不要遗漏

### 9.2 索引性能优化

| 机制 | 效果 |
|------|------|
| 批量嵌入（每批 30 条） | 请求数 ≈ chunk 数 / 30；逐条请求在建索引阶段不可接受 |
| 文件 MD5 哈希缓存 | 未修改文件跳过重新嵌入 |
| 向量库持久化 | 服务重启无需重建 |
| 增量重建 | (source, relative_path, md5) 三元组标识变更，仅对变更文件重新嵌入；关键词索引仍按全量语料重建 |
| 孤儿清理 | 源文件被删除时自动移除对应 chunk |
| BM25 全量重建 | 不做复杂增量，每次按全量语料重建并覆盖 pkl |

构建耗时取决于文件数、嵌入模型和是否本地部署，不同机器可差一个数量级，本文不给具体秒数；以本机实测为准。

### 9.3 多分支索引

如果项目有多个分支（如 main 开发分支和 release 发布分支），应索引所有分支：

- 每个 chunk 携带 `source` 元数据标记分支来源
- Scanner 配置多个源目录，分别扫描
- System Prompt 中说明分支差异，引导 LLM 对比回答
- 用户问 "XX 功能怎么实现的" → LLM 看到两个分支不同实现 → 分别说明

---

## 十、调试方法

### 10.1 日志

建议日志同时写入 stderr（终端可见）和文件（持久化备查）：

```python
def _log(msg):
    sys.stderr.write(f"[RAG] {msg}\n")
    sys.stderr.flush()  # 必须 flush
    with open("debug.log", "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
```

在 LLM 流式循环中记录：输入 context token 数、输出预算、finish_reason、输出字符数。这些是排查问题的关键数据。

### 10.2 端口检查

**最常见的静默故障**：旧进程占着端口，新启动的服务器绑定失败，请求打到旧代码上。

```bash
# 启动前检查
netstat -ano | grep :8001

# 杀掉所有占用进程
# Windows: 任务管理器或 taskkill /F /PID <id>
# Linux: lsof -ti:8001 | xargs kill -9
```

修改代码后行为不变，第一件事就是确认端口没有被旧进程占用。

### 10.3 系统代理

Windows 系统代理（`127.0.0.1:7890` 等）会拦截 requests 库发出的 **所有** HTTP 请求，包括 localhost。表现为：

- `curl http://localhost:11434/api/embed` 正常
- `python -c "import requests; requests.post(...)"` 返回 404

解决方案：
- `session.trust_env = False` 禁用环境变量代理
- `proxies={"http": "", "https": ""}` 显式绕过
- 或使用 `urllib.request`（不受 requests 代理逻辑影响）

### 10.4 检查清单

| 问题 | 检查方法 |
|------|---------|
| 修改不生效 | `netstat -ano \| grep PORT` — 旧进程是否还在 |
| 前端 JS 报错 | 浏览器 F12 Console + 确认 CDN 版本号 |
| 前端修改不生效 | Ctrl+Shift+R 硬刷新 + 检查 Cache-Control 头 |
| 流式中断 | 看服务端日志 finish_reason：length=token 不够，stop=正常结束 |
| 嵌入 API 404 | `curl` 测试模型名称是否正确（注意完整路径） |
| 回答太短 | 检查 context_max_tokens 是否挤占了输出空间 |
| 检索结果单一 | 同文件限制是否生效 |

---

## 十一、典型陷阱与解决方案

### 陷阱 1：上下文挤占输出空间

**现象**：`max_tokens` 设得很大但回答仍然很短。

**根因**：输入 context 占了 55K/64K，只剩 9K 给输出。

**教训**：`context_max_tokens` 不要超过总窗口的 50%。动态计算：`输出预算 = min(max_tokens, 总窗口 - 输入大小 - 缓冲)`。

### 陷阱 2：Chunk 边界逻辑错误

**现象**：分块超过文件大小或在单行上死循环。

**根因**：起始位置更新逻辑写错。

**教训**：测试覆盖 "空文件"、"单行文件"、"超长单行"、"正好等于 chunk_size"、"超大 Page/Class" 等边界。

### 陷阱 3：检索结果单一文件垄断

**现象**：所有检索结果都来自同一个大文件。

**根因**：大文件 chunk 多，命中概率自然高。

**教训**：对检索结果做同文件数量限制（每文件最多 3 个 chunk）。

### 陷阱 4：只依赖一种检索方式

**现象**：只用向量检索找不到精确函数名，只用关键词检索理解不了同义词。

**教训**：混合检索是必选项。RRF 是简单有效的融合方案。

### 陷阱 5：Prompt 引导方向错误

**现象**：写了"简洁明了" → LLM 回答过于简略，省略关键代码。

**教训**：代码问答中，"完整性" 和 "不过度省略" 比 "简洁" 重要。Prompt 明确：禁止省略、禁止写 "..."、每个要点独立完整。

### 陷阱 6：流式 Burst 效应

**现象**：输出一大段后停顿很久。

**教训**：前端缓冲 + 定时器平滑输出完全掩盖 Burst。

### 陷阱 7：配置改了不生效

**现象**：修改了分块大小等参数，行为没变化。

**根因**：函数默认参数硬编码了值，或需要重建索引才能生效。

**教训**：参数从全局配置对象读取；索引类参数变更后需要 `rebuild`。

### 陷阱 8：多进程端口占用

**现象**：kill 了服务器重启，行为完全不变。

**根因**：旧进程的子进程（uvicorn workers）仍在运行。

**教训**：`pkill` 不够，需要 `netstat` 确认端口真的释放了。

### 陷阱 9：CDN 版本漂移

**现象**：页面白屏或 JS 报错，代码看起来没问题。

**根因**：CDN latest 版本 API breaking change。

**教训**：所有 CDN 资源固定版本号。

### 陷阱 10：系统代理干扰本地 API

**现象**：`curl` 能访问 localhost 服务，Python requests 不能。

**根因**：系统/环境变量代理配置被 requests 库使用。

**教训**：本地服务调用显式绕过代理或使用 urllib。

---

## 十二、调参速查

| 症状 | 可能原因 | 调整方向 |
|------|----------|----------|
| 回答太简略 | max_tokens 太小、Prompt 没强调完整性 | 增大 max_tokens；强化 Prompt "完整" |
| 回答被截断 | context 太大挤占输出 | 减小 context_max_tokens（≤ 总窗口 50%） |
| LLM 编造不存在的内容 | Prompt 没强调 "只基于源码" | System Prompt 加入诚实约束 |
| 总是找不到相关代码 | chunk_size 太大、检索 top_k 太小 | 减小 chunk_size，增大初筛数量 |
| 找到了但回答不完整 | 没传全文件、chunk 上下文不够 | 开启全文件检索或增大扩展窗口（±300→±500） |
| 检索结果总来自同一文件 | 没做同文件限制 | 限制每文件最多 3 个 chunk |
| 索引构建太慢 | 逐条请求嵌入 | 改为批量请求（20-30 条/批） |
| 用户觉得输出卡 | 直接渲染 Burst 流 | 添加前端缓冲 + 打字机效果 |
| 中文问题匹配不到英文代码 | 没做查询扩展 | 补充术语映射表 |
| 会话历史太长导致幻觉 | 历史轮数太多 | 限制 3-5 轮 |
| 嵌入模型选型纠结 | 没做对比测试 | 固定问题集 + 相同流程对比 |
| 修改代码不生效 | 旧进程占端口 | `netstat` 检查 → 杀进程 → 重启 |

---

## 十三、快速启动检查清单

- [ ] 选定 LLM（上下文窗口 ≥ 64K，代码理解好，支持流式）
- [ ] 选定嵌入模型（选代码专用，支持批量，确认完整模型名称）
- [ ] 搭建向量库（持久化模式）+ 关键词检索引擎
- [ ] 实现结构感知分块（按语言类型在函数/类/组件边界处断开，含兜底策略）
- [ ] 实现混合检索（向量 + BM25 + RRF + 同文件限制 + 全文件补齐）
- [ ] 实现上下文溢出控制（动态 token 预算计算 + 截断策略）
- [ ] 设计 Prompt（短 System 硬护栏 + 重 User 上下文）
- [ ] 实现流式 API（同步生成器 + SSE + 心跳 + keep-alive）
- [ ] 实现前端（Markdown + 代码高亮 + 折叠 + 打字机缓冲 + 智能滚动）
- [ ] 固定 CDN 库版本号 + 静态文件禁用缓存
- [ ] 配置日志（stderr + 文件）
- [ ] 排除系统代理对 localhost API 的干扰
- [ ] 准备测试问题集（定位/实现/解释/列举/排查各 2-3 个）
- [ ] 跑测试 → 根据症状对照调参表修复
- [ ] 确认端口释放后再启动
