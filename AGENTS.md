# AGENTS.md

> 本文件面向 AI 编程助手。阅读前默认对该项目一无所知。

## 项目概述

这是一个基于检索增强生成（RAG）的**模拟联合国（MUN）规则与事实问答**应用，使用 Python 编写。数据来源是使用者放进 `source/` 目录的 `.html` / `.md` 资料：先用大模型把原始文件重写为结构化 JSON，再切块、向量化构建索引，最后通过“向量化检索 → 重排序 → 大模型流式生成答案”的流水线回答用户问题。

- 目标用户：参加模拟联合国活动、需要查询会议规则与文件事实的代表与组织者。
- 答案语言：简体中文。
- 答案必须基于文件原文，并在关键论断处标注出处（文档名、章节与真实原始链接），使用 Markdown 引用格式，例如 `> 文档名（章节） https://example.org/doc`。
- **用途边界**（由 `rag/engine.py` 的 `SYSTEM_PROMPT` 强制）：助手只做三件事——① 规则查询（议事规则、会议程序、表决方式、文件效力与格式）② 语言处理（针对用户已给出的文本做语法修改、润色、中译英，尤其是英语；不要求文件原文、不标注出处。三项都以"用户已给出的文本"为对象，从零写内容算代写）③ 信息查询（已收录文件中的人物、机构、会议、数据等事实）。决策类请求（投票建议、立场判断、策略分析）与代写文件类请求（决议草案、立场文件、工作文件、修正案、演讲稿、邮件、报告等成品文书）一律拒绝，回答固定的拒绝话术。语言处理与代写的分界见第一节第 2 条与第二节第 2 条。

主要入口：

- 解析原始资料：`python -m rag.parse`
- 构建索引：`python -m rag.ingest`
- 命令行问答：`python -m rag.cli`
- Web 聊天界面：`python -m rag.web`

## 数据流

```text
source/*.html|*.md
   │  python -m rag.parse   （Chat API 重写为结构化 JSON）
   ▼
source/json/<stem>.json
   │  python -m rag.ingest  （Embedding 向量化）
   ▼
rag_store/  （chunks.json + vectors.npy + meta.json）
   │  检索 → 重排 → 流式生成
   ▼
带出处链接的答案（cli / web）
```

## JSON 契约

`python -m rag.parse` 为 `source/` 下的每个 `.html` / `.md` 生成 `source/json/<输入文件 stem>.json`，文件内容是一个 JSON 数组，元素字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 条目唯一标识，形如 `pkunmun2025_manual_front_0001` |
| `source` | str | 人类可读的文档名，用于出处标注 |
| `section` | str | 章节路径，可为空串 |
| `content` | str | 忠实于原文的正文，禁止虚构 |
| `url` | str | 原始链接，缺失时写 `""` |
| `metadata` | dict | 附加信息，含 `language` / `document_year` / `chunk_type`，允许为空 dict |

禁止出现上述之外的顶层字段。

`source/test.json` 是**人工保留的 schema 样例**（条目演示了上述字段结构，其中未给出 `url`），不是任何 `.html` 的解析产物，**禁止覆盖或删除**；`parse` 只处理 `.html` / `.md`，不会写这个文件。

注意：结构化 JSON 必须放进 `source/json/`——那才是 `ingest` 的输入目录（`--input-dir` 默认值），放在 `source/` 根目录下的 JSON 不会被索引。另外 `source/test.json` 的前两个条目与 `source/json/北大学标PKUNMUN2025_学术标准手册.json` 的前两条 `id` 相同，若把 `--input-dir` 指向 `source/` 会让这两条重复入库。

当前 `source/json/北大学标PKUNMUN2025_学术标准手册.json` 的 299 个条目都填了同一个 `url`：`https://www.stringandstick.cn/rag/北大学标PKUNMUN2025_学术标准手册.pdf`，即 `source/北大学标PKUNMUN2025_学术标准手册.pdf` 由 `rag/web.py` 对外提供的地址（见「交互层」）。出处链接因此指向手册 PDF 原件；`url` 为空串时 `format_source` 会省略该片段，只标注文档名与章节。

## 项目结构

```text
.
├── requirements.txt          # Python 依赖
├── .env                      # 环境变量（API 密钥、服务地址），敏感文件
├── .gitignore                # 忽略 .env、rag_store/、source/ 等
├── AGENTS.md                 # 项目说明与 AI 助手指南
├── rag/                      # Python 包：RAG 核心代码
│   ├── __init__.py           # 包说明与模块索引
│   ├── config.py             # 配置加载（.env → Settings）
│   ├── parse.py              # 解析 source/ 原始资料并用大模型重写为结构化 JSON
│   ├── chunking.py           # 载入 source/json 结构化 JSON 并切分为文本块
│   ├── clients.py            # Embedding / Rerank / Chat 三个外部 API 客户端
│   ├── store.py              # 基于 numpy 的本地向量库存储与检索
│   ├── ingest.py             # 构建向量索引
│   ├── engine.py             # RAG 检索 + 生成管线
│   ├── cli.py                # 命令行入口
│   └── web.py                # FastAPI Web 服务与单文件前端
├── source/                   # 原始资料与解析产物
│   ├── test.json             # 人工保留的 schema 样例，勿覆盖
│   ├── 北大学标…手册.pdf      # 手册 PDF 原件，经 web 对外提供，作为出处链接
│   └── json/                 # 结构化 JSON：北大学标PKUNMUN2025_学术标准手册.json
└── rag_store/                # 构建好的向量索引（由 python -m rag.ingest 生成，可重建）
```

项目没有 `pyproject.toml`、`setup.py`、`package.json`、`Cargo.toml` 等构建文件，也没有测试文件或 README。

## 技术栈

- Python 3.10+
- 主要依赖（见 `requirements.txt`）：
  - `fastapi>=0.110` + `uvicorn>=0.29`：Web 服务
  - `openai>=1.30`：OpenAI 兼容的 Embedding 与 Chat 客户端（`parse` 的 JSON 重写也走 Chat 接口）
  - `numpy>=1.26`：向量矩阵运算与本地向量库
  - `httpx>=0.27`：Rerank 接口 HTTP 调用
  - `tqdm>=4.66`：索引构建进度条
  - `python-dotenv>=1.0`：从 `.env` 加载配置

## 运行时架构

1. **数据层**：`source/` 下由使用者放入的 `.html` / `.md` 原始资料，以及 `source/json/<stem>.json` 结构化重写产物（见上文 JSON 契约）。每个条目自带 `source`（文档名）、`section`（章节）与 `url`（原始链接）。

2. **解析**：`rag/parse.py`
   - 扫描 `source/` 下的 `.html` / `.md`，调用 Chat API 将原始内容重写为结构化 JSON 数组。
   - `content` 必须忠实原文、不得虚构；`url` 缺失时写 `""`。
   - 产物写入 `source/json/<输入文件 stem>.json`。

3. **分块**：`rag/chunking.py`
   - 载入 `source/json/` 下的结构化 JSON 并切分。
   - 默认 `chunk_size=800`、`overlap=100`；优先按空行（段落）边界切分，单段过长时按字符硬切，相邻块保留重叠。
   - 每个 `Chunk` 携带 `id`、`source`、`section`、`url`、`chunk_index`、`content`、`metadata`（`to_dict` 顺序即此序）；`Chunk.from_dict` 对缺失的 `section` / `url` / `metadata` 分别容错为 `""` / `""` / `{}`。

4. **向量化与索引**：`rag/ingest.py` → `rag/store.py`
   - 使用 bge-m3 类 Embedding 模型将文本块批量向量化。
   - 向量按行做 L2 归一化后保存为 `rag_store/vectors.npy`。
   - 文本块保存为 `rag_store/chunks.json`。
   - 元信息保存为 `rag_store/meta.json`，键为 `chunk_count`、`dim`、`sources`、`built_at`（`sources` 是去重排序后的 `chunk.source` 列表）。
   - 每次运行都会全量重建索引，没有增量更新。

5. **检索与生成**：`rag/engine.py`
   - 查询向量化 → 余弦相似度 top-30 召回 → bge-reranker 重排序取 top-6 → 丢弃重排分数低于 `RERANK_SCORE_THRESHOLD`（0.30）的段落 → 组装 prompt → kimi-for-coding 流式生成答案。
   - 候选中没有任何段落达到阈值时，`retrieve` 返回空列表且**不注入任何原文**；调用方仍会把控制权交给大模型（上下文里只有 `(未检索到任何文件原文)` 占位），由 `SYSTEM_PROMPT` 判断是拒绝越界请求、完成语言处理，还是回答「在已收录的文件中未找到相关内容」。这一取舍是为了让「语法修改 / 润色 / 中译英」这类不需要文件原文的正当用途可用。
   - `SYSTEM_PROMPT` 是本项目的行为边界所在：分「允许的用途」（规则查询 / 语言处理（语法修改、润色、中译英）/ 信息查询）、「必须拒绝的用途」（决策与判断、代写文件、无关请求，附固定拒绝话术）、「回答规则」三节。改动它等于改动产品能力范围，需同步本节与「项目概述」的用途边界说明。
   - 语言处理与代写文件的边界靠提示词里的一句限定维持：**语言处理的对象必须是用户已给出的文本**（第一节第 2 条），从零写内容归入代写并按第二节拒绝。放宽或收紧这条限制会直接改变产品能力，改提示词时留意两处要一起动。
   - 对外 API：`RAGEngine(settings, store_dir)`；`retrieve(question, top_k=30, top_n=6)` 返回 `{"chunk", "rerank_score", "vector_score"}` 列表；`build_messages(question, passages)`；`stream_answer(question, passages=None)`；模块级 `format_source(chunk)` 返回「文档名（章节） URL」、`format_source_title(chunk)` 返回「文档名（章节）」——`section` 或 `url` 为空时省略该片段，各片段以单空格连接后 `strip`。`cli.py` 与 `web.py` 都用它们生成出处展示。

6. **客户端**：`rag/clients.py`
   - `EmbeddingClient`：OpenAI 兼容接口，支持 `base64` 与 `float` 返回，可配置是否发送 `dimensions` 参数。
   - `RerankClient`：Cohere 兼容 `/rerank` 接口。
   - `ChatClient`：OpenAI 兼容流式对话接口，既用于 `parse` 的 JSON 重写，也用于最终答案生成。
   - 所有调用带 3 次指数退避重试（1s、2s、4s）。

7. **交互层**：
   - `rag/cli.py`：单次问答或交互式 REPL，答案后列出去重后的出处；检索为空时不再提前返回，仍调用大模型，因此语言处理与越界拒绝在 CLI 下同样生效（此时不打印「出处」段）。REPL 启动横幅写明支持范围。
   - `rag/web.py`：FastAPI 服务，提供 `/` 聊天页面、`POST /api/chat` SSE 流式回答（事件 `type` 为 `token` / `sources` / `error`，末尾 `data: [DONE]`）、`GET /api/stats` 索引统计（返回 `{"ready": true, **meta.json 内容}`）。`sources` 事件**仅在检索到段落时**发送，无检索结果时前端不会渲染出处栏。页面标题栏下方的提示行写明了「仅支持规则查询、语法修改/中译英润色、信息查询」。
   - `rag/web.py` 的 `GET /{filename}`（`serve_reference_file`）：把 `source/` 下的 PDF 原件作为静态文件发给浏览器，只放行 `.pdf` 且路径必须落在 `source/` 下（其余 404，含 `%2F` 编码的目录穿越）。因为 nginx 的 `location /rag/` 用 `proxy_pass http://127.0.0.1:5000/` 剥掉了 `/rag/` 前缀，应用根路径上的这个路由对外就表现为 `https://www.stringandstick.cn/rag/<文件名>.pdf`——索引里的 `url` 正是这个地址。新增需要被引用的 PDF 时，把它放进 `source/` 并把该地址填进对应 JSON 条目的 `url` 字段即可，不必改 nginx。

## 配置

所有配置通过 `.env` 文件加载，由 `rag/config.py` 统一管理。`.env` 是敏感文件，不应提交到版本控制。

必需的配置项：

| 环境变量 | 说明 |
|---------|------|
| `KIMI_API_KEY` | LLM（kimi-for-coding）API 密钥 |
| `EMBEDDING_BINDING_HOST` | Embedding 服务 base URL |
| `EMBEDDING_BINDING_API_KEY` | Embedding 服务 API 密钥 |
| `RERANK_BINDING_HOST` | Rerank 服务 URL |
| `RERANK_BINDING_API_KEY` | Rerank 服务 API 密钥 |

可选配置项（均有默认值）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `API_BASE_URL` | `https://api.kimi.com/coding/v1` | LLM base URL |
| `API_MODEL` | `kimi-for-coding` | LLM 模型名 |
| `LLM_THINKING_TYPE` | `disabled` | LLM thinking 模式 |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding 模型名 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `EMBEDDING_TOKEN_LIMIT` | `8192` | Embedding 输入 token 上限 |
| `EMBEDDING_SEND_DIM` | `False` | 是否向 Embedding 接口发送 `dimensions` 参数 |
| `EMBEDDING_USE_BASE64` | `True` | Embedding 返回是否使用 base64 编码 |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Rerank 模型名 |

当前 `rag_store/` 已构建：299 个文本块、1024 维，来源为 `北大学标PKUNMUN2025_学术标准手册`（由 `source/json/北大学标PKUNMUN2025_学术标准手册.json` 切分而来）。新增资料后需重跑 `python -m rag.parse` 与 `python -m rag.ingest`；在索引建好之前 `cli` 与 `web` 的提问都会失败，`GET /api/stats` 会返回 `ready: false`。

## 常用命令

### 安装依赖

```bash
pip install -r requirements.txt
```

### 解析原始资料

```bash
# 解析 source/ 下的 .html / .md，重写为 source/json/<stem>.json
python -m rag.parse

# 自定义输入/输出目录，或强制重解析
python -m rag.parse --input-dir ./source --output-dir ./source/json --force
```

需要先配置好 LLM（`KIMI_API_KEY` 等）才能运行；每个原始文件都会调用一次 Chat API。产物已存在且比输入文件新时会自动跳过，`--force` 可强制重解析。产物统一落在 `source/json/`，`source/test.json` 是人工保留的 schema 样例，`parse` 不会写它。

### 构建/重建索引

```bash
# 使用默认输入目录 ./source/json 与输出目录 ./rag_store
python -m rag.ingest

# 自定义目录与分块参数
python -m rag.ingest --input-dir ./source/json --store-dir ./rag_store --chunk-size 800 --overlap 100
```

### 新增资料（日常更新流程）

1. 把 `.html` / `.md` 文件放进 `source/`。
2. `python -m rag.parse` 生成/更新 `source/json/<stem>.json`。
3. `python -m rag.ingest` 全量重建 `rag_store/`。
4. 重启 Web 服务以载入新索引（见下节 `pm2 restart rag-web`）。

### 命令行问答

```bash
# 单次问答
python -m rag.cli "联合国大会的表决程序是怎样的?"

# 交互模式（输入 q / quit / exit 退出）
python -m rag.cli

# 使用自定义索引目录
python -m rag.cli --store-dir ./rag_store
```

### 启动 Web 服务

```bash
# 默认监听 127.0.0.1:5000
python -m rag.web

# 自定义主机与端口
python -m rag.web --host 0.0.0.0 --port 8080

# 自定义索引目录
python -m rag.web --store-dir ./rag_store
```

访问 `http://<host>:<port>/` 即可使用聊天页面。Web 服务提供 SSE 流式回答。

### 生产部署（nginx 子目录 + PM2）

线上通过 nginx 子目录 `https://www.stringandstick.cn/rag` 访问，配置位于
`/etc/nginx/sites-available/stringandstick.conf`（443 与 80 两个 server 块中各有一份
`/rag` location，`proxy_pass http://127.0.0.1:5000/`，已关闭缓冲以支持 SSE）。

注意：前端 JS 中的 `fetch('api/chat')` / `fetch('api/stats')` 是**相对路径**，
以便同时兼容直接访问（`127.0.0.1:5000/`）和子目录代理（`/rag/`），不要改回绝对路径。

服务以 PM2 托管（进程名 `rag-web`），运行在 **deploy 用户**的 PM2 实例下
（与 blog-api、cloudfile 等其他应用同一个实例；请勿用 root 的 PM2 托管）：

```bash
# 启动（deploy 用户，在项目根目录执行）
pm2 start .venv/bin/python --name rag-web -- -m rag.web --host 127.0.0.1 --port 5000
pm2 save   # 保存进程列表，配合 pm2-deploy systemd 服务开机自启

# 日常维护（root 下操作时需切换身份）
sudo -u deploy pm2 restart rag-web
sudo -u deploy pm2 logs rag-web
```

修改配置后执行 `nginx -t && systemctl reload nginx`。

## 代码组织与模块职责

| 文件 | 职责 |
|------|------|
| `rag/config.py` | 定义 `Settings` dataclass，从 `.env` 加载配置；定义 `ConfigurationError`。 |
| `rag/parse.py` | 扫描 `source/` 原始资料，调用 Chat API 重写为结构化 JSON 并写入 `source/json/`。 |
| `rag/chunking.py` | 定义 `Chunk` 数据类；载入 `source/json` 下的结构化 JSON 并切分为文本块。 |
| `rag/clients.py` | 封装三个外部 HTTP API 客户端，统一重试逻辑。 |
| `rag/store.py` | `VectorStore`：本地向量库，负责 save / load / search / meta。 |
| `rag/ingest.py` | 端到端索引构建脚本。 |
| `rag/engine.py` | `RAGEngine`：检索、重排、prompt 组装、流式生成；模块级 `format_source`。 |
| `rag/cli.py` | argparse 命令行入口。 |
| `rag/web.py` | FastAPI 应用与内嵌 HTML/JS 前端;回答气泡通过 CDN 引入 marked + DOMPurify 渲染 Markdown(CDN 不可用时回退纯文本)。 |

## 代码风格与开发约定

- 文件头统一使用 `#!/usr/bin/env python3` 与 `# -*- coding: utf-8 -*-`。
- 注释与文档字符串使用简体中文。
- 使用 `from __future__ import annotations` 支持延迟类型注解。
- 类型注解覆盖率较高，函数参数与返回值通常有类型标注。
- 常量使用全大写蛇形命名，如 `RETRIEVE_TOP_K`、`EMBED_BATCH_SIZE`。
- 配置项缺失时抛出 `ConfigurationError`；外部 API 重试失败后抛出 `APIClientError`。
- 关键数字硬编码在模块顶部常量中（如 `RETRIEVE_TOP_K=30`、`RERANK_TOP_N=6`、`RERANK_SCORE_THRESHOLD=0.30`）。这些检索参数目前不读 `.env`；如需免改代码调参，要先把它们接进 `Settings`。

## 测试

项目中没有单元测试、集成测试或测试框架配置。验证功能的方式是：

1. 运行 `python -m rag.parse`，确认 `source/json/` 下生成了结构合法的 JSON 数组。
2. 运行 `python -m rag.ingest` 成功构建索引（`rag_store/meta.json` 出现且 `chunk_count` 大于 0）。
3. 运行 `python -m rag.cli "一个模联问题"` 检查是否能正常检索并生成带出处链接的答案。
4. 运行 `python -m rag.cli "<一句有语病的英语>"`，确认「语法修改」在没有任何检索结果时也能作答。
5. 运行 `python -m rag.cli "把这句中文译成英语:<一句话>"` 与 `"帮我润色这句英语:<一句话>"`，确认中译英与润色同样可用（语言处理不依赖检索结果）。
6. 运行 `python -m rag.cli "帮我写一份决议草案"`、`"我该不该支持这个动议"` 与 `"帮我写一段英文发言稿"`，确认都得到固定的拒绝话术——最后一条用来验证「从零写内容」没有被中译英的口子放行。
7. 启动 `python -m rag.web` 并在浏览器中通过页面测试 SSE 回答。
8. 确认出处链接可用：`curl -sI "https://www.stringandstick.cn/rag/北大学标PKUNMUN2025_学术标准手册.pdf"` 应返回 200 与 `application/pdf`（GET / HEAD 都放行，带 `Range` 的请求返回 206）。

如需新增测试，建议自行创建 `tests/` 目录并引入 `pytest`。

## 安全注意事项

- `.env` 文件包含 API 密钥，已加入 `.gitignore`，切勿提交到版本控制，也不要把密钥写进日志或代码。
- `rag_store/` 与 `source/` 也已在 `.gitignore` 中，分别为可重新生成的索引数据与原始资料/解析产物；不要修改 `.gitignore`。
- `rag/web.py` 默认仅监听 `127.0.0.1`，如需暴露到公网请谨慎考虑鉴权与 HTTPS。
- Web 接口未做访问控制，生产环境部署时应添加反向代理、限流与身份验证。

## 已知限制

- 没有持久化数据库，所有索引与向量以本地文件形式存放在 `rag_store/`。
- 向量检索为全量矩阵乘法，索引规模增大时检索延迟会线性上升。
- 没有测试套件，也没有 CI/CD 配置。
- Web 前端为单文件内嵌 HTML/JS，没有前端构建流程；Markdown 渲染依赖 CDN 上的 marked + DOMPurify，CDN 不可用时回退纯文本。
- `parse` 依赖大模型重写原始资料，结果可能遗漏细节或引入偏差，且每个文件都要调用一次 API（耗时与计费）；入库前应人工抽查 `source/json/` 产物。
- 索引为全量重建（`ingest` 每次重扫全部 `source/json/`）；`parse` 有基于修改时间的增量跳过，但切分粒度由大模型决定，同一文件重解析的条目数会变化（实测同一输入一次 20 条、一次 18 条）。
- 重排分数阈值 `RERANK_SCORE_THRESHOLD=0.30` 是经验值：当前语料与 bge-reranker-v2-m3 下能把无关问题挡掉（实测无关问题重排最高分 0.0000–0.0001，相关问题 0.59–0.92）。换 rerank 模型或换语料后需重新标定，阈值过高会把弱相关但正确的段落一并滤掉。
- 出处链接取决于源资料是否给出 URL；缺失时 `url` 为空串，只能标注文档名与章节，`format_source` 会自动省略该片段。当前手册的 299 个条目都指向 `/rag/北大学标PKUNMUN2025_学术标准手册.pdf`，同一份文档共用一个链接（没有按章节/页码定位到具体页）。
- 用途边界（拒绝决策、拒绝代写文件）只靠 `SYSTEM_PROMPT` 约束，是提示词层面的软限制，不是访问控制：在提示词注入或强指令下仍可能被绕过，需要硬保证时应在服务层加拦截或人工审核。
- 检索为空时仍会调用一次大模型（为了支持不依赖原文的语法修改），因此无关问题也会消耗一次 API 调用并多几秒延迟；不再有「无检索结果就直接返回」的快速路径。同理，判定某个问题是否越界也由大模型完成，代码里没有关键词黑名单。
