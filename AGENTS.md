# AGENTS.md

> 本文件面向 AI 编程助手。阅读前默认对该项目一无所知。

## 项目概述

这是一个基于检索增强生成（RAG）的高中生物教材问答应用，使用 Python 编写。项目从 `md_output/` 目录下的单页 Markdown 教材文本构建向量索引，然后通过“向量化检索 → 重排序 → 大模型流式生成答案”的流水线回答用户问题。

- 目标用户：高中生/需要查询人教版高中生物教材的学习者。
- 答案语言：简体中文。
- 答案必须基于教材原文，并在关键论断处标注出处，格式为【《书名》第 N 页】。

主要入口：

- 命令行问答：`python -m rag.cli`
- Web 聊天界面：`python -m rag.web`
- 构建索引：`python -m rag.ingest`

## 项目结构

```text
.
├── requirements.txt          # Python 依赖
├── .env                      # 环境变量（API 密钥、服务地址），敏感文件
├── .gitignore                # 忽略 .env、rag_store/、md_output/ 等
├── AGENTS.md                 # 项目说明与 AI 助手指南
└── rag/                      # Python 包：RAG 核心代码
    ├── __init__.py           # 包说明与模块索引
    ├── config.py             # 配置加载（.env → Settings）
    ├── chunking.py           # 解析 Markdown 并切分为文本块
    ├── clients.py            # Embedding / Rerank / Chat 三个外部 API 客户端
    ├── store.py              # 基于 numpy 的本地向量库存储与检索
    ├── ingest.py             # 构建向量索引
    ├── engine.py             # RAG 检索 + 生成管线
    ├── cli.py                # 命令行入口
    └── web.py                # FastAPI Web 服务与单文件前端
├── md_output/                # 教材单页 Markdown 输入数据
└── rag_store/                # 构建好的向量索引（可重新生成）
```

项目没有 `pyproject.toml`、`setup.py`、`package.json`、`Cargo.toml` 等构建文件，也没有测试文件或 README。

## 技术栈

- Python 3.10+
- 主要依赖（见 `requirements.txt`）：
  - `fastapi>=0.110` + `uvicorn>=0.29`：Web 服务
  - `openai>=1.30`：OpenAI 兼容的 Embedding 与 Chat 客户端
  - `numpy>=1.26`：向量矩阵运算与本地向量库
  - `httpx>=0.27`：Rerank 接口 HTTP 调用
  - `tqdm>=4.66`：索引构建进度条
  - `python-dotenv>=1.0`：从 `.env` 加载配置

## 运行时架构

1. **数据层**：`md_output/<书名>/<书名><页码>.md`
   - 每页 Markdown 的正文位于 ````text ... ```` 代码块内。
   - 页码通过文件名中最后一组连续数字推断（如 `选必1教材笔记12.md` → 12）。
   - `index.md` 文件会被跳过。

2. **分块**：`rag/chunking.py`
   - 默认 `chunk_size=800`、`overlap=100`。
   - 优先按空行（段落）边界切分；单段过长时按字符硬切；相邻块保留重叠。
   - 每个 `Chunk` 携带 `id`、`book`、`page`、`chunk_index`、`content`、`metadata`。

3. **向量化与索引**：`rag/ingest.py` → `rag/store.py`
   - 使用 bge-m3 类 Embedding 模型将文本块批量向量化。
   - 向量按行做 L2 归一化后保存为 `rag_store/vectors.npy`。
   - 文本块保存为 `rag_store/chunks.json`。
   - 元信息保存为 `rag_store/meta.json`。

4. **检索与生成**：`rag/engine.py`
   - 查询向量化 → 余弦相似度 top-20 召回 → bge-reranker 重排序取 top-6 → 组装 prompt → kimi-for-coding 流式生成答案。

5. **客户端**：`rag/clients.py`
   - `EmbeddingClient`：OpenAI 兼容接口，支持 `base64` 与 `float` 返回，可配置是否发送 `dimensions` 参数。
   - `RerankClient`：Cohere 兼容 `/rerank` 接口。
   - `ChatClient`：OpenAI 兼容流式对话接口。
   - 所有调用带 3 次指数退避重试（1s、2s、4s）。

6. **交互层**：
   - `rag/cli.py`：单次问答或交互式 REPL。
   - `rag/web.py`：FastAPI 服务，提供 `/` 聊天页面、`POST /api/chat` SSE 流式回答、`GET /api/stats` 索引统计。

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
| `EMBEDDING_MODEL` | `bge-m3` | Embedding 模型名 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `EMBEDDING_TOKEN_LIMIT` | `8192` | Embedding 输入 token 上限 |
| `EMBEDDING_SEND_DIM` | `False` | 是否向 Embedding 接口发送 `dimensions` 参数 |
| `EMBEDDING_USE_BASE64` | `True` | Embedding 返回是否使用 base64 编码 |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Rerank 模型名 |

当前 `rag_store/meta.json` 显示索引包含 7 本书、1852 个文本块、1024 维向量，构建于 2026-07-25。

## 常用命令

### 安装依赖

```bash
pip install -r requirements.txt
```

### 构建/重建索引

```bash
# 使用默认输入目录 ./md_output 与输出目录 ./rag_store
python -m rag.ingest

# 自定义目录与分块参数
python -m rag.ingest --input-dir ./md_output --store-dir ./rag_store --chunk-size 800 --overlap 100
```

### 命令行问答

```bash
# 单次问答
python -m rag.cli "兴奋在神经元之间是如何传递的?"

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
| `rag/chunking.py` | 定义 `Chunk` 数据类；扫描 `md_output` 并按页分块。 |
| `rag/clients.py` | 封装三个外部 HTTP API 客户端，统一重试逻辑。 |
| `rag/store.py` | `VectorStore`：本地向量库，负责 save / load / search / meta。 |
| `rag/ingest.py` | 端到端索引构建脚本。 |
| `rag/engine.py` | `RAGEngine`：检索、重排、prompt 组装、流式生成。 |
| `rag/cli.py` | argparse 命令行入口。 |
| `rag/web.py` | FastAPI 应用与内嵌 HTML/JS 前端。 |

## 代码风格与开发约定

- 文件头统一使用 `#!/usr/bin/env python3` 与 `# -*- coding: utf-8 -*-`。
- 注释与文档字符串使用简体中文。
- 使用 `from __future__ import annotations` 支持延迟类型注解。
- 类型注解覆盖率较高，函数参数与返回值通常有类型标注。
- 常量使用全大写蛇形命名，如 `RETRIEVE_TOP_K`、`EMBED_BATCH_SIZE`。
- 配置项缺失时抛出 `ConfigurationError`；外部 API 重试失败后抛出 `APIClientError`。
- 关键数字硬编码在模块顶部常量中（如 `RETRIEVE_TOP_K=20`、`RERANK_TOP_N=6`）。

## 测试

项目中没有单元测试、集成测试或测试框架配置。验证功能的方式是：

1. 运行 `python -m rag.ingest` 成功构建索引。
2. 运行 `python -m rag.cli "一个生物学问题"` 检查是否能正常检索并生成带出处的答案。
3. 启动 `python -m rag.web` 并在浏览器中通过页面测试 SSE 回答。

如需新增测试，建议自行创建 `tests/` 目录并引入 `pytest`。

## 安全注意事项

- `.env` 文件包含 API 密钥，已加入 `.gitignore`，切勿提交到版本控制。
- `rag_store/` 与 `md_output/` 也已在 `.gitignore` 中，分别为可重新生成的索引数据与原始教材输入数据。
- `rag/web.py` 默认仅监听 `127.0.0.1`，如需暴露到公网请谨慎考虑鉴权与 HTTPS。
- Web 接口未做访问控制，生产环境部署时应添加反向代理、限流与身份验证。

## 已知限制

- 没有持久化数据库，所有索引与向量以本地文件形式存放在 `rag_store/`。
- 向量检索为全量矩阵乘法，索引规模增大时检索延迟会线性上升。
- 没有测试套件，也没有 CI/CD 配置。
- Web 前端为单文件内嵌 HTML/JS，没有前端构建流程。
