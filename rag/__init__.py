#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟联合国（MUN）规则与事实问答 RAG 应用。

模块:
    config    — 从 .env 加载 LLM / Embedding / Rerank 配置
    parse     — 解析 source/ 下的 .html/.md 并经大模型重写为结构化 JSON: python -m rag.parse
    chunking  — 载入 source/json 下的结构化 JSON 并切分为文本块
    clients   — Embedding / Rerank / Chat 三个 API 客户端
    store     — 基于 numpy 的本地向量库
    ingest    — 构建索引: python -m rag.ingest
    engine    — 检索 + 重排 + 生成管线
    cli       — 命令行问答: python -m rag.cli "问题"
    web       — Web 聊天界面: python -m rag.web
"""
