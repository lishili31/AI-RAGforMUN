#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高中生物教材 RAG 问答应用。

模块:
    config    — 从 .env 加载 LLM / Embedding / Rerank 配置
    chunking  — 解析 md_output 单页 Markdown 并分块
    clients   — Embedding / Rerank / Chat 三个 API 客户端
    store     — 基于 numpy 的本地向量库
    ingest    — 构建索引: python -m rag.ingest
    engine    — 检索 + 生成管线
    cli       — 命令行问答: python -m rag.cli "问题"
    web       — Web 聊天界面: python -m rag.web
"""
