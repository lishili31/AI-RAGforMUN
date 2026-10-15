#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 检索 + 生成管线。

流程: 问题 → bge-m3 向量化 → 余弦 top-K 召回 → bge-reranker-v2-m3 重排 top-N
      → 丢弃重排分数低于 RERANK_SCORE_THRESHOLD 的段落(全部低于时判定证据不足,不调用大模型)
      → 组装带出处链接的 prompt → kimi-for-coding 流式生成中文答案。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from .chunking import Chunk
from .clients import ChatClient, EmbeddingClient, RerankClient
from .config import Settings
from .store import VectorStore

logger = logging.getLogger("rag.engine")

RETRIEVE_TOP_K: int = 30   # 向量检索初始候选数
RERANK_TOP_N: int = 6      # 重排后送入 LLM 的段落数上限
RERANK_SCORE_THRESHOLD: float = 0.30  # 重排相关性阈值,低于该分数的段落不注入上下文

SYSTEM_PROMPT: str = (
    "你是一位模拟联合国(MUN)学术助手,为参会代表查询议事规则、会议程序与相关事实。\n\n"
    "## 回答规则\n"
    "1. 严格依据下方【检索到的文件原文】中的内容作答,不要编造文件之外的规则或事实;"
    "可以在忠于原文的前提下概括、重新组织表述。\n"
    "2. 使用简体中文,条理清晰,便于参会代表理解与引用;涉及议事规则术语时保持文件原表述"
    "(如 motion、point of order、quorum 等)。\n"
    "3. 每个关键论断的句末必须标注出处,使用 Markdown 引用格式(以 '> ' 开头),"
    "文件名称与被引用条目的链接已在每段原文开头给出,必须原样使用,不得杜撰链接,例如:\n"
    "> 来源:[《模联议事规则手册》](https://example.org/mun/rules)\n"
    "4. 若【检索到的文件原文】中没有足以回答问题的内容,请明确回答"
    "「在已收录的文件中未找到相关内容」,不要强行作答。\n"
    "5. 不要提及【检索到的文件原文】这个措辞本身,直接给出答案。"
)

USER_PROMPT_TEMPLATE: str = (
    "## 检索到的文件原文\n"
    "{passages}\n\n"
    "## 问题\n"
    "{question}"
)


def _split_labels(chunk: Chunk) -> tuple[str, str]:
    """拆分出 (文档名（章节）、链接),空片段保持空串"""
    source: str = (chunk.source or "").strip()
    section: str = (chunk.section or "").strip()
    url: str = (chunk.url or "").strip()
    if source and section:
        title = f"{source}（{section}）"
    else:
        title = source or section
    return title, url


class RAGEngine:
    """检索增强问答引擎(CLI 与 Web 共用)"""

    def __init__(self, settings: Settings, store_dir: Path) -> None:
        self.settings = settings
        self.store: VectorStore = VectorStore.load(store_dir)
        self.embedder = EmbeddingClient(settings)
        self.reranker = RerankClient(settings)
        self.chat = ChatClient(settings)

    def retrieve(
        self,
        question: str,
        top_k: int = RETRIEVE_TOP_K,
        top_n: int = RERANK_TOP_N,
    ) -> list[dict]:
        """
        检索并重排,返回用于生成答案的段落列表。

        重排分数低于 RERANK_SCORE_THRESHOLD 的段落会被丢弃;候选全部低于阈值时
        返回空列表,由调用方判定证据不足,不再注入无关内容。

        每项: {chunk, rerank_score, vector_score}
        """
        query_vector = self.embedder.embed_query(question)
        candidates: list[tuple[int, float]] = self.store.search(query_vector, top_k)
        if not candidates:
            return []

        documents: list[str] = [self.store.chunks[i].content for i, _ in candidates]
        ranked: list[tuple[int, float]] = self.reranker.rerank(
            question, documents, top_n
        )

        passages: list[dict] = []
        for doc_index, rerank_score in ranked:
            if rerank_score < RERANK_SCORE_THRESHOLD:
                continue
            chunk_index, vector_score = candidates[doc_index]
            passages.append(
                {
                    "chunk": self.store.chunks[chunk_index],
                    "rerank_score": rerank_score,
                    "vector_score": vector_score,
                }
            )
        if ranked and not passages:
            logger.info(
                "重排最高分 %.4f 低于阈值 %.2f,判定证据不足,不注入任何段落",
                max(score for _, score in ranked),
                RERANK_SCORE_THRESHOLD,
            )
        return passages

    def build_messages(self, question: str, passages: list[dict]) -> list[dict]:
        """组装发送给 LLM 的消息"""
        blocks: list[str] = []
        for idx, item in enumerate(passages, start=1):
            chunk: Chunk = item["chunk"]
            title, url = _split_labels(chunk)
            lines: list[str] = [f"【出处{idx}】{title}" if title else f"【出处{idx}】"]
            if url:
                lines.append(f"链接: {url}")
            lines.append(chunk.content)
            blocks.append("\n".join(lines))
        user_content: str = USER_PROMPT_TEMPLATE.format(
            passages="\n\n".join(blocks) if blocks else "(未检索到任何文件原文)",
            question=question,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def stream_answer(
        self, question: str, passages: Optional[list[dict]] = None
    ) -> Iterator[str]:
        """流式生成答案;passages 为空时自动检索"""
        if passages is None:
            passages = self.retrieve(question)
        messages = self.build_messages(question, passages)
        yield from self.chat.stream_chat(messages)


def format_source_title(chunk: Chunk) -> str:
    """出处标题(不含链接): 文档名（章节）,无章节时只有文档名"""
    return _split_labels(chunk)[0]


def format_source(chunk: Chunk) -> str:
    """出处展示格式: 文档名（章节） URL,空片段省略"""
    return " ".join(part for part in _split_labels(chunk) if part).strip()
