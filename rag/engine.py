#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 检索 + 生成管线。

流程: 问题 → bge-m3 向量化 → 余弦 top-K → bge-reranker-v2-m3 重排 top-N
      → 组装带出处标注的 prompt → kimi-for-coding 流式生成中文答案。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from .chunking import Chunk, get_display_book
from .clients import ChatClient, EmbeddingClient, RerankClient
from .config import Settings
from .store import VectorStore

logger = logging.getLogger("rag.engine")

RETRIEVE_TOP_K: int = 20   # 向量检索召回数
RERANK_TOP_N: int = 6      # 重排后送入 LLM 的段落数

SYSTEM_PROMPT: str = (
    "你是一位严谨的高中生物学习助手,正在帮助学生查阅权威的人教版高中生物教材。\n\n"
    "## 回答规则\n"
    "1. 严格依据下方【教材原文】中的内容作答,不要编造教材之外的知识;可以 reorganize、概括原文。\n"
    "2. 使用简体中文,条理清晰,适合高中生理解;涉及术语时保持教材原表述。\n"
    "3. 每个关键论断的句末必须标注出处,格式为【《书名》第N页】,"
    "例如:【《必修2：遗传与进化》第28页】。出处信息已在每段原文开头给出。\n"
    "4. 若【教材原文】中没有足以回答问题的内容,请明确回答"
    "「在教材中未找到相关内容」,不要强行作答。\n"
    "5. 不要提及「教材原文」四个字本身,直接给出答案。"
)

USER_PROMPT_TEMPLATE: str = (
    "## 教材原文\n"
    "{passages}\n\n"
    "## 问题\n"
    "{question}"
)


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
            chunk_index, vector_score = candidates[doc_index]
            passages.append(
                {
                    "chunk": self.store.chunks[chunk_index],
                    "rerank_score": rerank_score,
                    "vector_score": vector_score,
                }
            )
        return passages

    def build_messages(self, question: str, passages: list[dict]) -> list[dict]:
        """组装发送给 LLM 的消息"""
        blocks: list[str] = []
        for idx, item in enumerate(passages, start=1):
            chunk: Chunk = item["chunk"]
            display_book: str = get_display_book(chunk.book)
            blocks.append(
                f"【出处{idx}】《{display_book}》第{chunk.page}页\n{chunk.content}"
            )
        user_content: str = USER_PROMPT_TEMPLATE.format(
            passages="\n\n".join(blocks) if blocks else "(未检索到相关内容)",
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


def format_source(chunk: Chunk) -> str:
    """出处展示格式: 《书名》第N页"""
    return f"《{get_display_book(chunk.book)}》第{chunk.page}页"
