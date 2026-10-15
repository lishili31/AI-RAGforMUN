#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding / Rerank / Chat 三个外部 API 的客户端封装。

- Embedding: OpenAI 兼容接口(bge-m3),支持 base64/float 两种返回编码,
  是否发送 dimensions 参数由 EMBEDDING_SEND_DIM 控制。
- Rerank: Cohere 兼容接口(bge-reranker-v2-m3),POST {model, query, documents, top_n}。
- Chat: OpenAI 兼容对话接口(kimi-for-coding),支持流式(stream_chat)与非流式(complete_chat)。

所有调用带指数退避重试(1s, 2s, 4s)。
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Iterator, Optional

import httpx
import numpy as np
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from .config import Settings

logger = logging.getLogger("rag.clients")

EMBED_BATCH_SIZE: int = 32
MAX_RETRIES: int = 3
BACKOFF_BASE_SECONDS: float = 1.0
REQUEST_TIMEOUT_SECONDS: float = 120.0

# OpenAI embeddings 接口在 encoding_format="base64" 时返回 float32 字节的 base64 编码
_BASE64_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


class APIClientError(Exception):
    """外部 API 调用多次重试后仍失败"""


def _decode_embedding(raw, dim: int) -> list[float]:
    """将接口返回的向量统一解码为 float 列表(兼容 base64 字符串与 float 数组)"""
    if isinstance(raw, str):
        data: bytes = base64.b64decode(raw)
        vector = np.frombuffer(data, dtype=np.dtype("float32"))
        if vector.size != dim:
            raise APIClientError(
                f"Embedding 维度不符: 期望 {dim}, 实际解码得到 {vector.size}"
            )
        return vector.astype(np.float32).tolist()
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    raise APIClientError(f"无法识别的 embedding 返回类型: {type(raw)}")


class EmbeddingClient:
    """bge-m3 向量化客户端(OpenAI 兼容)"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_host,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        批量向量化文本,自动分批。

        Returns:
            shape 为 (len(texts), embedding_dim) 的 float32 数组
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch: list[str] = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.settings.embedding_dim):
            raise APIClientError(
                f"Embedding 返回形状异常: {result.shape}, "
                f"期望 ({len(texts)}, {self.settings.embedding_dim})"
            )
        return result

    def embed_query(self, text: str) -> np.ndarray:
        """向量化单条查询,返回 shape (dim,) 的向量"""
        return self.embed([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict = {
            "model": self.settings.embedding_model,
            "input": batch,
            "encoding_format": "base64" if self.settings.embedding_use_base64 else "float",
        }
        if self.settings.embedding_send_dim:
            kwargs["dimensions"] = self.settings.embedding_dim

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(**kwargs)
                ordered = sorted(response.data, key=lambda item: item.index)
                return [
                    _decode_embedding(item.embedding, self.settings.embedding_dim)
                    for item in ordered
                ]
            except (APITimeoutError, APIError, RateLimitError, httpx.HTTPError) as e:
                last_error = e
                wait: float = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding 请求失败(第 %d/%d 次): %s,%.0fs 后重试",
                    attempt, MAX_RETRIES, e, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise APIClientError(f"Embedding 请求在 {MAX_RETRIES} 次重试后失败: {last_error}")


class RerankClient:
    """bge-reranker-v2-m3 重排序客户端(Cohere 兼容 /rerank)"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """
        对候选文档重排序。

        Returns:
            (原始下标, 相关性分数) 列表,按分数降序,长度不超过 top_n
        """
        if not documents:
            return []

        payload: dict = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        headers: dict = {
            "Authorization": f"Bearer {self.settings.rerank_api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    response = client.post(
                        self.settings.rerank_host, json=payload, headers=headers
                    )
                response.raise_for_status()
                data = response.json()
                results = data.get("results") or []
                ranked: list[tuple[int, float]] = [
                    (int(item["index"]), float(item.get("relevance_score", 0.0)))
                    for item in results
                ]
                ranked.sort(key=lambda item: item[1], reverse=True)
                return ranked[:top_n]
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_error = e
                wait: float = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Rerank 请求失败(第 %d/%d 次): %s,%.0fs 后重试",
                    attempt, MAX_RETRIES, e, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise APIClientError(f"Rerank 请求在 {MAX_RETRIES} 次重试后失败: {last_error}")


class ChatClient:
    """kimi-for-coding 对话客户端(OpenAI 兼容,支持流式与非流式)"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def complete_chat(self, messages: list[dict]) -> str:
        """非流式对话,一次性返回完整回答文本(用于材料结构化解析)"""
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                extra_body: dict = {}
                if self.settings.llm_thinking_type:
                    extra_body["thinking"] = {"type": self.settings.llm_thinking_type}
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    stream=False,
                    extra_body=extra_body,
                )
                if not response.choices:
                    raise APIClientError("Chat 返回不含任何 choices")
                return response.choices[0].message.content or ""
            except (APITimeoutError, APIError, RateLimitError) as e:
                last_error = e
                wait: float = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Chat 请求失败(第 %d/%d 次): %s,%.0fs 后重试",
                    attempt, MAX_RETRIES, e, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise APIClientError(f"Chat 请求在 {MAX_RETRIES} 次重试后失败: {last_error}")

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """流式生成回答,逐段 yield 文本增量"""
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                extra_body: dict = {}
                if self.settings.llm_thinking_type:
                    extra_body["thinking"] = {"type": self.settings.llm_thinking_type}
                stream = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    stream=True,
                    extra_body=extra_body,
                )
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return
            except (APITimeoutError, APIError, RateLimitError) as e:
                last_error = e
                wait: float = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Chat 请求失败(第 %d/%d 次): %s,%.0fs 后重试",
                    attempt, MAX_RETRIES, e, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise APIClientError(f"Chat 请求在 {MAX_RETRIES} 次重试后失败: {last_error}")
