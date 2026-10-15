#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 numpy 的本地向量库。

存储内容(rag_store/ 目录):
    chunks.json  — 全部文本块(含来源、章节、链接、正文)
    vectors.npy  — 归一化后的 float32 向量矩阵 (N, dim)
    meta.json    — 块数、维度、来源列表、构建时间等元信息

向量在入库时已做 L2 归一化,检索即矩阵点积(等价余弦相似度)。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk

CHUNKS_FILENAME: str = "chunks.json"
VECTORS_FILENAME: str = "vectors.npy"
META_FILENAME: str = "meta.json"


def normalize(vectors: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化(零向量保持为零,避免除零)"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class VectorStore:
    """内存向量库:余弦相似度 top-k 检索"""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"块数({len(chunks)})与向量行数({vectors.shape[0]})不一致"
            )
        self.chunks: list[Chunk] = chunks
        self.vectors: np.ndarray = normalize(vectors.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """
        余弦相似度检索。

        Returns:
            (chunk 下标, 相似度) 列表,按相似度降序,长度不超过 top_k
        """
        if not self.chunks:
            return []
        query = query_vector.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        scores: np.ndarray = self.vectors @ query
        top_k = min(top_k, len(self.chunks))
        # argpartition 取前 top_k 再局部排序,避免全量排序
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(
            ((int(i), float(scores[i])) for i in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked

    def save(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / CHUNKS_FILENAME).write_text(
            json.dumps(
                [chunk.to_dict() for chunk in self.chunks],
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        np.save(store_dir / VECTORS_FILENAME, self.vectors)
        sources: list[str] = sorted({chunk.source for chunk in self.chunks})
        (store_dir / META_FILENAME).write_text(
            json.dumps(
                {
                    "chunk_count": len(self.chunks),
                    "dim": int(self.vectors.shape[1]) if len(self.chunks) else 0,
                    "sources": sources,
                    "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, store_dir: Path) -> "VectorStore":
        chunks_path = store_dir / CHUNKS_FILENAME
        vectors_path = store_dir / VECTORS_FILENAME
        if not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"索引不存在于 {store_dir},请先运行: python -m rag.ingest"
            )
        chunks = [
            Chunk.from_dict(item)
            for item in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        vectors: np.ndarray = np.load(vectors_path)
        store = cls.__new__(cls)
        store.chunks = chunks
        store.vectors = vectors  # 入库时已归一化
        if len(chunks) != vectors.shape[0]:
            raise ValueError("索引文件损坏: 块数与向量行数不一致")
        return store

    @staticmethod
    def meta(store_dir: Path) -> dict:
        meta_path = store_dir / META_FILENAME
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))
