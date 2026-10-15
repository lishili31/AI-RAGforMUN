#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 RAG 索引:读取 source/json 下的结构化 JSON → 分块 → 向量化 → 落盘到 rag_store/。

增量策略:向量缓存以文本块正文为键。构建时先复用现有索引中正文完全相同的向量,
只有新增或内容变化的文本才会调用 Embedding 接口,因此索引中每一行向量都与其正文
一一对应,已删除资料的向量也不会残留。现有索引的向量化模型或维度与当前配置不一致
(含旧版本构建、meta.json 中缺少 embedding_model 的索引)时,缓存整体失效并自动全量重建。

Usage:
    python -m rag.ingest                          # 增量构建(默认目录)
    python -m rag.ingest --input-dir ./source/json --store-dir ./rag_store
    python -m rag.ingest --force                  # 忽略缓存,强制全量向量化
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .chunking import Chunk, load_chunks, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from .clients import EmbeddingClient, EMBED_BATCH_SIZE
from .config import ConfigurationError, Settings
from .store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rag.ingest")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag.ingest",
        description="构建模拟联合国规则与事实问答 RAG 向量索引",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./source/json",
        help="结构化 JSON 根目录(默认: ./source/json)",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default="./rag_store",
        help="索引输出目录(默认: ./rag_store)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略现有向量缓存,对全部文本块重新向量化",
    )
    return parser.parse_args()


def load_vector_cache(
    store_dir: Path, settings: Settings
) -> tuple[dict[str, np.ndarray], int]:
    """
    读取现有索引,构建 正文 → 向量 的可复用缓存。

    Returns:
        (缓存, 现有索引块数)。索引不存在、向量模型/维度与当前配置不符或文件
        损坏时返回空缓存并说明原因,由调用方转为全量向量化。
    """
    meta: dict = VectorStore.meta(store_dir)
    if not meta:
        return {}, 0

    try:
        cached_model: str = str(meta.get("embedding_model") or "")
        cached_dim: int = int(meta.get("dim") or 0)
        previous_count: int = int(meta.get("chunk_count") or 0)
    except (TypeError, ValueError):
        logger.warning("现有索引 meta.json 字段异常,忽略缓存并全量向量化")
        return {}, 0

    if cached_model != settings.embedding_model:
        logger.warning(
            "现有索引由模型 %s 构建,当前模型为 %s,忽略缓存并全量向量化",
            cached_model or "(未知,索引由旧版本构建)",
            settings.embedding_model,
        )
        return {}, previous_count
    if cached_dim != settings.embedding_dim:
        logger.warning(
            "现有索引维度为 %d,当前配置为 %d,忽略缓存并全量向量化",
            cached_dim,
            settings.embedding_dim,
        )
        return {}, previous_count

    try:
        store = VectorStore.load(store_dir)
    except (OSError, ValueError) as e:
        logger.warning("读取现有索引失败(%s),忽略缓存并全量向量化", e)
        return {}, previous_count

    if store.vectors.shape[1] != settings.embedding_dim:
        logger.warning(
            "现有索引向量维度为 %d,当前配置为 %d,忽略缓存并全量向量化",
            store.vectors.shape[1],
            settings.embedding_dim,
        )
        return {}, previous_count

    cache: dict[str, np.ndarray] = {}
    for chunk, vector in zip(store.chunks, store.vectors):
        cache.setdefault(chunk.content, vector)
    return cache, previous_count


def main() -> None:
    args = parse_arguments()
    input_dir = Path(args.input_dir).resolve()
    store_dir = Path(args.store_dir).resolve()

    try:
        settings = Settings.from_env()
    except ConfigurationError as e:
        logger.error("%s", e)
        sys.exit(1)

    logger.info("扫描并分块: %s", input_dir)
    chunks: list[Chunk] = load_chunks(input_dir, args.chunk_size, args.overlap)
    if not chunks:
        logger.error("未切分出任何文本块,请检查输入目录")
        sys.exit(1)

    if args.force:
        logger.info("已指定 --force:忽略现有向量缓存,全量向量化")
        cache: dict[str, np.ndarray] = {}
        previous_count: int = 0
    else:
        cache, previous_count = load_vector_cache(store_dir, settings)

    reused: int = 0
    new_texts: list[str] = []
    queued: set[str] = set()
    per_source: dict[str, Counter] = {}
    for chunk in chunks:
        hit: bool = chunk.content in cache
        reused += 1 if hit else 0
        per_source.setdefault(chunk.source, Counter())["reused" if hit else "embedded"] += 1
        if not hit and chunk.content not in queued:
            queued.add(chunk.content)
            new_texts.append(chunk.content)

    if previous_count and previous_count != len(chunks):
        logger.info("文本块数量变化: %d → %d", previous_count, len(chunks))
    for source, counter in sorted(per_source.items()):
        logger.info(
            "  %s: %d 块(复用 %d,新向量化 %d)",
            source or "(未命名来源)",
            counter["reused"] + counter["embedded"],
            counter["reused"],
            counter["embedded"],
        )

    if new_texts:
        logger.info(
            "共 %d 个文本块,其中 %d 块需新向量化(去重后 %d 个文本),使用模型 %s...",
            len(chunks),
            len(chunks) - reused,
            len(new_texts),
            settings.embedding_model,
        )
        embedder = EmbeddingClient(settings)
        with tqdm(total=len(new_texts), desc="向量化") as bar:
            for start in range(0, len(new_texts), EMBED_BATCH_SIZE):
                batch = new_texts[start : start + EMBED_BATCH_SIZE]
                for text, vector in zip(batch, embedder.embed(batch)):
                    cache[text] = vector
                bar.update(len(batch))
    else:
        logger.info("共 %d 个文本块,全部命中向量缓存,跳过 Embedding 调用", len(chunks))

    matrix = np.vstack([cache[chunk.content] for chunk in chunks])
    store = VectorStore(chunks, matrix)
    store.save(store_dir, embedding_model=settings.embedding_model)

    logger.info("=" * 50)
    logger.info("索引构建完成!")
    logger.info(
        "  文本块: %d(复用向量 %d,新向量化 %d)",
        len(chunks),
        reused,
        len(chunks) - reused,
    )
    logger.info("  向量维度: %d", matrix.shape[1])
    logger.info("  存储目录: %s", store_dir)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
