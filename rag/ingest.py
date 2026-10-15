#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 RAG 索引:读取 source/json 下的结构化 JSON → 分块 → 批量向量化 → 落盘到 rag_store/。

Usage:
    python -m rag.ingest                          # 使用默认目录
    python -m rag.ingest --input-dir ./source/json --store-dir ./rag_store
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

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
    return parser.parse_args()


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

    source_counts = Counter(chunk.source for chunk in chunks)
    for source, count in sorted(source_counts.items()):
        logger.info("  %s: %d 块", source or "(未命名来源)", count)
    logger.info("共 %d 个文本块,开始向量化(模型 %s)...", len(chunks), settings.embedding_model)

    embedder = EmbeddingClient(settings)
    texts: list[str] = [chunk.content for chunk in chunks]

    vectors = []
    with tqdm(total=len(texts), desc="向量化") as bar:
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.append(embedder.embed(batch))
            bar.update(len(batch))

    import numpy as np

    matrix = np.vstack(vectors)
    store = VectorStore(chunks, matrix)
    store.save(store_dir)

    logger.info("=" * 50)
    logger.info("索引构建完成!")
    logger.info("  文本块: %d", len(chunks))
    logger.info("  向量维度: %d", matrix.shape[1])
    logger.info("  存储目录: %s", store_dir)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
