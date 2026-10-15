#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 source/json 下的结构化 JSON 文件并切分为 RAG 文本块。

输入文件约定(每个文件是一个 JSON 数组):
    [
      {
        "id": "pkunmun2025_manual_front_0001",
        "source": "北大学标PKUNMUN2025_学术标准手册",
        "section": "封面与出版信息 / 手册名称",
        "content": "……正文……",
        "url": "https://example.com/handbook",
        "metadata": {"language": "zh", "document_year": 2025, "chunk_type": "document_metadata"}
      },
      ...
    ]

分块策略:每个数组元素的 content 单独切分,优先按空行(段落)边界聚合,
单块不超过 chunk_size 字符,相邻块保留 overlap 字符重叠;切分出的文本块
继承该元素的 source / section / url / metadata。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100

logger = logging.getLogger("rag.chunking")


@dataclass
class Chunk:
    """RAG 文本块(携带来源文档、章节与原始链接用于引用)"""

    id: str
    source: str
    section: str
    url: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "section": self.section,
            "url": self.url,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=data["id"],
            source=data.get("source", ""),
            section=data.get("section", ""),
            url=data.get("url", ""),
            chunk_index=data["chunk_index"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


def get_display_source(source: str) -> str:
    """返回用于展示/引用的文档名(直通,便于统一调用点)"""
    return source


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    将一段文本按段落/长度切分为重叠块。

    1. 优先按空行(段落)边界聚合,不超过 chunk_size
    2. 单段超过 chunk_size 时按字符硬切
    3. 相邻块保留 overlap 字符重叠
    """
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buffer: str = ""

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        pieces.append(buffer.strip())
        buffer = buffer[-overlap:] if overlap > 0 and len(buffer) > overlap else ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                flush()
            step: int = max(chunk_size - overlap, 1)
            for i in range(0, len(para), step):
                pieces.append(para[i : i + chunk_size].strip())
            buffer = ""
            continue

        candidate: str = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > chunk_size:
            flush()
            buffer = para
        else:
            buffer = candidate

    flush()
    return [p for p in pieces if p]


def _read_str(element: dict, key: str, json_file: Path, element_index: int) -> str:
    """读取字符串字段,缺失或类型异常时回退为空串并告警"""
    value = element.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        logger.warning(
            "%s 第 %d 个元素的 %s 字段类型异常(%s),已按空值处理",
            json_file.name,
            element_index,
            key,
            type(value).__name__,
        )
        return ""
    return value.strip()


def load_chunks(
    input_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    扫描 input_dir 下所有 .json 并切分为文本块。

    单个文件 JSON 损坏或结构不符时跳过并告警,不影响其余文件;
    元素缺少有效 content 时跳过该元素。
    """
    chunks: list[Chunk] = []
    json_files: list[Path] = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.warning("目录 %s 下没有找到任何 .json 文件", input_dir)
        return chunks

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("跳过无法解析的文件 %s: %s", json_file.name, e)
            continue
        if not isinstance(data, list):
            logger.warning("跳过 %s: 顶层结构不是 JSON 数组", json_file.name)
            continue

        for element_index, element in enumerate(data, start=1):
            if not isinstance(element, dict):
                logger.warning(
                    "跳过 %s 第 %d 个元素: 不是 JSON 对象", json_file.name, element_index
                )
                continue

            content = element.get("content")
            if not isinstance(content, str) or not content.strip():
                logger.warning(
                    "跳过 %s 第 %d 个元素: content 缺失或为空", json_file.name, element_index
                )
                continue

            raw_id = element.get("id")
            element_id = raw_id.strip() if isinstance(raw_id, str) else ""
            if not element_id:
                element_id = f"{json_file.stem}-{element_index:04d}"

            metadata = element.get("metadata")
            if not isinstance(metadata, dict):
                if metadata is not None:
                    logger.warning(
                        "%s 第 %d 个元素的 metadata 类型异常(%s),已按空字典处理",
                        json_file.name,
                        element_index,
                        type(metadata).__name__,
                    )
                metadata = {}

            source = _read_str(element, "source", json_file, element_index)
            section = _read_str(element, "section", json_file, element_index)
            url = _read_str(element, "url", json_file, element_index)

            for idx, piece in enumerate(chunk_text(content, chunk_size, overlap)):
                chunks.append(
                    Chunk(
                        id=f"{element_id}-{idx:02d}",
                        source=source,
                        section=section,
                        url=url,
                        chunk_index=idx,
                        content=piece,
                        metadata=dict(metadata),
                    )
                )
    return chunks
