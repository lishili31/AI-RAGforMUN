#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 md_output 下的单页 Markdown 并切分为 RAG 文本块。

目录结构约定:
    md_output/<书名>/<书名><页码>.md
每页正文位于 ```text ... ``` 代码块中(image_to_text.py 的输出格式)。

分块策略与 prepare_for_rag.py 一致:优先按空行(段落)边界切分,
单块不超过 chunk_size 字符,相邻块保留 overlap 字符重叠。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100

# 目录名(即 Chunk.book)到完整显示书名的映射,用于 LLM 引用
BOOK_SUBTITLES: dict[str, str] = {
    "必修1": "必修1：分子与细胞",
    "必修2": "必修2：遗传与进化",
    "选择性必修1": "选择性必修1：稳态与调节",
    "选择性必修2": "选择性必修2：生物与环境",
    "选择性必修3": "选择性必修3：生物技术与工程",
    "课后习题和答案": "课后习题和答案",
    "高中生物学教材参考答案": "高中生物学教材参考答案",
}


@dataclass
class Chunk:
    """RAG 文本块(携带书名与页码用于引用)"""

    id: str
    book: str
    page: int
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "book": self.book,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=data["id"],
            book=data["book"],
            page=data["page"],
            chunk_index=data["chunk_index"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


def guess_page_number(filename: str) -> int:
    """从文件名中提取最后一组连续数字作为页码,如 '必修2:遗传与进化100.md' -> 100"""
    matches: list[str] = re.findall(r"(\d+)", Path(filename).stem)
    return int(matches[-1]) if matches else 0


def get_display_book(book: str) -> str:
    """根据目录名返回带副标题的完整显示书名"""
    return BOOK_SUBTITLES.get(book, book)


def extract_page_text(filepath: Path) -> str:
    """从单页 Markdown 中提取 ```text 代码块内的正文"""
    content: str = filepath.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)\n```", content, re.DOTALL)
    return match.group(1).strip() if match else ""


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


def scan_pages(input_dir: Path) -> list[tuple[str, int, Path]]:
    """
    扫描 input_dir 下所有书的单页 Markdown。

    Returns:
        (书名, 页码, 文件路径) 列表,按书名 + 页码排序
    """
    pages: list[tuple[str, int, Path]] = []
    for book_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for md_file in book_dir.glob("*.md"):
            if md_file.name == "index.md":
                continue
            pages.append((book_dir.name, guess_page_number(md_file.name), md_file))
    pages.sort(key=lambda item: (item[0], item[1]))
    return pages


def build_chunks(
    input_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """扫描目录并切分全部文本块"""
    chunks: list[Chunk] = []
    for book, page, md_file in scan_pages(input_dir):
        text: str = extract_page_text(md_file)
        if not text:
            continue
        for idx, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{book}-p{page:04d}-{idx:02d}",
                    book=book,
                    page=page,
                    chunk_index=idx,
                    content=piece,
                    metadata={"has_low_confidence": "【?】" in piece},
                )
            )
    return chunks
