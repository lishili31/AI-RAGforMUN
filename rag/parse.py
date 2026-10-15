#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
材料解析入口:把 source/ 下的 .html/.md 原始材料重写为结构化 JSON。

流程:
1. 用标准库抽取正文纯文本(HTML 用 html.parser 剥离标签,保留标题层级与表格结构)。
2. 抽取原始链接(HTML: <meta name="source"> → <link rel="canonical"> → 正文首个外站链接;
   Markdown: front-matter 的 url: / source:)。
3. 调用 Chat API(非流式)让模型重写为符合契约的 JSON 数组。
4. 白名单字段归一后写入 <输出目录>/<输入文件 stem>.json(先写临时文件再原子替换,
   解析失败绝不落半成品)。

产物条目字段(顺序固定):
    id / source / section / content / url / metadata

Usage:
    python -m rag.parse
    python -m rag.parse --input-dir ./source --output-dir ./source/json --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

from .clients import APIClientError, ChatClient
from .config import ConfigurationError, Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rag.parse")

DEFAULT_INPUT_DIR: str = "./source"
DEFAULT_OUTPUT_DIR: str = "./source/json"
SUPPORTED_SUFFIXES: tuple[str, ...] = (".html", ".htm", ".md")
SKIP_FILENAMES: frozenset[str] = frozenset({"index.md"})
MAX_INPUT_CHARS: int = 60000

SYSTEM_PROMPT: str = """你是模拟联合国(MUN)资料整理助手。把给定的原始材料文本重写为结构化 JSON 数组,供检索问答使用。

硬性要求:
1. 只使用原文信息,忠实原文;严禁虚构、推测、扩写原文没有的内容。
2. 必须完整保留原文出现的全部数字、日期、比率、条款编号、专有名词、机构名与术语(表格里的数字一个都不能丢)。
3. 每条记录是一个自包含的语义单元:content 用 1-3 句完整的话表述,脱离上下文也能读懂。
4. section 填该条内容所属的章节目录路径(如 "第二章 / 第三节 表决程序"),原文没有层级信息时填空字符串 ""。
5. id 使用 "<文件名>_<四位序号>" 形式,例如 test_0001,从 0001 开始按顺序递增。
6. source 填人类可读的文件名。
7. metadata.language 按正文语言判断,取值 "zh" 或 "en";metadata.document_year 为文档年份整数,取不到写 null;metadata.chunk_type 为内容类型(如 rule / procedure / definition / fact / figure / document_metadata)。
8. url 原样填写给定的文件链接,没有链接时写 ""。
9. 只输出 JSON 数组本身,不要输出解释、Markdown 代码围栏或任何额外文字。

每条记录的字段(不得增删字段,不得添加额外顶层字段):
{"id": "test_0001", "source": "示例文件", "section": "第二章 / 表决程序", "content": "……", "url": "https://example.org/doc", "metadata": {"language": "zh", "document_year": 2025, "chunk_type": "rule"}}"""

USER_PROMPT_TEMPLATE: str = """文件名: {filename}
文件链接(url): {url}

文件正文如下:
---
{content}
---
请把它重写为符合要求的结构化 JSON 数组。"""

RETRY_PROMPT_TEMPLATE: str = """上一次的输出无法解析为合法的 JSON 数组,错误信息: {error}

请重新输出完整、合法的 JSON 数组:严格遵守字段约定,不要输出任何额外文字或代码围栏。"""


class ParseError(Exception):
    """模型输出无法解析为符合契约的 JSON 数组"""


class _HTMLTextExtractor(HTMLParser):
    """抽取 HTML 正文纯文本,保留标题层级(h1-h6)与表格结构(行/单元格)"""

    BLOCK_TAGS: frozenset[str] = frozenset(
        {
            "p", "div", "section", "article", "header", "footer", "main", "aside",
            "nav", "blockquote", "pre", "figure", "figcaption", "dt", "dd", "li",
            "ul", "ol", "dl", "address", "form", "fieldset", "title",
        }
    )
    SKIP_TAGS: frozenset[str] = frozenset(
        {"script", "style", "noscript", "template", "iframe", "svg", "canvas"}
    )
    HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._buffer: list[str] = []
        self._cell_buffer: list[str] = []
        self._prefix: str = ""
        self._skip_depth: int = 0
        self._table_depth: int = 0
        self._in_cell: bool = False
        self._row: list[str] = []
        self.links: list[str] = []

    # -- 对外接口 --------------------------------------------------------
    def get_text(self) -> str:
        self._flush()
        collapsed: list[str] = []
        for line in self._lines:
            if not line and collapsed and not collapsed[-1]:
                continue
            collapsed.append(line)
        return "\n".join(collapsed).strip()

    # -- 内部工具 --------------------------------------------------------
    def _flush(self) -> None:
        text: str = re.sub(r"\s+", " ", " ".join(self._buffer)).strip()
        self._buffer.clear()
        if text:
            self._lines.append(f"{self._prefix}{text}")
            self._prefix = ""

    # -- HTMLParser 回调 -------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes: dict[str, str] = {
            key.lower(): (value or "") for key, value in attrs
        }

        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "a":
            href: str = attributes.get("href", "").strip()
            if href:
                self.links.append(href)
            return
        if tag == "br":
            self._flush()
            return
        if tag == "table":
            self._flush()
            self._table_depth += 1
            return
        if tag == "caption":
            self._flush()
            self._prefix = "表标题: "
            return
        if tag in ("thead", "tbody", "tfoot"):
            self._flush()
            return
        if tag == "tr":
            self._flush()
            self._row = []
            return
        if tag in ("td", "th"):
            self._flush()
            self._in_cell = True
            self._cell_buffer.clear()
            return
        if tag in self.HEADING_TAGS:
            self._flush()
            self._prefix = f"{'#' * int(tag[1])} "
            return
        if tag in self.BLOCK_TAGS:
            self._flush()

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        if tag.lower() == "br" and not self._skip_depth:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag in ("td", "th"):
            cell: str = re.sub(r"\s+", " ", " ".join(self._cell_buffer)).strip()
            self._row.append(cell)
            self._cell_buffer.clear()
            self._in_cell = False
            return
        if tag == "tr":
            self._flush()
            if self._row:
                self._lines.append("| " + " | ".join(self._row) + " |")
            self._row = []
            return
        if tag == "table":
            self._flush()
            self._table_depth = max(0, self._table_depth - 1)
            self._lines.append("")
            return
        if tag in self.HEADING_TAGS or tag in self.BLOCK_TAGS or tag == "caption":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_cell:
            self._cell_buffer.append(data)
        else:
            self._buffer.append(data)


def extract_html(raw: str) -> tuple[str, str]:
    """返回 (正文纯文本, 首个原始链接)"""
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    return parser.get_text(), _extract_html_url(raw, parser.links)


def _extract_attr(tag: str, name: str) -> str:
    """从单个标签文本中读取属性值(兼容单双引号与属性顺序)"""
    match = re.search(rf"""{name}\s*=\s*"([^"]*)\"""", tag, flags=re.IGNORECASE)
    if not match:
        match = re.search(rf"""{name}\s*=\s*'([^']*)'""", tag, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_html_url(raw: str, links: list[str]) -> str:
    """HTML 链接抽取: <meta name="source"> → <link rel="canonical"> → 正文首个外站链接"""
    for tag in re.findall(r"<meta\b[^>]*>", raw, flags=re.IGNORECASE):
        if _extract_attr(tag, "name").lower() == "source":
            content: str = _extract_attr(tag, "content")
            if content:
                return content

    for tag in re.findall(r"<link\b[^>]*>", raw, flags=re.IGNORECASE):
        if "canonical" in _extract_attr(tag, "rel").lower():
            href: str = _extract_attr(tag, "href")
            if href:
                return href

    for link in links:
        if re.match(r"^https?://", link, flags=re.IGNORECASE):
            return link
    return ""


def extract_markdown_url(raw: str) -> str:
    """Markdown 链接抽取: front-matter 的 url: 或 source:"""
    if not raw.startswith("---"):
        return ""
    end: int = raw.find("\n---", 3)
    if end == -1:
        return ""

    front: dict[str, str] = {}
    for line in raw[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        front[key.strip().lower()] = value.strip().strip('"').strip("'")
    return front.get("url") or front.get("source") or ""


def read_document(path: Path) -> tuple[str, str]:
    """读取输入文件,返回 (正文纯文本, 原始链接)"""
    raw: str = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        return extract_html(raw)
    return raw, extract_markdown_url(raw)


def build_messages(filename: str, text: str, url: str) -> list[dict]:
    """组装解析用对话消息"""
    user_content: str = USER_PROMPT_TEMPLATE.format(
        filename=filename, url=url or "(无)", content=text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_json_array(reply: str) -> str:
    """去掉 ```json 代码围栏,再截取第一个 '[' 到最后一个 ']' 之间的内容"""
    text: str = reply.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start: int = text.find("[")
    end: int = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ParseError("未能定位 JSON 数组,输出中找不到成对的 []")
    return text[start : end + 1]


def parse_model_output(reply: str, stem: str) -> list[dict]:
    """解析并校验模型输出;不合法时抛出 ParseError"""
    payload: str = extract_json_array(reply)
    try:
        data: Any = json.loads(payload)
    except ValueError as e:
        raise ParseError(f"JSON 解析失败: {e}") from e

    if not isinstance(data, list):
        raise ParseError(f"顶层不是 JSON 数组,而是 {type(data).__name__}")
    if not data:
        raise ParseError("条目数为 0")

    problems: list[str] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            problems.append(f"第 {index} 条不是 JSON 对象")
            continue
        for field in ("id", "source", "content"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"第 {index} 条缺少或为空: {field}")
    if problems:
        raise ParseError("; ".join(problems[:5]))
    return data


def _normalize_year(value: Any) -> Optional[int]:
    """年份归一: 整数原样,数字字符串转整数,缺失/取不到写 None(JSON null)"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_records(records: list[dict], stem: str, url: str) -> list[dict]:
    """字段白名单与类型归一: id 强制为 <stem>_<四位序号>, url 全文共用, metadata 保证为 dict"""
    normalized: list[dict] = []
    for index, item in enumerate(records, start=1):
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        normalized.append(
            {
                "id": f"{stem}_{index:04d}",
                "source": str(item.get("source") or "").strip(),
                "section": str(item.get("section") or "").strip(),
                "content": str(item.get("content") or "").strip(),
                "url": url,
                "metadata": {
                    "language": str(metadata.get("language") or "").strip(),
                    "document_year": _normalize_year(metadata.get("document_year")),
                    "chunk_type": str(metadata.get("chunk_type") or "").strip(),
                },
            }
        )
    return normalized


def write_records(path: Path, records: list[dict]) -> None:
    """原子写盘: 先写同目录临时文件再替换,避免留下半成品"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, path)


def parse_document(path: Path, chat: ChatClient) -> list[dict]:
    """解析单个文件: 抽文本 → 调模型 → 校验;解析失败带错误信息重试一次"""
    text, url = read_document(path)

    if not url:
        logger.warning(
            "%s: 未找到原始链接(meta source / canonical / 正文外链 / front-matter),url 写空字符串",
            path.name,
        )
    if not text.strip():
        raise ParseError("抽取到的正文为空")
    if len(text) > MAX_INPUT_CHARS:
        logger.warning(
            "%s: 正文 %d 字符超过上限 %d,已截断后再交模型处理",
            path.name, len(text), MAX_INPUT_CHARS,
        )
        text = text[:MAX_INPUT_CHARS]

    messages: list[dict] = build_messages(path.name, text, url)
    reply: str = chat.complete_chat(messages)
    try:
        records = parse_model_output(reply, path.stem)
    except ParseError as e:
        logger.warning("%s: 首次输出解析失败(%s),带错误信息重试一次", path.name, e)
        retry_messages: list[dict] = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": RETRY_PROMPT_TEMPLATE.format(error=e)},
        ]
        records = parse_model_output(chat.complete_chat(retry_messages), path.stem)

    return normalize_records(records, path.stem, url)


def discover_inputs(input_dir: Path, output_dir: Path) -> list[Path]:
    """扫描输入目录(不递归),跳过输出目录自身、index.md 与非 .html/.htm/.md 文件"""
    files: list[Path] = []
    for entry in sorted(input_dir.iterdir()):
        resolved: Path = entry.resolve()
        if output_dir == resolved or output_dir in resolved.parents:
            continue
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if entry.name.lower() in SKIP_FILENAMES:
            continue
        files.append(entry)
    return files


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag.parse",
        description="把模联原始材料(HTML/Markdown)重写为结构化 JSON",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"原始材料目录(默认: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"JSON 产物目录(默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新解析,忽略已存在且较新的产物",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_dir: Path = Path(args.input_dir).resolve()
    output_dir: Path = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        logger.error("输入目录不存在: %s", input_dir)
        sys.exit(1)

    try:
        settings = Settings.from_env()
    except ConfigurationError as e:
        logger.error("%s", e)
        sys.exit(1)

    files: list[Path] = discover_inputs(input_dir, output_dir)
    if not files:
        logger.error("未找到 .html/.htm/.md 材料文件: %s", input_dir)
        sys.exit(1)

    logger.info("发现 %d 个材料文件,输出目录: %s", len(files), output_dir)
    chat = ChatClient(settings)

    total: int = 0
    written: int = 0
    failed: int = 0
    results: list[tuple[str, str]] = []
    for path in files:
        target: Path = output_dir / f"{path.stem}.json"
        if target.exists() and not args.force and target.stat().st_mtime > path.stat().st_mtime:
            logger.info("%s: 已存在，跳过（--force 可强制重解析）", path.name)
            results.append((path.name, "已存在，跳过"))
            continue

        try:
            records = parse_document(path, chat)
        except (APIClientError, ParseError, OSError) as e:
            logger.warning("%s: 解析失败，跳过该文件: %s", path.name, e)
            results.append((path.name, f"失败({e})"))
            failed += 1
            continue

        write_records(target, records)
        logger.info("%s: %d 条 → %s", path.name, len(records), target)
        results.append((path.name, f"{len(records)} 条"))
        total += len(records)
        written += 1

    logger.info("=" * 50)
    for name, summary in results:
        logger.info("  %s: %s", name, summary)
    logger.info(
        "共处理 %d 个文件, 新写入 %d 个文件, 失败 %d 个, 总条目数 %d",
        len(files), written, failed, total,
    )
    logger.info("=" * 50)

    if written == 0 and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
