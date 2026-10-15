#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行问答入口。

Usage:
    python -m rag.cli "决议草案的提交需要多少赞成票?"   # 单次问答
    python -m rag.cli                                  # 交互模式(输入 q 退出)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigurationError, Settings
from .engine import RAGEngine, format_source

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rag.cli")


def answer_once(engine: RAGEngine, question: str) -> None:
    """检索 → 流式输出答案 → 列出处"""
    print("\n检索中...", end="", flush=True)
    passages = engine.retrieve(question)
    print("\r" + " " * 20 + "\r", end="", flush=True)

    if not passages:
        print("已收录的资料中证据不足,无法回答该问题。")
        return

    print("回答:\n")
    for token in engine.stream_answer(question, passages):
        print(token, end="", flush=True)
    print("\n")

    print("出处:")
    seen: set[str] = set()
    for item in passages:
        source = format_source(item["chunk"])
        if source not in seen:
            seen.add(source)
            print(f"  - {source} (相关度 {item['rerank_score']:.3f})")
    print()


def repl(engine: RAGEngine) -> None:
    print("模联资料问答(输入 q 或 quit 退出)")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break
        try:
            answer_once(engine, question)
        except Exception as e:  # noqa: BLE001 - CLI 交互场景,兜底提示后继续
            print(f"\n出错了: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag.cli", description="模拟联合国资料 RAG 命令行问答"
    )
    parser.add_argument("question", nargs="?", default="", help="要提问的问题(省略则进入交互模式)")
    parser.add_argument("--store-dir", type=str, default="./rag_store", help="索引目录")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        settings = Settings.from_env()
        engine = RAGEngine(settings, Path(args.store_dir).resolve())
    except (ConfigurationError, FileNotFoundError) as e:
        print(f"启动失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        answer_once(engine, args.question)
    else:
        repl(engine)


if __name__ == "__main__":
    main()
