#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 .env 加载 RAG 应用配置(LLM / Embedding / Rerank)。

所有配置项与 .env.example 对齐;缺少必需项时抛出 ConfigurationError。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_LLM_BASE_URL: str = "https://api.kimi.com/coding/v1"
DEFAULT_LLM_MODEL: str = "kimi-for-coding"


class ConfigurationError(Exception):
    """配置缺失或非法"""


def _required(name: str) -> str:
    value: str = os.environ.get(name, "").strip().strip('"').strip("'")
    if not value:
        raise ConfigurationError(
            f"未设置 {name} 环境变量。请在 .env 文件中配置(参考 .env.example)。"
        )
    return value


def _optional(name: str, default: str) -> str:
    value: str = os.environ.get(name, "").strip().strip('"').strip("'")
    return value or default


def _optional_bool(name: str, default: bool) -> bool:
    value: str = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _optional_int(name: str, default: int) -> int:
    value: str = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class Settings:
    """RAG 应用全部配置"""

    # LLM(生成答案)
    llm_api_key: str
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL

    # Embedding(向量化)
    embedding_host: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_token_limit: int = 8192
    embedding_send_dim: bool = False
    embedding_use_base64: bool = True

    # Rerank(重排序, Cohere 兼容接口)
    rerank_host: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "bge-reranker-v2-m3"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Settings":
        env_path: Path = env_file or (PROJECT_ROOT / ".env")
        load_dotenv(env_path)

        return cls(
            llm_api_key=_required("KIMI_API_KEY"),
            llm_base_url=_optional("API_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_model=_optional("API_MODEL", DEFAULT_LLM_MODEL),
            embedding_host=_required("EMBEDDING_BINDING_HOST"),
            embedding_api_key=_required("EMBEDDING_BINDING_API_KEY"),
            embedding_model=_optional("EMBEDDING_MODEL", "bge-m3"),
            embedding_dim=_optional_int("EMBEDDING_DIM", 1024),
            embedding_token_limit=_optional_int("EMBEDDING_TOKEN_LIMIT", 8192),
            embedding_send_dim=_optional_bool("EMBEDDING_SEND_DIM", False),
            embedding_use_base64=_optional_bool("EMBEDDING_USE_BASE64", True),
            rerank_host=_required("RERANK_BINDING_HOST"),
            rerank_api_key=_required("RERANK_BINDING_API_KEY"),
            rerank_model=_optional("RERANK_MODEL", "bge-reranker-v2-m3"),
        )
