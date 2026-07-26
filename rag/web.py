#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web 聊天界面(FastAPI,无前端构建,单文件页面)。

Usage:
    python -m rag.web [--host 127.0.0.1] [--port 5000]

接口:
    GET  /            聊天页面
    POST /api/chat    {"question": "..."} → SSE 流式回答(token / sources / error)
    GET  /api/stats   索引统计信息
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, ConfigurationError, Settings
from .engine import RAGEngine, format_source
from .store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rag.web")

STORE_DIR: Path = PROJECT_ROOT / "rag_store"

app = FastAPI(title="高中生物教材问答")
_engine: Optional[RAGEngine] = None
_store_dir: Path = STORE_DIR


class ChatRequest(BaseModel):
    question: str


def get_engine() -> RAGEngine:
    """惰性加载引擎(首次请求时加载索引与客户端)"""
    global _engine
    if _engine is None:
        settings = Settings.from_env()
        _engine = RAGEngine(settings, _store_dir)
    return _engine


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def chat_event_stream(question: str) -> Iterator[str]:
    try:
        engine = get_engine()
    except (ConfigurationError, FileNotFoundError) as e:
        yield sse({"type": "error", "message": f"服务未就绪: {e}"})
        return

    try:
        passages = engine.retrieve(question)
    except Exception as e:  # noqa: BLE001 - 检索失败需反馈到前端
        logger.exception("检索失败")
        yield sse({"type": "error", "message": f"检索失败: {e}"})
        return

    if not passages:
        yield sse({"type": "token", "content": "未检索到相关教材内容,请换个问法试试。"})
        yield "data: [DONE]\n\n"
        return

    try:
        for token in engine.stream_answer(question, passages):
            yield sse({"type": "token", "content": token})
    except Exception as e:  # noqa: BLE001 - 生成中断需反馈到前端
        logger.exception("生成失败")
        yield sse({"type": "error", "message": f"生成失败: {e}"})
        return

    sources = [
        {
            "source": format_source(item["chunk"]),
            "score": round(item["rerank_score"], 4),
        }
        for item in passages
    ]
    yield sse({"type": "sources", "sources": sources})
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        def empty() -> Iterator[str]:
            yield sse({"type": "error", "message": "问题不能为空"})
        return StreamingResponse(empty(), media_type="text/event-stream")
    return StreamingResponse(
        chat_event_stream(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stats")
def api_stats() -> dict:
    meta = VectorStore.meta(_store_dir)
    if not meta:
        return {"ready": False, "message": "索引未构建,请先运行 python -m rag.ingest"}
    return {"ready": True, **meta}


INDEX_HTML: str = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>高中生物教材问答</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f0f2f5; height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: #2f6b4f; color: #fff; padding: 12px 20px;
    display: flex; align-items: baseline; gap: 12px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; opacity: .8; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; }
  .msg { max-width: 820px; margin: 0 auto 16px; display: flex; }
  .msg .bubble {
    padding: 12px 16px; border-radius: 10px; line-height: 1.8;
    white-space: pre-wrap; word-break: break-word; font-size: 15px;
  }
  .msg.user { justify-content: flex-end; }
  .msg.user .bubble { background: #2f6b4f; color: #fff; border-bottom-right-radius: 2px; }
  .msg.bot .bubble { background: #fff; border: 1px solid #e4e7eb; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .sources { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #ddd; font-size: 13px; color: #666; }
  .sources .tag {
    display: inline-block; background: #eef5f1; color: #2f6b4f;
    border-radius: 4px; padding: 2px 8px; margin: 3px 6px 0 0;
  }
  .error { color: #c0392b; }
  #input-bar {
    background: #fff; border-top: 1px solid #e4e7eb; padding: 12px 20px;
  }
  #input-bar .row { max-width: 820px; margin: 0 auto; display: flex; gap: 10px; }
  #question {
    flex: 1; border: 1px solid #d0d5db; border-radius: 8px;
    padding: 10px 14px; font-size: 15px; outline: none;
  }
  #question:focus { border-color: #2f6b4f; }
  #send {
    background: #2f6b4f; color: #fff; border: none; border-radius: 8px;
    padding: 0 22px; font-size: 15px; cursor: pointer;
  }
  #send:disabled { background: #9db8ac; cursor: not-allowed; }
  .hint { max-width: 820px; margin: 6px auto 0; font-size: 12px; color: #999; }
</style>
</head>
<body>
<header>
  <h1>高中生物教材问答</h1>
  <span id="stats">基于人教版教材 · RAG 检索增强</span>
</header>
<div id="chat"></div>
<div id="input-bar">
  <div class="row">
    <input id="question" placeholder="例如:兴奋在神经元之间是如何传递的?" autofocus>
    <button id="send">提问</button>
  </div>
  <div class="hint">答案由教材原文检索生成,出处以【《书名》第N页】标注 · Enter 发送</div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('question');
const sendBtn = document.getElementById('send');
let busy = false;

fetch('/api/stats').then(r => r.json()).then(s => {
  if (s.ready) document.getElementById('stats').textContent =
    `已索引 ${s.chunk_count} 个片段 · ${s.books.length} 本教材`;
}).catch(() => {});

function addMsg(role) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function ask() {
  const question = input.value.trim();
  if (!question || busy) return;
  busy = true; sendBtn.disabled = true; input.value = '';
  addMsg('user').textContent = question;
  const bubble = addMsg('bot');
  bubble.textContent = '检索中…';

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', answer = '', sourcesBox = null;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const events = buffer.split('\\n\\n');
      buffer = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        const data = JSON.parse(payload);
        if (data.type === 'token') {
          if (!answer) bubble.textContent = '';
          answer += data.content;
          bubble.textContent = answer;
        } else if (data.type === 'sources') {
          sourcesBox = document.createElement('div');
          sourcesBox.className = 'sources';
          sourcesBox.innerHTML = '出处: ' + data.sources.map(s =>
            `<span class="tag">${s.source}</span>`).join('');
          bubble.appendChild(sourcesBox);
        } else if (data.type === 'error') {
          bubble.innerHTML = `<span class="error">${data.message}</span>`;
        }
        chat.scrollTop = chat.scrollHeight;
      }
    }
  } catch (e) {
    bubble.innerHTML = `<span class="error">请求失败: ${e.message}</span>`;
  }
  busy = false; sendBtn.disabled = false; input.focus();
}

sendBtn.addEventListener('click', ask);
input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rag.web", description="高中生物教材 RAG Web 问答")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--store-dir", type=str, default=str(STORE_DIR), help="索引目录")
    return parser.parse_args()


def main() -> None:
    global _store_dir
    args = parse_arguments()
    _store_dir = Path(args.store_dir).resolve()

    if not (VectorStore.meta(_store_dir)):
        logger.warning("索引不存在于 %s,首次提问将失败;请先运行 python -m rag.ingest", _store_dir)

    import uvicorn

    logger.info("启动 Web 服务: http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
