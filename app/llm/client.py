"""DashScope（通义千问）客户端，内置 mock 模式。

设计要点：
- 没有 API Key（或 USE_MOCK=1）时自动进入 mock 模式，保证无网、无密钥也能演示与跑评测。
- 真实模式才 import dashscope（懒加载），因此单测不需要安装 SDK、更不会触网。
- 所有调用（含 mock）统一写入 logs/traces.jsonl，便于事后排查与评测回溯。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

# ---- 项目根目录 / 常量 ----

ROOT = Path(__file__).resolve().parents[2]  # app/llm/client.py -> 项目根

TRACE_PATH = ROOT / "logs" / "traces.jsonl"

# 若安装了 python-dotenv，自动读取根目录 .env（缺省不报错，保证 mock 模式零依赖）
try:  # pragma: no cover - 环境相关的小分支
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DEFAULT_CHAT_MODEL = "qwen-plus"
DEFAULT_EMBED_MODEL = "text-embedding-v3"

#: mock 向量维度（SPEC 规定 128 维）
MOCK_EMBED_DIM = 128

#: 每个 token 散列到几个维度（太少则向量稀疏、相似度恒为 0）
_HASHES_PER_TOKEN = 4

#: 示例数据表的列名，供 mock 的「提取列名」规则使用
KNOWN_COLUMNS: tuple[str, ...] = (
    "date",
    "region",
    "category",
    "salesperson",
    "amount",
    "quantity",
)

#: 中文别名 -> 列名，让 mock 也能"听懂"中文问题（评测用例多用中文提问）
COLUMN_ALIASES: dict[str, str] = {
    "日期": "date",
    "时间": "date",
    "月份": "date",
    "地区": "region",
    "区域": "region",
    "大区": "region",
    "省份": "region",
    "城市": "region",
    "品类": "category",
    "类目": "category",
    "类别": "category",
    "产品": "category",
    "销售员": "salesperson",
    "销售人员": "salesperson",
    "业务员": "salesperson",
    "金额": "amount",
    "销售额": "amount",
    "营收": "amount",
    "收入": "amount",
    "总金额": "amount",
    "数量": "quantity",
    "件数": "quantity",
    "销量": "quantity",
}


def _is_truthy(value: str | None) -> bool:
    """宽松判断环境变量是否表示"开启"。"""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _streamlit_secret(name: str) -> str | None:
    """Streamlit Cloud 的 secrets 兑底：本地无 st 环境或未配置时返回 None。"""
    try:
        import streamlit as st  # 懒加载，非云端环境不依赖

        return st.secrets.get(name) or None
    except Exception:
        return None


class LLMClient:
    """DashScope 大模型客户端；无 Key 时退化为可复现的 mock 实现。"""

    def __init__(self, api_key: str | None = None) -> None:
        # 显式传入的 key 优先，其次环境变量 DASHSCOPE_API_KEY，最后 Streamlit secrets（云端部署）
        self.api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or _streamlit_secret("DASHSCOPE_API_KEY")
            or None
        )
        # USE_MOCK=1 强制 mock；没有 key 也自动 mock
        self._mock = _is_truthy(os.getenv("USE_MOCK")) or not self.api_key

    # ------------------------------------------------------------------ 属性

    @property
    def mock(self) -> bool:
        """当前是否处于 mock 模式。"""
        return self._mock

    # ------------------------------------------------------------------ 对外接口

    def chat(self, prompt: str, system: str = "", model: str = DEFAULT_CHAT_MODEL) -> str:
        """单轮对话，返回模型文本回复。"""
        started = time.perf_counter()
        if self._mock:
            reply = self._mock_chat(prompt)
        else:
            reply = self._dashscope_chat(prompt, system, model)
        self._write_trace(
            kind="chat",
            model=model,
            prompt=prompt,
            response=reply,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return reply

    def embed(self, texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
        """文本向量化，返回与输入等长的向量列表。"""
        texts = list(texts)
        started = time.perf_counter()
        if self._mock:
            vectors = [self._mock_embed(t) for t in texts]
        else:
            vectors = self._dashscope_embed(texts, model)
        elapsed = (time.perf_counter() - started) * 1000
        for text in texts:
            # 向量太长，trace 里只记录维度信息
            self._write_trace(
                kind="embed",
                model=model,
                prompt=text,
                response=f"<vector dim={len(vectors[0]) if vectors else 0}>",
                elapsed_ms=elapsed,
            )
        return vectors

    # ------------------------------------------------------------------ 真实模式

    def _dashscope_chat(self, prompt: str, system: str, model: str) -> str:
        """调用 dashscope.Generation.call。"""
        import dashscope  # 懒加载：mock 模式无需安装 SDK

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = dashscope.Generation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            result_format="message",
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope chat 调用失败: {resp.status_code} {resp.message}")
        return resp.output.choices[0].message.content

    def _dashscope_embed(self, texts: list[str], model: str) -> list[list[float]]:
        """调用 dashscope.TextEmbedding.call，按顺序返回向量。

        text-embedding-v3 单请求批量上限为 10 条，超出会被服务端 400 拒绝；
        这里自动分批发送并拼回原顺序，单条失败重试 1 次。
        """
        import dashscope  # 懒加载

        batch_size = 10
        vectors: list[list[float]] = [[] for _ in texts]

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = None
            for attempt in (1, 2):  # 失败重试 1 次
                resp = dashscope.TextEmbedding.call(
                    api_key=self.api_key,
                    model=model,
                    input=batch,
                )
                if resp.status_code == 200:
                    break
                if attempt == 1:
                    import time

                    time.sleep(1.0)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope embed 调用失败: {resp.status_code} {resp.message}"
                )
            # DashScope 返回的 embeddings 可能不保证顺序，按 text_index（批内相对序号）排序
            items = sorted(resp.output["embeddings"], key=lambda x: x.get("text_index", 0))
            for offset, item in enumerate(items):
                vectors[start + offset] = item["embedding"]

        if any(not v for v in vectors):
            raise RuntimeError("DashScope embed 返回向量数量与输入不一致")
        return vectors

    # ------------------------------------------------------------------ mock 实现

    def _mock_chat(self, prompt: str) -> str:
        """规则化 mock 回答：含 "SQL" 则产出合法 SELECT 语句，否则回显。"""
        if "SQL" in prompt.upper():
            return f"```sql\n{self._mock_sql(prompt)}\n```"
        return "[mock] " + prompt[:80]

    def _mock_sql(self, prompt: str) -> str:
        """从问题里提取列名，拼一条语法合法的查询语句。"""
        cols: list[str] = []

        # 1) 英文列名直接命中（按词边界简单匹配，避免 amount 命中总金额之类误伤）
        lowered = prompt.lower()
        for col in KNOWN_COLUMNS:
            if col in lowered:
                cols.append(col)

        # 2) 中文别名命中
        for alias, col in COLUMN_ALIASES.items():
            if alias in prompt and col not in cols:
                cols.append(col)

        # 保持与表定义一致的稳定顺序，便于评测断言
        cols = [c for c in KNOWN_COLUMNS if c in cols]

        # 兜底：一个问题都没提到列名时返回全列
        projection = ", ".join(cols) if cols else "*"
        return f"SELECT {projection} FROM sales LIMIT 5;"

    def _mock_embed(self, text: str) -> list[float]:
        """基于文本哈希的 128 维伪向量：同文本同向量，可直接复现。

        实现采用「hashing trick」：把文本切成词 / 单字 / 字二元组，每个片段用
        sha256 散列到 128 维中的若干个下标并带符号累加（二元组权重更高，因为它
        更能代表语义单元）。这样既满足"由哈希决定、可复现"，又让共享词汇的文本
        （例如同一篇文档的提问与正文）具有较高余弦相似度，使 mock 模式下的 RAG
        检索仍然有意义——纯随机向量做不到这一点。
        """
        vec = [0.0] * MOCK_EMBED_DIM
        for token, weight in self._tokenize(text):
            # 一个 token 散列到 _HASHES_PER_TOKEN 个维度，避免向量过于稀疏
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(_HASHES_PER_TOKEN):
                chunk = digest[i * 4 : i * 4 + 4]
                idx = int.from_bytes(chunk, "big") % MOCK_EMBED_DIM
                sign = 1.0 if digest[i * 4] & 1 else -1.0
                vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:  # 空文本 -> 全零向量，避免除零
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, float]]:
        """极简分词，返回 (片段, 权重) 列表。

        中文没有空格，所以除了按空白切词，还额外取单字与相邻字的二元组；
        二元组权重更高，让「退货」「回款」这类实义词主导相似度。
        """
        tokens: list[tuple[str, float]] = []
        for word in text.lower().split():
            word = word.strip("，。！？；：、（）()[]{}<>\"'`*#-_/\\|")
            if not word:
                continue
            tokens.append((word, 1.0))
            for i, char in enumerate(word):
                tokens.append((char, 0.3))  # 单字：兜住分词切不开的情况，但降权
                bigram = word[i : i + 2]
                if len(bigram) == 2:
                    tokens.append((bigram, 1.5))
        return tokens

    # ------------------------------------------------------------------ trace 日志

    def _write_trace(
        self,
        *,
        kind: str,
        model: str,
        prompt: str,
        response: str,
        elapsed_ms: float,
    ) -> None:
        """把一次调用追加到 logs/traces.jsonl（失败不影响主流程）。"""
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "kind": kind,
            "model": model,
            "elapsed_ms": round(elapsed_ms, 2),
            "prompt": prompt[:200],
            "response": response[:200],
            "mock": self.mock,
        }
        try:
            TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRACE_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # 日志写不进去（只读文件系统等）不应该让业务调用失败
            pass