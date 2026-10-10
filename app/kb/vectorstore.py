"""轻量向量库（VectorStore）——基于 numpy 的余弦相似度检索。

持久化：
- `.cache/vectors.npz`：矩阵文件，每行一个向量（numpy memmap 友好格式）
- `.cache/vectors_meta.json`：文本与元数据索引（text ↔ row_id）

接口：
- ``build(chunks)` / ``add(chunks)`：从 Chunk 对象向量化并写入磁盘
- ``query(text, k=4)``：返回 ``(distance, [Chunk])`` 前 k 个最相似块
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.kb.chunker import Chunk
from app.llm.client import LLMClient

# ---- 路径常量 ----

ROOT = Path(__file__).resolve().parents[2]  # 项目根
_CACHE_DIR = ROOT / ".cache"
_NPZ_PATH = _CACHE_DIR / "vectors.npz"
_META_PATH = _CACHE_DIR / "vectors_meta.json"

# ---- mock embedder（本地计算，避免引入 LLMClient 循环依赖） ----


def _mock_embed(text: str, dim: int = 128) -> list[float]:
    """复现 LLMClient._mock_embed 的逻辑，保证向量库不触网。"""
    MOCK_EMBED_DIM = dim
    _HASHES_PER_TOKEN = 4

    def _tokenize(t: str):
        tokens: list[tuple[str, float]] = []
        for word in t.lower().split():
            word = word.strip("，。！？；：、（）()[]{}<>\"'`*#-_/\\|")
            if not word:
                continue
            tokens.append((word, 1.0))
            for i, char in enumerate(word):
                tokens.append((char, 0.3))
                bigram = word[i : i + 2]
                if len(bigram) == 2:
                    tokens.append((bigram, 1.5))
        return tokens

    vec = [0.0] * MOCK_EMBED_DIM
    for token, weight in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(_HASHES_PER_TOKEN):
            chunk = digest[i * 4 : i * 4 + 4]
            idx = int.from_bytes(chunk, "big") % MOCK_EMBED_DIM
            sign = 1.0 if digest[i * 4] & 1 else -1.0
            vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


# ------------------------------------------------------------------ 元数据 IO


def _load_meta() -> dict[str, Any]:
    """读取元数据 JSON：``{"texts": [...], "sources": [...], "headings": [...]}``。"""
    if _META_PATH.exists():
        raw = json.loads(_META_PATH.read_text(encoding="utf-8"))
        return {
            "texts": raw.get("texts", []),
            "sources": raw.get("sources", []),
            "headings": raw.get("headings", []),
        }
    return {"texts": [], "sources": [], "headings": []}


def _save_meta(meta: dict[str, Any]) -> None:
    """将元数据序列化到 JSON。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------ 核心类


class VectorStore:
    """基于 numpy 的轻量向量存储与余弦相似度检索。

    内部用两个并行数组维护：
    - ``self.vectors``：shape ``(n, dim)`` 的 ``float32`` 矩阵
    - ``self.meta``：dict of lists（texts / sources / headings）

    每次 ``build()`` 从头构建（覆盖）。提供 ``persist()``, ``load()``
    实现跨进程持久化。
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        #: shape (n, dim) 的矩阵
        self.vectors: np.ndarray | None = None
        #: metadata
        self.meta: dict[str, Any] = {"texts": [], "sources": [], "headings": []}
        #: chunks 缓存（在 query 时用于还原 Chunk 对象）
        self._chunks_cache: list[Chunk] | None = None

    # --------------------------------------------------------------- build / add

    def build(self, chunks: list[Chunk]) -> None:
        """一次性构建向量库：对所有 chunk 做 embed → 存矩阵。

        如果 LLMClient 处于真实模式，使用远程 API；否则走本地 mock 伪嵌入。
        构建完成后自动持久化。
        """
        texts = [c.text for c in chunks]
        sources = [c.source for c in chunks]
        headings = [c.heading for c in chunks]

        if self.client.mock:
            # mock 模式：本地逐条计算
            vectors_list = [_mock_embed(t) for t in texts]
        else:
            # 真实模式：批量调用 DashScope
            vectors_list = self.client.embed(texts)

        self.vectors = np.array(vectors_list, dtype=np.float32)
        self.meta = {"texts": texts, "sources": sources, "headings": headings}
        self._chunks_cache = list(chunks)
        self.persist()

    def persist(self) -> None:
        """将当前内存中的向量与元数据写回磁盘。"""
        if self.vectors is None or self.vectors.shape[0] == 0:
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_NPZ_PATH, vectors=self.vectors)
        _save_meta(self.meta)

    @classmethod
    def load(cls) -> "VectorStore":
        """从磁盘加载已持久化的向量库。"""
        store = cls()
        if not _NPZ_PATH.exists() or not _META_PATH.exists():
            return store
        data = np.load(str(_NPZ_PATH))
        store.vectors = data["vectors"]
        store.meta = _load_meta()
        store._chunks_cache = None  # rebuild on first query
        return store

    # ------------------------------------------------------------------- query

    def query(self, text: str, k: int = 4) -> list[tuple[float, Chunk]]:
        """检索与 ``text`` 最相似的 top-k 文本块。

        Returns:
            ``[(similarity, Chunk), ...]`` 按相似度降序排列。

        相似度为余弦相似度（归一化后的点积），范围 [-1, 1]。
        """
        # 确保有数据
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []

        # 生成查询向量
        if isinstance(self.client, LLMClient) and self.client.mock:
            q_vec = np.array(_mock_embed(text), dtype=np.float32)
        else:
            try:
                emb = self.client.embed([text])[0]
                q_vec = np.array(emb, dtype=np.float32)
            except Exception:  # noqa: BLE001 - 网络失败返回空结果
                return []

        # 余弦相似度（假设向量已归一化，直接点积）
        scores = self.vectors.dot(q_vec)

        # top-k 索引（降序）
        n = len(scores)
        indices = np.argsort(-scores)[:k]

        results: list[tuple[float, Chunk]] = []
        for idx in indices:
            score = float(scores[idx])
            # 重建 Chunk 对象
            if self._chunks_cache and idx < len(self._chunks_cache):
                chunk = self._chunks_cache[idx]
            else:
                chunk = Chunk(
                    text=self.meta["texts"][idx] if idx < len(self.meta["texts"]) else "",
                    source=self.meta["sources"][idx] if idx < len(self.meta["sources"]) else "",
                    heading=self.meta["headings"][idx] if idx < len(self.meta["headings"]) else "",
                )
            results.append((score, chunk))
        return results

    def rebuild_chunks_cache(self) -> None:
        """在 query 之前，从 meta 重新构建 chunks 缓存（防止跨进程失效）。"""
        if self._chunks_cache is not None:
            return
        self._chunks_cache = [
            Chunk(
                text=text,
                source=source,
                heading=heading,
            )
            for text, source, heading in zip(
                self.meta["texts"], self.meta["sources"], self.meta["headings"]
            )
        ]


# ------------------------------------------------------------------ 便捷函数

#: 单例式全局 store（按需 load/build），方便 rag.py 使用
_global_store: VectorStore | None = None


def get_store(client: LLMClient | None = None) -> VectorStore:
    """获取或创建全局向量库实例。"""
    global _global_store
    if _global_store is None or _global_store.vectors is None:
        _global_store = VectorStore.load()
        if _global_store.vectors is None:
            _global_store = VectorStore(client or LLMClient())
    return _global_store


def build_store(chunks: list[Chunk], client: LLMClient | None = None) -> VectorStore:
    """快捷构建并返回 store（build → persist → load）。"""
    store = VectorStore(client or LLMClient())
    store.build(chunks)
    return store
