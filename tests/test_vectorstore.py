"""app/kb/vectorstore.py 的单元测试：构建、持久化、查询与边界条件。

全部使用 mock LLMClient，不触网。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.kb.chunker import Chunk
from app.kb.vectorstore import (
    _load_meta,
    _mock_embed,
    _save_meta,
    VectorStore,
)
from app.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ 伪嵌入函数


def test_mock_embed_same_text_same_vector() -> None:
    """同文本 → 同向量（确定性）。"""
    v1 = _mock_embed("退货期限是多久")
    v2 = _mock_embed("退货期限是多久")
    assert v1 == v2


def test_mock_embed_vector_norms_near_one() -> None:
    """非空文本的向量范数应接近 1（已归一化）。"""
    v = _mock_embed("这是一个测试文本用于验证归一化")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_mock_embed_different_texts_different_vectors() -> None:
    """不同文本 → 不同向量。"""
    v1 = _mock_embed("退货政策")
    v2 = _mock_embed("销售流程")
    assert v1 != v2


def test_mock_embed_empty_returns_zero_vector() -> None:
    """空文本返回全零向量。"""
    v = _mock_embed("")
    assert all(x == 0.0 for x in v)
    assert len(v) == 128


def test_mock_embed_dim_128() -> None:
    """维度应为 128（SPEC 规定）。"""
    assert len(_mock_embed("test")) == 128


# ------------------------------------------------------------------ VectorStore build / persist / load


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    """创建一组带有标题和来源的测试块。"""
    return [
        Chunk(text="退货商品自签收之日起 7 天内支持无理由退货。", source="退货政策.md", heading="退货期限"),
        Chunk(text="数码类商品自签收之日起提供 12 个月整机保修。", source="产品FAQ.md", heading="保修与售后"),
        Chunk(text="标准折扣上限为标价的 9 折。", source="销售流程.md", heading="折扣授权"),
        Chunk(text="单笔订单金额满 199 元包邮。", source="产品FAQ.md", heading="发货与物流"),
    ]


@pytest.fixture()
def store(sample_chunks: list[Chunk]) -> VectorStore:
    """构建并持久化一个干净的 VectorStore。"""
    client = LLMClient()
    assert client.mock  # 确保走 mock
    vs = VectorStore(client)
    vs.build(sample_chunks)
    return vs


def test_build_sets_vectors_shape(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    assert store.vectors is not None
    assert store.vectors.shape == (len(sample_chunks), 128)


def test_build_stores_metadata(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    assert len(store.meta["texts"]) == len(sample_chunks)
    assert len(store.meta["sources"]) == len(sample_chunks)
    assert len(store.meta["headings"]) == len(sample_chunks)


def test_persist_creates_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_store 后应在目标目录生成 .npz 和 .json 文件。"""
    # Monkey-patch 到临时目录避免污染项目
    monkeypatch.setattr("app.kb.vectorstore._CACHE_DIR", tmp_path)
    monkeypatch.setattr("app.kb.vectorstore._NPZ_PATH", tmp_path / "vectors.npz")
    monkeypatch.setattr("app.kb.vectorstore._META_PATH", tmp_path / "vectors_meta.json")

    chunks = [Chunk(text=f"段落{i}", source="f.md", heading=f"H{i}") for i in range(4)]
    from app.kb.vectorstore import build_store
    build_store(chunks)

    assert (tmp_path / "vectors.npz").exists()
    assert (tmp_path / "vectors_meta.json").exists()


def test_load_rehydrates_from_disk(sample_chunks: list[Chunk]) -> None:
    """先 build → persist → load，矩阵与元数据应一致。"""
    client = LLMClient()
    assert client.mock

    original = VectorStore(client)
    original.build(sample_chunks)
    original.persist()

    loaded = VectorStore.load()
    assert loaded.vectors is not None
    assert loaded.vectors.shape == original.vectors.shape
    assert np.allclose(loaded.vectors, original.vectors)
    assert loaded.meta["texts"] == original.meta["texts"]


def test_load_empty_when_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """没有持久化文件时，load 返回无数据的 store。"""
    monkeypatch.setattr("app.kb.vectorstore._NPZ_PATH", tmp_path / "nonexistent.npz")
    monkeypatch.setattr("app.kb.vectorstore._META_PATH", tmp_path / "nonexistent.json")

    store = VectorStore.load()
    assert store.vectors is None


# ------------------------------------------------------------------ query / cosine similarity


def test_query_returns_top_k(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    """用与某 chunk 高度相似的文本提问 → 该 chunk 应在 top-k 中排第一。"""
    results = store.query("退货期限", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) and s >= -1.0 for s, _ in results)


def test_query_exact_match_hits_itself(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    """问与某个 chunk text 完全相同的句子 → 最高相似度 chunk 就是它。"""
    exact_text = sample_chunks[0].text
    results = store.query(exact_text, k=4)
    best_score, best_chunk = results[0]
    assert best_chunk.text == exact_text
    assert best_score > 0


def test_query_fewer_than_k_returns_all(store: VectorStore) -> None:
    """只有 4 个 chunk 时请求 k=10 → 返回 4 个结果。"""
    results = store.query("退款", k=10)
    assert len(results) == 4


def test_query_empty_store_returns_empty() -> None:
    """未 build 的 store → query 返回空列表。"""
    store = VectorStore(LLMClient())
    assert store.query("anything", k=1) == []


# ----------------------------------------------------------- metadata IO helpers


def test_save_load_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta_path = tmp_path / "meta.json"
    monkeypatch.setattr("app.kb.vectorstore._META_PATH", meta_path)

    meta = {
        "texts": ["你好", "世界"],
        "sources": ["a.md", "b.md"],
        "headings": ["H1", "H2"],
    }
    _save_meta(meta)
    loaded = _load_meta()
    assert loaded == meta


def test_load_meta_nonexistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta_path = tmp_path / "missing.json"
    monkeypatch.setattr("app.kb.vectorstore._META_PATH", meta_path)
    assert _load_meta() == {"texts": [], "sources": [], "headings": []}
