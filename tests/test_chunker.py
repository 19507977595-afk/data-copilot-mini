"""app/kb/chunker.py 的单元测试：分块结构、大小约束与边界条件。

全部在纯文本层面运行，不触网、不引入 LLMClient。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.kb.chunker import Chunk, MAX_CHUNK_SIZE, chunk_document, chunk_files


ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ 基本分块


def test_empty_text_returns_empty() -> None:
    """空文档不应产生任何块。"""
    chunks = chunk_document("")
    assert chunks == []


def test_whitespace_only_returns_empty() -> None:
    """仅空白字符的文档也不应产生块。"""
    chunks = chunk_document("   \n\n  ")
    assert chunks == []


def test_single_section_splits_on_blank_lines() -> None:
    """一个二级标题 + 多个段落 → 按段落累积切分。"""
    para = "这是一段话。" * 30  # ~150 字
    text = "# 标题\n## 小节\n\n" + para * 6  # ~900 字 → 应切成多块
    chunks = chunk_document(text)
    headings = {c.heading for c in chunks}
    assert "小节" in headings
    assert len(chunks) >= 2


def test_no_h2_heading_treated_as_root() -> None:
    """没有 ## 标题时，整篇作为一块（如果不超过 MAX_CHUNK_SIZE）。"""
    text = "这是一段没有标题的文字。\n" * 5
    chunks = chunk_document(text)
    assert all(c.heading == "" for c in chunks)
    assert len(chunks) >= 1


def test_multiple_sections_have_correct_headings() -> None:
    """多节文档：每个块的 heading 应匹配其所属 section。"""
    text = (
        "## 第一节\n\n"
        "第一节的文字。" * 20 + "\n\n"
        "## 第二节\n\n"
        "第二节的文字。" * 20
    )
    chunks = chunk_document(text)
    headings = {c.heading for c in chunks}
    assert "第一节" in headings
    assert "第二节" in headings
    assert headings <= {"第一节", "第二节"}


# ----------------------------------------------------------- 块大小约束


def test_chunk_size_within_bounds() -> None:
    """每块的长度应在 MAX_CHUNK_SIZE 以内。"""
    long_text = "重复段落。" * 200  # ~800 字 → 应被滑动窗口切分
    chunks = chunk_document(long_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_SIZE, f"块过长({len(c.text)}): ...{c.text[-30:]}"


def test_chunk_overlap_proportional() -> None:
    """超长文本经滑动窗口切分后，相邻块应有内容重叠。"""
    long = "ABCD" * 200  # 800 字，无换行符，走滑动窗口路径
    chunks = chunk_document(long)
    if len(chunks) >= 2:
        overlap_start = MAX_CHUNK_SIZE - 50
        overlap = chunks[0].text[overlap_start:MAX_CHUNK_SIZE]
        assert overlap == chunks[1].text[:50], "滑动窗口重叠部分不一致"


# ----------------------------------------------------------- 文件批量分块


def test_chunk_files_reads_from_disk() -> None:
    """用真实的 kb 文件验证：至少返回若干块，且每块带 source。"""
    paths = ["data/kb/退货政策.md", "data/kb/产品FAQ.md"]
    chunks = chunk_files(paths)
    assert len(chunks) >= 2
    sources = {c.source for c in chunks}
    assert any("退货政策" in s for s in sources)
    assert any("产品FAQ" in s for s in sources)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_files_with_absolute_path(tmp_path: Path) -> None:
    """用临时文件验证绝对路径也能正确分块。"""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "# 测试\n\n## 章节 A\n\n这是章节A的内容。" * 20 + "\n\n"
        "## 章节 B\n\n这是章节B的内容。" * 20,
        encoding="utf-8",
    )
    chunks = chunk_files([str(md_file)])
    assert len(chunks) >= 2
    headings = {c.heading for c in chunks}
    assert "章节 A" in headings
    assert "章节 B" in headings


def test_chunk_files_empty_list() -> None:
    chunks = chunk_files([])
    assert chunks == []


# ----------------------------------------------------------- Chunk dataclass


def test_chunk_ordering() -> None:
    """Chunk 可排序：同源文件按 idx 升序。"""
    c1 = Chunk(text="a", source="f", heading="", idx=0)
    c2 = Chunk(text="b", source="f", heading="", idx=1)
    assert c1 < c2
