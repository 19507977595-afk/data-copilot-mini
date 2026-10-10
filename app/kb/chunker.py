"""文档分块（Chunker）——按标题 / 空行切段，目标 200-500 字，重叠 50 字。

设计规范：
- 先按 Markdown 标题（`## xxx`）把文档拆成多个 section；
- 若某个 section 不足 200 字则合并到上一块，超过 500 字则按空行进一步拆分；
- 对超长块（>500 字）使用滑动窗口（步长 = 500-50 = 450 字）切成多段；
- 每一块记录 `source`（来源文件名）和 `heading`（所属标题）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---- 常量 ----

#: 单块最大字数（超出后按滑动窗口切分）
MAX_CHUNK_SIZE = 500

#: 滑动窗口重叠字数
CHUNK_OVERLAP = 50


@dataclass(order=True)
class Chunk:
    """一个文本块。

    Attributes:
        text:       块内纯文本。
        source:     来源文件相对路径。
        heading:    所属的二级标题（如 "退货期限"），无标题则为 ""。
        idx:        在同源文件中的序号，用于排序与比较。
    """

    text: str
    source: str
    heading: str = ""
    idx: int = field(default=0, compare=True)


def _read_file(path: Path) -> str:
    """读文件，相对路径转为基于项目根部的绝对路径。"""
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 核心 API


def chunk_text(text: str, *, source: str = "", heading: str = "") -> list[Chunk]:
    """将一段纯文本切分成小块。

    如果文本长度 ≤ MAX_CHUNK_SIZE 直接返回一块；否则以 MAX_CHUNK_SIZE
    为窗口、重叠 CHUNK_OVERLAP 字做滑动切分。每块带上传入的 source / heading。
    """
    cleaned = _normalize(text)
    if len(cleaned) <= MAX_CHUNK_SIZE:
        return [Chunk(text=cleaned, source=source, heading=heading, idx=0)]
    chunks: list[Chunk] = []
    step = MAX_CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(cleaned), step)):
        window = cleaned[start : start + MAX_CHUNK_SIZE]
        chunks.append(Chunk(text=window, source=source, heading=heading, idx=i))
    return chunks


def chunk_document(text: str, *, source: str = "") -> list[Chunk]:
    """按标题 / 空行将一篇 Markdown 文档分块。

    流程：
    1. 用正则按 `## .*?` 切分出若干 section；
    2. 每个 section 按空行（连续 ≥2 个换行）拆成段落；
    3. 逐段累积，累计到 > MAX_CHUNK_SIZE 时触发滑动切分；
    4. 末尾剩余段落 flush 出去。

    超长段落（> MAX_CHUNK_SIZE）独立走滑动窗口切分，不阻塞后续段落处理。
    """
    if not text.strip():
        return []

    sections = _split_into_sections(text)
    chunks: list[Chunk] = []
    for heading, body in sections:
        if not body.strip():
            continue
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        accumulator: list[str] = []
        acc_len = 0
        for para in paragraphs:
            para_len = len(para)
            # 单独一段就超长 → 立即独立切分，不等待积累
            if para_len > MAX_CHUNK_SIZE:
                if accumulator:
                    merged = "\n\n".join(accumulator)
                    chunks.extend(chunk_text(merged, source=source, heading=heading))
                    accumulator.clear()
                    acc_len = 0
                chunks.extend(chunk_text(para, source=source, heading=heading))
                continue
            # 加上这一段是否超限
            if acc_len + para_len > MAX_CHUNK_SIZE and accumulator:
                merged = "\n\n".join(accumulator)
                chunks.extend(chunk_text(merged, source=source, heading=heading))
                accumulator = [para]
                acc_len = para_len
            else:
                accumulator.append(para)
                acc_len += para_len
        # flush 尾部 accumulation
        if accumulator:
            merged = "\n\n".join(accumulator)
            chunks.extend(chunk_text(merged, source=source, heading=heading))
    return chunks


def chunk_files(paths: Iterable[str | Path], base: str | Path | None = None) -> list[Chunk]:
    """批量分块多个文档文件。

    Args:
        paths: 文件路径（可以是相对于项目根的相对路径或绝对路径）。
        base:  当路径是相对路径时的基准目录（缺省为项目根）。

    Returns:
        所有块的列表，按文件顺序、块内顺序排列。
    """
    all_chunks: list[Chunk] = []
    root = Path(__file__).resolve().parents[2] if base is None else Path(base)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        # 跨盘路径无法计算 relative_to，退化为文件名
        if base is None:
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
        else:
            rel = str(path)
        text = _read_file(path)
        all_chunks.extend(chunk_document(text, source=rel))
    return all_chunks


# ------------------------------------------------------------------ 内部工具


def _normalize(text: str) -> str:
    """清理空白：压缩连续空白为单空格，去首尾空白。"""
    return re.sub(r"\s+", " ", text).strip()


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """将 Markdown 文本按 `## heading` 切分为 (heading, body) 对。

    如果全文没有任何二级标题，返回 `[("", full_text)]`。
    """
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections
