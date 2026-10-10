"""RAG 问答模块——检索增强生成。

流程：
1. ``rag.build(document_paths)``：读取文档 → chunker 分块 → vectorstore 构建向量库
2. ``rag.answer(question, k=4)``：用 question 做 top-k 检索 → 拼装上下文 prompt → chat → 返回带引用来源的回答

回答格式在内部组装；外部也可只调用 ``rag.retrieve(question, k)`` 获取原始检索结果。

prompt 模板放在模块常量 ``RETRIEVAL_PROMPT_TEMPLATE``，方便评测调整。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.kb.chunker import Chunk, chunk_files
from app.kb.vectorstore import VectorStore, build_store
from app.llm.client import LLMClient

# ---- 路径常量 ----

ROOT = Path(__file__).resolve().parents[2]

# ---- Prompt 模板（评测可调） ----

#: RAG 检索上下文 + 用户问题的拼接模板
RETRIEVAL_SYSTEM = """你是一个智能助手。请基于以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请坦诚告知"抱歉，参考资料中未找到相关信息"。
回答要求简洁、准确，并在末尾附上参考来源。"""

RETRIEVAL_USER_TEMPLATE = """参考资料：
{context}

---
用户问题：{question}

请基于以上资料回答问题，并在回答末尾列出参考来源（格式：[文件名]）。
"""


class RAG:
    """知识库 RAG 问答器。

    Attributes:
        store:      底层向量库实例（已持久化到 .cache/）。
        client:     LLM 客户端（chat 阶段使用）。
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self.store: VectorStore | None = None
        #: 是否已经 build（快速检查）
        self._built = False

    # --------------------------------------------------------------- build

    def build(self, document_paths: list[str | Path]) -> None:
        """对一组文档执行完整 pipeline：切块 → 向量化 → 持久化。

        Args:
            document_paths: 文档文件路径列表（相对于项目根或绝对路径）。
        """
        chunks = chunk_files(document_paths, base=str(ROOT))
        if not chunks:
            raise ValueError("分块结果为空，请检查文档路径")
        self.store = build_store(chunks, self.client)
        self._built = True

    # ----------------------------------------------------------------- query

    def retrieve(self, question: str, k: int = 4) -> list[tuple[float, Chunk]]:
        """纯检索：返回与问题最相似的 k 个文本块。

        Returns:
            ``[(score, Chunk), ...]``，按余弦相似度降序排列。
        """
        if not self._built or self.store is None:
            return []
        return self.store.query(question, k=k)

    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        """端到端 RAG 问答。

        返回字典包含：
        - ``answer``: 模型生成的回答文本
        - ``sources``: 参考来源列表 ``[{"text": ..., "source": ..., "heading": ...}, ...]``
        - ``chunks``: 检索到的全部 chunk 对象（可用于调试）

        Args:
            question: 用户问题。
            k: 检索数量。
        """
        results = self.retrieve(question, k=k)
        if not results:
            return {
                "answer": "抱歉，知识库为空，请先调用 build() 加载文档。",
                "sources": [],
                "chunks": [],
            }

        # 拼 context：每个 chunk 一行
        context_lines: list[str] = []
        sources: list[dict[str, str]] = []
        for score, chunk in results:
            line = f"[{chunk.source}] {chunk.heading}: {chunk.text}"
            context_lines.append(line)
            sources.append({
                "source": chunk.source,
                "heading": chunk.heading,
                "similarity": round(score, 4),
            })

        context = "\n\n".join(context_lines)
        prompt = RETRIEVAL_USER_TEMPLATE.format(context=context, question=question)

        reply = self.client.chat(
            prompt=prompt,
            system=RETRIEVAL_SYSTEM,
        )

        return {
            "answer": reply,
            "sources": sources,
            "chunks": [c for _, c in results],
        }


# ------------------------------------------------------------------ 便捷函数

def rag_answer(question: str, *, documents: list[str] | None = None, k: int = 4) -> dict[str, Any]:
    """一次性快捷回答（自动 build + answer）。

    适用于一次性查询场景。对于重复查询应优先使用 ``RAG`` 类直接操作 store。

    Args:
        question:   用户问题。
        documents:  文档路径列表；缺省使用 ``data/kb/*.md``。
        k:          检索数量。
    """
    kb_dir = ROOT / "data" / "kb"
    if documents is None:
        documents = sorted(str(p) for p in kb_dir.glob("*.md"))
    rag = RAG()
    rag.build(documents)
    return rag.answer(question, k=k)
