"""自然语言 → SQL → DuckDB 执行（NL2SQL）。

流程：LLM 生成 SQL → 安全校验（只允许 SELECT、表名限定）→ DuckDB 对注册表执行
→ 返回 DataFrame。执行失败时把报错回喂给 LLM 重试 1 次，重试后仍失败则只报错不抛异常。

安全是底线：本模块不执行任何非 SELECT 语句；校验逻辑在 `validate_sql` 集中管理，
并通过单元测试锁定（防堆叠语句、防未注册表、防注释混淆、防 DDL/DML）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.llm.client import KNOWN_COLUMNS, LLMClient

#: 默认注册进 DuckDB 的表名（数据文件 sales.csv -> 表 sales）
DEFAULT_TABLE = "sales"

#: 明确禁止出现在语句中的关键词（词边界匹配，避免误伤普通列名）
FORBIDDEN_KEYWORDS = (
    "delete",
    "insert",
    "update",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "copy",
    "import",
    "export",
    "pragma",
    "call",
    "load",
    "install",
)

#: 系统提示词：把表结构讲清楚（真实模式才真正被模型看到；mock 只看 user 消息）
SYSTEM_PROMPT = (
    "你是数据查询助手，把用户的中文问题翻译成对 sales 表（销售明细）的 DuckDB SQL 查询。\n"
    "表结构：\n"
    f"- {KNOWN_COLUMNS[0]} TEXT 销售日期\n"
    f"- {KNOWN_COLUMNS[1]} TEXT 地区\n"
    f"- {KNOWN_COLUMNS[2]} TEXT 品类\n"
    f"- {KNOWN_COLUMNS[3]} TEXT 销售员\n"
    f"- {KNOWN_COLUMNS[4]} DOUBLE 销售金额\n"
    f"- {KNOWN_COLUMNS[5]} INTEGER 销售数量\n"
    "硬性要求：只允许 SELECT 查询；FROM 的表名必须是 sales；"
    "不允许注释 / 堆叠语句 / 任何修改数据的语句；不要臆造表中不存在的列。"
)

#: 代码块 / 内嵌 SQL 的提取正则
_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


class SQLValidationError(ValueError):
    """SQL 未通过安全校验时抛出，message 说明具体原因。"""


class SQLExecutionError(RuntimeError):
    """DuckDB 执行失败时抛出（把底层报错包装成可回喂给 LLM 的消息）。"""


@dataclass
class NL2SQLResult:
    """一次 NL2SQL 查询的结果快照（含失败时的报错，便于上层展示）。"""

    question: str = ""
    #: 最终采用（或最后一次尝试）的 SQL；生成失败时为 None
    sql: str | None = None
    #: 执行结果；失败时为 None
    df: pd.DataFrame | None = field(default=None)
    #: 最后一次报错（重试后仍失败才非空）
    error: str | None = None
    #: 实际尝试次数（1 = 一次成功；2 = 首次失败重试过一次）
    attempts: int = 0
    #: 用例编号（评测时填充，页面调用可留空）
    case_id: str | None = None

    @property
    def success(self) -> bool:
        """是否拿到了可用的执行结果。"""
        return self.error is None and self.df is not None


# ------------------------------------------------------------------ 提取与校验


def extract_sql(text: str | None) -> str | None:
    """从 LLM 回答中提取 SQL 语句。

    优先取 ```sql 代码块；没有代码块（或块内不是 SQL）时，从整段文本里
    搜第一条以 SELECT 开头的语句，截到第一个分号。最终结果必须是以
    SELECT 开头的语句，否则返回 None（触发重试）。
    """
    if not text:
        return None
    block = _SQL_BLOCK_RE.search(text)
    candidate = block.group(1).strip() if block else ""

    if not candidate.lower().startswith("select"):
        inner = re.search(r"\bselect\b[\s\S]*?(?:;|$)", text, re.I)
        candidate = inner.group(0).strip() if inner else ""

    candidate = candidate.strip().rstrip(";").strip()
    if not candidate.lower().startswith("select"):
        return None
    return candidate


def validate_sql(sql: str, registered_tables: Iterable[str] = (DEFAULT_TABLE,)) -> str:
    """安全校验并规整一条 SQL，只允许单条 SELECT。

    Args:
        sql: 待校验语句（可带结尾分号）。
        registered_tables: 允许被查询的表名集合（大小写敏感比较）。

    Returns:
        规整后的 SQL（统一去掉首尾空白、补上结尾分号）。

    Raises:
        SQLValidationError: 违反任意安全规则，message 说明原因。
    """
    if not sql or not sql.strip():
        raise SQLValidationError("SQL 为空，拒绝执行")

    statement = sql.strip()
    lower = statement.lower()

    # 1) 只允许 SELECT 开头
    if not lower.startswith("select"):
        raise SQLValidationError(
            f"只允许 SELECT 查询，实际以 {statement[:24]!r} 开头"
        )

    # 2) 去掉结尾分号后不允许再出现分号（防 `SELECT ...; DROP ...` 堆叠语句）
    body = statement.rstrip().rstrip(";")
    if ";" in body:
        raise SQLValidationError("语句中存在多余分号，疑似多条语句堆叠，已拒绝")

    # 3) 禁止注释（防把危险语句藏进注释绕过词法检查）
    if "--" in body or "/*" in body or "#" in body:
        raise SQLValidationError("不允许使用 SQL 注释")

    # 4) 禁止 DDL / DML / 其它危险关键字
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lower):
            raise SQLValidationError(f"检测到被禁止的关键字 {keyword!r}（只允许 SELECT）")

    # 5) FROM / JOIN 后第一个标识符必须是已注册的表（允许 sales / "sales" / main.sales）
    tables = {t.lower() for t in registered_tables}
    for match in re.finditer(r"\b(?:from|join)\s+([^\s,;()]+)", body.lower()):
        ident = match.group(1).strip('"`[]').split(".")[-1]
        if ident.startswith("("):  # 子查询之类的复杂来源，一律不放开
            raise SQLValidationError("不支持 FROM/JOIN 子查询或表达式，表名必须是普通表")
        if ident not in tables:
            raise SQLValidationError(
                f"表 {ident!r} 未注册，禁止查询（可用表：{sorted(tables)}）"
            )

    return body + ";"


# ------------------------------------------------------------------ 执行


def run_sql(df: pd.DataFrame, sql: str, *, table_name: str = DEFAULT_TABLE) -> pd.DataFrame:
    """把 DataFrame 注册为指定表，用 DuckDB 执行 SQL 并返回结果 DataFrame。

    Raises:
        SQLExecutionError: DuckDB 执行失败（消息可直接回喂给 LLM 让它修正）。
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.register(table_name, df)
        return con.execute(sql).fetchdf()
    except Exception as exc:  # @except 只包住 DuckDB 的调用，避免吞掉自身编程错误
        raise SQLExecutionError(f"DuckDB 执行失败：{exc}") from exc
    finally:
        con.close()


# ------------------------------------------------------------------ 提示词与主入口


def build_prompt(question: str, error: str | None = None) -> str:
    """构造 user 消息。含 "SQL" 字样使 mock 客户端命中规则；带 error 即重试轮。"""
    prefix = (
        "请把下面的中文问题翻译成一条合法的 SQL 查询（只允许 SELECT，只查 sales 表，不要注释）。"
    )
    if error:
        prefix += (
            "\n上一次生成的 SQL 执行失败，报错如下，请修正这段 SQL 后重新输出：\n"
            f"{error}"
        )
    return f"{prefix}\n问题：{question}\nSQL："


def ask_nl2sql(
    question: str,
    *,
    df: pd.DataFrame | None = None,
    csv_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
    client: LLMClient | None = None,
    retries: int = 1,
) -> NL2SQLResult:
    """把自然语言问题转成 SQL 并在 DuckDB 上执行，失败时回喂报错重试。

    Args:
        question: 中文自然语言问题。
        df: 待查询的数据（注册为 table_name）。与 csv_path 二选一。
        csv_path: CSV 路径，由本函数读成 DataFrame。
        table_name: 注册进 DuckDB 的表名（默认 sales）。
        client: LLM 客户端；缺省新建（遵循 USE_MOCK 环境）。
        retries: 失败后最多重试次数（SPEC 约定为 1，即总共最多 2 次尝试）。

    Returns:
        NL2SQLResult：成功时 df/sql 就位；失败时填 error，不抛异常。
    """
    if df is None:
        if csv_path is None:
            raise ValueError("必须提供 df 或 csv_path 之一")
        df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("待查询的数据为空，无法执行 SQL")

    client = client or LLMClient()
    result = NL2SQLResult(question=question)
    error: str | None = None

    for attempt in range(retries + 1):
        result.attempts = attempt + 1
        reply = client.chat(build_prompt(question, error=error), system=SYSTEM_PROMPT)
        generated = extract_sql(reply)

        if generated is None:
            error = f"未能从回答中提取到 SQL（回答开头：{reply[:80]!r}）"
            continue

        try:
            result.sql = validate_sql(generated, registered_tables=[table_name])
            result.df = run_sql(df, result.sql, table_name=table_name)
            result.error = None
            return result
        except (SQLValidationError, SQLExecutionError) as exc:
            error = f"{exc}"

    result.error = error  # 重试耗尽，保留最后一次报错
    return result