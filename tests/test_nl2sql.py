"""app/analysis/nl2sql.py 的单元测试。

重点锁定三条安全红线（只 SELECT / 防堆叠 / 表名限定）与「一次失败重试」
的行为，以及 mock 模式下整链路（LLM 生成 -> 校验 -> DuckDB 执行）可复现。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.analysis.nl2sql import (
    SQLValidationError,
    ask_nl2sql,
    build_prompt,
    extract_sql,
    run_sql,
    validate_sql,
)

ROOT = Path(__file__).resolve().parents[1]
SALES_CSV = ROOT / "data" / "sales.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(SALES_CSV)


# ----------------------------------------------------------- 提示词与提取


def test_build_prompt_contains_sql_keyword() -> None:
    """prompt 里必须带 "SQL" 字样，mock 客户端才会走生成 SQL 的规则。"""
    assert "SQL" in build_prompt("查一下金额")


def test_build_prompt_retry_mentions_error() -> None:
    prompt = build_prompt("查一下金额", error="表 'users' 未注册")
    assert "上一次生成的 SQL 执行失败" in prompt
    assert "未注册" in prompt


def test_extract_sql_from_fence() -> None:
    reply = "好的，这是 SQL：\n```sql\nSELECT region, amount FROM sales LIMIT 5;\n```"
    assert extract_sql(reply) == "SELECT region, amount FROM sales LIMIT 5"


def test_extract_sql_bare_statement() -> None:
    # 没包代码块的裸语句也可以提取，去掉结尾分号
    assert extract_sql("select region from sales limit 5;") == "select region from sales limit 5"


def test_extract_sql_with_prose_prefix() -> None:
    reply = "你要的查询是这样的：SELECT region FROM sales LIMIT 3; 这就是结果"
    assert extract_sql(reply) == "SELECT region FROM sales LIMIT 3"


def test_extract_sql_returns_none_for_garbage() -> None:
    assert extract_sql("[mock] 抱歉，我不知道这个问题的答案") is None
    assert extract_sql("") is None
    assert extract_sql(None) is None


# ----------------------------------------------------------- 安全校验


def test_validate_accepts_normal_select() -> None:
    sql = validate_sql("SELECT region, amount FROM sales LIMIT 5;")
    assert sql == "SELECT region, amount FROM sales LIMIT 5;"


def test_validate_accepts_lowercase_and_qualified() -> None:
    for sql in (
        "select region from sales limit 5",
        'SELECT * FROM "sales" LIMIT 1',
        "SELECT * FROM main.sales LIMIT 1",
        "SELECT s.region FROM sales AS s LIMIT 3",
    ):
        validated = validate_sql(sql)
        assert validated.endswith(";")


def test_validate_rejects_non_select() -> None:
    with pytest.raises(SQLValidationError, match="只允许 SELECT"):
        validate_sql("DELETE FROM sales")


def test_validate_rejects_stacked_statements() -> None:
    with pytest.raises(SQLValidationError, match="堆叠"):
        validate_sql("SELECT * FROM sales; DROP TABLE sales;")


def test_validate_rejects_comments() -> None:
    with pytest.raises(SQLValidationError, match="注释"):
        validate_sql("SELECT * FROM sales -- 注释")


def test_validate_rejects_unknown_table() -> None:
    with pytest.raises(SQLValidationError, match="未注册"):
        validate_sql("SELECT * FROM users WHERE amount > 100")


def test_validate_rejects_forbidden_keyword() -> None:
    with pytest.raises(SQLValidationError, match="禁止"):
        validate_sql("SELECT * FROM sales WHERE amount > 0 CREATE TABLE hack")


def test_validate_empty_sql_rejected() -> None:
    with pytest.raises(SQLValidationError, match="空"):
        validate_sql("   ")


# ----------------------------------------------------------- DuckDB 执行


def test_run_sql_returns_dataframe(df: pd.DataFrame) -> None:
    out = run_sql(df, "SELECT region, amount FROM sales LIMIT 5;")
    assert out.shape == (5, 2)
    assert list(out.columns) == ["region", "amount"]


def test_run_sql_invalid_statement_raises(df: pd.DataFrame) -> None:
    from app.analysis.nl2sql import SQLExecutionError

    with pytest.raises(SQLExecutionError):
        run_sql(df, "SELECT nonexistent_col FROM sales")


# ----------------------------------------------------------- 主入口（mock 全链路）


def test_ask_nl2sql_mock_basic(df: pd.DataFrame) -> None:
    """mock 模式下问题命中中文别名，生成稳定 SQL 并执行成功。"""
    result = ask_nl2sql("查看前5条销售记录的销售地区和金额。", df=df)
    assert result.success
    assert result.attempts == 1
    assert result.sql == "SELECT region, amount FROM sales LIMIT 5;"
    assert result.df is not None and result.df.shape == (5, 2)
    assert list(result.df.columns) == ["region", "amount"]


def test_ask_nl2sql_accepts_csv_path() -> None:
    result = ask_nl2sql("列出前5条订单的日期和地区。", csv_path=SALES_CSV)
    assert result.success
    assert list(result.df.columns) == ["date", "region"]


def test_ask_nl2sql_requires_input() -> None:
    with pytest.raises(ValueError, match="df 或 csv_path"):
        ask_nl2sql("随便一个问题")


# ----------------------------------------------------------- 一次失败重试


class _FakeClient:
    """按队列回放回答的假客户端：用来注入失败、验证重试路径。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []  # (system, prompt)

    @property
    def mock(self) -> bool:
        return True

    def chat(self, prompt: str, system: str = "", model: str = "qwen-plus") -> str:
        self.calls.append((system, prompt))
        return self._replies.pop(0)


def test_retry_after_no_sql(df: pd.DataFrame) -> None:
    """第一次回答里没有 SQL（mock 没命中规则），重试后拿到合法 SQL。"""
    client = _FakeClient(
        [
            "[mock] 抱歉，我不太确定",
            "```sql\nSELECT region, amount FROM sales LIMIT 5;\n```",
        ]
    )
    result = ask_nl2sql("查看前5条销售记录的销售地区和金额。", df=df, client=client)
    assert result.success
    assert result.attempts == 2
    assert "上一次生成的 SQL 执行失败" in client.calls[1][1]  # 报错回喂 LLM


def test_retry_after_invalid_table(df: pd.DataFrame) -> None:
    """第一次生成了未注册表的 SQL，校验失败回喂报错后重试成功。"""
    client = _FakeClient(
        [
            "```sql\nSELECT * FROM users;\n```",
            "```sql\nSELECT region, amount FROM sales LIMIT 5;\n```",
        ]
    )
    result = ask_nl2sql("查看前5条销售记录的销售地区和金额。", df=df, client=client)
    assert result.success
    assert result.attempts == 2
    assert "未注册" in client.calls[1][1]


def test_retry_exhausted_records_error(df: pd.DataFrame) -> None:
    client = _FakeClient(["[mock] 抱歉", "[mock] 还是不行"])
    result = ask_nl2sql("某问题", df=df, client=client)
    assert not result.success
    assert result.df is None
    assert result.attempts == 2  # retries=1 -> 最多尝试 2 次
    assert result.error and "未能从回答中提取到 SQL" in result.error