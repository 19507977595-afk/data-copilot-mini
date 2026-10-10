"""eval/harness.py 的单元测试（全程 mock，零触网）。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import eval.harness as harness
from eval.harness import (
    SuiteError,
    _col_match,
    _row_agreement,
    load_cases,
    run_nl2sql_suite,
    run_rag_suite,
    run_suite,
    write_report,
)


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    return pd.read_csv(harness.DEFAULT_DATA)


@pytest.fixture()
def mock_client():
    from app.llm.client import LLMClient

    return LLMClient()


# ------------------------------------------------------------------ 用例加载


def test_load_cases_nl2sql_all_valid():
    cases = load_cases("nl2sql")
    assert len(cases) >= 6
    for c in cases:
        assert {"id", "question", "gold_sql"} <= set(c)


def test_load_cases_rag_all_valid():
    cases = load_cases("rag")
    assert len(cases) >= 6
    for c in cases:
        assert {"id", "question", "expect_source_contains"} <= set(c)


def test_load_cases_missing_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "CASE_DIR", tmp_path)
    with pytest.raises(SuiteError):
        load_cases("nope")


def test_load_cases_bad_json(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "CASE_DIR", tmp_path)
    (tmp_path / "broken.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(SuiteError):
        load_cases("broken")


# ------------------------------------------------------------------ 指标函数


def test_col_match_perfect_and_empty():
    assert _col_match(["a", "b"], ["b", "a"]) == 1.0
    assert _col_match([], []) == 1.0
    assert _col_match(["a"], []) == 0.0


def test_col_match_jaccard_partial():
    # 交集 {a}，并集 {a,b,c} -> 1/3
    assert _col_match(["a", "b"], ["a", "c"]) == pytest.approx(1 / 3)


def test_row_agreement_none_inputs():
    df = pd.DataFrame({"a": [1]})
    assert _row_agreement(None, df) == 0.0
    assert _row_agreement(df, None) == 0.0


def test_row_agreement_shared_columns_only():
    pred = pd.DataFrame({"a": [1, 2], "x": [9, 9]})
    gold = pd.DataFrame({"a": [1, 2], "y": [8, 8]})
    assert _row_agreement(pred, gold) == 1.0  # 只比较共享列 a
    bad = pd.DataFrame({"b": [1, 2]})
    assert _row_agreement(pred, bad) == 0.0  # 无共享列


# ------------------------------------------------------------------ 套件执行


def test_nl2sql_suite_mock_passes(sales_df, mock_client):
    report = run_nl2sql_suite(mock_client, sales_df, load_cases("nl2sql"))
    assert report["suite"] == "nl2sql"
    assert report["summary"]["failed"] == 0
    assert report["metrics"]["exec_success_rate"] == 1.0


def test_rag_suite_mock_retrieval_hits(mock_client):
    report = run_rag_suite(mock_client, pd.DataFrame(), load_cases("rag"))
    assert report["suite"] == "rag"
    assert report["metrics"]["retrieval_hit_rate"] == 1.0
    for case in report["per_case"]:
        assert case["pass"] is True


def test_run_suite_registered():
    assert set(harness.SUITE_RUNNERS) == {"nl2sql", "rag"}


def test_run_all_skips_missing(monkeypatch, tmp_path, sales_df, mock_client):
    monkeypatch.setattr(harness, "CASE_DIR", tmp_path)
    # 两个套件用例文件都不存在 -> 全部 skip，不抛异常
    assert run_suite_missing_safe(mock_client, sales_df) == []


def run_suite_missing_safe(client, df):
    return harness.run_all(client, df)


# ------------------------------------------------------------------ 报告落盘


def test_write_report_json(tmp_path, sales_df, mock_client):
    monkeypatch_dir = tmp_path / "reports"
    original = harness.REPORT_DIR
    harness.REPORT_DIR = monkeypatch_dir
    try:
        report = run_nl2sql_suite(mock_client, sales_df, load_cases("nl2sql")[:2])
        path = write_report([report], "mock")
        assert path.exists() and path.suffix == ".json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["mode"] == "mock"
        assert payload["suites"][0]["suite"] == "nl2sql"
    finally:
        harness.REPORT_DIR = original


def test_main_cli_smoke(tmp_path, monkeypatch, capsys):
    original = harness.REPORT_DIR
    harness.REPORT_DIR = tmp_path / "reports"
    try:
        rc = harness.main(["run", "--suite", "nl2sql"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "评测摘要" in out
        assert "完整报告已写入" in out
    finally:
        harness.REPORT_DIR = original
