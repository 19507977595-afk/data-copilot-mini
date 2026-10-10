"""AI Harness —— 评测运行器（M2 最小版：仅 NL2SQL 套件）。

用法：
    python -m eval.harness run --suite nl2sql
    python -m eval.harness run                # 默认 all，跑所有已有套件

每条用例独立跑：LLM 生成 SQL（mock 模式可复现）→ 安全校验 → DuckDB 执行；
与 gold_sql 的执行结果比较列匹配与行集一致，输出每条 PASS/FAIL 与聚合指标，
并落盘一份 JSON 报告（eval/reports/report-<ts>.json）。全程不触网。

RAG 套件在 M3 落地后自动接入（套件注册表 SUITES 里加一行即可）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.analysis.nl2sql import ask_nl2sql, run_sql
from app.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "eval" / "cases"
REPORT_DIR = ROOT / "eval" / "reports"
DEFAULT_DATA = ROOT / "data" / "sales.csv"
TABLE = "sales"


class SuiteError(RuntimeError):
    """某套件无法运行（如用例文件缺失）时抛出，交给 CLI 友好提示。"""


# ------------------------------------------------------------------ 用例加载


def load_cases(suite: str) -> list[dict[str, Any]]:
    """读取 eval/cases/<suite>.jsonl；文件缺失抛 SuiteError。"""
    path = CASE_DIR / f"{suite}.jsonl"
    if not path.exists():
        raise SuiteError(f"用例文件不存在：{path}")
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    if not cases:
        raise SuiteError(f"用例文件为空：{path}")
    return cases


# ------------------------------------------------------------------ 指标计算


def _col_match(pred_cols: list[str], expect_cols: list[str]) -> float:
    """结果列匹配率：两侧列名的 Jaccard 系数（0~1，完全一致为 1）。"""
    if not expect_cols:
        return 1.0 if not pred_cols else 0.0
    pred, expect = set(pred_cols), set(expect_cols)
    return len(pred & expect) / len(pred | expect)


def _row_agreement(pred: pd.DataFrame | None, gold: pd.DataFrame | None) -> float:
    """行集一致率：两个结果集在共享列上的行元组 Jaccard 系数。

    列不同（如 gold 有聚合列）时只比较两边都存在的列，保证指标可计算。
    """
    if pred is None or gold is None:
        return 0.0
    shared = [c for c in gold.columns if c in pred.columns]
    if not shared:
        return 0.0
    p = _row_keys(pred, shared)
    g = _row_keys(gold, shared)
    if not g:
        return 0.0
    return len(p & g) / len(p | g)


def _row_keys(df: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    """把若干列转成行的规范化字符串元组，用于集合比较。"""
    return {tuple("" if pd.isna(v) else str(v) for v in row) for row in df[columns].itertuples(index=False)}


# ------------------------------------------------------------------ NL2SQL 套件


def run_nl2sql_suite(client: LLMClient, df: pd.DataFrame, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """逐条跑 NL2SQL 用例，返回套件级结果（含每条用例明细）。"""
    per_case: list[dict[str, Any]] = []
    exec_ok = 0
    col_ratio = 0.0
    row_ratio = 0.0
    keyword_hit = 0

    for case in cases:
        case_id = case.get("id", "?")
        question = case["question"]
        gold_sql = case["gold_sql"]

        result = ask_nl2sql(question, df=df, client=client, retries=1)
        pred_cols = list(result.df.columns) if result.df is not None else []

        # gold 语句的执行结果：作为"标准答案"与本轮结果比较
        gold_df = None
        gold_error = None
        try:
            gold_df = run_sql(df, gold_sql, table_name=TABLE)
        except Exception as exc:  # noqa: BLE001 - gold 写错也要能出报告
            gold_error = f"{exc}"

        expect_cols = case.get("expect_columns") or (list(gold_df.columns) if gold_df is not None else [])
        col = _col_match(pred_cols, expect_cols)
        row = _row_agreement(result.df, gold_df)
        generated = result.sql or ""
        hit = all(k.lower() in generated.lower() for k in case.get("expect_sql_contains", []))
        passed = result.success and col == 1.0 and row == 1.0 and hit

        exec_ok += 1 if result.success else 0
        col_ratio += col
        row_ratio += row
        keyword_hit += 1 if hit else 0

        per_case.append(
            {
                "id": case_id,
                "question": question,
                "pass": bool(passed),
                "success": bool(result.success),
                "sql": generated,
                "pred_columns": pred_cols,
                "expect_columns": expect_cols,
                "column_match": col,
                "row_agreement": row,
                "keyword_hit": bool(hit),
                "gold_sql": gold_sql,
                "gold_error": gold_error,
                "error": result.error,
                "attempts": result.attempts,
            }
        )

    n = len(cases)
    return {
        "suite": "nl2sql",
        "n_cases": n,
        "per_case": per_case,
        "metrics": {
            "exec_success_rate": exec_ok / n,
            "column_match_rate": col_ratio / n,
            "row_agreement_rate": row_ratio / n,
            "keyword_hit_rate": keyword_hit / n,
        },
        "summary": {
            "total": n,
            "passed": sum(1 for c in per_case if c["pass"]),
            "failed": sum(1 for c in per_case if not c["pass"]),
        },
    }


# ------------------------------------------------------------------ 套件注册表


#: 套件名 -> 运行函数。M3 的 rag.jsonl 与 RAG 套件就绪后在此追加一行即可
SUITE_RUNNERS: dict[str, Any] = {
    "nl2sql": run_nl2sql_suite,
}


def run_suite(name: str, client: LLMClient, df: pd.DataFrame) -> dict[str, Any]:
    """运行单个套件（含用例加载）。"""
    runner = SUITE_RUNNERS[name]
    return runner(client, df, load_cases(name))


def run_all(client: LLMClient, df: pd.DataFrame) -> list[dict[str, Any]]:
    """跑所有已就绪的套件；用例文件缺失的套件跳过并提示。"""
    output = []
    for name in SUITE_RUNNERS:
        try:
            output.append(run_suite(name, client, df))
        except SuiteError as exc:
            print(f"[skip] 套件 {name}：{exc}（该套件用例后续里程碑落地后可跑）")
    return output


# ------------------------------------------------------------------ 报告输出


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_markdown(suite_report: dict[str, Any], mode: str) -> None:
    """在控制台打印 markdown 格式的评测摘要。"""
    metrics = suite_report["metrics"]
    summary = suite_report["summary"]
    print(f"### NL2SQL 评测摘要（{mode} 模式，共 {summary['total']} 条用例）")
    print(f"- PASS {summary['passed']} / FAIL {summary['failed']}")
    print(f"- SQL 执行成功率：{_format_pct(metrics['exec_success_rate'])}")
    print(f"- 结果列匹配率：{_format_pct(metrics['column_match_rate'])}")
    print(f"- 行集一致率：{_format_pct(metrics['row_agreement_rate'])}")
    print()
    print("| 用例 | 问题 | 状态 | 列匹配 | 行一致 | SQL |")
    print("|------|------|------|--------|--------|-----|")
    for case in suite_report["per_case"]:
        status = "PASS" if case["pass"] else "FAIL"
        sql = (case["sql"] or "").replace("|", "\\|")
        print(
            f"| {case['id']} | {case['question']} | {status} "
            f"| {case['column_match']:.2f} | {case['row_agreement']:.2f} | {sql} |"
        )


def write_report(reports: list[dict[str, Any]], mode: str) -> Path:
    """把一次运行的全部套件结果落盘为 eval/reports/report-<ts>.json。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"report-{ts}.json"
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "mock": mode == "mock",
        "suites": reports,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.harness",
        description="AI Harness：对 AI 能力（NL2SQL 等）做可回归的离线评测（mock 模式可跑）。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="运行评测套件")
    run_parser.add_argument(
        "--suite",
        default="all",
        choices=["all", *SUITE_RUNNERS.keys()],
        help="要运行的套件（默认 all = 全部已就绪套件）",
    )
    run_parser.add_argument("--data", default=str(DEFAULT_DATA), help="待查询的数据 CSV 路径")
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"未知命令 {args.command!r}")

    client = LLMClient()
    mode = "mock" if client.mock else "dashscope"
    df = pd.read_csv(args.data)

    if args.suite == "all":
        reports = run_all(client, df)
    else:
        try:
            reports = [run_suite(args.suite, client, df)]
        except SuiteError as exc:
            print(f"[error] 套件 {args.suite} 无法运行：{exc}", file=sys.stderr)
            return 1

    for report in reports:
        print_markdown(report, mode)

    report_path = write_report(reports, mode)
    print(f"\n完整报告已写入：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())