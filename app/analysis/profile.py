"""CSV 自动数据剖析（profile）。

对一张销售类 CSV 做一次「体检」：形状、dtype、缺失率、重复行率、
数值列的五数概括（describe）、低基数类别列的 Top 分布。

返回结构化的 dict，供 Streamlit 页直接渲染，也方便测试逐字段断言。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

#: 列的唯一值数量不超过该值时，才做 Top 分布统计（防止把高基数列刷屏）
DEFAULT_MAX_CARDINALITY = 12
#: 每列 Top 分布最多展示几个取值
DEFAULT_N_TOP = 5


def _to_number(value: Any) -> float | None:
    """把 pandas 描述统计量转成可 JSON 序列化的浮点数；NaN 记 None。"""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _describe_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict]:
    """数值列的 count/mean/std/min/四分位/max 五数概括。"""
    out: dict[str, dict] = {}
    for col in numeric_cols:
        stats = df[col].describe()
        out[col] = {
            "count": int(stats["count"]),
            "mean": _to_number(stats["mean"]),
            "std": _to_number(stats["std"]),
            "min": _to_number(stats["min"]),
            "q25": _to_number(stats["25%"]),
            "q50": _to_number(stats["50%"]),
            "q75": _to_number(stats["75%"]),
            "max": _to_number(stats["max"]),
        }
    return out


def _top_values(df: pd.DataFrame, *, n_top: int, max_cardinality: int) -> dict[str, dict]:
    """低基数文本列的取值分布（含 NaN 时以 "nan" 为键）。"""
    out: dict[str, dict] = {}
    for col in df.columns:
        series = df[col]
        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if not is_text:
            continue
        value_counts = series.value_counts(dropna=False)
        if len(value_counts) > max_cardinality:  # 高基数（如日期/姓名）不展示
            continue
        out[col] = {str(k): int(v) for k, v in value_counts.head(n_top).items()}
    return out


def profile_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    n_top: int = DEFAULT_N_TOP,
    max_cardinality: int = DEFAULT_MAX_CARDINALITY,
) -> dict[str, Any]:
    """剖析一张 CSV，返回结构化描述。

    Args:
        path: CSV 路径（绝对或相对当前工作目录）。
        encoding: 文件编码，默认 utf-8。
        n_top: Top 分布每列展示的取值个数。
        max_cardinality: 超过该唯一值数的文本列不统计分布。

    Returns:
        dict，键为 shape / columns / dtypes / missing / duplicates /
        numeric / top_values。数值统一用 float，缺失率是 0~1 的小数。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容为空（含全空行）。
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    if df.empty:
        raise ValueError(f"数据文件为空，没有可剖析的行：{csv_path}")

    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]

    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        # 缺失率：0.0 ~ 1.0
        "missing": {str(c): round(float(df[c].isna().mean()), 6) for c in df.columns},
        "duplicates": {
            "count": int(df.duplicated().sum()),
            "rate": round(float(df.duplicated().mean()), 6),
        },
        "numeric": {
            "columns": numeric_cols,
            "describe": _describe_numeric(df, numeric_cols),
        },
        "top_values": _top_values(df, n_top=n_top, max_cardinality=max_cardinality),
    }