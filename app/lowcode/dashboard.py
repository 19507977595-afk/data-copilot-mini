"""配置 → Streamlit 看板页渲染（低代码内核之二）。

本模块没有任何业务判断：读 ``charts`` 画图、读 ``filters`` 生成筛选控件、
读 ``transforms`` 重算数据集。改 YAML 即改页面，Python 一行都不用动。

典型用法（app/main.py 的看板页）：

    cfg = load_pipeline("configs/sales_dashboard.yaml")
    df = pd.read_csv(cfg.data)
    datasets = run_pipeline(df, cfg.transforms)
    render_dashboard(cfg, datasets)          # 控件 + 图表，一步到位

筛选逻辑：控件作用于 ``raw``，然后**重跑管道**，这样 monthly 等聚合数据集会跟着
筛选联动（否则对聚合结果筛 region 会因缺少 region 列而失败）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import RAW_DATASET, ChartSpec, FilterSpec, PipelineConfig
from app.lowcode.pipeline import aggregate, run_pipeline

# ---- streamlit 是"页面运行时"依赖：导入失败也不该让模块本身炸掉（便于无头 import）----
try:  # pragma: no cover - 取决于本机是否装了 streamlit
    import streamlit as st
except ImportError as _exc:  # pragma: no cover
    st = None
    _ST_IMPORT_ERROR: ImportError | None = _exc
else:
    _ST_IMPORT_ERROR = None

#: 「全部」选项的显示文本（select 控件用）
ALL_OPTION = "（全部）"


def _st():
    """取 streamlit 模块；未安装时给出可执行的安装提示。"""
    if st is None:  # pragma: no cover - 仅在缺依赖时触发
        raise RuntimeError("未安装 streamlit，请先执行 pip install -r requirements.txt") from _ST_IMPORT_ERROR
    return st


# ------------------------------------------------------------------ 图表


def value_column(chart: ChartSpec) -> str:
    """bar/pie 用 ``group_by + metric`` 现场聚合时，结果列的名字。"""
    if chart.y:
        return chart.y
    assert chart.metric is not None  # ChartSpec 校验保证 group_by 与 metric 同时出现
    return f"{chart.metric.fn}({chart.metric.col})"


def prepare_chart_frame(chart: ChartSpec, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """取出图表要用的数据：直接引用数据集，或对 ``group_by + metric`` 现场聚合。"""
    if chart.data not in datasets:
        raise KeyError(f"数据集 '{chart.data}' 不存在（可用：{sorted(datasets)}）")
    df = datasets[chart.data]

    if chart.group_by and chart.metric:
        return aggregate(df, [chart.group_by], {value_column(chart): chart.metric})
    return df


def figure_from_chart(chart: ChartSpec, datasets: dict[str, pd.DataFrame]):
    """把一条 ChartSpec 变成 Plotly Figure（line / bar / pie）。"""
    import plotly.express as px  # 懒加载：只在真正要画图时导入

    df = prepare_chart_frame(chart, datasets)
    title = chart.title or None

    if df.empty:
        fig = px.scatter(title=f"{title or chart.type}（无数据）")
        fig.add_annotation(text="当前筛选条件下没有数据", showarrow=False)
        return fig

    if chart.type == "line":
        fig = px.line(df, x=chart.x, y=chart.y, title=title, markers=True)
    elif chart.type == "bar":
        # group_by 模式下面板数据已在 prepare_chart_frame 里聚合好
        x = chart.x or chart.group_by
        y = chart.y or value_column(chart)
        fig = px.bar(df, x=x, y=y, title=title)
    else:  # pie
        fig = px.pie(df, names=chart.x, values=chart.y, title=title)

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
    return fig


# ------------------------------------------------------------------ 筛选控件


def _unique_values(df: pd.DataFrame, field: str) -> list[Any]:
    """某列的候选值（去空、保持出现顺序）。"""
    return [v for v in pd.unique(df[field].dropna())]


def collect_selections(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    *,
    key_prefix: str = "",
) -> dict[str, Any]:
    """按 ``filters`` 渲染控件，返回 ``{field: 选中值}``。

    只在原数据里真实存在的字段上生成控件（配置写错字段不会让整页崩掉）。
    """
    st = _st()
    selections: dict[str, Any] = {}
    missing: list[str] = []

    for spec in cfg.filters:
        if spec.field not in df.columns:
            missing.append(spec.field)
            continue

        key = f"{key_prefix}filter_{spec.field}"
        values = _unique_values(df, spec.field)

        if spec.kind == "multiselect":
            picked = st.multiselect(spec.field, options=values, default=[], key=key)
            if picked:
                selections[spec.field] = picked
        elif spec.kind == "select":
            picked = st.selectbox(spec.field, options=[ALL_OPTION, *values], key=key)
            if picked != ALL_OPTION:
                selections[spec.field] = picked
        elif spec.kind == "range":
            numeric = pd.to_numeric(df[spec.field], errors="coerce").dropna()
            if numeric.empty:
                continue
            low, high = float(numeric.min()), float(numeric.max())
            picked = st.slider(spec.field, min_value=low, max_value=high, value=(low, high), key=key)
            if picked != (low, high):
                selections[spec.field] = picked
        else:  # date_range
            dates = pd.to_datetime(df[spec.field], errors="coerce").dropna()
            if dates.empty:
                continue
            low, high = dates.min().date(), dates.max().date()
            picked = st.date_input(spec.field, value=(low, high), key=key)
            if isinstance(picked, (list, tuple)) and len(picked) == 2 and tuple(picked) != (low, high):
                selections[spec.field] = tuple(picked)

    if missing:
        st.caption(f"⚠️ 配置里的筛选字段在原数据中不存在，已跳过：{missing}")
    return selections


def apply_selections(
    df: pd.DataFrame,
    filters: list[FilterSpec],
    selections: dict[str, Any],
) -> pd.DataFrame:
    """把选中的筛选条件应用到 DataFrame（未选中任何值的控件不过滤）。"""
    out = df
    for spec in filters:
        if spec.field not in selections or spec.field not in out.columns:
            continue
        picked = selections[spec.field]
        series = out[spec.field]

        if spec.kind == "range" and isinstance(picked, (list, tuple)) and len(picked) == 2:
            numeric = pd.to_numeric(series, errors="coerce")
            out = out[(numeric >= picked[0]) & (numeric <= picked[1])]
        elif spec.kind == "date_range" and isinstance(picked, (list, tuple)) and len(picked) == 2:
            dates = pd.to_datetime(series, errors="coerce")
            start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
            out = out[(dates >= start) & (dates <= end)]
        else:
            out = out[series.isin(list(picked))]
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ 主入口


def build_datasets(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    selections: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """先按筛选条件裁剪 raw，再重跑 transforms —— 让聚合数据集跟着筛选联动。"""
    raw = df if not selections else apply_selections(df, cfg.filters, selections)
    return run_pipeline(raw, cfg.transforms)


def render_dashboard(
    cfg: PipelineConfig,
    datasets: dict[str, pd.DataFrame],
    *,
    selections: dict[str, Any] | None = None,
    key_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """按配置渲染整页看板：筛选控件 + 若干 Plotly 图。

    Args:
        cfg: 已校验的看板配置。
        datasets: ``run_pipeline`` 的输出（至少含 ``raw``）。
        selections: 外部传入的筛选结果；为 None 时由本函数渲染控件收集。
        key_prefix: 控件 key 前缀（同一页面多次渲染时避免 key 冲突）。

    Returns:
        真正参与渲染的数据集（若应用了筛选，是重跑管道后的版本）。
    """
    st = _st()
    raw = datasets.get(RAW_DATASET)

    effective = datasets
    if raw is not None:
        selections = (
            collect_selections(cfg, raw, key_prefix=key_prefix) if selections is None else selections
        )
        effective = build_datasets(cfg, raw, selections)
        rows = len(effective[RAW_DATASET])
        st.caption(f"当前筛选后行数：{rows} / {len(raw)}")

    if not cfg.charts:
        st.info("这份配置还没有定义 charts，去左边加一条试试。")
        return effective

    columns = st.columns(2) if len(cfg.charts) > 1 else [st.container()]
    for idx, chart in enumerate(cfg.charts):
        with columns[idx % len(columns)]:
            st.plotly_chart(
                figure_from_chart(chart, effective),
                use_container_width=True,
                key=f"{key_prefix}chart_{idx}",
            )
    return effective