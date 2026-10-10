"""配置驱动的 transform 管道——低代码内核。

一份 YAML 里的 ``transforms`` 会被逐条执行：每一步吃上一步的输出，产出一个以
``id`` 命名的新数据集；``raw`` 永远是最原始的数据，可被任何图表直接引用。

支持的算子（与 app/config.py 的 ``TRANSFORM_OPS`` 一一对应）：

- ``filter``    简单表达式过滤：``field operator value``（==、!=、>、>=、<、<=、in、contains…）
- ``derive``    从日期列派生 ``month`` / ``year`` / ``quarter`` / ``month`` / ``weekday`` 等
- ``aggregate`` ``group_by`` + ``metrics``（sum/mean/count/min/max）

设计上刻意不引入任何业务概念：本模块只认「列名 + 算子」，所以把销售换成设备、
把 amount 换成温度，YAML 改一改照样跑——这才是低代码。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from app.config import RAW_DATASET, MetricSpec, TransformSpec

#: 可派生（derive）的日期部件 -> (中文说明, 生成函数)
#: 用 target 名字查表；``month`` 是销售看板最常用的粒度，故默认取 %Y-%m。
DATE_PARTS = ("month", "year", "quarter", "week", "weekday", "day")

#: 供「自动派生」寻找日期源列时的候选列名（按优先级）
_DATE_SOURCE_HINTS = ("date", "日期", "时间", "datetime", "day")

#: filter 支持的比较算子
FILTER_OPERATORS = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not in",
    "contains",
    "startswith",
)


class PipelineError(ValueError):
    """管道执行失败（未知算子、缺列、类型不匹配等），message 里带字段名。"""


# ------------------------------------------------------------------ 工具函数


def derive_dates(series: pd.Series, part: str) -> pd.Series:
    """把一列日期文本派生为指定部件。

    Args:
        series: 原始日期列（字符串或 datetime，无法解析的值会成为 NaT）。
        part: ``month`` / ``year`` / ``quarter`` / ``week`` / ``weekday`` / ``day``。

    Returns:
        派生后的 Series：month 形如 ``2024-03``，weekday 形如 ``周一``，
        其余为整数。

    Raises:
        PipelineError: part 不在支持列表内。
    """
    key = str(part).strip().lower()
    if key not in DATE_PARTS:
        raise PipelineError(
            f"target: 不支持的派生列 '{part}'（支持：{'/'.join(DATE_PARTS)}）"
        )

    dt = pd.to_datetime(series, errors="coerce")
    if key == "month":
        return dt.dt.strftime("%Y-%m")
    if key == "year":
        return dt.dt.year
    if key == "quarter":
        return dt.dt.quarter
    if key == "week":
        return dt.dt.strftime("%Y-W%V")
    if key == "day":
        return dt.dt.day
    # weekday：1=周一 … 7=周日，映射成中文更直观
    names = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}
    return dt.dt.isoweekday().map(names)


def find_date_column(df: pd.DataFrame) -> str | None:
    """在 df 中找一个像日期的列（用于 group_by 里的 month 自动派生）。"""
    for name in _DATE_SOURCE_HINTS:
        if name in df.columns:
            return name
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    return None


def _coerce_values(value: Any) -> list[Any]:
    """把 ``in`` 这类算子的 value 统一成列表（支持 ``a,b`` 字符串写法）。"""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def apply_filter(df: pd.DataFrame, spec: TransformSpec) -> pd.DataFrame:
    """按 ``field operator value`` 过滤行（保留全部列）。"""
    field = str(spec.field)
    if field not in df.columns:
        raise PipelineError(
            f"filter.field: 列 '{field}' 不存在（可用列：{list(df.columns)}）"
        )

    operator = (spec.operator or "==").strip().lower()
    if operator not in FILTER_OPERATORS:
        raise PipelineError(
            f"filter.operator: 未知算子 '{spec.operator}'（支持：{'/'.join(FILTER_OPERATORS)}）"
        )

    series = df[field]
    value = spec.value

    if operator == "==":
        mask = series == value
    elif operator == "!=":
        mask = series != value
    elif operator == ">":
        mask = series > value
    elif operator == ">=":
        mask = series >= value
    elif operator == "<":
        mask = series < value
    elif operator == "<=":
        mask = series <= value
    elif operator == "in":
        mask = series.isin(_coerce_values(value))
    elif operator == "not in":
        mask = ~series.isin(_coerce_values(value))
    elif operator == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    else:  # startswith
        mask = series.astype(str).str.startswith(str(value), na=False)

    return df[mask].reset_index(drop=True)


def aggregate(
    df: pd.DataFrame,
    group_by: Iterable[str],
    metrics: Mapping[str, MetricSpec | Mapping[str, str]],
) -> pd.DataFrame:
    """按 ``group_by`` 分组，用 ``metrics`` 做聚合，返回「分组列 + 指标列」的宽表。

    metrics 的键是输出列名，值是 ``{col, fn}``（或已校验的 MetricSpec）。
    分组列里出现 ``month`` 这类派生命中而源数据没有时，会自动从日期列派生。
    """
    groups = [str(g) for g in group_by]
    if not groups:
        raise PipelineError("aggregate.group_by: 不能为空")

    work = df
    for name in groups:
        if name not in work.columns:
            work = _auto_derive(work, name)

    if not metrics:
        raise PipelineError("aggregate.metrics: 不能为空")

    named: dict[str, Any] = {}
    for out_name, spec in metrics.items():
        col, fn = _metric_parts(spec, out_name)
        if col not in work.columns:
            raise PipelineError(
                f"aggregate.metrics.{out_name}.col: 列 '{col}' 不存在"
                f"（可用列：{list(work.columns)}）"
            )
        named[str(out_name)] = pd.NamedAgg(column=col, aggfunc=fn)

    result = (
        work.groupby(groups, dropna=False, observed=True)
        .agg(**named)
        .reset_index()
        .sort_values(groups)
        .reset_index(drop=True)
    )
    return result


def _metric_parts(spec: MetricSpec | Mapping[str, str], out_name: str) -> tuple[str, str]:
    """取出 (列名, 聚合函数)，兼容 pydantic 对象与原始 dict。"""
    if isinstance(spec, MetricSpec):
        return spec.col, spec.fn
    if isinstance(spec, Mapping):
        try:
            return str(spec["col"]), str(spec["fn"])
        except KeyError as exc:
            raise PipelineError(f"aggregate.metrics.{out_name}: 缺少字段 {exc.args[0]}") from exc
    raise PipelineError(f"aggregate.metrics.{out_name}: 必须是 {{col, fn}} 映射")


def _auto_derive(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """为缺失的分组列做隐式派生（仅当名字是日期部件且源表里有日期列）。"""
    if name not in DATE_PARTS:
        raise PipelineError(f"aggregate.group_by: 列 '{name}' 不存在（可用列：{list(df.columns)}）")
    source = find_date_column(df)
    if source is None:
        raise PipelineError(
            f"aggregate.group_by: 列 '{name}' 不存在，且找不到可用于派生的日期列"
        )
    out = df.copy()
    out[name] = derive_dates(out[source], name)
    return out


def apply_derive(df: pd.DataFrame, spec: TransformSpec) -> pd.DataFrame:
    """新增一列派生日期部件（不修改原始列）。"""
    source = str(spec.source)
    target = str(spec.target)
    if source not in df.columns:
        raise PipelineError(f"derive.source: 列 '{source}' 不存在（可用列：{list(df.columns)}）")

    out = df.copy()
    out[target] = derive_dates(out[source], target)
    return out


def apply_transform(df: pd.DataFrame, spec: TransformSpec) -> pd.DataFrame:
    """执行一步变换，按 ``op`` 分发。"""
    if spec.op == "filter":
        return apply_filter(df, spec)
    if spec.op == "derive":
        return apply_derive(df, spec)
    if spec.op == "aggregate":
        return aggregate(df, spec.group_by or [], spec.metrics or {})
    # 理论上 config.py 已拦住未知 op，这里兜底并给出同样的字段路径报错
    raise PipelineError(f"op: 未知算子 '{spec.op}'")


# ------------------------------------------------------------------ 主入口


def run_pipeline(
    df: pd.DataFrame,
    transforms: Iterable[TransformSpec | Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """按顺序执行全部变换，返回 ``{数据集名: DataFrame}``。

    - ``raw`` 始终是传入数据的副本（不会被任何变换污染）；
    - 每步变换的输入是**上一步的输出**，输出以该步 ``id`` 命名；
    - 因此后定义的步骤可以引用前面派生出来的列 / 过滤后的行。

    Args:
        df: 原始数据。
        transforms: TransformSpec 列表（也接受等价的 dict，便于测试与脚本调用）。

    Raises:
        PipelineError: 某一步执行失败（消息含出错的字段路径）。
    """
    datasets: dict[str, pd.DataFrame] = {RAW_DATASET: df.copy()}
    current = df.copy()

    for spec in transforms:
        spec = _as_spec(spec)
        try:
            current = apply_transform(current, spec)
        except PipelineError as exc:
            # 补上是哪一步出的错，页面/CLI 里能直接定位到 YAML 的第几个 transform
            raise PipelineError(f"transforms[{spec.id}]: {exc}") from exc
        datasets[spec.id] = current

    return datasets


def _as_spec(spec: TransformSpec | Mapping[str, Any]) -> TransformSpec:
    """把 dict 形式的 transform 交给 pydantic 校验成 TransformSpec。"""
    if isinstance(spec, TransformSpec):
        return spec
    return TransformSpec.model_validate(dict(spec))