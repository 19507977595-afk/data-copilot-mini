"""低代码配置加载与校验。

一份 YAML 描述「数据 + 变换 + 图表 + 筛选器」，本模块负责把它解析成强类型的
PipelineConfig。改 YAML 即改看板——这是低代码卖点的起点。

校验策略：用 pydantic 管类型，用自定义 validator 给出**带字段路径的中文报错**，
例如 `transforms[0].op: 未知算子 'groupby'（支持：aggregate/derive/filter）`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# 支持的算子 / 图表 / 筛选控件种类（SPEC 契约，改这里 = 扩展低代码能力）
TRANSFORM_OPS = ("filter", "derive", "aggregate")
CHART_TYPES = ("line", "bar", "pie")
AGG_FUNCS = ("sum", "mean", "count", "min", "max")
FILTER_KINDS = ("multiselect", "select", "range", "date_range")

#: 原始数据集的名字，任何 transforms 之前的数据都叫 raw
RAW_DATASET = "raw"


class ConfigError(ValueError):
    """配置不合法时抛出，message 里必带出错的字段路径。"""


class MetricSpec(BaseModel):
    """聚合指标：对某列做某种聚合。"""

    model_config = ConfigDict(extra="forbid")

    col: str = Field(min_length=1)
    fn: Literal["sum", "mean", "count", "min", "max"]

    @model_validator(mode="after")
    def _check_fn(self) -> "MetricSpec":
        if self.fn not in AGG_FUNCS:  # pragma: no cover - Literal 已兜住，双重保险
            raise ConfigError(f"metric.fn: 未知聚合函数 '{self.fn}'（支持：{'/'.join(AGG_FUNCS)}）")
        return self


class TransformSpec(BaseModel):
    """一步数据变换，输出一个以 id 命名的新数据集。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    op: str
    # filter
    field: str | None = None
    operator: str | None = None
    value: Any = None
    # derive
    source: str | None = None
    target: str | None = None
    # aggregate
    group_by: list[str] | None = None
    metrics: dict[str, MetricSpec] | None = None

    @model_validator(mode="after")
    def _check_op(self) -> "TransformSpec":
        if self.op not in TRANSFORM_OPS:
            raise ConfigError(
                f"op: 未知算子 '{self.op}'（支持：{'/'.join(TRANSFORM_OPS)}）"
            )
        if self.op == "filter" and not self.field:
            raise ConfigError("op=filter 时必须提供 field 字段")
        if self.op == "derive" and not (self.source and self.target):
            raise ConfigError("op=derive 时必须提供 source 与 target 字段")
        if self.op == "aggregate":
            if not self.group_by:
                raise ConfigError("op=aggregate 时必须提供 group_by 字段")
            if not self.metrics:
                raise ConfigError("op=aggregate 时必须提供 metrics 字段")
        return self


class ChartSpec(BaseModel):
    """一张图表的声明式配置。"""

    model_config = ConfigDict(extra="forbid")

    type: str
    data: str = RAW_DATASET
    title: str = ""
    # line / pie 用 x + y
    x: str | None = None
    y: str | None = None
    # bar 可以直接按列聚合
    group_by: str | None = None
    metric: MetricSpec | None = None

    @model_validator(mode="after")
    def _check_chart(self) -> "ChartSpec":
        if self.type not in CHART_TYPES:
            raise ConfigError(
                f"type: 未知图表类型 '{self.type}'（支持：{'/'.join(CHART_TYPES)}）"
            )
        if self.group_by and not self.metric:
            raise ConfigError("设置 group_by 时必须同时提供 metric")
        if not self.group_by and not (self.x and self.y):
            raise ConfigError("图表必须提供 (x, y) 或 (group_by, metric)")
        return self


class FilterSpec(BaseModel):
    """全局筛选控件。"""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    kind: str = "multiselect"

    @model_validator(mode="after")
    def _check_kind(self) -> "FilterSpec":
        if self.kind not in FILTER_KINDS:
            raise ConfigError(
                f"kind: 未知筛选控件 '{self.kind}'（支持：{'/'.join(FILTER_KINDS)}）"
            )
        return self


class PipelineConfig(BaseModel):
    """一份完整的低代码看板配置。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    data: str = Field(min_length=1)
    transforms: list[TransformSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_refs(self) -> "PipelineConfig":
        """交叉校验：图表引用的数据集必须存在，id 不能重复。"""
        ids = [t.id for t in self.transforms]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ConfigError(f"transforms.id: 数据集 id 重复 {sorted(duplicates)}")

        known = {RAW_DATASET, *ids}
        for idx, chart in enumerate(self.charts):
            if chart.data not in known:
                raise ConfigError(
                    f"charts[{idx}].data: 数据集 '{chart.data}' 不存在"
                    f"（可用：{sorted(known)}）"
                )
        return self

    # ---- 便捷访问 ----

    @property
    def dataset_ids(self) -> list[str]:
        """所有可用数据集名（含 raw），按定义顺序。"""
        return [RAW_DATASET, *[t.id for t in self.transforms]]


def _format_pydantic_error(exc: ValidationError) -> str:
    """把 pydantic 的错误列表压成一行行 `字段: 说明`。"""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        msg = err["msg"]
        # 枚举类错误（Literal）自带信息但看不到实际传入值，这里补上，排错更快
        if err["type"] != "missing" and "input" in err:
            msg = f"值 {err['input']!r} 不合法：{msg}"
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def load_pipeline(path: str | Path) -> PipelineConfig:
    """读取并校验 YAML 配置。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ConfigError: YAML 语法错误或字段不合法（消息含字段名）。
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{cfg_path} YAML 解析失败：{exc}") from exc

    if raw is None:
        raise ConfigError(f"{cfg_path} 内容为空，至少需要 name 与 data 字段")
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} 顶层必须是映射（key: value），实际是 {type(raw).__name__}")

    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{cfg_path} 校验失败：\n{_format_pydantic_error(exc)}") from exc
    except ConfigError as exc:
        # 自定义 validator 抛的错：前面加上文件名，便于定位
        raise ConfigError(f"{cfg_path} 校验失败：{exc}") from exc