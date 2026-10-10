"""app/config.py 的单元测试：YAML 加载、schema 校验、报错可读性。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import (
    RAW_DATASET,
    ChartSpec,
    ConfigError,
    PipelineConfig,
    TransformSpec,
    load_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "sales_dashboard.yaml"


def write_config(tmp_path: Path, payload: dict, name: str = "cfg.yaml") -> Path:
    """把 dict 写成 YAML，返回路径。"""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


# ----------------------------------------------------------- 示例配置


def test_load_example_config() -> None:
    """仓库自带的示例配置必须能加载，且字段符合 SPEC。"""
    cfg = load_pipeline(EXAMPLE_CONFIG)

    assert cfg.name == "销售驾驶舱"
    assert cfg.data == "data/sales.csv"


def test_example_transforms() -> None:
    cfg = load_pipeline(EXAMPLE_CONFIG)

    assert len(cfg.transforms) == 1
    monthly = cfg.transforms[0]
    assert monthly.id == "monthly"
    assert monthly.op == "aggregate"
    assert monthly.group_by == ["month"]
    assert monthly.metrics is not None
    assert {k: (m.col, m.fn) for k, m in monthly.metrics.items()} == {
        "total_amount": ("amount", "sum"),
        "avg_amount": ("amount", "mean"),
    }


def test_example_charts_and_filters() -> None:
    cfg = load_pipeline(EXAMPLE_CONFIG)

    assert [c.type for c in cfg.charts] == ["line", "bar"]

    line = cfg.charts[0]
    assert (line.data, line.x, line.y) == ("monthly", "month", "total_amount")
    assert line.title == "月度销售额"

    bar = cfg.charts[1]
    assert bar.data == RAW_DATASET
    assert bar.group_by == "region"
    assert bar.metric is not None and bar.metric.fn == "sum"

    assert len(cfg.filters) == 1
    assert cfg.filters[0].field == "region"
    assert cfg.filters[0].kind == "multiselect"


def test_dataset_ids() -> None:
    cfg = load_pipeline(EXAMPLE_CONFIG)
    assert cfg.dataset_ids == ["raw", "monthly"]


def test_load_accepts_str_path() -> None:
    """load_pipeline 同时接受 str 与 Path。"""
    assert load_pipeline(str(EXAMPLE_CONFIG)).name == "销售驾驶舱"


# ----------------------------------------------------------- 默认值与最小配置


def test_minimal_config_defaults(tmp_path: Path) -> None:
    """只给 name + data 时，其余字段退化为空列表。"""
    path = write_config(tmp_path, {"name": "极简", "data": "data/sales.csv"})
    cfg = load_pipeline(path)

    assert cfg.transforms == []
    assert cfg.charts == []
    assert cfg.filters == []
    assert cfg.dataset_ids == ["raw"]


# ----------------------------------------------------------- 错误路径


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_pipeline(tmp_path / "not-exist.yaml")
    assert "配置文件不存在" in str(exc.value)


def test_unknown_op_error_mentions_field(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "data/sales.csv",
        "transforms": [{"id": "a", "op": "groupby"}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))

    msg = str(exc.value)
    assert "未知算子" in msg and "groupby" in msg
    assert "op" in msg  # 报错点名了出错字段


def test_missing_required_field_mentions_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, {"data": "data/sales.csv"}))
    assert "name" in str(exc.value)


def test_missing_data_field_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, {"name": "x"}))
    assert "data" in str(exc.value)


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    payload = {"name": "x", "data": "d.csv", "layout": "wide"}
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "layout" in str(exc.value)


def test_unknown_chart_type_raises(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "charts": [{"type": "scatter3d", "x": "a", "y": "b"}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "未知图表类型" in str(exc.value)


def test_chart_unknown_dataset_raises(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "transforms": [{"id": "monthly", "op": "aggregate", "group_by": ["month"],
                        "metrics": {"t": {"col": "amount", "fn": "sum"}}}],
        "charts": [{"type": "line", "data": "weekly", "x": "a", "y": "b"}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))

    msg = str(exc.value)
    assert "weekly" in msg and "charts[0].data" in msg


def test_duplicate_transform_ids_raise(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "transforms": [
            {"id": "dup", "op": "aggregate", "group_by": ["region"],
             "metrics": {"t": {"col": "amount", "fn": "sum"}}},
            {"id": "dup", "op": "aggregate", "group_by": ["region"],
             "metrics": {"t": {"col": "amount", "fn": "sum"}}},
        ],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "重复" in str(exc.value)


def test_aggregate_requires_metrics(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "transforms": [{"id": "a", "op": "aggregate", "group_by": ["region"]}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "metrics" in str(exc.value)


def test_filter_requires_field(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "transforms": [{"id": "a", "op": "filter", "operator": ">", "value": 1}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "field" in str(exc.value)


def test_bar_chart_requires_metric_with_group_by(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "charts": [{"type": "bar", "data": "raw", "group_by": "region"}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "metric" in str(exc.value)


def test_unknown_metric_fn_raises(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "transforms": [
            {"id": "a", "op": "aggregate", "group_by": ["region"],
             "metrics": {"t": {"col": "amount", "fn": "median"}}}
        ],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "median" in str(exc.value)


def test_unknown_filter_kind_raises(tmp_path: Path) -> None:
    payload = {
        "name": "x",
        "data": "d.csv",
        "filters": [{"field": "region", "kind": "checkbox"}],
    }
    with pytest.raises(ConfigError) as exc:
        load_pipeline(write_config(tmp_path, payload))
    assert "未知筛选控件" in str(exc.value)


def test_broken_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: [不闭合\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_pipeline(path)
    assert "YAML 解析失败" in str(exc.value)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_pipeline(path)
    assert "内容为空" in str(exc.value)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_pipeline(path)
    assert "顶层必须是映射" in str(exc.value)


# ----------------------------------------------------------- 直接构造 model


def test_models_constructible_directly() -> None:
    """模型可脱离 YAML 直接构造，方便测试与程序化生成配置。"""
    cfg = PipelineConfig(
        name="直构",
        data="data/sales.csv",
        transforms=[TransformSpec(id="m", op="derive", source="date", target="month")],
        charts=[ChartSpec(type="pie", data="raw", x="category", y="amount")],
    )
    assert cfg.dataset_ids == ["raw", "m"]