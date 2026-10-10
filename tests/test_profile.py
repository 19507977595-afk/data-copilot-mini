"""app/analysis/profile.py 的单元测试：数据剖析的结构与正确性。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.profile import profile_csv

ROOT = Path(__file__).resolve().parents[1]
SALES_CSV = ROOT / "data" / "sales.csv"


def profile() -> dict:
    return profile_csv(SALES_CSV)


# ----------------------------------------------------------- 形状与列


def test_shape_and_columns() -> None:
    p = profile()
    assert p["shape"] == {"rows": 600, "columns": 6}
    assert p["columns"] == ["date", "region", "category", "salesperson", "amount", "quantity"]


def test_dtypes_type_names() -> None:
    p = profile()
    assert "float" in p["dtypes"]["amount"]
    assert "int" in p["dtypes"]["quantity"]
    assert set(p["dtypes"]) == set(p["columns"])


# ----------------------------------------------------------- 缺失与重复


def test_clean_file_no_missing() -> None:
    p = profile()
    assert all(rate == 0.0 for rate in p["missing"].values())


def test_clean_file_no_duplicate_rows() -> None:
    p = profile()
    assert p["duplicates"] == {"count": 0, "rate": 0.0}


def test_missing_and_dup_rates(tmp_path: Path) -> None:
    """手工构造含缺失与重复行的 CSV，验证两个比率算得对。"""
    csv_path = tmp_path / "dirty.csv"
    csv_path.write_text(
        "date,region,amount\n"
        "2024-01-01,华南,100\n"
        "2024-01-02,,200\n"      # region 缺失
        "2024-01-03,华东,200\n"
        "2024-01-02,,200\n",     # 与第 2 行完全重复
        encoding="utf-8",
    )
    p = profile_csv(csv_path)
    assert p["shape"] == {"rows": 4, "columns": 3}
    assert p["missing"]["region"] == pytest.approx(0.5)
    assert p["missing"]["amount"] == 0.0
    assert p["duplicates"]["count"] == 1
    assert p["duplicates"]["rate"] == pytest.approx(0.25)


# ----------------------------------------------------------- 数值列描述


def test_numeric_describe_structure() -> None:
    p = profile()
    assert p["numeric"]["columns"] == ["amount", "quantity"]

    markup = {"count", "mean", "std", "min", "q25", "q50", "q75", "max"}
    for col in ("amount", "quantity"):
        assert markup <= set(p["numeric"]["describe"][col]), col


def test_numeric_describe_sane_values() -> None:
    p = profile()
    amount = p["numeric"]["describe"]["amount"]
    assert amount["count"] == 600
    assert amount["min"] >= 0
    assert amount["max"] > amount["min"]
    # 四分位必须单调
    assert amount["q25"] <= amount["q50"] <= amount["q75"] <= amount["max"]

    qty = p["numeric"]["describe"]["quantity"]
    assert qty["count"] == 600
    assert qty["min"] >= 1


# ----------------------------------------------------------- Top 分布


def test_top_values_regions() -> None:
    p = profile()
    regions = p["top_values"]["region"]
    assert set(regions) == {"华东", "华中", "华北", "华南", "西南"}
    assert sum(regions.values()) == 600
    assert max(regions.values()) <= 600


def test_top_values_categories() -> None:
    p = profile()
    cats = p["top_values"]["category"]
    assert set(cats) == {"家居", "数码", "服饰", "美妆", "食品"}
    assert sum(cats.values()) == 600


def test_high_cardinality_columns_skipped() -> None:
    """日期（~600 个唯一值）与销售员（15 个）不该出现在 Top 分布里。"""
    p = profile()
    assert "date" not in p["top_values"]
    assert "salesperson" not in p["top_values"]


# ----------------------------------------------------------- 错误路径


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        profile_csv(tmp_path / "not-exist.csv")
    assert "不存在" in str(exc.value)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("date,region", encoding="utf-8")  # 只有表头没有行
    with pytest.raises(ValueError) as exc:
        profile_csv(path)
    assert "为空" in str(exc.value)