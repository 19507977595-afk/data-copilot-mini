"""生成示例销售数据 data/sales.csv（固定随机种子，可重复产出同样数据）。

用法：
    python scripts/gen_data.py            # 生成 data/sales.csv（600 行）
    python scripts/gen_data.py --rows 800 # 自定义行数

列：date, region, category, salesperson, amount, quantity
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGIONS: dict[str, list[str]] = {
    "华东": ["王磊", "李婷", "陈浩"],
    "华北": ["张伟", "刘洋", "赵敏"],
    "华南": ["黄俊", "周慧", "吴凯"],
    "西南": ["杨帆", "何静", "罗磊"],
    "华中": ["徐涛", "孙倩", "郑楠"],
}

#: 品类 -> (单价区间, 数量区间)，让金额与品类、数量有合理关联
CATEGORIES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "数码": ((800, 6000), (1, 4)),
    "家居": ((150, 1500), (1, 8)),
    "服饰": ((80, 800), (2, 15)),
    "食品": ((20, 260), (5, 40)),
    "美妆": ((60, 600), (1, 10)),
}

#: 简单季节性：年末旺季、年初淡季
MONTH_WEIGHT = [0.85, 0.80, 0.95, 1.00, 1.05, 1.10, 1.00, 0.98, 1.12, 1.20, 1.35, 1.45]

FIELDNAMES = ["date", "region", "category", "salesperson", "amount", "quantity"]


def generate(rows: int = 600, seed: int = 42, start: date = date(2024, 1, 1)) -> list[dict]:
    """生成 `rows` 条销售记录。"""
    rng = random.Random(seed)
    region_names = list(REGIONS)
    category_names = list(CATEGORIES)
    records: list[dict] = []

    for _ in range(rows):
        # 按月份权重选日期，制造季节性波动
        month = rng.choices(range(1, 13), weights=MONTH_WEIGHT, k=1)[0]
        day = rng.randint(1, 28)
        when = date(start.year, month, day)

        region = rng.choice(region_names)
        salesperson = rng.choice(REGIONS[region])
        category = rng.choice(category_names)
        (lo_price, hi_price), (lo_qty, hi_qty) = CATEGORIES[category]

        quantity = rng.randint(lo_qty, hi_qty)
        unit_price = round(rng.uniform(lo_price, hi_price), 2)
        amount = round(unit_price * quantity, 2)

        records.append(
            {
                "date": when.isoformat(),
                "region": region,
                "category": category,
                "salesperson": salesperson,
                "amount": amount,
                "quantity": quantity,
            }
        )

    records.sort(key=lambda r: (r["date"], r["region"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="生成示例销售数据")
    parser.add_argument("--rows", type=int, default=600, help="生成行数（默认 600，>=500）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "sales.csv",
        help="输出路径（默认 data/sales.csv）",
    )
    args = parser.parse_args()

    if args.rows < 500:
        raise SystemExit("行数需 >= 500（SPEC 要求）")

    records = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    total = sum(r["amount"] for r in records)
    print(f"已生成 {len(records)} 行 -> {args.out}")
    print(f"日期范围：{records[0]['date']} ~ {records[-1]['date']}，总销售额：{total:,.2f}")


if __name__ == "__main__":
    main()