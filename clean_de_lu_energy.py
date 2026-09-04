# -*- coding: utf-8 -*-
"""清洗 OPSD 德国-卢森堡（DE_LU）能源数据。

删除“无数据”空壳行与窗口外杂点，但保留真实的历史极端电价事件。

用法：
    python clean_de_lu_energy.py             # 保留真实电价尖峰
    python clean_de_lu_energy.py --drop-high  # 同时删除电价 > 500 EUR/MWh
"""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "de_lu_energy.csv")
OUT = os.path.join(HERE, "de_lu_energy_clean.csv")
LOGF = os.path.join(HERE, "de_lu_cleaning_log.md")

VALUE_COLS = [
    "DE_LU_load_actual_entsoe_transparency",
    "DE_LU_load_forecast_entsoe_transparency",
    "DE_LU_price_day_ahead",
    "DE_LU_solar_generation_actual",
    "DE_LU_wind_generation_actual",
    "DE_LU_wind_offshore_generation_actual",
    "DE_LU_wind_onshore_generation_actual",
]
PRICE = "DE_LU_price_day_ahead"
WINDOW_START = "2018-01-01 00:00:00+00:00"
WINDOW_END = "2020-09-30 23:00:00+00:00"
HIGH_PRICE = 500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-high", action="store_true", help="删除日前电价 > 500 EUR/MWh")
    args = ap.parse_args()

    df = pd.read_csv(SRC, parse_dates=["utc_timestamp"])
    n_orig = len(df)
    lines = ["# de_lu_energy 数据清洗日志", ""]
    lines.append("- 生成时间：" + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("- 原始行数：" + str(n_orig))
    lines.append("- 时间范围：" + str(df["utc_timestamp"].min()) + " → " + str(df["utc_timestamp"].max()))
    lines.append("")

    empty_mask = df[VALUE_COLS].isna().all(axis=1)
    n_empty = int(empty_mask.sum())
    df = df[~empty_mask].copy()
    lines.append("## 1. 删除空数据行")
    lines.append("- 规则：所有数据字段均为空的行（OPSD 未提供数据的时段）")
    lines.append("- 删除：" + str(n_empty) + " 行")
    lines.append("")

    start = pd.Timestamp(WINDOW_START)
    end = pd.Timestamp(WINDOW_END)
    in_window = (df["utc_timestamp"] >= start) & (df["utc_timestamp"] <= end)
    n_out = int((~in_window).sum())
    df = df[in_window].copy()
    lines.append("## 2. 限定分析窗口")
    lines.append("- 窗口：" + WINDOW_START + " → " + WINDOW_END)
    lines.append("- 删除窗口外行：" + str(n_out) + " 行（含 2016/2017 杂点）")
    lines.append("")

    high = df[df[PRICE].notna() & (df[PRICE] > HIGH_PRICE)].copy()
    lines.append("## 3. 极端高电价事件")
    lines.append("- 规则：日前电价 > " + str(HIGH_PRICE) + " EUR/MWh，共 " + str(len(high)) + " 条")
    lines.append("- 判断：这些是真实的历史市场事件（如 2019-04-11 的德国日前电价纪录），不是测量误差，默认保留。")
    lines.append("")
    lines.append("| UTC 时间 | 日前电价 (EUR/MWh) |")
    lines.append("|---|---|")
    for _, r in high.iterrows():
        lines.append("| " + str(r["utc_timestamp"]) + " | " + "{:.2f}".format(r[PRICE]) + " |")
    lines.append("")

    if args.drop_high:
        drop_high = df[PRICE].notna() & (df[PRICE] > HIGH_PRICE)
        df = df[~drop_high].copy()
        lines.append("- 已按 --drop-high 删除高电价行：" + str(int(drop_high.sum())) + " 行")
        lines.append("")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    df.to_csv(OUT, index=False)

    price = df[PRICE].dropna()
    lines.append("## 4. 清洗结果")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append("| 清洗后行数 | " + str(len(df)) + " |")
    lines.append("| 时间范围 | " + str(df["utc_timestamp"].min()) + " → " + str(df["utc_timestamp"].max()) + " |")
    lines.append("| 电价非空 | " + str(price.shape[0]) + " |")
    lines.append("| 电价最小 / 中位 / 最大 | " + "{:.2f}".format(price.min()) + " / "
                 + "{:.2f}".format(price.median()) + " / " + "{:.2f}".format(price.max()) + " |")
    lines.append("| 负电价小时数 | " + str(int((price < 0).sum()))
                 + "（占非空 " + "{:.2f}".format((price < 0).mean() * 100) + "%） |")
    lines.append("")
    lines.append("## 5. 说明")
    lines.append("- 空数据行对应 OPSD 原始序列中未提供数据的时段，直接删除。")
    lines.append("- 保留的真实电价尖峰对“价格预测 + 新能源消纳”分析是有价值的研究样本。")
    lines.append("- 如需供基线模型使用、刻意排除尖峰，运行 python clean_de_lu_energy.py --drop-high。")
    lines.append("")

    with open(LOGF, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("清洗完成：" + str(n_orig) + " -> " + str(len(df)) + " 行")
    print("输出：" + OUT)
    print("日志：" + LOGF)


if __name__ == "__main__":
    main()