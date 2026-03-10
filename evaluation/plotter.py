"""
评测结果绘图。
"""
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


SCORE_COLS = ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_results(progress_file: Path) -> List[dict]:
    if not progress_file.exists():
        return []

    with open(progress_file, "r", encoding="utf-8") as file:
        content = file.read().strip()

    results: List[dict] = []
    # 使用 JSONDecoder 来解析连续的 JSON 对象
    decoder = json.JSONDecoder()
    index = 0
    while index < len(content):
        # 跳过空白字符
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        try:
            data, index = decoder.raw_decode(content, index)
            results.append(data)
        except json.JSONDecodeError:
            # 如果解析失败，尝试下一行
            break

    return results


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("评测统计摘要")
    print("=" * 60)
    print(df.groupby(["System", "Type"])[SCORE_COLS].mean().round(2))
    print("\n")
    print(df.groupby("System")[SCORE_COLS].mean().round(2))
    print("=" * 60)


def plot_results(config: Config = None, results: List[dict] = None) -> None:
    """
    生成评估结果的图表。

    数据源优先级：
    1. 参数传入的 results
    2. final_evaluation_results.csv（修正后的标准数据）
    """
    if config is None:
        from config import default_config
        config = default_config

    if results is None:
        # 统一从 CSV 文件读取（这是经过修正的标准数据）
        csv_path = config.paths.output_dir / "final_evaluation_results.csv"
        if not csv_path.exists():
            print("没有找到评测数据文件: final_evaluation_results.csv")
            print("请先运行评估: python main.py evaluate")
            return

        results = pd.read_csv(csv_path).to_dict(orient="records")

    if not results:
        print("没有找到评测数据，无法绘图")
        return

    df = pd.DataFrame(results)
    for column in SCORE_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    setup_plot_style()
    config.paths.charts_dir.mkdir(parents=True, exist_ok=True)

    chart_specs = [
        ("Score_Comprehensiveness", "viridis", "chart_comprehensiveness.png", "Comprehensiveness Comparison (1-10 Scale)"),
        ("Score_Faithfulness", "magma", "chart_faithfulness.png", "Faithfulness Comparison (1-10 Scale)"),
        ("Score_Relevance", "coolwarm", "chart_relevance.png", "Relevance Comparison (1-10 Scale)"),
    ]

    for score_key, palette, filename, title in chart_specs:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="System", y=score_key, hue="Type", palette=palette)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("System")
        plt.ylabel("Score")
        plt.ylim(0, 10.5)
        plt.tight_layout()
        chart_path = config.paths.charts_dir / filename
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"图表已生成: {chart_path}")

    plt.figure(figsize=(10, 8))
    heatmap_data = df.groupby("System")[SCORE_COLS].mean()
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, vmin=0, vmax=10)
    plt.title("Average Scores Heatmap by System", fontsize=14, fontweight="bold")
    plt.tight_layout()
    heatmap_path = config.paths.charts_dir / "chart_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"图表已生成: {heatmap_path}")

    print_summary(df)
