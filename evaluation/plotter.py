"""
评测结果绘图模块
"""
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

SCORE_COLS = ['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']


def setup_plot_style():
    """设置绘图样式和中文字体"""
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


def load_results(progress_file: Path) -> List[dict]:
    """从 jsonl 文件加载评测结果"""
    results = []
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except json.JSONDecodeError:
                    continue
    return results


def plot_results(config: Config = None, results: List[dict] = None):
    """绘制所有评测图表"""
    if config is None:
        from config import default_config
        config = default_config

    # 加载数据
    if results is None:
        progress_file = config.paths.output_dir / "evaluation_progress.jsonl"
        results = load_results(progress_file)

    if not results:
        print("没有找到评测数据，无法绘图")
        return

    print(f"成功加载 {len(results)} 条评测记录")

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 转换分数列为数字类型
    for col in SCORE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 设置绘图样式
    setup_plot_style()

    # 确保输出目录存在
    charts_dir = config.paths.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. 完整性对比 (Comprehensiveness)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="System", y="Score_Comprehensiveness", hue="Type", palette="viridis")
    plt.title("Comprehensiveness Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("System", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 10.5)
    plt.legend(title="Type")
    plt.tight_layout()
    chart_path = charts_dir / "chart_comprehensiveness.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✓ 图表已生成: {chart_path}")
    plt.close()

    # 2. 忠实度对比 (Faithfulness)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="System", y="Score_Faithfulness", hue="Type", palette="magma")
    plt.title("Faithfulness Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("System", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 10.5)
    plt.legend(title="Type")
    plt.tight_layout()
    chart_path = charts_dir / "chart_faithfulness.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✓ 图表已生成: {chart_path}")
    plt.close()

    # 3. 相关性对比 (Relevance)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="System", y="Score_Relevance", hue="Type", palette="coolwarm")
    plt.title("Relevance Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("System", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 10.5)
    plt.legend(title="Type")
    plt.tight_layout()
    chart_path = charts_dir / "chart_relevance.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✓ 图表已生成: {chart_path}")
    plt.close()

    # 4. 热力图
    plt.figure(figsize=(10, 8))
    heatmap_data = df.groupby('System')[SCORE_COLS].mean()
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=0.5, vmin=0, vmax=10)
    plt.title("Average Scores Heatmap by System", fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = charts_dir / "chart_heatmap.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✓ 图表已生成: {chart_path}")
    plt.close()

    # 打印统计摘要
    print_summary(df)


def print_summary(df: pd.DataFrame):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("评测统计摘要")
    print("=" * 60)

    # 按系统和类型分组统计
    print("\n【按系统和类型分组的平均分】")
    print(df.groupby(['System', 'Type'])[SCORE_COLS].mean().round(2))

    # 按系统整体统计
    print("\n【按系统整体平均分】")
    print(df.groupby('System')[SCORE_COLS].mean().round(2))

    # 总体统计
    print("\n【总体统计】")
    print(df[SCORE_COLS].describe().round(2))

    print("=" * 60 + "\n")


if __name__ == "__main__":
    plot_results()
