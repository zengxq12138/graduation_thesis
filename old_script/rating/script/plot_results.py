"""
从 evaluation_progress.jsonl 读取评测结果并绘制图表
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 配置 =================
PROGRESS_FILE = "../output/fixed_evaluation_progress.jsonl"
SCORE_COLS = ['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']


def load_results(filename):
    """从 jsonl 文件加载评测结果"""
    results = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                results.append(data)
            except json.JSONDecodeError:
                continue
    return results


def setup_plot_style():
    """设置绘图样式和中文字体"""
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


def plot_charts(df):
    """绘制所有图表"""
    
    # 1. 完整性对比 (Comprehensiveness)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="System", y="Score_Comprehensiveness", hue="Type", palette="viridis")
    plt.title("Comprehensiveness Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("System", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 10.5)
    plt.legend(title="Type")
    plt.tight_layout()
    plt.savefig("../output/chart_comprehensiveness.png", dpi=300)
    print("✓ 图表已生成: ../output/chart_comprehensiveness.png")
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
    plt.savefig("../output/chart_faithfulness.png", dpi=300)
    print("✓ 图表已生成: ../output/chart_faithfulness.png")
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
    plt.savefig("../output/chart_relevance.png", dpi=300)
    print("✓ 图表已生成: ../output/chart_relevance.png")
    plt.close()

    # 4. 综合雷达图/热力图 (可选)
    plt.figure(figsize=(10, 8))
    summary = df.groupby(['System', 'Type'])[SCORE_COLS].mean().reset_index()
    summary_pivot = summary.pivot(index='System', columns='Type', values=SCORE_COLS)
    
    # 创建热力图数据
    heatmap_data = df.groupby('System')[SCORE_COLS].mean()
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                linewidths=0.5, vmin=0, vmax=10)
    plt.title("Average Scores Heatmap by System", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../output/chart_heatmap.png", dpi=300)
    print("✓ 图表已生成: ../output/chart_heatmap.png")
    plt.close()


def print_summary(df):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("评测统计摘要")
    print("=" * 60)
    
    # 按系统和类型分组统计
    summary = df.groupby(['System', 'Type'])[SCORE_COLS].agg(['mean', 'std', 'count'])
    print("\n【按系统和类型分组的平均分】")
    print(df.groupby(['System', 'Type'])[SCORE_COLS].mean().round(2))
    
    # 按系统整体统计
    print("\n【按系统整体平均分】")
    print(df.groupby('System')[SCORE_COLS].mean().round(2))
    
    # 总体统计
    print("\n【总体统计】")
    print(df[SCORE_COLS].describe().round(2))
    
    print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("从 evaluation_progress.jsonl 加载评测结果并绘图")
    print("=" * 60)
    
    # 1. 加载数据
    results = load_results(PROGRESS_FILE)
    if not results:
        print(f"错误：无法从 {PROGRESS_FILE} 加载数据或文件为空")
        return
    
    print(f"✓ 成功加载 {len(results)} 条评测记录")
    
    # 2. 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 3. 转换分数列为数字类型
    for col in SCORE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 4. 设置绘图样式
    setup_plot_style()
    
    # 5. 绘制图表
    plot_charts(df)
    
    # 6. 打印统计摘要
    print_summary(df)
    
    print("绑制完成！")


if __name__ == "__main__":
    main()
