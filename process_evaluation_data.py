#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理 fixed_evaluation_progress.jsonl 文件
将多个连续的JSON对象解析并转换为CSV格式
"""
import json
import csv
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config


def parse_json_objects(file_path: Path):
    """
    解析包含多个JSON对象的文件
    返回所有JSON对象的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用json.JSONDecoder来解析多个JSON对象
    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    content_length = len(content)

    print(f"文件总长度: {content_length} 字符")

    while idx < content_length:
        # 跳过空白字符
        while idx < content_length and content[idx].isspace():
            idx += 1

        if idx >= content_length:
            break

        try:
            # 尝试从当前位置解析JSON对象
            obj, end_idx = decoder.raw_decode(content, idx)
            objects.append(obj)
            idx = end_idx

            if len(objects) % 100 == 0:
                print(f"已解析 {len(objects)} 个对象...")

        except json.JSONDecodeError as e:
            print(f"解析错误在位置 {idx}: {e}")
            print(f"附近内容: {content[max(0, idx-50):min(content_length, idx+50)]}")
            # 尝试找到下一个 '{' 继续
            next_brace = content.find('{', idx)
            if next_brace == -1:
                break
            idx = next_brace

    print(f"总共解析了 {len(objects)} 个JSON对象")
    return objects


def save_to_csv(data: list, output_path: Path):
    """
    将数据列表保存为CSV文件
    """
    if not data:
        print("没有数据可保存")
        return

    # 定义列顺序
    fieldnames = [
        'System', 'Type', 'Question', 'Method',
        'Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance',
        'Reason'
    ]

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV文件已保存到: {output_path}")


def calculate_statistics(data: list):
    """
    计算统计数据
    """
    df = pd.DataFrame(data)

    print("\n" + "="*60)
    print("统计数据分析")
    print("="*60)

    # 总记录数
    print(f"\n总记录数: {len(df)}")

    # 按系统和类型分组统计
    print("\n按系统和类型分组的记录数:")
    grouped = df.groupby(['System', 'Type']).size().reset_index(name='Count')
    print(grouped.to_string(index=False))

    # 计算各方法的平均分数
    print("\n各方法的平均分数:")
    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        for test_type in ['A', 'B']:
            subset = df[(df['System'] == system) & (df['Type'] == test_type)]
            if len(subset) > 0:
                faith_mean = subset['Score_Faithfulness'].mean()
                comp_mean = subset['Score_Comprehensiveness'].mean()
                rel_mean = subset['Score_Relevance'].mean()

                print(f"\n{system} - 测试集{test_type}:")
                print(f"  忠实度: {faith_mean:.2f}")
                print(f"  完整性: {comp_mean:.2f}")
                print(f"  相关性: {rel_mean:.2f}")
                print(f"  记录数: {len(subset)}")

    # 计算整体平均分数
    print("\n各方法整体平均分数:")
    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        subset = df[df['System'] == system]
        if len(subset) > 0:
            faith_mean = subset['Score_Faithfulness'].mean()
            comp_mean = subset['Score_Comprehensiveness'].mean()
            rel_mean = subset['Score_Relevance'].mean()
            overall_mean = (faith_mean + comp_mean + rel_mean) / 3

            print(f"\n{system}:")
            print(f"  忠实度: {faith_mean:.2f}")
            print(f"  完整性: {comp_mean:.2f}")
            print(f"  相关性: {rel_mean:.2f}")
            print(f"  综合得分: {overall_mean:.2f}")
            print(f"  总记录数: {len(subset)}")

    # 统计低分案例（<5分）
    print("\n低分案例统计（得分<5分）:")
    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        subset = df[df['System'] == system]
        faith_low = len(subset[subset['Score_Faithfulness'] < 5])
        comp_low = len(subset[subset['Score_Comprehensiveness'] < 5])
        rel_low = len(subset[subset['Score_Relevance'] < 5])

        print(f"\n{system}:")
        print(f"  忠实度<5: {faith_low} ({faith_low/len(subset)*100:.1f}%)")
        print(f"  完整性<5: {comp_low} ({comp_low/len(subset)*100:.1f}%)")
        print(f"  相关性<5: {rel_low} ({rel_low/len(subset)*100:.1f}%)")

    return df


def main():
    """主函数"""
    config = Config()

    # 输入文件路径
    input_file = config.paths.output_dir / "fixed_evaluation_progress.jsonl"

    # 输出文件路径
    output_csv = config.paths.output_dir / "final_evaluation_results.csv"

    print("="*60)
    print("处理 fixed_evaluation_progress.jsonl 文件")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_csv}")
    print()

    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return

    # 解析JSON对象
    print("开始解析JSON对象...")
    data = parse_json_objects(input_file)

    if not data:
        print("错误: 没有解析到任何数据")
        return

    # 保存为CSV
    print("\n保存为CSV文件...")
    save_to_csv(data, output_csv)

    # 计算统计数据
    df = calculate_statistics(data)

    # 额外保存统计数据到JSON文件
    stats_file = config.paths.output_dir / "statistics_summary.json"
    stats = {}

    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        stats[system] = {}
        for test_type in ['A', 'B']:
            subset = df[(df['System'] == system) & (df['Type'] == test_type)]
            if len(subset) > 0:
                stats[system][f'testset_{test_type}'] = {
                    'faithfulness': round(subset['Score_Faithfulness'].mean(), 2),
                    'comprehensiveness': round(subset['Score_Comprehensiveness'].mean(), 2),
                    'relevance': round(subset['Score_Relevance'].mean(), 2),
                    'count': int(len(subset))
                }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n统计数据已保存到: {stats_file}")
    print("\n处理完成！")


if __name__ == "__main__":
    main()