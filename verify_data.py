#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证论文中使用的数据是否准确
"""
import pandas as pd
import json

def verify_thesis_data():
    """验证论文数据"""

    print("="*80)
    print("论文数据验证")
    print("="*80)

    # 读取CSV数据
    df = pd.read_csv('output/final_evaluation_results.csv')

    # 验证表4-1的数据
    print("\n验证表4-1：三种方法在测试集A上的评分统计")
    print("-" * 80)

    test_a_data = {
        'Pure LLM': {'faith': 6.05, 'comp': 6.95, 'rel': 8.60},
        'Naive RAG': {'faith': 8.65, 'comp': 8.40, 'rel': 8.60},
        'LightRAG': {'faith': 9.35, 'comp': 9.40, 'rel': 9.60}
    }

    for system, expected in test_a_data.items():
        actual = df[(df['System'] == system.lower().replace(' ', '_')) & (df['Type'] == 'A')]
        actual_faith = actual['Score_Faithfulness'].mean()
        actual_comp = actual['Score_Comprehensiveness'].mean()
        actual_rel = actual['Score_Relevance'].mean()

        print(f"\n{system}:")
        print(f"  忠实度 - 论文: {expected['faith']}, 实际: {actual_faith:.2f}, {'OK' if abs(expected['faith'] - actual_faith) < 0.01 else 'ERROR'}")
        print(f"  完整性 - 论文: {expected['comp']}, 实际: {actual_comp:.2f}, {'OK' if abs(expected['comp'] - actual_comp) < 0.01 else 'ERROR'}")
        print(f"  相关性 - 论文: {expected['rel']}, 实际: {actual_rel:.2f}, {'OK' if abs(expected['rel'] - actual_rel) < 0.01 else 'ERROR'}")

    # 验证表4-2的数据
    print("\n\n验证表4-2：三种方法在测试集B上的评分统计")
    print("-" * 80)

    test_b_data = {
        'Pure LLM': {'faith': 5.73, 'comp': 6.53, 'rel': 7.80},
        'Naive RAG': {'faith': 7.77, 'comp': 7.90, 'rel': 8.50},
        'LightRAG': {'faith': 8.53, 'comp': 8.60, 'rel': 9.40}
    }

    for system, expected in test_b_data.items():
        actual = df[(df['System'] == system.lower().replace(' ', '_')) & (df['Type'] == 'B')]
        actual_faith = actual['Score_Faithfulness'].mean()
        actual_comp = actual['Score_Comprehensiveness'].mean()
        actual_rel = actual['Score_Relevance'].mean()

        print(f"\n{system}:")
        print(f"  忠实度 - 论文: {expected['faith']}, 实际: {actual_faith:.2f}, {'OK' if abs(expected['faith'] - actual_faith) < 0.01 else 'ERROR'}")
        print(f"  完整性 - 论文: {expected['comp']}, 实际: {actual_comp:.2f}, {'OK' if abs(expected['comp'] - actual_comp) < 0.01 else 'ERROR'}")
        print(f"  相关性 - 论文: {expected['rel']}, 实际: {actual_rel:.2f}, {'OK' if abs(expected['rel'] - actual_rel) < 0.01 else 'ERROR'}")

    # 验证表4-3的整体数据
    print("\n\n验证表4-3：三种方法整体性能对比")
    print("-" * 80)

    overall_data = {
        'Pure LLM': {'faith': 5.86, 'comp': 6.70, 'rel': 8.12, 'overall': 6.89},
        'Naive RAG': {'faith': 8.12, 'comp': 8.10, 'rel': 8.54, 'overall': 8.25},
        'LightRAG': {'faith': 8.86, 'comp': 8.92, 'rel': 9.48, 'overall': 9.09}
    }

    for system, expected in overall_data.items():
        actual = df[df['System'] == system.lower().replace(' ', '_')]
        actual_faith = actual['Score_Faithfulness'].mean()
        actual_comp = actual['Score_Comprehensiveness'].mean()
        actual_rel = actual['Score_Relevance'].mean()
        actual_overall = (actual_faith + actual_comp + actual_rel) / 3

        print(f"\n{system}:")
        print(f"  忠实度 - 论文: {expected['faith']}, 实际: {actual_faith:.2f}, {'OK' if abs(expected['faith'] - actual_faith) < 0.01 else 'ERROR'}")
        print(f"  完整性 - 论文: {expected['comp']}, 实际: {actual_comp:.2f}, {'OK' if abs(expected['comp'] - actual_comp) < 0.01 else 'ERROR'}")
        print(f"  相关性 - 论文: {expected['rel']}, 实际: {actual_rel:.2f}, {'OK' if abs(expected['rel'] - actual_rel) < 0.01 else 'ERROR'}")
        print(f"  综合得分 - 论文: {expected['overall']}, 实际: {actual_overall:.2f}, {'✓' if abs(expected['overall'] - actual_overall) < 0.01 else '✗'}")

    # 验证表4-4的数据
    print("\n\n验证表4-4：事实型问题 vs 推理型问题综合得分对比")
    print("-" * 80)

    question_type_data = {
        '事实型问题（测试集A）': {'Pure LLM': 7.20, 'Naive RAG': 8.55, 'LightRAG': 9.45},
        '推理型问题（测试集B）': {'Pure LLM': 6.69, 'Naive RAG': 8.06, 'LightRAG': 8.84}
    }

    for qtype, expected in question_type_data.items():
        test_type = 'A' if 'A' in qtype else 'B'
        print(f"\n{qtype}:")

        for system in ['Pure LLM', 'Naive RAG', 'LightRAG']:
            actual = df[(df['System'] == system.lower().replace(' ', '_')) & (df['Type'] == test_type)]
            actual_score = actual[['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']].mean().mean()

            print(f"  {system} - 论文: {expected[system]}, 实际: {actual_score:.2f}, {'OK' if abs(expected[system] - actual_score) < 0.01 else 'ERROR'}")

    # 验证低分案例统计
    print("\n\n验证表4-5：各方法低分案例统计（得分<5分）")
    print("-" * 80)

    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        subset = df[df['System'] == system]
        total = len(subset)
        faith_low = len(subset[subset['Score_Faithfulness'] < 5])
        comp_low = len(subset[subset['Score_Comprehensiveness'] < 5])
        rel_low = len(subset[subset['Score_Relevance'] < 5])

        print(f"\n{system}:")
        print(f"  忠实度<5: {faith_low} ({faith_low/total*100:.1f}%)")
        print(f"  完整性<5: {comp_low} ({comp_low/total*100:.1f}%)")
        print(f"  相关性<5: {rel_low} ({rel_low/total*100:.1f}%)")

    # 验证关键百分比
    print("\n\n验证关键数据百分比")
    print("-" * 80)

    # LightRAG相比Pure LLM的提升
    light_faith = df[df['System'] == 'light_rag']['Score_Faithfulness'].mean()
    pure_faith = df[df['System'] == 'pure_llm']['Score_Faithfulness'].mean()
    faith_improvement = (light_faith - pure_faith) / pure_faith * 100

    print(f"\nLightRAG相比Pure LLM忠实度提升: {faith_improvement:.1f}%")
    print(f"  论文中: 50.9%")
    print(f"  实际值: {faith_improvement:.1f}%")
    print(f"  {'OK' if abs(faith_improvement - 50.9) < 1 else 'ERROR'}")

    light_comp = df[df['System'] == 'light_rag']['Score_Comprehensiveness'].mean()
    pure_comp = df[df['System'] == 'pure_llm']['Score_Comprehensiveness'].mean()
    comp_improvement = (light_comp - pure_comp) / pure_comp * 100

    print(f"\nLightRAG相比Pure LLM完整性提升: {comp_improvement:.1f}%")
    print(f"  论文中: 33.1%")
    print(f"  实际值: {comp_improvement:.1f}%")
    print(f"  {'OK' if abs(comp_improvement - 33.1) < 1 else 'ERROR'}")

    overall_light = df[df['System'] == 'light_rag'][['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']].mean().mean()
    overall_pure = df[df['System'] == 'pure_llm'][['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']].mean().mean()
    overall_improvement = (overall_light - overall_pure) / overall_pure * 100

    print(f"\nLightRAG相比Pure LLM综合得分提升: {overall_improvement:.1f}%")
    print(f"  论文中: 31.8%")
    print(f"  实际值: {overall_improvement:.1f}%")
    print(f"  {'OK' if abs(overall_improvement - 31.8) < 1 else 'ERROR'}")

    print("\n" + "="*80)
    print("验证完成")
    print("="*80)


if __name__ == "__main__":
    verify_thesis_data()