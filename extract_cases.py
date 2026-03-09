#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取典型案例用于论文
"""
import pandas as pd
from pathlib import Path
import json

def extract_cases():
    """提取典型案例"""

    # 读取CSV数据
    df = pd.read_csv('output/final_evaluation_results.csv')

    print("="*80)
    print("提取典型案例")
    print("="*80)

    # 案例1：知识幻觉纠正 - 寻找Pure LLM得分很低但LightRAG得分高的问题
    print("\n案例1：知识幻觉纠正")
    print("-" * 80)

    # 找到Pure LLM忠实度最低的几个问题
    pure_llm_low = df[(df['System'] == 'pure_llm') & (df['Type'] == 'A')].nsmallest(5, 'Score_Faithfulness')

    for idx, row in pure_llm_low.iterrows():
        question = row['Question']
        print(f"\n问题: {question}")

        # 找到同一问题的其他方法答案
        for system in ['pure_llm', 'naive_rag', 'light_rag']:
            record = df[(df['System'] == system) & (df['Question'] == question)]
            if len(record) > 0:
                r = record.iloc[0]
                print(f"  {system}: 忠{r['Score_Faithfulness']:.0f}/完{r['Score_Comprehensiveness']:.0f}/相{r['Score_Relevance']:.0f}")
        break  # 只显示第一个案例

    # 案例2：完整性对比 - 推理型问题
    print("\n\n案例2：完整性对比 - 推理型问题")
    print("-" * 80)

    # 找到LightRAG完整性明显高于Pure LLM的问题
    test_b = df[df['Type'] == 'B']

    for question in test_b['Question'].unique()[:5]:  # 查看前5个推理型问题
        print(f"\n问题: {question}")

        scores = {}
        for system in ['pure_llm', 'naive_rag', 'light_rag']:
            record = df[(df['System'] == system) & (df['Question'] == question)]
            if len(record) > 0:
                r = record.iloc[0]
                scores[system] = {
                    'faith': r['Score_Faithfulness'],
                    'comp': r['Score_Comprehensiveness'],
                    'rel': r['Score_Relevance']
                }
                print(f"  {system}: 忠{r['Score_Faithfulness']:.0f}/完{r['Score_Comprehensiveness']:.0f}/相{r['Score_Relevance']:.0f}")

    # 案例3：对比分析问题
    print("\n\n案例3：对比分析问题")
    print("-" * 80)

    # 查找包含"比较"、"区别"等关键词的问题
    comparison_questions = df[df['Question'].str.contains('比较|区别|不同', na=False)]

    if len(comparison_questions) > 0:
        q = comparison_questions.iloc[0]['Question']
        print(f"\n问题: {q}")

        for system in ['pure_llm', 'naive_rag', 'light_rag']:
            record = df[(df['System'] == system) & (df['Question'] == q)]
            if len(record) > 0:
                r = record.iloc[0]
                print(f"  {system}: 忠{r['Score_Faithfulness']:.0f}/完{r['Score_Comprehensiveness']:.0f}/相{r['Score_Relevance']:.0f}")

    # 案例4：最高分和最低分对比
    print("\n\n案例4：极端分数案例")
    print("-" * 80)

    # 找到LightRAG得分最高的问题
    light_rag_best = df[(df['System'] == 'light_rag') & (df['Type'] == 'A')].nlargest(3, 'Score_Faithfulness')

    print("\nLightRAG表现最好的问题:")
    for idx, row in light_rag_best.iterrows():
        print(f"\n问题: {row['Question']}")
        print(f"  LightRAG: 忠{row['Score_Faithfulness']:.0f}/完{row['Score_Comprehensiveness']:.0f}/相{row['Score_Relevance']:.0f}")

        # 对比其他方法
        for system in ['pure_llm', 'naive_rag']:
            record = df[(df['System'] == system) & (df['Question'] == row['Question'])]
            if len(record) > 0:
                r = record.iloc[0]
                print(f"  {system}: 忠{r['Score_Faithfulness']:.0f}/完{r['Score_Comprehensiveness']:.0f}/相{r['Score_Relevance']:.0f}")
        break  # 只显示第一个

    # 案例5：失败案例分析
    print("\n\n案例5：各方法失败案例")
    print("-" * 80)

    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        print(f"\n{system} 得分最低的问题:")

        worst = df[df['System'] == system].nsmallest(1, 'Score_Faithfulness')
        if len(worst) > 0:
            row = worst.iloc[0]
            print(f"  问题: {row['Question']}")
            print(f"  得分: 忠{row['Score_Faithfulness']:.0f}/完{row['Score_Comprehensiveness']:.0f}/相{row['Score_Relevance']:.0f}")
            print(f"  原因: {row['Reason'][:100]}...")

    # 输出一些具体的问答对
    print("\n\n" + "="*80)
    print("具体问答示例")
    print("="*80)

    # 选择一个有代表性的问题
    sample_question = "杨梅癌肿病（杨梅疮）的病原菌属于哪类细菌？"

    print(f"\n问题: {sample_question}")
    print("\n各方法回答:")

    for system in ['pure_llm', 'naive_rag', 'light_rag']:
        record = df[(df['System'] == system) & (df['Question'] == sample_question)]
        if len(record) > 0:
            r = record.iloc[0]
            print(f"\n{system}:")
            print(f"  评分: 忠实度{r['Score_Faithfulness']:.0f} | 完整性{r['Score_Comprehensiveness']:.0f} | 相关性{r['Score_Relevance']:.0f}")
            print(f"  评语: {r['Reason']}")


if __name__ == "__main__":
    extract_cases()