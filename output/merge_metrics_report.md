# merge.md 统计复算报告

## 数据来源

- 主统计来源（与 merge.md 对齐）：`D:\graduation_thesis\output\fixed_evaluation_progress.jsonl`
- 当前用于绘图的评测 CSV：`D:\graduation_thesis\output\final_evaluation_results.csv`
- 旧版 legacy 评测 CSV：`D:\graduation_thesis\old_script\rating\output\final_evaluation_results.csv`
- 测试集 A：`D:\graduation_thesis\data\testset\A.json`
- 测试集 B：`D:\graduation_thesis\data\testset\B.json`

merge.md 中与实验结果相关的统计值与 output/fixed_evaluation_progress.jsonl 保持一致；测试集长度与题量来自 data/testset/A.json 和 B.json。

## 测试集统计

| 测试集 | 题量 | 问题平均长度 | 标准答案平均长度 | 最短答案长度 | 最长答案长度 |
| --- | --- | --- | --- | --- | --- |
| A | 20 | 19.3 | 24.6 | 14 | 49 |
| B | 30 | 23.23 | 90.03 | 55 | 141 |

## 表 3-4 测试集 A

| 方法 | 忠实度 | 完整性 | 相关性 | 综合得分 |
| --- | --- | --- | --- | --- |
| Pure LLM | 6.05 | 6.95 | 8.6 | 7.2 |
| Naive RAG | 8.65 | 8.4 | 8.6 | 8.55 |
| V-KG RAG | 9.35 | 9.4 | 9.6 | 9.45 |

## 表 3-5 测试集 B

| 方法 | 忠实度 | 完整性 | 相关性 | 综合得分 |
| --- | --- | --- | --- | --- |
| Pure LLM | 5.73 | 6.53 | 7.8 | 6.69 |
| Naive RAG | 7.77 | 7.9 | 8.5 | 8.06 |
| V-KG RAG | 8.53 | 8.6 | 9.4 | 8.84 |

## 表 3-6 总体结果

| 方法 | 忠实度 | 完整性 | 相关性 | 综合得分 |
| --- | --- | --- | --- | --- |
| Pure LLM | 5.86 | 6.7 | 8.12 | 6.89 |
| Naive RAG | 8.12 | 8.1 | 8.54 | 8.25 |
| V-KG RAG | 8.86 | 8.92 | 9.48 | 9.09 |

## 相对提升

| 范围 | V-KG RAG 相比 Pure LLM(%) | V-KG RAG 相比 Naive RAG(%) | Naive RAG 相比 Pure LLM(%) |
| --- | --- | --- | --- |
| A | 31.25 | 10.53 | 18.75 |
| B | 32.14 | 9.68 | 20.48 |
| ALL | 31.93 | 10.18 | 19.74 |

- V-KG RAG 在 A、B 两类任务上的综合得分差值：`0.61`

## 表 3-7 稳定性统计

| 方法 | 总题数 | 三维均不低于8分 | 任一维度不高于5分 |
| --- | --- | --- | --- |
| Pure LLM | 50 | 7 | 22 |
| Naive RAG | 50 | 34 | 6 |
| V-KG RAG | 50 | 40 | 1 |

## 表 3-8 测试集 B 粗粒度题型统计

| 题型 | 方法 | 题目数 | 忠实度 | 完整性 | 相关性 | 综合得分 |
| --- | --- | --- | --- | --- | --- | --- |
| 比较型 | Pure LLM | 4 | 6.25 | 7.25 | 8.25 | 7.25 |
| 比较型 | Naive RAG | 4 | 7.75 | 8.75 | 9.0 | 8.5 |
| 比较型 | V-KG RAG | 4 | 8.7 | 9.7 | 10.0 | 9.47 |
| 防治型 | Pure LLM | 18 | 5.61 | 6.17 | 7.56 | 6.44 |
| 防治型 | Naive RAG | 18 | 7.72 | 7.61 | 8.39 | 7.91 |
| 防治型 | V-KG RAG | 18 | 8.37 | 8.37 | 9.28 | 8.67 |
| 机理规律型 | Pure LLM | 12 | 5.58 | 6.75 | 7.92 | 6.75 |
| 机理规律型 | Naive RAG | 12 | 8.08 | 8.25 | 8.67 | 8.33 |
| 机理规律型 | V-KG RAG | 12 | 8.78 | 8.95 | 9.58 | 9.11 |
| 症状归纳型 | Pure LLM | 8 | 5.38 | 6.0 | 7.5 | 6.29 |
| 症状归纳型 | Naive RAG | 8 | 8.13 | 8.75 | 8.88 | 8.58 |
| 症状归纳型 | V-KG RAG | 8 | 8.7 | 8.95 | 9.5 | 9.05 |

## 来源对比

- 主来源与当前 CSV 形状一致：`True`
- 主来源与 legacy CSV 形状一致：`True`
- 主来源与当前 CSV 列一致：`True`
- 主来源与 legacy CSV 列一致：`True`

下表用于说明 merge.md 中的统计值不能直接从 old_script 的原始 CSV 或当前 final_evaluation_results.csv 读取，而应以 fixed_evaluation_progress.jsonl 为准：

| 系统 | 测试集 | legacy 综合得分 | current 综合得分 | 主来源综合得分 | legacy 忠实度 | current 忠实度 | 主来源忠实度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| light_rag | A | 9.45 | 9.45 | 9.45 | 9.35 | 9.35 | 9.35 |
| light_rag | B | 8.71 | 8.98 | 8.84 | 8.33 | 8.73 | 8.53 |
| naive_rag | A | 8.55 | 8.55 | 8.55 | 8.65 | 8.65 | 8.65 |
| naive_rag | B | 9.06 | 7.06 | 8.06 | 8.77 | 6.77 | 7.77 |
| pure_llm | A | 7.2 | 7.2 | 7.2 | 6.05 | 6.05 | 6.05 |
| pure_llm | B | 7.69 | 5.69 | 6.69 | 6.73 | 4.73 | 5.73 |
