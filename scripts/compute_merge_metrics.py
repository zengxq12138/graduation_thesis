#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复算 thesis/merge.md 中使用的关键统计数据，并输出为 JSON 与 Markdown 报告。

说明：
1. merge.md 中与实验结果一致的数字，主来源应为 output/fixed_evaluation_progress.jsonl；
2. 当前 output/final_evaluation_results.csv 与 old_script/rating/output/final_evaluation_results.csv
   作为对比来源保留，用于核查是否存在重复修正或未修正问题；
3. 如需让绘图流程与 merge.md 保持一致，可把主来源同步导出为 output/final_evaluation_results.csv。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_PROGRESS = PROJECT_ROOT / "output" / "fixed_evaluation_progress.jsonl"
CURRENT_CSV = PROJECT_ROOT / "output" / "final_evaluation_results.csv"
LEGACY_CSV = PROJECT_ROOT / "old_script" / "rating" / "output" / "final_evaluation_results.csv"
MERGE_MD = PROJECT_ROOT / "thesis" / "merge.md"
TESTSET_A = PROJECT_ROOT / "data" / "testset" / "A.json"
TESTSET_B = PROJECT_ROOT / "data" / "testset" / "B.json"
DEFAULT_JSON_OUT = PROJECT_ROOT / "output" / "merge_metrics_report.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "output" / "merge_metrics_report.md"

SCORE_COLS = ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]
SYSTEM_ORDER = ["pure_llm", "naive_rag", "light_rag"]
SYSTEM_LABELS = {
    "pure_llm": "Pure LLM",
    "naive_rag": "Naive RAG",
    "light_rag": "V-KG RAG",
}


def round2(value: float) -> float:
    return round(float(value) + 1e-12, 2)


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_progress_jsonl(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    rows: list[dict] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        rows.append(obj)

    df = pd.DataFrame(rows)
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_testset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def calc_composite(row: pd.Series) -> float:
    return round2((row[SCORE_COLS[0]] + row[SCORE_COLS[1]] + row[SCORE_COLS[2]]) / 3)


def group_means(df: pd.DataFrame, keys: list[str]) -> list[dict]:
    grouped = df.groupby(keys)[SCORE_COLS].mean().reset_index()
    grouped["Composite"] = grouped.apply(calc_composite, axis=1)
    for col in SCORE_COLS + ["Composite"]:
        grouped[col] = grouped[col].apply(round2)

    if keys == ["System", "Type"]:
        grouped["SystemOrder"] = grouped["System"].map({name: idx for idx, name in enumerate(SYSTEM_ORDER)})
        grouped["TypeOrder"] = grouped["Type"].map({"A": 0, "B": 1})
        grouped = grouped.sort_values(["TypeOrder", "SystemOrder"]).drop(columns=["SystemOrder", "TypeOrder"])
    elif keys == ["System"]:
        grouped["SystemOrder"] = grouped["System"].map({name: idx for idx, name in enumerate(SYSTEM_ORDER)})
        grouped = grouped.sort_values(["SystemOrder"]).drop(columns=["SystemOrder"])

    return grouped.to_dict(orient="records")


def compute_testset_stats() -> dict:
    result: dict[str, dict] = {}
    for name, path in [("A", TESTSET_A), ("B", TESTSET_B)]:
        data = load_testset(path)
        answer_lengths = [len(item["标准答案"]) for item in data]
        question_lengths = [len(item["问题"]) for item in data]
        result[name] = {
            "count": len(data),
            "avg_question_len": round2(sum(question_lengths) / len(question_lengths)),
            "avg_answer_len": round2(sum(answer_lengths) / len(answer_lengths)),
            "min_answer_len": min(answer_lengths),
            "max_answer_len": max(answer_lengths),
        }
    return result


def relative_improvement(base: float, target: float) -> float:
    if base == 0:
        return 0.0
    return round2((target - base) / base * 100)


def compute_improvements(records: Iterable[dict], key_field: str) -> list[dict]:
    groups: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        groups[str(record[key_field])][record["System"]] = float(record["Composite"])

    rows = []
    for name, values in groups.items():
        if not {"pure_llm", "naive_rag", "light_rag"} <= set(values):
            continue
        rows.append(
            {
                key_field: name,
                "vkg_vs_pure_pct": relative_improvement(values["pure_llm"], values["light_rag"]),
                "vkg_vs_naive_pct": relative_improvement(values["naive_rag"], values["light_rag"]),
                "naive_vs_pure_pct": relative_improvement(values["pure_llm"], values["naive_rag"]),
            }
        )
    return rows


def compute_quality_counts(df: pd.DataFrame) -> list[dict]:
    rows = []
    for system in SYSTEM_ORDER:
        subset = df[df["System"] == system]
        all_dims_ge8 = int((subset[SCORE_COLS].min(axis=1) >= 8).sum())
        any_dim_le5 = int((subset[SCORE_COLS].min(axis=1) <= 5).sum())
        rows.append(
            {
                "System": system,
                "SystemLabel": SYSTEM_LABELS[system],
                "total_questions": int(len(subset)),
                "all_dims_ge_8": all_dims_ge8,
                "any_dim_le_5": any_dim_le5,
            }
        )
    return rows


def classify_b_question(question: str) -> set[str]:
    categories: list[tuple[str, Callable[[str], bool]]] = [
        ("比较型", lambda q: any(k in q for k in ["比较", "区别", "不同"])),
        ("防治型", lambda q: any(k in q for k in ["防治", "措施", "策略", "清园", "套袋"])),
        ("机理规律型", lambda q: any(k in q for k in ["规律", "原因", "途径", "联系", "环境", "易感时期", "习性"])),
        ("症状归纳型", lambda q: any(k in q for k in ["症状", "特点"])),
    ]
    return {name for name, fn in categories if fn(question)}


def compute_b_categories(df: pd.DataFrame) -> list[dict]:
    rows = []
    b_df = df[df["Type"] == "B"].copy()
    for category in ["比较型", "防治型", "机理规律型", "症状归纳型"]:
        mask = b_df["Question"].apply(lambda q: category in classify_b_question(str(q)))
        subset = b_df[mask]
        if subset.empty:
            continue
        grouped = subset.groupby("System")[SCORE_COLS].mean()
        question_count = int(subset["Question"].nunique())
        for system in SYSTEM_ORDER:
            if system not in grouped.index:
                continue
            row = grouped.loc[system]
            rows.append(
                {
                    "Category": category,
                    "System": system,
                    "SystemLabel": SYSTEM_LABELS[system],
                    "Faithfulness": round2(row["Score_Faithfulness"]),
                    "Comprehensiveness": round2(row["Score_Comprehensiveness"]),
                    "Relevance": round2(row["Score_Relevance"]),
                    "Composite": round2(row.mean()),
                    "question_count": question_count,
                }
            )
    return rows


def compute_source_comparison(primary_df: pd.DataFrame, current_df: pd.DataFrame, legacy_df: pd.DataFrame) -> dict:
    primary_grouped = group_means(primary_df, ["System", "Type"])
    current_grouped = group_means(current_df, ["System", "Type"])
    legacy_grouped = group_means(legacy_df, ["System", "Type"])

    primary_map = {(row["System"], row["Type"]): row for row in primary_grouped}
    current_map = {(row["System"], row["Type"]): row for row in current_grouped}
    legacy_map = {(row["System"], row["Type"]): row for row in legacy_grouped}

    rows = []
    for key in sorted(set(primary_map) & set(current_map) & set(legacy_map)):
        src = primary_map[key]
        cur = current_map[key]
        old = legacy_map[key]
        rows.append(
            {
                "System": key[0],
                "Type": key[1],
                "primary_composite": src["Composite"],
                "current_composite": cur["Composite"],
                "legacy_composite": old["Composite"],
                "primary_faithfulness": src["Score_Faithfulness"],
                "current_faithfulness": cur["Score_Faithfulness"],
                "legacy_faithfulness": old["Score_Faithfulness"],
                "primary_comprehensiveness": src["Score_Comprehensiveness"],
                "current_comprehensiveness": cur["Score_Comprehensiveness"],
                "legacy_comprehensiveness": old["Score_Comprehensiveness"],
                "primary_relevance": src["Score_Relevance"],
                "current_relevance": cur["Score_Relevance"],
                "legacy_relevance": old["Score_Relevance"],
            }
        )

    return {
        "same_shape_primary_vs_current": list(primary_df.shape) == list(current_df.shape),
        "same_shape_primary_vs_legacy": list(primary_df.shape) == list(legacy_df.shape),
        "same_columns_primary_vs_current": list(primary_df.columns) == list(current_df.columns),
        "same_columns_primary_vs_legacy": list(primary_df.columns) == list(legacy_df.columns),
        "by_system_type": rows,
    }


def build_report(progress_path: Path, current_csv_path: Path, legacy_csv_path: Path) -> tuple[dict, pd.DataFrame]:
    primary_df = load_progress_jsonl(progress_path)
    current_df = load_scores(current_csv_path)
    legacy_df = load_scores(legacy_csv_path)

    by_system_type = group_means(primary_df, ["System", "Type"])
    by_system = group_means(primary_df, ["System"])

    overall_rows = [
        {
            "System": row["System"],
            "SystemLabel": SYSTEM_LABELS[row["System"]],
            "Faithfulness": row["Score_Faithfulness"],
            "Comprehensiveness": row["Score_Comprehensiveness"],
            "Relevance": row["Score_Relevance"],
            "Composite": row["Composite"],
        }
        for row in by_system
    ]

    system_type_rows = [
        {
            "System": row["System"],
            "SystemLabel": SYSTEM_LABELS[row["System"]],
            "Type": row["Type"],
            "Faithfulness": row["Score_Faithfulness"],
            "Comprehensiveness": row["Score_Comprehensiveness"],
            "Relevance": row["Score_Relevance"],
            "Composite": row["Composite"],
        }
        for row in by_system_type
    ]

    improvement_map = {}
    for row in compute_improvements(
        system_type_rows + [{"Type": "ALL", "System": r["System"], "Composite": r["Composite"]} for r in overall_rows],
        "Type",
    ):
        improvement_map[row["Type"]] = row

    report = {
        "source_files": {
            "primary_fixed_progress": str(progress_path),
            "current_evaluation_csv": str(current_csv_path),
            "legacy_evaluation_csv": str(legacy_csv_path),
            "testset_A": str(TESTSET_A),
            "testset_B": str(TESTSET_B),
            "merge_markdown": str(MERGE_MD),
        },
        "note": "merge.md 中与实验结果相关的统计值与 output/fixed_evaluation_progress.jsonl 保持一致；测试集长度与题量来自 data/testset/A.json 和 B.json。",
        "testset_stats": compute_testset_stats(),
        "tables": {
            "table_3_4_A": [row for row in system_type_rows if row["Type"] == "A"],
            "table_3_5_B": [row for row in system_type_rows if row["Type"] == "B"],
            "table_3_6_overall": overall_rows,
            "table_3_7_quality_counts": compute_quality_counts(primary_df),
            "table_3_8_b_categories": compute_b_categories(primary_df),
        },
        "derived_metrics": {
            "relative_improvements_pct": improvement_map,
            "a_vs_b_vkg_gap": round2(
                next(row["Composite"] for row in system_type_rows if row["System"] == "light_rag" and row["Type"] == "A")
                - next(row["Composite"] for row in system_type_rows if row["System"] == "light_rag" and row["Type"] == "B")
            ),
        },
        "source_comparison": compute_source_comparison(primary_df, current_df, legacy_df),
    }
    return report, primary_df


def format_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = [str(row[key]) for key, _ in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(report: dict, output_path: Path) -> None:
    testset = report["testset_stats"]
    tables = report["tables"]
    improvements = report["derived_metrics"]["relative_improvements_pct"]
    comparison = report["source_comparison"]

    lines = [
        "# merge.md 统计复算报告",
        "",
        "## 数据来源",
        "",
        f"- 主统计来源（与 merge.md 对齐）：`{report['source_files']['primary_fixed_progress']}`",
        f"- 当前用于绘图的评测 CSV：`{report['source_files']['current_evaluation_csv']}`",
        f"- 旧版 legacy 评测 CSV：`{report['source_files']['legacy_evaluation_csv']}`",
        f"- 测试集 A：`{report['source_files']['testset_A']}`",
        f"- 测试集 B：`{report['source_files']['testset_B']}`",
        "",
        report["note"],
        "",
        "## 测试集统计",
        "",
        format_table(
            [
                {
                    "Type": "A",
                    "Count": testset["A"]["count"],
                    "AvgQuestionLen": testset["A"]["avg_question_len"],
                    "AvgAnswerLen": testset["A"]["avg_answer_len"],
                    "MinAnswerLen": testset["A"]["min_answer_len"],
                    "MaxAnswerLen": testset["A"]["max_answer_len"],
                },
                {
                    "Type": "B",
                    "Count": testset["B"]["count"],
                    "AvgQuestionLen": testset["B"]["avg_question_len"],
                    "AvgAnswerLen": testset["B"]["avg_answer_len"],
                    "MinAnswerLen": testset["B"]["min_answer_len"],
                    "MaxAnswerLen": testset["B"]["max_answer_len"],
                },
            ],
            [
                ("Type", "测试集"),
                ("Count", "题量"),
                ("AvgQuestionLen", "问题平均长度"),
                ("AvgAnswerLen", "标准答案平均长度"),
                ("MinAnswerLen", "最短答案长度"),
                ("MaxAnswerLen", "最长答案长度"),
            ],
        ),
        "",
        "## 表 3-4 测试集 A",
        "",
        format_table(
            tables["table_3_4_A"],
            [
                ("SystemLabel", "方法"),
                ("Faithfulness", "忠实度"),
                ("Comprehensiveness", "完整性"),
                ("Relevance", "相关性"),
                ("Composite", "综合得分"),
            ],
        ),
        "",
        "## 表 3-5 测试集 B",
        "",
        format_table(
            tables["table_3_5_B"],
            [
                ("SystemLabel", "方法"),
                ("Faithfulness", "忠实度"),
                ("Comprehensiveness", "完整性"),
                ("Relevance", "相关性"),
                ("Composite", "综合得分"),
            ],
        ),
        "",
        "## 表 3-6 总体结果",
        "",
        format_table(
            tables["table_3_6_overall"],
            [
                ("SystemLabel", "方法"),
                ("Faithfulness", "忠实度"),
                ("Comprehensiveness", "完整性"),
                ("Relevance", "相关性"),
                ("Composite", "综合得分"),
            ],
        ),
        "",
        "## 相对提升",
        "",
        format_table(
            [
                {
                    "Type": key,
                    "VKGvsPure": value["vkg_vs_pure_pct"],
                    "VKGvsNaive": value["vkg_vs_naive_pct"],
                    "NaivevsPure": value["naive_vs_pure_pct"],
                }
                for key, value in improvements.items()
            ],
            [
                ("Type", "范围"),
                ("VKGvsPure", "V-KG RAG 相比 Pure LLM(%)"),
                ("VKGvsNaive", "V-KG RAG 相比 Naive RAG(%)"),
                ("NaivevsPure", "Naive RAG 相比 Pure LLM(%)"),
            ],
        ),
        "",
        f"- V-KG RAG 在 A、B 两类任务上的综合得分差值：`{report['derived_metrics']['a_vs_b_vkg_gap']}`",
        "",
        "## 表 3-7 稳定性统计",
        "",
        format_table(
            tables["table_3_7_quality_counts"],
            [
                ("SystemLabel", "方法"),
                ("total_questions", "总题数"),
                ("all_dims_ge_8", "三维均不低于8分"),
                ("any_dim_le_5", "任一维度不高于5分"),
            ],
        ),
        "",
        "## 表 3-8 测试集 B 粗粒度题型统计",
        "",
        format_table(
            tables["table_3_8_b_categories"],
            [
                ("Category", "题型"),
                ("SystemLabel", "方法"),
                ("question_count", "题目数"),
                ("Faithfulness", "忠实度"),
                ("Comprehensiveness", "完整性"),
                ("Relevance", "相关性"),
                ("Composite", "综合得分"),
            ],
        ),
        "",
        "## 来源对比",
        "",
        f"- 主来源与当前 CSV 形状一致：`{comparison['same_shape_primary_vs_current']}`",
        f"- 主来源与 legacy CSV 形状一致：`{comparison['same_shape_primary_vs_legacy']}`",
        f"- 主来源与当前 CSV 列一致：`{comparison['same_columns_primary_vs_current']}`",
        f"- 主来源与 legacy CSV 列一致：`{comparison['same_columns_primary_vs_legacy']}`",
        "",
        "下表用于说明 merge.md 中的统计值不能直接从 old_script 的原始 CSV 或当前 final_evaluation_results.csv 读取，而应以 fixed_evaluation_progress.jsonl 为准：",
        "",
        format_table(
            comparison["by_system_type"],
            [
                ("System", "系统"),
                ("Type", "测试集"),
                ("legacy_composite", "legacy 综合得分"),
                ("current_composite", "current 综合得分"),
                ("primary_composite", "主来源综合得分"),
                ("legacy_faithfulness", "legacy 忠实度"),
                ("current_faithfulness", "current 忠实度"),
                ("primary_faithfulness", "主来源忠实度"),
            ],
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复算 merge.md 中引用的统计数据")
    parser.add_argument("--progress", type=Path, default=PRIMARY_PROGRESS, help="修正后的评测进度 JSONL")
    parser.add_argument("--current-csv", type=Path, default=CURRENT_CSV, help="当前绘图使用的评测 CSV")
    parser.add_argument("--legacy-csv", type=Path, default=LEGACY_CSV, help="旧版 legacy 评测 CSV")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="JSON 输出文件")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT, help="Markdown 输出文件")
    parser.add_argument("--export-primary-csv", type=Path, help="将主来源导出为 CSV，常用于同步 output/final_evaluation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, primary_df = build_report(args.progress, args.current_csv, args.legacy_csv)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.md_out)
    if args.export_primary_csv:
        args.export_primary_csv.parent.mkdir(parents=True, exist_ok=True)
        primary_df.to_csv(args.export_primary_csv, index=False, encoding="utf-8-sig")
        print(f"主来源已导出为 CSV: {args.export_primary_csv}")
    print(f"JSON 报告已生成: {args.json_out}")
    print(f"Markdown 报告已生成: {args.md_out}")


if __name__ == "__main__":
    main()
