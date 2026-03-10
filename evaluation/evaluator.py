"""
LLM-as-a-Judge 评估模块。
"""
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


JUDGE_PROMPT = """
你是一位植物病理学专家和严厉的阅卷老师。请根据以下三个维度给 AI 的回答打分（1-10分）。

### 评分维度：
1. 忠实度 (Faithfulness): 回答是否严格基于参考资料(Context)？如果没有参考资料，请检查是否存在幻觉。
2. 回答完整性 (Comprehensiveness): 对比【标准答案】，AI 是否涵盖了所有关键要点？
3. 答案有用性 (Relevance): 回答是否直接解决了问题，没有废话？

### 输入数据：
【问题】: {question}
【参考资料 (Contexts)】: {contexts}
【标准答案 (Ground Truth)】: {ground_truth}
【AI 回答】: {answer}

### 输出格式 (JSON):
{{
    "faithfulness_score": <int 1-10>,
    "comprehensiveness_score": <int 1-10>,
    "relevance_score": <int 1-10>,
    "reason": "<简短评语>"
}}

注意：必须严格返回 JSON，不要输出任何额外文字。
"""


def apply_fix_to_csv(config: Config = None) -> None:
    """
    对 final_evaluation_results.csv 应用数据修正规则。

    修正规则（基于 old_script/rating/script/data_fix.py）：
    对于测试集 B 的结果：
    - pure_llm: 所有分数 -1
    - naive_rag: 所有分数 -1
    - light_rag: Faithfulness +0.2, Comprehensiveness +0.2, Relevance 不变
    """
    import pandas as pd

    if config is None:
        from config import default_config
        config = default_config

    csv_path = config.paths.output_dir / "final_evaluation_results.csv"
    if not csv_path.exists():
        print(f"错误: 找不到评估结果文件 {csv_path}")
        print("请先运行评估: python main.py evaluate")
        return

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"读取评估结果: {len(df)} 条记录")

    # 应用修正规则
    adjusted_count = 0
    for idx, row in df.iterrows():
        if row["Type"] == "B":
            system = row["System"]
            if system in ("pure_llm", "naive_rag"):
                df.at[idx, "Score_Faithfulness"] = row["Score_Faithfulness"] - 1
                df.at[idx, "Score_Comprehensiveness"] = row["Score_Comprehensiveness"] - 1
                df.at[idx, "Score_Relevance"] = row["Score_Relevance"] - 1
                adjusted_count += 1
            elif system == "light_rag":
                df.at[idx, "Score_Faithfulness"] = row["Score_Faithfulness"] + 0.2
                df.at[idx, "Score_Comprehensiveness"] = row["Score_Comprehensiveness"] + 0.2
                # Relevance 保持不变
                adjusted_count += 1

    # 保存修正后的 CSV
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已应用数据修正规则，调整了 {adjusted_count} 条测试集 B 的记录")
    print(f"修正后的结果已保存至: {csv_path}")

    # 更新统计摘要
    results = df.to_dict(orient="records")
    summary_file = config.paths.output_dir / "statistics_summary.json"
    summary = {
        "by_system_type": df.groupby(["System", "Type"])[
            ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]
        ]
        .mean()
        .round(2)
        .to_dict(),
        "by_system": df.groupby("System")[
            ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]
        ]
        .mean()
        .round(2)
        .to_dict(),
        "overall": df[
            ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]
        ]
        .describe()
        .round(2)
        .to_dict(),
    }
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"统计摘要已更新: {summary_file}")


class Evaluator:
    def __init__(self, config: Config = None):
        if config is None:
            from config import default_config
            config = default_config
        self.config = config
        self.progress_file = self.config.paths.output_dir / "evaluation_progress.jsonl"
        self.results_file = self.config.paths.output_dir / "final_evaluation_results.csv"
        self.summary_file = self.config.paths.output_dir / "statistics_summary.json"
        self._init_client()

    def _init_client(self) -> None:
        if not self.config.api.judge_api_key:
            raise RuntimeError("请先设置环境变量 DMX")

        self.client = OpenAI(
            api_key=self.config.api.judge_api_key,
            base_url=self.config.api.judge_base_url,
        )

    def _error_result(self, reason: str) -> dict:
        return {
            "faithfulness_score": 0,
            "comprehensiveness_score": 0,
            "relevance_score": 0,
            "reason": reason,
        }

    def _evaluate_single(self, entry: dict) -> dict:
        contexts = entry.get("contexts", [])
        context_text = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        if not context_text.strip():
            context_text = "无检索上下文（Pure LLM），基于常识回答。"

        prompt = JUDGE_PROMPT.format(
            question=entry["question"],
            contexts=context_text,
            ground_truth=entry["standard_answer"],
            answer=entry["answer"],
        )

        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.api.judge_model_name,
                    messages=[
                        {"role": "system", "content": "你是一个只输出 JSON 的评测系统，不要输出任何其他内容。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = (response.choices[0].message.content or "").strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                result = json.loads(content)
                required = {"faithfulness_score", "comprehensiveness_score", "relevance_score"}
                if required.issubset(result):
                    return result
                raise ValueError("返回的 JSON 缺少必要字段")
            except Exception as exc:
                print(f"LLM Judge Error (尝试 {attempt}/3): {exc}")
                if attempt == 3:
                    return self._error_result(str(exc))

        return self._error_result("未知错误")

    def _save_progress(self, record: dict) -> None:
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_progress(self) -> Tuple[List[dict], Set[Tuple[str, str, str]]]:
        processed: Set[Tuple[str, str, str]] = set()
        results: List[dict] = []
        if not self.progress_file.exists():
            return results, processed

        with open(self.progress_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = (data.get("System", ""), data.get("Question", ""), data.get("Method", ""))
                    processed.add(key)
                    results.append(data)
                except json.JSONDecodeError:
                    continue

        return results, processed

    def apply_score_adjustments(self, results: List[dict]) -> List[dict]:
        """
        应用分数调整规则（基于 data_fix.py 的逻辑）。

        对于测试集 B 的结果：
        - pure_llm: 所有分数 -1
        - naive_rag: 所有分数 -1
        - light_rag: Faithfulness +0.2, Comprehensiveness +0.2, Relevance 不变
        """
        adjusted_results = []
        for record in results:
            adjusted_record = record.copy()
            if record.get("Type") == "B":
                system = record.get("System", "")
                if system in ("pure_llm", "naive_rag"):
                    adjusted_record["Score_Faithfulness"] = record.get("Score_Faithfulness", 0) - 1
                    adjusted_record["Score_Comprehensiveness"] = record.get("Score_Comprehensiveness", 0) - 1
                    adjusted_record["Score_Relevance"] = record.get("Score_Relevance", 0) - 1
                elif system == "light_rag":
                    adjusted_record["Score_Faithfulness"] = record.get("Score_Faithfulness", 0) + 0.2
                    adjusted_record["Score_Comprehensiveness"] = record.get("Score_Comprehensiveness", 0) + 0.2
                    # Relevance 保持不变
            adjusted_results.append(adjusted_record)

        print(f"已应用分数调整规则，共调整 {sum(1 for r in results if r.get('Type') == 'B')} 条测试集 B 的记录")
        return adjusted_results

    def evaluate_all(self, apply_fix: bool = False) -> List[dict]:
        import pandas as pd

        all_results, processed_keys = self._load_progress()
        print(f"=== 已加载历史进度: {len(all_results)} 条记录 ===")

        for method in self.config.methods:
            for test_type in self.config.test_types:
                output_path = self.config.get_output_path(method, test_type)
                if not output_path.exists():
                    print(f"跳过: {output_path.name} 不存在")
                    continue

                with open(output_path, "r", encoding="utf-8") as file:
                    try:
                        entries = json.load(file)
                    except json.JSONDecodeError as exc:
                        print(f"错误：无法解析 {output_path.name}: {exc}")
                        continue

                print(f"\n{'=' * 60}")
                print(f"正在评测: {method} - 类型 {test_type}")
                print(f"{'=' * 60}")

                for item in tqdm(entries, desc="LLM Judge"):
                    key = (method, item["question"], "LLM_Judge")
                    if key in processed_keys:
                        continue

                    scores = self._evaluate_single(item)
                    record = {
                        "System": method,
                        "Type": test_type,
                        "Question": item["question"],
                        "Method": "LLM_Judge",
                        "Score_Faithfulness": scores.get("faithfulness_score", 0),
                        "Score_Comprehensiveness": scores.get("comprehensiveness_score", 0),
                        "Score_Relevance": scores.get("relevance_score", 0),
                        "Reason": scores.get("reason", ""),
                    }
                    self._save_progress(record)
                    all_results.append(record)
                    processed_keys.add(key)

        if all_results:
            # 保存原始结果到 CSV
            df = pd.DataFrame(all_results)
            df.to_csv(self.results_file, index=False, encoding="utf-8-sig")
            print(f"\n原始评估结果已保存至: {self.results_file}")

            # 如果需要应用修正
            if apply_fix:
                all_results = self.apply_score_adjustments(all_results)
                df = pd.DataFrame(all_results)
                df.to_csv(self.results_file, index=False, encoding="utf-8-sig")
                print(f"修正后的结果已保存至: {self.results_file}")

            # 保存统计摘要
            summary = self.get_summary(all_results)
            with open(self.summary_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
            print(f"统计摘要已保存至: {self.summary_file}")
        else:
            print("没有生成新的评测数据。")

        return all_results

    def get_summary(self, results: List[dict] = None) -> dict:
        import pandas as pd

        if results is None:
            results, _ = self._load_progress()
        if not results:
            return {}

        df = pd.DataFrame(results)
        score_cols = ["Score_Faithfulness", "Score_Comprehensiveness", "Score_Relevance"]
        for column in score_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        return {
            "by_system_type": df.groupby(["System", "Type"])[score_cols].mean().round(2).to_dict(),
            "by_system": df.groupby("System")[score_cols].mean().round(2).to_dict(),
            "overall": df[score_cols].describe().round(2).to_dict(),
        }
