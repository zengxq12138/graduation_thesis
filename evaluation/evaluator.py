"""
LLM Judge 评估模块
"""
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

# 评分 Prompt 模板
JUDGE_PROMPT = """
你是一位植物病理学专家和严厉的阅卷老师。请根据以下三个维度给 AI 的回答打分（1-10分）。

### 评分维度：
1. **忠实度 (Faithfulness)**: 回答是否严格基于参考资料(Context)？如果没有参考资料，请检查是否存在幻觉。
2. **回答完整性 (Comprehensiveness)**: 对比【标准答案】，AI 是否涵盖了所有关键要点？(这是核心指标)
3. **答案有用性 (Relevance)**: 回答是否直接解决了问题，没有废话？

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

注意：必须严格返回 JSON 格式，不要有任何额外文字。
"""


class Evaluator:
    """LLM Judge 评估器"""

    def __init__(self, config: Config = None):
        if config is None:
            from config import default_config
            config = default_config
        self.config = config
        self._init_client()

        # 进度文件路径
        self.progress_file = config.paths.output_dir / "evaluation_progress.jsonl"
        self.results_file = config.paths.output_dir / "final_evaluation_results.csv"

    def _init_client(self):
        """初始化评分模型客户端"""
        api_key = self.config.api.judge_api_key
        if not api_key:
            raise ValueError("请设置环境变量 ZAI_API_KEY 或 OPENAI_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api.judge_base_url
        )

    def _evaluate_single(self, entry: dict) -> dict:
        """评估单条记录"""
        ctx = entry.get("contexts", [])
        ctx_str = "\n".join(ctx) if isinstance(ctx, list) else str(ctx)
        if not ctx_str or ctx == [""]:
            ctx_str = "无检索上下文 (Pure LLM)，基于常识回答"

        prompt = JUDGE_PROMPT.format(
            question=entry['question'],
            contexts=ctx_str,
            ground_truth=entry['standard_answer'],
            answer=entry['answer']
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.api.judge_model_name,
                    messages=[
                        {"role": "system", "content": "你是一个只输出 JSON 的评测系统，不要输出任何其他内容。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content.strip()
                # 清理可能的 markdown 标记
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()

                result = json.loads(content)
                print(result)

                # 验证必要字段
                required_fields = ["faithfulness_score", "comprehensiveness_score", "relevance_score"]
                if all(field in result for field in required_fields):
                    return result
                else:
                    raise ValueError("返回的 JSON 缺少必要字段")

            except json.JSONDecodeError as e:
                print(f"JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return self._error_result(f"JSON解析失败: {str(e)}")
            except Exception as e:
                print(f"LLM Judge Error (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return self._error_result(f"Error: {str(e)}")

        return self._error_result("未知错误")

    def _error_result(self, reason: str) -> dict:
        """返回错误结果"""
        return {
            "faithfulness_score": 0,
            "comprehensiveness_score": 0,
            "relevance_score": 0,
            "reason": reason
        }

    def _save_progress(self, record: dict):
        """实时追加保存一条记录"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_progress(self) -> Tuple[List[dict], Set[Tuple]]:
        """加载已完成的进度"""
        processed = set()
        results = []
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        key = (data["System"], data["Question"], data["Method"])
                        processed.add(key)
                        results.append(data)
                    except:
                        continue
        return results, processed

    def evaluate_all(self) -> List[dict]:
        """评估所有方法的所有测试集"""
        import pandas as pd

        # 加载历史进度
        all_results, processed_keys = self._load_progress()
        print(f"=== 已加载历史进度: {len(all_results)} 条记录 ===")
        print("=== 开始评测流程 ===\n")

        for method in self.config.methods:
            for test_type in self.config.test_types:
                output_path = self.config.get_output_path(method, test_type)

                if not output_path.exists():
                    print(f"跳过: {output_path.name} 不存在")
                    continue

                print(f"\n{'=' * 60}")
                print(f"正在评测: {method} - 类型 {test_type}")
                print(f"{'=' * 60}")

                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"错误：无法读取 {output_path.name}: {e}")
                    continue

                if not data:
                    print(f"警告：{output_path.name} 为空")
                    continue

                # LLM Judge 评分
                print(f"  > 正在运行 LLM Judge ({len(data)} 条)...")
                for item in tqdm(data, desc="LLM Judge"):
                    key = (method, item['question'], "LLM_Judge")
                    if key in processed_keys:
                        continue

                    scores = self._evaluate_single(item)

                    record = {
                        "System": method,
                        "Type": test_type,
                        "Question": item['question'],
                        "Method": "LLM_Judge",
                        "Score_Faithfulness": scores.get('faithfulness_score', 0),
                        "Score_Comprehensiveness": scores.get('comprehensiveness_score', 0),
                        "Score_Relevance": scores.get('relevance_score', 0),
                        "Reason": scores.get('reason', '')
                    }

                    self._save_progress(record)
                    all_results.append(record)
                    processed_keys.add(key)

        # 保存最终结果为 CSV
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(self.results_file, index=False, encoding='utf-8-sig')
            print(f"\n{'=' * 60}")
            print(f"评测完成！共 {len(df)} 条记录")
            print(f"结果已保存至: {self.results_file}")
            print(f"{'=' * 60}\n")
        else:
            print("\n没有生成评测数据。")

        return all_results

    def get_summary(self, results: List[dict] = None) -> dict:
        """获取评测摘要"""
        import pandas as pd

        if results is None:
            results, _ = self._load_progress()

        if not results:
            return {}

        df = pd.DataFrame(results)
        score_cols = ['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']

        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        summary = {
            "by_system_type": df.groupby(['System', 'Type'])[score_cols].mean().round(2).to_dict(),
            "by_system": df.groupby('System')[score_cols].mean().round(2).to_dict(),
            "overall": df[score_cols].describe().round(2).to_dict()
        }

        return summary
