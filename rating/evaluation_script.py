import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI

# ================= 配置部分 =================
# 1. API Key 配置
API_KEY = os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZAI_API_KEY 或 OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = API_KEY

# 2. API 配置
API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4"  # 裁判模型

# 3. 定义文件路径模式
DATA_DIR = "testset"
SYSTEMS = ["pure_llm", "naive_rag", "light_rag"]
TYPES = ["A", "B"]

# ================= LLM Judge 部分 =================
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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


def evaluate_with_llm(entry):
    """使用自定义 Prompt 让大模型打分"""
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
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个只输出 JSON 的评测系统，不要输出任何其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            # 尝试清理可能的 markdown 标记
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(content)
            
            # 验证必要字段
            required_fields = ["faithfulness_score", "comprehensiveness_score", "relevance_score"]
            if all(field in result for field in required_fields):
                return result
            else:
                raise ValueError("返回的 JSON 缺少必要字段")
                
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            print(f"返回内容: {content[:200]}")
            if attempt == max_retries - 1:
                return {
                    "faithfulness_score": 0,
                    "comprehensiveness_score": 0,
                    "relevance_score": 0,
                    "reason": f"JSON解析失败: {str(e)}"
                }
        except Exception as e:
            print(f"LLM Judge Error (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "faithfulness_score": 0,
                    "comprehensiveness_score": 0,
                    "relevance_score": 0,
                    "reason": f"Error: {str(e)}"
                }


# ================= 工具函数：实时保存 =================
def save_progress(record, filename="evaluation_progress.jsonl"):
    """实时追加写入一条记录"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_progress(filename="evaluation_progress.jsonl"):
    """读取已完成的进度"""
    processed = set()
    results = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (data["System"], data["Question"], data["Method"])
                    processed.add(key)
                    results.append(data)
                except:
                    continue
    return results, processed


# ================= 主流程 =================
def main():
    # 1. 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录 {DATA_DIR} 不存在！")
        print(f"请创建目录并放入数据文件")
        return
    
    # 2. 读取之前的进度
    progress_file = "output/evaluation_progress.jsonl"
    all_results, processed_keys = load_progress(progress_file)
    print(f"=== 已加载历史进度: {len(all_results)} 条记录 ===")
    print("=== 开始评测流程 ===\n")

    for sys_name in SYSTEMS:
        for q_type in TYPES:
            filename = f"{sys_name}_output_{q_type}.json"
            filepath = os.path.join(DATA_DIR, filename)

            if not os.path.exists(filepath):
                print(f"跳过: {filename} 不存在")
                continue

            print(f"\n{'='*60}")
            print(f"正在评测: {sys_name} - 类型 {q_type}")
            print(f"{'='*60}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"错误：无法读取 {filename}: {e}")
                continue

            if not data:
                print(f"警告：{filename} 为空")
                continue

            # --- LLM Judge ---
            print(f"  > 正在运行 Custom LLM Judge ({len(data)} 条)...")
            for item in tqdm(data, desc="LLM Judge"):
                if (sys_name, item['question'], "LLM_Judge") in processed_keys:
                    continue

                scores = evaluate_with_llm(item)

                record = {
                    "System": sys_name,
                    "Type": q_type,
                    "Question": item['question'],
                    "Method": "LLM_Judge",
                    "Score_Faithfulness": scores.get('faithfulness_score', 0),
                    "Score_Comprehensiveness": scores.get('comprehensiveness_score', 0),
                    "Score_Relevance": scores.get('relevance_score', 0),
                    "Reason": scores.get('reason', '')
                }

                save_progress(record, progress_file)
                all_results.append(record)
                processed_keys.add((sys_name, item['question'], "LLM_Judge"))

    # ================= 绘图与保存 =================
    if not all_results:
        print("\n没有生成评测数据。")
        return

    df = pd.DataFrame(all_results)
    df.to_csv("output/final_evaluation_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print(f"评测完成！共 {len(df)} 条记录")
    print(f"结果已保存至: output/final_evaluation_results.csv")
    print(f"{'='*60}\n")

    # 绘图逻辑
    df_judge = df.copy()

    if not df_judge.empty:
        # 转换列为数字类型
        cols = ['Score_Faithfulness', 'Score_Comprehensiveness', 'Score_Relevance']
        for col in cols:
            df_judge[col] = pd.to_numeric(df_judge[col], errors='coerce').fillna(0)

        # 设置绘图样式
        sns.set_theme(style="whitegrid")
        
        # 尝试设置中文字体（可选）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

        # 1. 完整性对比
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_judge, x="System", y="Score_Comprehensiveness", hue="Type", palette="viridis")
        plt.title("Comprehensiveness Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
        plt.xlabel("System", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 10.5)
        plt.legend(title="Type")
        plt.tight_layout()
        plt.savefig("output/chart_comprehensiveness.png", dpi=300)
        print("✓ 图表已生成: output/chart_comprehensiveness.png")

        # 2. 忠实度对比
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_judge, x="System", y="Score_Faithfulness", hue="Type", palette="magma")
        plt.title("Faithfulness Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
        plt.xlabel("System", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 10.5)
        plt.legend(title="Type")
        plt.tight_layout()
        plt.savefig("output/chart_faithfulness.png", dpi=300)
        print("✓ 图表已生成: output/chart_faithfulness.png")
        
        # 3. 相关性对比
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_judge, x="System", y="Score_Relevance", hue="Type", palette="coolwarm")
        plt.title("Relevance Comparison (1-10 Scale)", fontsize=14, fontweight='bold')
        plt.xlabel("System", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 10.5)
        plt.legend(title="Type")
        plt.tight_layout()
        plt.savefig("output/chart_relevance.png", dpi=300)
        print("✓ 图表已生成: output/chart_relevance.png")

        # 4. 统计摘要
        print("\n" + "="*60)
        print("评测统计摘要")
        print("="*60)
        summary = df_judge.groupby(['System', 'Type'])[cols].mean()
        print(summary.round(2))
        print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
