
import json
import os
import traceback

from openai import OpenAI


def build_prompt(question: str, max_chars: int) -> str:
    # 控制回答长度并只输出答案
    return (
        f"你是一个简洁的助手。请用中文回答以下问题,"
        "只返回答案内容，也不要包含标点前后的空行。\n\n"
        f"问题：{question}\n\n只返回答案："
    )


def get_response(client: OpenAI, prompt: str, temperature: float = 0.0,model_name: str = "qwen3-max" ) -> str:
    response = client.chat.completions.create(
    model=model_name,
                    messages=[
                        {"role": "system", "content": "你需要扮演一个果园病虫害的专家"},
                        {"role": "user", "content": prompt},
                    ],
        temperature=temperature,
        stream=False,  
    )
    return response.choices[0].message.content

class records:
    def __init__(self):
        self.data = []

    def add_record(self, question, answer, standard_answer):
        self.data.append({"question": question, "answer": answer, "standard_answer": standard_answer})





if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 OPENAI_API_KEY，然后重试。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 数据集 A 测试
    with open("testset/A.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

        results = records()
        for item in test_data:
            question = item.get("问题", "")
            standard_answer = item.get("标准答案", "")
            try:
                prompt = build_prompt(question, max_chars=100)
                answer = get_response(client, prompt)
                print(f"Q: {question}\nA: {answer}\n")
            except Exception as e:
                print(f"Error processing question: {question}\n{e}")
                traceback.print_exc()
                answer = "Error occurred during processing."
            results.add_record(question, answer, standard_answer)
        # 将结果写入到 testset/output_A.json
        with open("testset/output_A.json", "w", encoding="utf-8") as f:
            json.dump(results.data, f, ensure_ascii=False, indent=2)


    # 数据集 B 测试
    with open("testset/B.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
        results = records()
        for item in test_data:
            question = item.get("问题", "")
            standard_answer = item.get("标准答案", "")
            try:
                prompt= build_prompt(question, max_chars=350)
                answer = get_response(client, prompt)
                print(f"Q: {question}\nA: {answer}\n")
            except Exception as e:
                print(f"Error processing question: {question}")
                traceback.print_exc()
                answer = "Error occurred during processing."
            results.add_record(question, answer, standard_answer)
        # 将结果写入到 testset/output_B.json
        with open("testset/output_B.json", "w", encoding="utf-8") as f:
            json.dump(results.data, f, ensure_ascii=False, indent=2)

    print("Testing completed.")
