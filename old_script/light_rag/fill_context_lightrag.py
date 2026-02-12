import requests
import json
import os
from tqdm import tqdm

# LightRAG 服务地址
URL = "http://127.0.0.1:9621/query"


def get_lightrag_context(question, session):
    """
    调用 LightRAG API 获取检索到的上下文 (Context)。
    通过设置 'only_need_context': True 来获取原始检索内容。
    """
    payload = {
        "query": question,
        "mode": "mix",  # 保持和你生成答案时一致的模式 (mix/local/global)
        "only_need_context": True,
        "top_k":5,
        "chunk_top_k":5
    }

    try:
        response = session.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # LightRAG 返回的 context 通常是一个拼接好的长字符串
        # 为了适配 Ragas，我们需要将其放入列表
        context_text = result.get("response", "")
        if not context_text:
            return []
        return [context_text]

    except Exception as e:
        print(f"Error retrieving context for '{question}': {e}")
        return []


def process_file(file_path):
    print(f"正在处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 使用 Session 提高 HTTP 请求效率
    with requests.Session() as session:
        for item in tqdm(data, desc="Fetching Contexts"):
            # 如果已经有 contexts 且不为空，跳过
            if "contexts" in item and item["contexts"]:
                continue

            question = item.get("question", "")
            if not question:
                continue

            # 获取 Context 并回填
            contexts = get_lightrag_context(question, session)
            item["contexts"] = contexts

    # 覆盖保存
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"完成！已更新 {file_path}\n")


if __name__ == "__main__":
    # 请确保这里的文件名是你 LightRAG 生成的输出文件
    # 假设 LightRAG 生成的是 output_A.json 和 output_B.json
    # 如果你的文件名不同，请修改此处
    files_to_process = [
        "testset/output_A.json",
        "testset/output_B.json"
    ]

    for fp in files_to_process:
        process_file(fp)