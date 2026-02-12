import os
import json
from embedchain import App
from tqdm import tqdm

# === 1. 配置部分 (必须与你生成答案时的配置完全一致) ===
# 环境变量配置
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Embedchain 配置
config = {
    'llm': {
        'provider': 'openai',
        'config': {
            'model': 'qwen3-max',
        }
    },
    'embedder': {
        'provider': 'openai',
        'config': {
            'model': 'text-embedding-v4',
            'vector_dimension': 1024,
        }
    },
    'vectordb': {
        'provider': 'chroma',
        'config': {
            'collection_name': 'orchard-pest-rag',
            'dir': 'db',
            'batch_size': 10,  # DashScope text-embedding-v4 限制每批最多 10 条
        }
    }
}

# === 2. 初始化 App ===
print("正在初始化 Embedchain App 并加载向量数据库...")
app = App.from_config(config=config)

# === 2.1 检查向量数据库是否为空，若为空则重置去重记录并重新导入文档 ===
txt_path = "./经济果林病虫害防治手册.txt"
db_count = app.db.count()
if db_count == 0:
    print(f"向量数据库为空（{db_count} 条），正在重置并导入文档: {txt_path} ...")
    if os.path.exists(txt_path):
        # 先 reset，清除 embedchain 内部的去重记录和向量库，防止被跳过
        app.reset()
        # reset 后需要重新初始化 App（reset 会清空 collection）
        app = App.from_config(config=config)
        app.add(txt_path)
        new_count = app.db.count()
        print(f"文档导入完成！当前文档数: {new_count}")
        if new_count == 0:
            print("警告: 导入后文档数仍为 0，请检查 OPENAI_API_KEY 和 embedding API 是否可用。")
    else:
        print(f"错误: 找不到文件 {txt_path}，请确保文件在当前目录下。")
else:
    print(f"向量数据库已有 {db_count} 条文档，跳过导入。")


def process_file(file_path):
    print(f"正在处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Retrieving Chunks"):
        # 如果已经有 contexts 且不为空，跳过
        if "contexts" in item and item["contexts"]:
            continue

        question = item.get("question", "")
        if not question:
            continue

        try:
            # embedchain 的 search 方法返回 list[dict]，每个 dict 含 'context' 和 'metadata'
            results = app.search(question, num_documents=5)

            # 提取每个结果中的 context 文本
            retrieved_chunks = [r["context"] for r in results if "context" in r]

            item["contexts"] = retrieved_chunks

        except Exception as e:
            print(f"Error retrieving for '{question}': {e}")
            item["contexts"] = []

    # 覆盖保存
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"完成！已更新 {file_path}\n")


if __name__ == "__main__":
    # 请确保这里的文件名是你 Naive RAG 生成的输出文件
    # 假设 Naive RAG 也跑了 A 和 B，或者你有专门的文件名
    # 请务必区分 LightRAG 的输出文件和 Naive RAG 的输出文件，不要混淆
    files_to_process = [
        "testset/output_A.json",  # 请修改为你实际的 Naive RAG 输出文件名
        "testset/output_B.json"
    ]

    for fp in files_to_process:
        process_file(fp)